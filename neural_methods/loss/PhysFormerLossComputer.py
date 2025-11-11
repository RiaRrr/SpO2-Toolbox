'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/blob/main/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    # Reshape the labels tensor to match the shape of inputs
    labels = labels.view(1, -1)
    
    # Compute the KL Div Loss
    criterion = nn.KLDivLoss(reduction='sum')
    loss = criterion(F.log_softmax(inputs, dim=-1), labels)
    return loss
 
class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        # Ensure all tensors are on the same device/dtype as output
        dev = output.device
        dtype = output.dtype
        two_pi_n_over_N = (2 * math.pi * torch.arange(0, N, dtype=dtype, device=dev)) / N
        hanning = torch.from_numpy(np.hanning(N)).to(device=dev, dtype=dtype).view(1, -1)

        k = k.to(device=dev, dtype=dtype)

        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).to(device=dev, dtype=dtype)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                            + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        # move bpm_range to same device/dtype
        bpm_range = bpm_range.to(device=output.device, dtype=output.dtype)
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4] Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator      
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=inputs.dtype, device=inputs.device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=inputs.dtype, device=inputs.device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=inputs.dtype, device=inputs.device)

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs, std):
        target_distribution = [normal_sampling(int(target), i, std) for i in range(40, 180)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.tensor(target_distribution, dtype=inputs.dtype, device=inputs.device)
        
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        
        bpm_range = torch.arange(40, 180, dtype=inputs.dtype, device=inputs.device)

        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        
        fre_distribution = ca/torch.sum(ca)
        loss_distribution_kl = kl_loss(fre_distribution, target_distribution)
        
        whole_max_val, whole_max_idx = ca.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        return loss_distribution_kl, F.cross_entropy(ca, (target-bpm_range[0]).view(1).type(torch.long)),  torch.abs(target[0]-bpm_range[0]-whole_max_idx)
        