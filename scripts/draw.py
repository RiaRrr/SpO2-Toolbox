import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# ============================
# 配置
# ============================
csv_path = "/root/jjt/SpO2-Toolbox/runs/exp/SpO2_PhysNet_FFT_RGBA_raw_RR_SPO2_TaskRR_SizeW64_SizeH64_ClipLength900_DataTypeRaw_DataAugNone_LabelTypeRaw_Crop_faceFalse_BackendHC_Large_boxFalse_Large_size1.5_Dyamic_DetFalse_det_len32_Median_face_boxFalse/log/test_result_20251202_214429.csv"
save_dir = "plots"
segment_len = 300
max_segments = 5
# ============================

os.makedirs(save_dir, exist_ok=True)

def parse_tensor_str(t):
    """
    解析 CSV 中的 prediction/label 字段，支持多行 tensor。
    """
    if not isinstance(t, str):
        return None

    # 去掉换行、多个空格
    t_clean = t.replace("\n", " ").replace("\r", " ")
    t_clean = re.sub(r"\s+", " ", t_clean)

    # 提取最内层 tensor([...])
    match = re.search(r"tensor\(\s*\[([^\]]+)\]\s*\)", t_clean)
    if not match:
        return None

    arr_str = match.group(1)
    try:
        values = [float(x) for x in arr_str.split(",") if x.strip() != ""]
        return values
    except:
        return None


df = pd.read_csv(csv_path)

for idx, row in df.iterrows():

    pred = parse_tensor_str(row["prediction"])
    label = parse_tensor_str(row["label"])

    if pred is None or label is None:
        print(f"[Warning] parse failed at index {idx}")
        continue

    # 对齐长度
    L = min(len(pred), len(label))
    pred = pred[:L]
    label = label[:L]

    total_len = L
    num_segments = min(max_segments, total_len // segment_len)

    for s in range(num_segments):

        start = s * segment_len
        end = start + segment_len

        seg_pred = pred[start:end]
        seg_label = label[start:end]

        plt.figure(figsize=(12, 4))
        plt.plot(seg_label, label='GT Label', linewidth=1.2)
        plt.plot(seg_pred, label='Prediction', linewidth=1.2)
        plt.title(f"Index {idx} Segment {s} ({start}-{end})")
        plt.legend()
        plt.tight_layout()

        filename = f"{save_dir}/idx{idx}_seg{s}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

        print(f"Saved: {filename}")

print("Done!")
