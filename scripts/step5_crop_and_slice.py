import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= 终极配置区域 =================
INPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_FINAL")
OUTPUT_DIR = Path(r"F:\Dinov2_data\final_slices_cor")
EXCEL_PATH = Path(r"F:\Dinov2_data\ALL_COR_N4\分类.xlsx")

ROBUST_PERCENTILE = (0.5, 99.5)
CONTEXT_EXPANSION = 2


# ===========================================

def volume_robust_norm(volume_3d):
    """局部 Z 轴的 Min-Max 归一化 (排除背景极值干扰)"""
    normalized_vol = np.zeros_like(volume_3d, dtype=np.float32)

    for c in range(3):
        channel_data = volume_3d[..., c].astype(np.float32)
        valid_pixels = channel_data[channel_data > 0]
        if len(valid_pixels) > 0:
            p_min, p_max = np.percentile(valid_pixels, ROBUST_PERCENTILE)
        else:
            p_min, p_max = 0, 0

        channel_data = np.clip(channel_data, p_min, p_max)
        div = p_max - p_min

        if div > 0:
            channel_data = (channel_data - p_min) / div
        else:
            channel_data = np.zeros_like(channel_data)

        normalized_vol[..., c] = channel_data

    return normalized_vol


def process_one_case(case_dir, is_labeled_group, save_root):
    case_id = case_dir.name

    path_w = case_dir / "water.nii.gz"
    path_t1 = case_dir / "t1.nii.gz"
    path_f = case_dir / "fat.nii.gz"
    path_l = case_dir / "label.nii.gz"
    path_bbox = case_dir / "pelvis_bbox_mask.nii.gz"

    if not all(p.exists() for p in [path_w, path_t1, path_f, path_l, path_bbox]):
        return False, 0, "缺少必要文件"

    arr_w = sitk.GetArrayFromImage(sitk.ReadImage(str(path_w), sitk.sitkFloat32))
    arr_t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(path_t1), sitk.sitkFloat32))
    arr_f = sitk.GetArrayFromImage(sitk.ReadImage(str(path_f), sitk.sitkFloat32))
    arr_l = sitk.GetArrayFromImage(sitk.ReadImage(str(path_l), sitk.sitkUInt8))
    arr_bbox = sitk.GetArrayFromImage(sitk.ReadImage(str(path_bbox), sitk.sitkUInt8))

    b_z, b_y, b_x = np.where(arr_bbox > 0)
    if len(b_y) == 0:
        return False, 0, "BBox 为空"

    min_y, max_y = b_y.min(), b_y.max() + 1
    min_x, max_x = b_x.min(), b_x.max() + 1
    z_len = arr_w.shape[0]

    # ================= 核心分流逻辑 (提前计算 Z 轴边界) =================
    if is_labeled_group:
        # 高速向量化操作：寻找有病灶的 Z 轴索引
        label_z_indices = np.where(np.max(arr_l, axis=(1, 2)) > 0)[0]
        if len(label_z_indices) == 0:
            return False, 0, "🚨 警告：该 Label 组病人标签为空！已安全跳过。"

        start_idx = max(0, label_z_indices.min() - CONTEXT_EXPANSION)
        end_idx = min(z_len, label_z_indices.max() + CONTEXT_EXPANSION + 1)
        out_case_dir = save_root / "label" / case_id
    else:
        # Unlabel 组直接取 BBox 的 Z 轴边界
        start_idx = b_z.min()
        end_idx = b_z.max() + 1
        out_case_dir = save_root / "unlabel" / case_id

    # ================= 物理裁剪 (Z, Y, X 三维同步裁剪) =================
    crop_w = arr_w[start_idx:end_idx, min_y:max_y, min_x:max_x]
    crop_t1 = arr_t1[start_idx:end_idx, min_y:max_y, min_x:max_x]
    crop_f = arr_f[start_idx:end_idx, min_y:max_y, min_x:max_x]
    crop_l = arr_l[start_idx:end_idx, min_y:max_y, min_x:max_x]

    # 多模态堆叠 & 局部归一化 (大幅节省内存，归一化更精准)
    volume_raw = np.stack([crop_w, crop_t1, crop_f], axis=-1)
    volume_norm = volume_robust_norm(volume_raw)

    out_case_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    # 保存切片 (i 是截取数组的相对索引，actual_z 是图像真实的切片编号)
    for i, actual_z in enumerate(range(start_idx, end_idx)):
        img_slice = volume_norm[i]
        mask_slice = crop_l[i]

        np.save(out_case_dir / f"{case_id}_slice_{actual_z:03d}_data.npy", img_slice)
        np.save(out_case_dir / f"{case_id}_slice_{actual_z:03d}_label.npy", mask_slice)
        saved_count += 1

    return True, saved_count, "Success"


def main():
    if not INPUT_DIR.exists():
        print(f"❌ 找不到输入目录: {INPUT_DIR}")
        return

    if not EXCEL_PATH.exists():
        print(f"❌ 找不到 CSV 文件: {EXCEL_PATH}")
        return

    # O(1) 查找优化：使用 Set 集合
    df = pd.read_excel(EXCEL_PATH)
    if 'label' not in df.columns:
        print("❌ CSV 文件中没有找到名为 'label' 的列！")
        return

    labeled_cases_set = set(df['label'].dropna().astype(str).str.strip())

    (OUTPUT_DIR / "label").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "unlabel").mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])

    print("\n" + "=" * 50)
    print("🚀 启动双轨制兵工厂 V2.0：[三维物理裁剪 + 纯净归一化]")
    print(f"📝 发现 Label 组名单: {len(labeled_cases_set)} 例")
    print("=" * 50 + "\n")

    count_label = 0
    count_unlabel = 0

    for case_dir in tqdm(case_dirs, desc="正在加工切片"):
        case_id = case_dir.name
        is_labeled_group = (case_id in labeled_cases_set)

        success, count, msg = process_one_case(case_dir, is_labeled_group, OUTPUT_DIR)

        if success:
            if is_labeled_group:
                count_label += count
            else:
                count_unlabel += count
        else:
            print(f"\n⚠️ {case_id} 跳过: {msg}")

    print("\n" + "=" * 50)
    print("🎉 双轨大一统预处理彻底杀青！")
    print(f"   - 🎯 Label 组 (策略A) 生成切片: {count_label} 张")
    print(f"   - 🛡️ Unlabel 组 (策略B) 生成切片: {count_unlabel} 张")
    print(f"   - 📂 总计产出: {count_label + count_unlabel} 张！")


if __name__ == "__main__":
    main()