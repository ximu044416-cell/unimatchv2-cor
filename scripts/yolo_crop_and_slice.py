import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 终极配置区域 =================
# 刚刚用迁移脚本建好的、包含 train/val/unlabel 的新基地
INPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_YOLO")

# ★ 最终的终点站：准备喂给 DINOv2 的 NPY 纯净数据集 ★
OUTPUT_DIR = Path(r"F:\Dinov2_data\final_slices_YOLO_cor")

ROBUST_PERCENTILE = (0.5, 99.5)
CONTEXT_EXPANSION = 2
MARGIN = 35  # YOLO 框向外的物理缓冲边距


# ===========================================

def volume_robust_norm(volume_3d):
    """【100% 原版复刻】局部 Z 轴的 Min-Max 归一化 (排除背景极值干扰)"""
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


def get_global_yolo_tube(yolo_dir, img_w, img_h):
    """【全新升级】读取病人所有的 YOLO TXT，生成统一的 3D 外接管道"""
    global_xmin, global_ymin = float('inf'), float('inf')
    global_xmax, global_ymax = 0, 0
    valid_boxes_found = False

    if not yolo_dir.exists():
        return None

    for txt_file in yolo_dir.glob("*.txt"):
        if txt_file.stat().st_size > 0:
            with open(txt_file, 'r') as f:
                content = f.read().strip()
                if content:
                    coords = list(map(int, content.split(',')))
                    global_xmin = min(global_xmin, coords[0])
                    global_ymin = min(global_ymin, coords[1])
                    global_xmax = max(global_xmax, coords[2])
                    global_ymax = max(global_ymax, coords[3])
                    valid_boxes_found = True

    if not valid_boxes_found:
        return None

    # 施加 Margin 并加上物理边界锁，绝不越界报错！
    min_x = max(0, global_xmin - MARGIN)
    min_y = max(0, global_ymin - MARGIN)
    max_x = min(img_w, global_xmax + MARGIN)
    max_y = min(img_h, global_ymax + MARGIN)

    return min_x, min_y, max_x, max_y


def process_one_case(case_dir, split_name, save_root):
    case_id = case_dir.name

    # 兼容大小写的灵活读取
    def get_nii_path(names):
        for name in names:
            p = case_dir / name
            if p.exists(): return p
        return None

    path_w = get_nii_path(["water.nii.gz", "Water.nii.gz"])
    path_t1 = get_nii_path(["t1.nii.gz", "T1.nii.gz"])
    path_f = get_nii_path(["fat.nii.gz", "Fat.nii.gz"])
    path_l = get_nii_path(["label.nii.gz", "Mask.nii.gz"])
    yolo_dir = case_dir / "yolo_boxes"

    if not all([path_w, path_t1, path_f, path_l]):
        return False, 0, "缺少 NIfTI 序列文件"

    # 读取 3D 矩阵
    arr_w = sitk.GetArrayFromImage(sitk.ReadImage(str(path_w), sitk.sitkFloat32))
    arr_t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(path_t1), sitk.sitkFloat32))
    arr_f = sitk.GetArrayFromImage(sitk.ReadImage(str(path_f), sitk.sitkFloat32))
    arr_l = sitk.GetArrayFromImage(sitk.ReadImage(str(path_l), sitk.sitkUInt8))

    z_len, img_h, img_w = arr_w.shape

    # ================= 1. X-Y 轴：获取统一 3D 管道 =================
    tube_coords = get_global_yolo_tube(yolo_dir, img_w, img_h)
    if tube_coords is None:
        return False, 0, "未找到有效的 YOLO 坐标"
    min_x, min_y, max_x, max_y = tube_coords

    # ================= 2. Z 轴：严格复刻你的分流逻辑 =================
    if split_name in ['train', 'val']:
        # 【100% 原版】寻找有病灶的 Z 轴索引并上下扩充
        label_z_indices = np.where(np.max(arr_l, axis=(1, 2)) > 0)[0]
        if len(label_z_indices) == 0:
            return False, 0, "🚨 该病人标签为空！已安全跳过。"

        start_idx = max(0, label_z_indices.min() - CONTEXT_EXPANSION)
        end_idx = min(z_len, label_z_indices.max() + CONTEXT_EXPANSION + 1)
    else:
        # 【最高指令】Unlabel 组：直接提取全部！不删减任何一层！
        start_idx = 0
        end_idx = z_len

    # ================= 3. 物理裁剪 (Z, Y, X 三维同步裁剪) =================
    crop_w = arr_w[start_idx:end_idx, min_y:max_y, min_x:max_x]
    crop_t1 = arr_t1[start_idx:end_idx, min_y:max_y, min_x:max_x]
    crop_f = arr_f[start_idx:end_idx, min_y:max_y, min_x:max_x]
    crop_l = arr_l[start_idx:end_idx, min_y:max_y, min_x:max_x]

    # 【100% 原版】多模态堆叠 & 局部 Z 轴归一化
    volume_raw = np.stack([crop_w, crop_t1, crop_f], axis=-1)
    volume_norm = volume_robust_norm(volume_raw)

    # 建立对应分类的输出目录
    out_case_dir = save_root / split_name / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    # ================= 4. 保存为纯净 NPY 切片 =================
    for i, actual_z in enumerate(range(start_idx, end_idx)):
        img_slice = volume_norm[i]
        mask_slice = crop_l[i]

        np.save(out_case_dir / f"{case_id}_slice_{actual_z:03d}_data.npy", img_slice)
        np.save(out_case_dir / f"{case_id}_slice_{actual_z:03d}_label.npy", mask_slice)
        saved_count += 1

    return True, saved_count, "Success"


def main():
    if not INPUT_DIR.exists():
        print(f"❌ 找不到输入总目录: {INPUT_DIR}")
        return

    print("\n" + "=" * 55)
    print("🚀 启动终极数据兵工厂：[YOLO 3D管道 + 纯净归一化]")
    print("=" * 55 + "\n")

    stats = {'train': 0, 'val': 0, 'unlabel': 0}

    for split_name in ['train', 'val', 'unlabel']:
        split_dir = INPUT_DIR / split_name
        if not split_dir.exists(): continue

        case_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        if not case_dirs: continue

        print(f"\n📂 正在加工 [{split_name.upper()}] 组数据...")

        for case_dir in tqdm(case_dirs, desc=f"切片进度"):
            success, count, msg = process_one_case(case_dir, split_name, OUTPUT_DIR)
            if success:
                stats[split_name] += count
            else:
                print(f"\n⚠️ {case_dir.name} 跳过: {msg}")

    print("\n" + "=" * 50)
    print("🎉 终极纯净版数据集彻底杀青！")
    print(f"   - 🏋️ Train 组 (原版扩充法) 生成: {stats['train']} 张")
    print(f"   - 🧪 Val 组 (原版扩充法) 生成: {stats['val']} 张")
    print(f"   - 👻 Unlabel 组 (全序列无损) 生成: {stats['unlabel']} 张")
    print(f"   - 📦 终极产出: {sum(stats.values())} 张极致纯净图！")
    print(f"👉 它们已静静躺在 {OUTPUT_DIR}，等待大模型的召唤！")


if __name__ == "__main__":
    main()