import SimpleITK as sitk
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ================= 配置 =================
DATA_DIR = Path(r"F:\Dinov2_data\ALL_COR_N4")
CSV_PATH = Path(r"F:\Dinov2_data\global_geometry_report1.csv")
QC_OUTPUT_DIR = Path(r"F:\Dinov2_data\QC_130_Perfect_With_Label")


# =======================================

def normalize_image(img_arr):
    """使用 1% 和 99% 分位数进行截断归一化"""
    p1 = np.percentile(img_arr, 1)
    p99 = np.percentile(img_arr, 99)
    img_arr = np.clip(img_arr, p1, p99)
    # 防止除以0
    img_arr = (img_arr - p1) / (p99 - p1 + 1e-8) * 255.0
    return img_arr.astype(np.uint8)


def main():
    print("🚀 启动 130 例完美数据视觉质检 (RGB融合 + Label高亮) ...")
    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取 CSV，筛选 T1_Match? 为 True 的病人
    df = pd.read_csv(CSV_PATH)
    perfect_cases = df[df['T1_Match?'].astype(str).str.strip() == 'True']['Patient_ID'].tolist()

    print(f"✅ 找到 {len(perfect_cases)} 个 T1_Match=True 的病例。")

    for case_id in tqdm(perfect_cases, desc="Generating QC Images"):
        case_dir = DATA_DIR / str(case_id)
        path_w = case_dir / "water.nii.gz"
        path_t1 = case_dir / "t1.nii.gz"
        path_label = case_dir / "label.nii.gz"  # 新增：读取Label

        if not path_w.exists() or not path_t1.exists() or not path_label.exists():
            continue

        # 1. 读取图像
        img_w = sitk.ReadImage(str(path_w), sitk.sitkFloat32)
        img_t1 = sitk.ReadImage(str(path_t1), sitk.sitkFloat32)
        img_label = sitk.ReadImage(str(path_label), sitk.sitkUInt8)

        # 2. 转换为 Numpy 数组
        arr_w = sitk.GetArrayFromImage(img_w)
        arr_t1 = sitk.GetArrayFromImage(img_t1)
        arr_label = sitk.GetArrayFromImage(img_label)

        # 3. 寻找包含 Label 的中心层
        # (比单纯找几何中心更智能：尽量找有病灶的那一层展示)
        z_indices = np.where(arr_label > 0)[0]
        if len(z_indices) > 0:
            best_z = z_indices[len(z_indices) // 2]  # 病灶的中间层
        else:
            best_z = arr_w.shape[0] // 2  # 如果这人没有病灶(全0)，就取几何中间层

        slice_w = arr_w[best_z, :, :]
        slice_t1 = arr_t1[best_z, :, :]
        slice_label = arr_label[best_z, :, :]

        # 4. 灰度归一化
        norm_w = normalize_image(slice_w)
        norm_t1 = normalize_image(slice_t1)

        # 5. 创建 RGB 图像
        h, w = norm_w.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = norm_w  # R: Water (水肿会偏红)
        rgb[:, :, 1] = norm_t1  # G: T1 (骨皮质轮廓)

        # 6. 把 Label 叠加成高亮的蓝色
        label_mask = slice_label > 0
        rgb[label_mask, 2] = 255  # B通道拉满 (亮蓝色)
        # 可选：让红色通道也在label处变亮一点，这样会显示为洋红色/紫色，更醒目
        rgb[label_mask, 0] = np.maximum(rgb[label_mask, 0], 150)

        # 保存为 JPG
        img_pil = Image.fromarray(rgb)
        img_pil.save(QC_OUTPUT_DIR / f"{case_id}_QC_slice_z{best_z}.jpg")

    print(f"\n🎉 质检图生成完毕！请打开文件夹: {QC_OUTPUT_DIR}")
    print("💡 视觉指南:")
    print("   1. 骨头边缘发黄 = T1和Water对齐良好。")
    print("   2. 出现明显红绿错开的双眼皮 = 对齐失败。")
    print("   3. 亮蓝紫色的色块 = 你的 Label！看看它是不是完美覆盖在红色的 Water 水肿上！")


if __name__ == "__main__":
    main()