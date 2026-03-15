import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ================= 核心配置 =================
OUT_130_DIR = Path(r"F:\Dinov2_data\ALL_COR_130_FIXED")
CSV_130_PATH = OUT_130_DIR / "label_bounds_report_130.csv"


# ===========================================

def normalize_image(img_arr):
    p1 = np.percentile(img_arr, 1)
    p99 = np.percentile(img_arr, 99)
    img_arr = np.clip(img_arr, p1, p99)
    img_arr = (img_arr - p1) / (p99 - p1 + 1e-8) * 255.0
    return img_arr.astype(np.uint8)


def fix_overflow_bbox(case_id):
    case_dir = OUT_130_DIR / case_id

    # 1. 读取已经存好的全套数据
    img_f = sitk.ReadImage(str(case_dir / "fat.nii.gz"), sitk.sitkFloat32)
    img_w = sitk.ReadImage(str(case_dir / "water.nii.gz"), sitk.sitkFloat32)
    img_t1 = sitk.ReadImage(str(case_dir / "t1.nii.gz"), sitk.sitkFloat32)
    img_l = sitk.ReadImage(str(case_dir / "label.nii.gz"), sitk.sitkUInt8)
    img_bbox = sitk.ReadImage(str(case_dir / "pelvis_bbox_mask.nii.gz"), sitk.sitkUInt8)

    arr_f, arr_w, arr_t1, arr_l, arr_bbox = map(sitk.GetArrayFromImage, [img_f, img_w, img_t1, img_l, img_bbox])
    z_len, y_len, x_len = arr_f.shape

    # 2. 获取当前框和 Label 的真实边界
    l_z, l_y, l_x = np.where(arr_l > 0)
    b_z, b_y, b_x = np.where(arr_bbox > 0)

    # 3. 核心修复：全方位动态撑爆黄框！
    # 只要 Label 超出当前 BBox 边界，就向外扩张，并附加 5 像素安全缓冲带
    new_min_z = min(l_z.min(), b_z.min())
    new_max_z = max(l_z.max(), b_z.max())

    new_min_y = min(max(0, l_y.min() - 5), b_y.min())
    new_max_y = max(min(y_len, l_y.max() + 6), b_y.max())

    # X 轴稍微特殊：扩张后依然要保持左右对称（完美居中）
    new_min_x_raw = min(max(0, l_x.min() - 5), b_x.min())
    new_max_x_raw = max(min(x_len, l_x.max() + 6), b_x.max())

    center_x = (new_min_x_raw + new_max_x_raw) // 2
    half_w = max(center_x - new_min_x_raw, new_max_x_raw - center_x)
    new_min_x = max(0, center_x - half_w)
    new_max_x = min(x_len, center_x + half_w + 1)

    # 4. 生成全新的 3D 护城河
    arr_bbox_new = np.zeros_like(arr_f, dtype=np.uint8)
    arr_bbox_new[new_min_z:new_max_z + 1, new_min_y:new_max_y, new_min_x:new_max_x] = 1

    img_bbox_new = sitk.GetImageFromArray(arr_bbox_new)
    img_bbox_new.CopyInformation(img_f)

    # 5. 原地覆盖 BBox
    sitk.WriteImage(img_bbox_new, str(case_dir / "pelvis_bbox_mask.nii.gz"))

    # 6. 原地重写质检图
    seg_dir = case_dir / "seg"
    rgb_dir = case_dir / "rgb"

    for z in range(z_len):
        slice_f, slice_w, slice_t1 = arr_f[z, :, :], arr_w[z, :, :], arr_t1[z, :, :]
        slice_l, slice_bbox = arr_l[z, :, :], arr_bbox_new[z, :, :]

        norm_f = normalize_image(slice_f)
        norm_w = normalize_image(slice_w)
        norm_t1 = normalize_image(slice_t1)

        # 刷新 SEG
        rgb_seg = np.zeros((y_len, x_len, 3), dtype=np.uint8)
        rgb_seg[:, :, 0] = rgb_seg[:, :, 1] = rgb_seg[:, :, 2] = norm_f

        if (slice_bbox > 0).any():
            rgb_seg[slice_bbox > 0, 0] = (norm_f[slice_bbox > 0] * 0.6 + 255 * 0.4).astype(np.uint8)
            rgb_seg[slice_bbox > 0, 1] = (norm_f[slice_bbox > 0] * 0.6 + 255 * 0.4).astype(np.uint8)
            rgb_seg[slice_bbox > 0, 2] = (norm_f[slice_bbox > 0] * 0.6).astype(np.uint8)

        if (slice_l > 0).any():
            rgb_seg[slice_l > 0] = [0, 100, 255]
        Image.fromarray(rgb_seg).save(seg_dir / f"slice_{z:02d}.jpg")

        # 刷新 RGB (因为 BBox 变了，顺手更新一下保证一致性)
        rgb_reg = np.zeros((y_len, x_len, 3), dtype=np.uint8)
        rgb_reg[:, :, 0] = norm_w
        rgb_reg[:, :, 1] = norm_t1
        if (slice_l > 0).any():
            rgb_reg[slice_l > 0] = [0, 100, 255]
        Image.fromarray(rgb_reg).save(rgb_dir / f"slice_{z:02d}.jpg")

    return True


def main():
    print("🔍 正在读取 130 例报告，揪出 Label 溢出的叛徒...")
    df = pd.read_csv(CSV_130_PATH)

    # 兼容 Pandas 解析为布尔值或字符串 'True' 的情况
    bad_mask = (df['Label_Out_Of_Bounds'] == True) | (df['Label_Out_Of_Bounds'].astype(str).str.strip() == 'True')
    bad_cases = df[bad_mask]['Patient_ID'].astype(str).tolist()

    if not bad_cases:
        print("🎉 报告里没有溢出病例！你是不是看错啦？")
        return

    print(f"🚨 锁定 {len(bad_cases)} 个溢出病例！启动[全方位强行拉伸]程序...\n")

    for idx, case_id in enumerate(bad_cases, 1):
        print(f"[{idx}/{len(bad_cases)}] 正在修复: {case_id} ...", end=" ", flush=True)
        fix_overflow_bbox(case_id)
        print("✅ 护城河已扩建完毕！")

    print("\n" + "=" * 50)
    print("🎉 10 个叛徒已全部被镇压！")
    print(
        "👉 快去 ALL_COR_130_FIXED 找其中一两个病例，打开他们的 seg 文件夹，看看原本漏出来的蓝色病灶是不是已经被黄框狠狠包裹住了！")


if __name__ == "__main__":
    main()