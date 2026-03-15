import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import time

# ================= 核心配置 =================
INPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_N4")
OUT_74_DIR = Path(r"F:\Dinov2_data\ALL_COR_74_FIXED")
OUT_130_DIR = Path(r"F:\Dinov2_data\ALL_COR_130_FIXED")
CSV_PATH = Path(r"F:\Dinov2_data\global_geometry_report1.csv")

PAD_Y = 10
PAD_X = 70

# 那 3 个需要打补丁的特殊病例
OUTLIERS_3 = ["0004025950_20220804193615", "0008826735_20220730183740", "0010650523_20220921204012"]


# ===========================================

def normalize_image(img_arr):
    p1 = np.percentile(img_arr, 1)
    p99 = np.percentile(img_arr, 99)
    img_arr = np.clip(img_arr, p1, p99)
    img_arr = (img_arr - p1) / (p99 - p1 + 1e-8) * 255.0
    return img_arr.astype(np.uint8)


def get_standard_bbox(arr_f, arr_l):
    """提取标准自适应 BBox，并返回坐标和是否溢出的标记"""
    z_len, y_len, x_len = arr_f.shape
    x_start, x_end = int(x_len * 0.10), int(x_len * 0.90)
    y_start, y_end = int(y_len * 0.35), int(y_len * 0.75)

    arr_roi = np.zeros_like(arr_f)
    arr_roi[:, y_start:y_end, x_start:x_end] = arr_f[:, y_start:y_end, x_start:x_end]

    img_roi = sitk.GetImageFromArray(arr_roi)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    arr_mask_raw = sitk.GetArrayFromImage(otsu.Execute(img_roi))

    z_ind, y_ind, x_ind = np.where(arr_mask_raw > 0)
    if len(z_ind) == 0:
        return None, None, None, None, None, None, True

    min_z, max_z = z_ind.min(), z_ind.max()
    min_y = max(0, y_ind.min() - PAD_Y)
    max_y = min(y_len, y_ind.max() + PAD_Y + 1)
    min_x = max(0, x_ind.min() - PAD_X)
    max_x_raw = min(x_len, x_ind.max() + PAD_X + 1)

    center_x = (min_x + max_x_raw) // 2
    half_w = max(center_x - min_x, max_x_raw - center_x)
    min_x = max(0, center_x - half_w)
    max_x = min(x_len, center_x + half_w + 1)

    return min_z, max_z, min_y, max_y, min_x, max_x, False


def generate_qc_images(case_out_dir, arr_f, arr_w, arr_t1, arr_l, arr_bbox):
    """生成 SEG 和 RGB 质检图"""
    seg_dir = case_out_dir / "seg"
    rgb_dir = case_out_dir / "rgb"
    seg_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    z_len, y_len, x_len = arr_f.shape
    for z in range(z_len):
        slice_f = arr_f[z, :, :]
        slice_w = arr_w[z, :, :]
        slice_t1 = arr_t1[z, :, :]
        slice_l = arr_l[z, :, :]
        slice_bbox = arr_bbox[z, :, :]

        norm_f = normalize_image(slice_f)
        norm_w = normalize_image(slice_w)
        norm_t1 = normalize_image(slice_t1)

        # 1. SEG 图
        rgb_seg = np.zeros((y_len, x_len, 3), dtype=np.uint8)
        rgb_seg[:, :, 0] = rgb_seg[:, :, 1] = rgb_seg[:, :, 2] = norm_f

        yellow_mask = slice_bbox > 0
        if yellow_mask.any():
            alpha = 0.4
            rgb_seg[yellow_mask, 0] = (norm_f[yellow_mask] * (1 - alpha) + 255 * alpha).astype(np.uint8)
            rgb_seg[yellow_mask, 1] = (norm_f[yellow_mask] * (1 - alpha) + 255 * alpha).astype(np.uint8)
            rgb_seg[yellow_mask, 2] = (norm_f[yellow_mask] * (1 - alpha) + 0 * alpha).astype(np.uint8)

        label_mask = slice_l > 0
        if label_mask.any():
            rgb_seg[label_mask, 0] = 0;
            rgb_seg[label_mask, 1] = 100;
            rgb_seg[label_mask, 2] = 255
        Image.fromarray(rgb_seg).save(seg_dir / f"slice_{z:02d}.jpg")

        # 2. RGB 图
        rgb_reg = np.zeros((y_len, x_len, 3), dtype=np.uint8)
        rgb_reg[:, :, 0] = norm_w
        rgb_reg[:, :, 1] = norm_t1
        if label_mask.any():
            rgb_reg[label_mask, 0] = 0;
            rgb_reg[label_mask, 1] = 100;
            rgb_reg[label_mask, 2] = 255
        Image.fromarray(rgb_reg).save(rgb_dir / f"slice_{z:02d}.jpg")


# ================= 轨 1：刺头专属修复 =================
def process_outlier(case_id):
    case_in_dir = INPUT_DIR / case_id
    case_out_dir = OUT_74_DIR / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)  # 🛠️ 修复 B：加上目录创建守卫

    img_f = sitk.ReadImage(str(case_in_dir / "fat.nii.gz"), sitk.sitkFloat32)
    img_w = sitk.ReadImage(str(case_in_dir / "water.nii.gz"), sitk.sitkFloat32)
    img_t1 = sitk.ReadImage(str(case_in_dir / "t1.nii.gz"), sitk.sitkFloat32)
    img_l = sitk.ReadImage(str(case_in_dir / "label.nii.gz"), sitk.sitkUInt8)

    arr_f, arr_w, arr_t1, arr_l = map(sitk.GetArrayFromImage, [img_f, img_w, img_t1, img_l])

    min_z, max_z, min_y, max_y, min_x, max_x, failed = get_standard_bbox(arr_f, arr_l)

    # 🛠️ 修复 A：加入对 failed 状态的拦截
    if failed:
        return {"Patient_ID": case_id, "Label_Out_Of_Bounds": None, "Status": "Failed_Mask"}  # 🛠️ 修复 C：返回 None 保持数据列纯净

    # 🔥 核心补丁：动态向上拉伸顶盖
    label_z, label_y, label_x = np.where(arr_l > 0)
    if len(label_y) > 0:
        label_min_y = label_y.min()
        if label_min_y < min_y:
            print(f" [补丁拉伸] 顶盖强行抬高 {min_y - label_min_y + 5} 像素...", end=" ")
            min_y = max(0, label_min_y - 5)

    arr_bbox = np.zeros_like(arr_f, dtype=np.uint8)
    arr_bbox[min_z:max_z + 1, min_y:max_y, min_x:max_x] = 1

    img_bbox = sitk.GetImageFromArray(arr_bbox)
    img_bbox.CopyInformation(img_f)

    # 重新配准
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricFixedMask(img_bbox)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=100)
    R.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(img_f, img_t1, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)

    try:
        final_transform = R.Execute(img_f, img_t1)
    except Exception as e:
        print(f"\n❌ 配准抛出异常: {str(e)}")
        return {"Patient_ID": case_id, "Label_Out_Of_Bounds": None, "Status": "Reg_Failed"}

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_f)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(np.float64(img_t1.GetPixel(0, 0, 0)))
    resampler.SetTransform(final_transform)
    img_t1_fixed = resampler.Execute(img_t1)

    sitk.WriteImage(img_t1_fixed, str(case_out_dir / "t1.nii.gz"))
    sitk.WriteImage(img_bbox, str(case_out_dir / "pelvis_bbox_mask.nii.gz"))

    arr_t1_fixed = sitk.GetArrayFromImage(img_t1_fixed)
    generate_qc_images(case_out_dir, arr_f, arr_w, arr_t1_fixed, arr_l, arr_bbox)

    label_out = bool(np.any((arr_l > 0) & (arr_bbox == 0)))
    return {"Patient_ID": case_id, "Label_Out_Of_Bounds": label_out, "Status": "Patched_Success"}


# ================= 轨 2：130 例建档 =================
def process_good_case(case_id):
    case_in_dir = INPUT_DIR / case_id
    case_out_dir = OUT_130_DIR / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)

    img_f = sitk.ReadImage(str(case_in_dir / "fat.nii.gz"), sitk.sitkFloat32)
    img_w = sitk.ReadImage(str(case_in_dir / "water.nii.gz"), sitk.sitkFloat32)
    img_t1 = sitk.ReadImage(str(case_in_dir / "t1.nii.gz"), sitk.sitkFloat32)
    img_l = sitk.ReadImage(str(case_in_dir / "label.nii.gz"), sitk.sitkUInt8)

    arr_f, arr_w, arr_t1, arr_l = map(sitk.GetArrayFromImage, [img_f, img_w, img_t1, img_l])

    min_z, max_z, min_y, max_y, min_x, max_x, failed = get_standard_bbox(arr_f, arr_l)

    if failed:
        # 🛠️ 修复 C：提取失败时返回 None
        return {"Patient_ID": case_id, "Label_Out_Of_Bounds": None, "Status": "Failed_Mask"}

    arr_bbox = np.zeros_like(arr_f, dtype=np.uint8)
    arr_bbox[min_z:max_z + 1, min_y:max_y, min_x:max_x] = 1

    img_bbox = sitk.GetImageFromArray(arr_bbox)
    img_bbox.CopyInformation(img_f)

    sitk.WriteImage(img_f, str(case_out_dir / "fat.nii.gz"))
    sitk.WriteImage(img_w, str(case_out_dir / "water.nii.gz"))
    sitk.WriteImage(img_l, str(case_out_dir / "label.nii.gz"))
    sitk.WriteImage(img_t1, str(case_out_dir / "t1.nii.gz"))
    sitk.WriteImage(img_bbox, str(case_out_dir / "pelvis_bbox_mask.nii.gz"))

    generate_qc_images(case_out_dir, arr_f, arr_w, arr_t1, arr_l, arr_bbox)

    label_out = bool(np.any((arr_l > 0) & (arr_bbox == 0)))
    return {"Patient_ID": case_id, "Label_Out_Of_Bounds": label_out, "Status": "Copied_Success"}


# ================= 主控制台 =================
def main():
    OUT_130_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("🛠️ 任务 1：启动 3 例刺头动态拉伸与重配准流水线...")
    res_74 = []
    for idx, case_id in enumerate(OUTLIERS_3, 1):
        print(f"[{idx}/3] 重新打补丁: {case_id} ...", end=" ", flush=True)
        res = process_outlier(case_id)
        if res["Label_Out_Of_Bounds"] is True:
            print("🚨 仍然溢出！(需人工核查)", end=" ")
        print(f"✅ 完成！({res['Status']})")
        res_74.append(res)

    pd.DataFrame(res_74).to_csv(OUT_74_DIR / "patched_3_cases_report.csv", index=False)

    print("\n" + "=" * 50)
    print("📦 任务 2：启动 130 例好数据统一画框建档流水线...")
    df = pd.read_csv(CSV_PATH)
    good_cases = df[df['T1_Match?'].astype(str).str.strip() == 'True']['Patient_ID'].tolist()

    res_130 = []
    start_time = time.time()
    for idx, case_id in enumerate(good_cases, 1):
        case_id = str(case_id).strip()
        print(f"[{idx}/{len(good_cases)}] 正在建档: {case_id} ...", end=" ", flush=True)
        res = process_good_case(case_id)
        if res["Status"] == "Copied_Success":
            if res["Label_Out_Of_Bounds"] is True:
                print("🚨 警告: 该好数据的 Label 竟然溢出了！", end=" ")
            print("✅ 归档成功！")
        else:
            print(f"❌ 处理失败！({res['Status']})")
        res_130.append(res)

    pd.DataFrame(res_130).to_csv(OUT_130_DIR / "label_bounds_report_130.csv", index=False)

    mins = (time.time() - start_time) / 60
    print("\n" + "=" * 50)
    print(f"🎉 你的 204 例数据大一统战役已全面胜利！130 例处理耗时: {mins:.2f} 分钟。")
    print(f"📁 3例补丁报告已放入: {OUT_74_DIR}")
    print(f"📁 130例的完整数据和报告已全部归入: {OUT_130_DIR}")


if __name__ == "__main__":
    main()