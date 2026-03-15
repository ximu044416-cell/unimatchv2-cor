import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import time

# ================= 核心配置 =================
INPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_N4")
OUTPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_74_FIXED")
CSV_PATH = Path(r"F:\Dinov2_data\global_geometry_report1.csv")

PAD_Y = 10  # Y轴上下扩展量
PAD_X = 70  # 左右宽屏扩展量


# ===========================================

def normalize_image(img_arr):
    p1 = np.percentile(img_arr, 1)
    p99 = np.percentile(img_arr, 99)
    img_arr = np.clip(img_arr, p1, p99)
    img_arr = (img_arr - p1) / (p99 - p1 + 1e-8) * 255.0
    return img_arr.astype(np.uint8)


def process_case(case_id):
    # 🛠️ 修复 1：强制转换 ID 为字符串，彻底杜绝 WindowsPath 与 int 拼接崩溃
    case_id = str(case_id).strip()

    case_in_dir = INPUT_DIR / case_id
    case_out_dir = OUTPUT_DIR / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)

    seg_dir = case_out_dir / "seg"
    rgb_dir = case_out_dir / "rgb"
    seg_dir.mkdir(exist_ok=True)
    rgb_dir.mkdir(exist_ok=True)

    path_w = str(case_in_dir / "water.nii.gz")
    path_f = str(case_in_dir / "fat.nii.gz")
    path_l = str(case_in_dir / "label.nii.gz")
    path_t1 = str(case_in_dir / "t1.nii.gz")

    if not all(Path(p).exists() for p in [path_w, path_f, path_l, path_t1]):
        print(f"⚠️ {case_id} 缺少必要文件，跳过。")
        return {"Patient_ID": case_id, "Label_Out_Of_Bounds": "Error", "Status": "Missing_Files"}

    # 1. 读取数据
    img_f = sitk.ReadImage(path_f, sitk.sitkFloat32)
    img_w = sitk.ReadImage(path_w, sitk.sitkFloat32)
    img_t1 = sitk.ReadImage(path_t1, sitk.sitkFloat32)
    img_l = sitk.ReadImage(path_l, sitk.sitkUInt8)

    arr_f = sitk.GetArrayFromImage(img_f)
    arr_w = sitk.GetArrayFromImage(img_w)
    arr_l = sitk.GetArrayFromImage(img_l)
    z_len, y_len, x_len = arr_f.shape

    # ================= 阶段一：动态生成黄金框 (BBox Mask) =================
    x_start, x_end = int(x_len * 0.10), int(x_len * 0.90)
    y_start, y_end = int(y_len * 0.35), int(y_len * 0.75)
    arr_roi = np.zeros_like(arr_f)
    arr_roi[:, y_start:y_end, x_start:x_end] = arr_f[:, y_start:y_end, x_start:x_end]

    img_roi = sitk.GetImageFromArray(arr_roi)
    img_roi.CopyInformation(img_f)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    arr_mask_raw = sitk.GetArrayFromImage(otsu.Execute(img_roi))

    z_ind, y_ind, x_ind = np.where(arr_mask_raw > 0)
    if len(z_ind) == 0:
        return {"Patient_ID": case_id, "Label_Out_Of_Bounds": "Error", "Status": "Failed_Mask"}

    # 🛠️ 修复 2：彻底消灭“差一错误”，为 max_y 和 max_x 加上精确的 +1
    min_z, max_z = z_ind.min(), z_ind.max()
    min_y = max(0, y_ind.min() - PAD_Y)
    max_y = min(y_len, y_ind.max() + PAD_Y + 1)  # 补上 +1
    min_x = max(0, x_ind.min() - PAD_X)
    max_x_raw = min(x_len, x_ind.max() + PAD_X + 1)  # 补上 +1

    # 重新计算对称中心
    center_x = (min_x + max_x_raw) // 2
    half_w = max(center_x - min_x, max_x_raw - center_x)
    min_x = max(0, center_x - half_w)
    max_x = min(x_len, center_x + half_w + 1)  # 补上 +1

    arr_bbox = np.zeros_like(arr_f, dtype=np.uint8)
    arr_bbox[min_z:max_z + 1, min_y:max_y, min_x:max_x] = 1

    img_bbox = sitk.GetImageFromArray(arr_bbox)
    img_bbox.CopyInformation(img_f)

    # 检测 Label 是否溢出方块
    label_out_of_bounds = bool(np.any((arr_l > 0) & (arr_bbox == 0)))

    # ================= 阶段二：终极刚性配准 (T1 找 Fat) =================
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricFixedMask(img_bbox)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=100, gradientMagnitudeTolerance=1e-8
    )
    R.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        img_f, img_t1, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)

    # 🛠️ 修复 3：加入 Try...Except 护体套件，防止某一颗老鼠屎坏了一锅粥！
    try:
        final_transform = R.Execute(img_f, img_t1)
    except Exception as e:
        print(f"\n❌ {case_id} 配准引擎抛出异常: {str(e)}")
        # 如果配准失败，优雅地返回失败状态，流水线继续下一个病人
        return {"Patient_ID": case_id, "Label_Out_Of_Bounds": "Error", "Status": "Reg_Failed"}

    # ================= 阶段三：B 样条高阶重采样 =================
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_f)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(np.float64(img_t1.GetPixel(0, 0, 0)))
    resampler.SetTransform(final_transform)

    img_t1_fixed = resampler.Execute(img_t1)

    # ================= 阶段四：保存最终 NIfTI 文件 =================
    sitk.WriteImage(img_f, str(case_out_dir / "fat.nii.gz"))
    sitk.WriteImage(img_w, str(case_out_dir / "water.nii.gz"))
    sitk.WriteImage(img_l, str(case_out_dir / "label.nii.gz"))
    sitk.WriteImage(img_t1_fixed, str(case_out_dir / "t1.nii.gz"))
    sitk.WriteImage(img_bbox, str(case_out_dir / "pelvis_bbox_mask.nii.gz"))

    # ================= 阶段五：生成双重质检图 =================
    arr_t1_fixed = sitk.GetArrayFromImage(img_t1_fixed)

    for z in range(z_len):
        slice_f = arr_f[z, :, :]
        slice_w = arr_w[z, :, :]
        slice_t1 = arr_t1_fixed[z, :, :]
        slice_l = arr_l[z, :, :]
        slice_bbox = arr_bbox[z, :, :]

        norm_f = normalize_image(slice_f)
        norm_w = normalize_image(slice_w)
        norm_t1 = normalize_image(slice_t1)

        # 1. SEG 质检图
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
            rgb_seg[label_mask, 0] = 0
            rgb_seg[label_mask, 1] = 100
            rgb_seg[label_mask, 2] = 255

        Image.fromarray(rgb_seg).save(seg_dir / f"slice_{z:02d}.jpg")

        # 2. RGB 质检图
        rgb_reg = np.zeros((y_len, x_len, 3), dtype=np.uint8)
        rgb_reg[:, :, 0] = norm_w
        rgb_reg[:, :, 1] = norm_t1
        if label_mask.any():
            rgb_reg[label_mask, 0] = 0
            rgb_reg[label_mask, 1] = 100
            rgb_reg[label_mask, 2] = 255

        Image.fromarray(rgb_reg).save(rgb_dir / f"slice_{z:02d}.jpg")

    return {"Patient_ID": case_id, "Label_Out_Of_Bounds": label_out_of_bounds, "Status": "Success"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # 强制将列转换为字符串再做判断，配合 process_case 里的保护
    bad_cases = df[df['T1_Match?'].astype(str).str.strip() != 'True']['Patient_ID'].tolist()

    print(f"🚀 发现 {len(bad_cases)} 个 T1 错位病例！启动工业级抗跌落手术流水线...")

    results = []
    start_time = time.time()

    for idx, case_id in enumerate(bad_cases, 1):
        print(f"[{idx}/{len(bad_cases)}] 正在手术: {case_id} ...", end=" ", flush=True)
        res = process_case(case_id)
        if isinstance(res, dict):
            if res["Status"] == "Success":
                if res["Label_Out_Of_Bounds"]:
                    print("🚨 警告: Label 溢出框外！", end=" ")
                print("✅ 成功！")
            else:
                print(f"❌ 失败！(原因: {res['Status']})")
            results.append(res)
        else:
            print("❌ 提取失败！")

    out_df = pd.DataFrame(results)
    csv_out = OUTPUT_DIR / "label_bounds_report.csv"
    out_df.to_csv(csv_out, index=False)

    end_time = time.time()
    mins = (end_time - start_time) / 60

    print("\n" + "=" * 50)
    print(f"🎉 74 例超级大决战圆满完成！总耗时: {mins:.2f} 分钟。")
    print(f"📁 修复后的完美数据全在: {OUTPUT_DIR}")
    print(f"📊 工业级质检报告已生成: {csv_out}")


if __name__ == "__main__":
    main()