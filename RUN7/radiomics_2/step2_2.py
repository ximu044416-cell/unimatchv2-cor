import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 路径设置
# =========================================================
WORK_DIR = Path(r"/RUN7/radiomics_2")
TRAIN_DIR = WORK_DIR / "train"
TEST_DIR = WORK_DIR / "test"
THRESHOLD_ROOT = WORK_DIR / "thresholds"

# radiomics yaml（沿用你现有的）
YAML_PATH = Path(r"/downstreamtasks/radiomics/radiomics_features.yaml")

# 输出目录
OUT_DIR = WORK_DIR / "radiomics_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TRAIN_GT = OUT_DIR / "Train_GT_Features.csv"
OUTPUT_TEST_GT = OUT_DIR / "Test_GT_Features.csv"
OUTPUT_THRESHOLD_SUMMARY = OUT_DIR / "Threshold_Extraction_Summary.csv"

# 扫描阈值（与你 step1 一致）
THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]


# =========================================================
# 工具函数
# =========================================================
def apply_pure_zscore(sitk_image):
    """
    仅对 >0 的真实组织做 Z-score，并保留原始空间信息。
    """
    img_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    valid_pixels = img_array[img_array > 0]

    if len(valid_pixels) == 0:
        return sitk_image

    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)
    if std_val < 1e-5:
        std_val = 1e-5

    img_normalized = (img_array - mean_val) / std_val
    img_normalized[img_array <= 0] = img_normalized.min() - 1.0

    new_sitk_image = sitk.GetImageFromArray(img_normalized)
    new_sitk_image.CopyInformation(sitk_image)
    return new_sitk_image


def collect_feature_keys(extractor, example_img, example_mask):
    """
    用一个非空 mask 先跑一次，拿到完整特征列名。
    """
    feats = extractor.execute(example_img, example_mask)
    feature_keys = []
    for k in feats.keys():
        if k.startswith("original_") or k.startswith("log-") or k.startswith("wavelet-"):
            feature_keys.append(k)
    return sorted(feature_keys)


def extract_one_case(extractor, sitk_img_norm, sitk_mask, patient_id, feature_keys):
    """
    对单病例提特征；如果 mask 全空，则返回 NaN 特征 + 元信息。
    """
    mask_array = sitk.GetArrayFromImage(sitk_mask)
    mask_volume = int(np.sum(mask_array))
    is_zero = int(mask_volume == 0)

    row = {
        "Patient_ID": patient_id,
        "MaskVolume": mask_volume,
        "IsZeroMask": is_zero
    }

    if is_zero:
        for k in feature_keys:
            row[k] = np.nan
        return row

    feats = extractor.execute(sitk_img_norm, sitk_mask)
    for k in feature_keys:
        row[k] = feats.get(k, np.nan)

    return row


def get_example_nonzero_case_from_train():
    """
    从 train GT 中找到一个非空病例，用于初始化完整特征列名。
    """
    for patient_dir in TRAIN_DIR.iterdir():
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        img_path = patient_dir / f"{patient_id}_Image.nii.gz"
        gt_path = patient_dir / f"{patient_id}_GT.nii.gz"

        if not (img_path.exists() and gt_path.exists()):
            continue

        sitk_img = sitk.ReadImage(str(img_path))
        sitk_gt = sitk.ReadImage(str(gt_path))
        gt_array = sitk.GetArrayFromImage(sitk_gt)

        if np.sum(gt_array) > 0:
            sitk_img_norm = apply_pure_zscore(sitk_img)
            return sitk_img_norm, sitk_gt

    raise RuntimeError("Train GT 中没有找到非空 mask，无法初始化 feature keys。")


# =========================================================
# Part A：Train GT
# =========================================================
def extract_train_gt(extractor, feature_keys):
    print("\n==============================")
    print("Part A: 提取 Train GT 特征")
    print("==============================")

    patient_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    print(f"📊 Train 患者数: {len(patient_dirs)}")

    rows = []

    for patient_dir in tqdm(patient_dirs, desc="Train GT"):
        patient_id = patient_dir.name
        img_path = patient_dir / f"{patient_id}_Image.nii.gz"
        gt_path = patient_dir / f"{patient_id}_GT.nii.gz"

        if not (img_path.exists() and gt_path.exists()):
            print(f"⚠️ 缺文件，跳过: {patient_id}")
            continue

        try:
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_gt = sitk.ReadImage(str(gt_path))
            sitk_img_norm = apply_pure_zscore(sitk_img)

            row = extract_one_case(extractor, sitk_img_norm, sitk_gt, patient_id, feature_keys)

            # Train GT 为空的一般直接跳过，保持和你原始 step2 一致
            if row["IsZeroMask"] == 1:
                print(f"⚠️ Train GT 全空，跳过: {patient_id}")
                continue

            rows.append(row)

        except Exception as e:
            print(f"❌ Train GT 提取失败: {patient_id}, 原因: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TRAIN_GT, index=False, encoding="utf-8-sig")
    print(f"✅ Train GT 特征保存完成: {OUTPUT_TRAIN_GT}")
    print(f"   shape = {df.shape}")


# =========================================================
# Part B：Test GT
# =========================================================
def extract_test_gt(extractor, feature_keys):
    print("\n==============================")
    print("Part B: 提取 Test GT 特征")
    print("==============================")

    patient_dirs = [d for d in TEST_DIR.iterdir() if d.is_dir()]
    print(f"📊 Test 患者数: {len(patient_dirs)}")

    rows = []

    for patient_dir in tqdm(patient_dirs, desc="Test GT"):
        patient_id = patient_dir.name
        img_path = patient_dir / f"{patient_id}_Image.nii.gz"
        gt_path = patient_dir / f"{patient_id}_GT.nii.gz"

        if not (img_path.exists() and gt_path.exists()):
            print(f"⚠️ 缺文件，跳过: {patient_id}")
            continue

        try:
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_gt = sitk.ReadImage(str(gt_path))
            sitk_img_norm = apply_pure_zscore(sitk_img)

            row = extract_one_case(extractor, sitk_img_norm, sitk_gt, patient_id, feature_keys)

            # 和旧逻辑一样：GT 全空的 test case 不进入 radiomics downstream
            if row["IsZeroMask"] == 1:
                print(f"⚠️ Test GT 全空，跳过: {patient_id}")
                continue

            rows.append(row)

        except Exception as e:
            print(f"❌ Test GT 提取失败: {patient_id}, 原因: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TEST_GT, index=False, encoding="utf-8-sig")
    print(f"✅ Test GT 特征保存完成: {OUTPUT_TEST_GT}")
    print(f"   shape = {df.shape}")


# =========================================================
# Part C：按 threshold 提取 Test Pred
# =========================================================
def extract_test_pred_by_threshold(extractor, feature_keys):
    print("\n========================================")
    print("Part C: 按 threshold 提取 Test Pred 特征")
    print("========================================")

    summary_rows = []

    for thr in THRESHOLDS:
        thr_tag = f"thr_{int(thr * 100):03d}"
        thr_test_dir = THRESHOLD_ROOT / thr_tag / "test"
        out_csv = OUT_DIR / f"Test_Pred_Features_{thr_tag}.csv"

        if not thr_test_dir.exists():
            print(f"⚠️ 缺少目录，跳过: {thr_test_dir}")
            continue

        patient_dirs = [d for d in thr_test_dir.iterdir() if d.is_dir()]
        print(f"\n📌 Threshold = {thr:.2f} | 患者数 = {len(patient_dirs)}")

        rows = []
        zero_count = 0

        for patient_dir in tqdm(patient_dirs, desc=f"Pred {thr:.2f}"):
            patient_id = patient_dir.name

            # image / gt 来自主 test 目录
            base_test_dir = TEST_DIR / patient_id
            img_path = base_test_dir / f"{patient_id}_Image.nii.gz"
            gt_path = base_test_dir / f"{patient_id}_GT.nii.gz"
            pred_path = patient_dir / f"{patient_id}_Pred.nii.gz"

            if not (img_path.exists() and gt_path.exists() and pred_path.exists()):
                print(f"⚠️ 缺文件，跳过: {patient_id}")
                continue

            try:
                sitk_img = sitk.ReadImage(str(img_path))
                sitk_gt = sitk.ReadImage(str(gt_path))
                sitk_pred = sitk.ReadImage(str(pred_path))
                sitk_img_norm = apply_pure_zscore(sitk_img)

                # 和旧逻辑一样：GT 全空不进入 downstream radiomics
                gt_array = sitk.GetArrayFromImage(sitk_gt)
                if np.sum(gt_array) == 0:
                    continue

                row = extract_one_case(extractor, sitk_img_norm, sitk_pred, patient_id, feature_keys)

                if row["IsZeroMask"] == 1:
                    zero_count += 1

                rows.append(row)

            except Exception as e:
                print(f"❌ Pred 提取失败: thr={thr:.2f}, patient={patient_id}, 原因: {e}")

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        summary_rows.append({
            "Threshold": thr,
            "TotalCases": len(df),
            "ZeroMaskCases": int((df["IsZeroMask"] == 1).sum()) if len(df) > 0 else 0,
            "NonZeroCases": int((df["IsZeroMask"] == 0).sum()) if len(df) > 0 else 0,
            "OutputCSV": str(out_csv)
        })

        print(f"✅ 已保存: {out_csv}")
        print(f"   Zero-mask = {zero_count}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_THRESHOLD_SUMMARY, index=False, encoding="utf-8-sig")
    print(f"\n🎉 Threshold summary 已保存: {OUTPUT_THRESHOLD_SUMMARY}")


# =========================================================
# 主程序
# =========================================================
def main():
    print("🚀 启动 RUN7 radiomics_2 Step2（PyRadiomics 环境）")

    if not YAML_PATH.exists():
        raise FileNotFoundError(f"找不到 radiomics 配置文件: {YAML_PATH}")

    extractor = featureextractor.RadiomicsFeatureExtractor(str(YAML_PATH))
    print("✅ PyRadiomics 提取器初始化成功")

    # 先从一个非空 train GT 病例拿到完整特征列
    example_img, example_mask = get_example_nonzero_case_from_train()
    feature_keys = collect_feature_keys(extractor, example_img, example_mask)
    print(f"✅ 已锁定特征列数: {len(feature_keys)}")

    extract_train_gt(extractor, feature_keys)
    extract_test_gt(extractor, feature_keys)
    extract_test_pred_by_threshold(extractor, feature_keys)

    print("\n🎉 Step2 全部完成。")


if __name__ == "__main__":
    main()