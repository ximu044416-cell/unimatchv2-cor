import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =========================================================
# 路径
# =========================================================
OUT_ROOT = Path(r"F:\cor\RUN8\down_8_NEW\down_2_threshold")
RECON_DIR = OUT_ROOT / "reconstructed_dev164"

CLINICAL_DIR = OUT_ROOT / "clinical"
CONFIG_DIR = OUT_ROOT / "config"
FEATURE_DIR = OUT_ROOT / "features_dev164"

CLINICAL_MASTER_FILE = CLINICAL_DIR / "clinical_info_all.xlsx"
YAML_PATH = CONFIG_DIR / "radiomics_features.yaml"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
CLINICAL_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

OUTPUT_GT = FEATURE_DIR / "GT_Features_Dev164.csv"
OUTPUT_DEV_CLINICAL = CLINICAL_DIR / "clinical_info_dev164.xlsx"
OUTPUT_TEMPLATE_JSON = FEATURE_DIR / "feature_template_columns.json"


def apply_pure_zscore(sitk_image):
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


def collect_dev164_ids():
    patient_dirs = sorted([p for p in RECON_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if len(patient_dirs) == 0:
        raise RuntimeError(f"❌ reconstructed_dev164 为空：{RECON_DIR}")
    return [p.name for p in patient_dirs]


def main():
    print("🚀 启动 RUN8 Step3：dev164 多阈值影像组学提取")
    print(f"📂 RECON_DIR   = {RECON_DIR}")
    print(f"📄 CLINICAL    = {CLINICAL_MASTER_FILE}")
    print(f"📄 YAML        = {YAML_PATH}")

    if not CLINICAL_MASTER_FILE.exists():
        raise FileNotFoundError(f"❌ 找不到总临床表：{CLINICAL_MASTER_FILE}")
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到 radiomics yaml：{YAML_PATH}")

    extractor = featureextractor.RadiomicsFeatureExtractor(str(YAML_PATH))
    dev_ids = collect_dev164_ids()

    # -----------------------------
    # 0. 导出 dev164 临床表
    # -----------------------------
    df_clinical_all = pd.read_excel(CLINICAL_MASTER_FILE)
    if "Patient_ID" not in df_clinical_all.columns:
        raise KeyError("❌ clinical_info_all.xlsx 缺少 Patient_ID 列")

    df_clinical_all["Patient_ID"] = df_clinical_all["Patient_ID"].astype(str).str.strip()
    df_clinical_dev = df_clinical_all[df_clinical_all["Patient_ID"].isin(dev_ids)].copy()
    df_clinical_dev.to_excel(OUTPUT_DEV_CLINICAL, index=False)
    print(f"✅ 已导出 dev164 临床表：{OUTPUT_DEV_CLINICAL} | n={len(df_clinical_dev)}")

    # -----------------------------
    # 1. 提取 dev164 GT 特征
    # -----------------------------
    gt_features_list = []
    gt_feature_template = None

    patient_dirs = sorted([p for p in RECON_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])

    print(f"\n📊 开始提取 dev164 GT 特征，共 {len(patient_dirs)} 名患者...")
    for patient_dir in tqdm(patient_dirs, desc="GT features"):
        patient_id = patient_dir.name
        img_path = patient_dir / f"{patient_id}_Image.nii.gz"
        gt_path = patient_dir / f"{patient_id}_GT.nii.gz"

        if not (img_path.exists() and gt_path.exists()):
            continue

        try:
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_gt = sitk.ReadImage(str(gt_path))

            gt_array = sitk.GetArrayFromImage(sitk_gt)
            if np.sum(gt_array) == 0:
                # GT 空掩码无法提特征，保持和你旧逻辑一致：跳过
                continue

            sitk_img_norm = apply_pure_zscore(sitk_img)
            features = extractor.execute(sitk_img_norm, sitk_gt)

            patient_feat = {"Patient_ID": patient_id}
            for key, value in features.items():
                if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                    patient_feat[key] = value

            gt_features_list.append(patient_feat)

            if gt_feature_template is None:
                gt_feature_template = [k for k in patient_feat.keys() if k != "Patient_ID"]

        except Exception as e:
            print(f"⚠️ GT 提取失败：{patient_id} | {e}")

    if len(gt_features_list) == 0:
        raise RuntimeError("❌ 没有提取到任何 GT 特征")

    df_gt = pd.DataFrame(gt_features_list)
    df_gt.to_csv(OUTPUT_GT, index=False, encoding="utf-8-sig")
    print(f"✅ GT 特征已保存：{OUTPUT_GT} | n={len(df_gt)}")

    with open(OUTPUT_TEMPLATE_JSON, "w", encoding="utf-8") as f:
        json.dump(gt_feature_template, f, ensure_ascii=False, indent=2)

    # -----------------------------
    # 2. 提取各阈值 Pred 特征
    # -----------------------------
    for t in THRESHOLDS:
        t_str = f"{int(t * 100):03d}"
        out_csv = FEATURE_DIR / f"Pred_Features_{t_str}.csv"

        print(f"\n⚙️ 正在提取阈值 {t:.2f} 的 Pred 特征...")
        pred_features_list = []

        for patient_dir in tqdm(patient_dirs, desc=f"Pred {t_str}"):
            patient_id = patient_dir.name
            img_path = patient_dir / f"{patient_id}_Image.nii.gz"
            pred_path = patient_dir / f"{patient_id}_Pred_{t_str}.nii.gz"

            if not (img_path.exists() and pred_path.exists()):
                continue

            # 只对 GT 成功提取过的患者做 Pred 提取，保证特征模板一致
            if patient_id not in set(df_gt["Patient_ID"].astype(str)):
                continue

            try:
                sitk_img = sitk.ReadImage(str(img_path))
                sitk_pred = sitk.ReadImage(str(pred_path))
                sitk_img_norm = apply_pure_zscore(sitk_img)
                pred_array = sitk.GetArrayFromImage(sitk_pred)

                patient_feat_pred = {"Patient_ID": patient_id}

                # 空掩码：按旧逻辑直接补 0
                if np.sum(pred_array) == 0:
                    for key in gt_feature_template:
                        patient_feat_pred[key] = 0.0
                else:
                    features_pred = extractor.execute(sitk_img_norm, sitk_pred)
                    for key, value in features_pred.items():
                        if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                            patient_feat_pred[key] = value

                    for key in gt_feature_template:
                        if key not in patient_feat_pred:
                            patient_feat_pred[key] = 0.0

                pred_features_list.append(patient_feat_pred)

            except Exception:
                # 保持旧逻辑：失败就跳过，下一步用 left join 对齐并 rescue
                pass

        df_pred = pd.DataFrame(pred_features_list)
        df_pred.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"💾 阈值 {t:.2f} 的 Pred 特征已保存：{out_csv} | n={len(df_pred)}")

    print("\n🎉 RUN8 Step3 完成。")


if __name__ == "__main__":
    main()