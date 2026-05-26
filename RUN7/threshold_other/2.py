import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


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


def extract_run7_radiomics():
    # ================= 1. 路径修改为 threshold_other =================
    WORK_DIR = Path(r"/RUN7/threshold_other")
    TRAIN_DIR = WORK_DIR / "train"
    TEST_DIR = WORK_DIR / "test"

    YAML_PATH = Path(r"/downstreamtasks/radiomics/radiomics_features.yaml")

    OUTPUT_TRAIN_GT = WORK_DIR / "Train_GT_Features.csv"
    OUTPUT_TEST_GT = WORK_DIR / "Test_GT_Features.csv"

    # 多阈值列表
    THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]

    print("🚀 启动 RUN7 Step2: Train GT + Test(GT & 多阈值 Pred) 一体化组学提取引擎...")

    if not YAML_PATH.exists():
        print(f"❌ 错误：找不到配置文件 {YAML_PATH}！")
        return

    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(str(YAML_PATH))
        print("✅ PyRadiomics 提取器初始化成功。")
    except Exception as e:
        print(f"❌ 提取器初始化失败: {e}")
        return

    # =========================================================
    # Part A: Train GT
    # =========================================================
    train_patient_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    print(f"\n📊 在 Train 文件夹中共发现 {len(train_patient_dirs)} 名患者。准备提取 Train GT 特征...\n")

    train_features_list = []

    for patient_dir in tqdm(train_patient_dirs, desc="Train GT 特征提取进度"):
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
                continue

            sitk_img_norm = apply_pure_zscore(sitk_img)
            features = extractor.execute(sitk_img_norm, sitk_gt)

            patient_feat = {"Patient_ID": patient_id}
            for key, value in features.items():
                if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                    patient_feat[key] = value

            train_features_list.append(patient_feat)

        except Exception as e:
            print(f"❌ 患者 {patient_id} Train GT 提取失败，原因: {e}")

    if len(train_features_list) > 0:
        df_train = pd.DataFrame(train_features_list)
        df_train.to_csv(OUTPUT_TRAIN_GT, index=False)
        print(f"\n✅ Train GT 特征已保存到: {OUTPUT_TRAIN_GT}")

    # =========================================================
    # Part B: Test GT
    # =========================================================
    test_patient_dirs = [d for d in TEST_DIR.iterdir() if d.is_dir()]
    print(f"\n📊 在 Test 文件夹中共发现 {len(test_patient_dirs)} 名患者。准备提取 Test GT 特征...\n")

    gt_features_list = []
    for patient_dir in tqdm(test_patient_dirs, desc="Test GT 特征提取进度"):
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
                continue

            sitk_img_norm = apply_pure_zscore(sitk_img)
            features_gt = extractor.execute(sitk_img_norm, sitk_gt)
            patient_feat_gt = {"Patient_ID": patient_id}
            for key, value in features_gt.items():
                if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                    patient_feat_gt[key] = value
            gt_features_list.append(patient_feat_gt)
        except Exception as e:
            pass

    if len(gt_features_list) > 0:
        df_gt = pd.DataFrame(gt_features_list)
        df_gt.to_csv(OUTPUT_TEST_GT, index=False)
        print(f"✅ Test GT 特征已保存到: {OUTPUT_TEST_GT}")

    # =========================================================
    # Part C: Test Pred (批量遍历各个阈值)
    # =========================================================
    for t in THRESHOLDS:
        t_str = f"{int(t * 100):03d}"
        OUTPUT_TEST_PRED = WORK_DIR / f"Test_Pred_Features_{t_str}.csv"
        print(f"\n⚙️ 正在提取阈值 {t:.2f} 的 Pred 特征...")

        pred_features_list = []
        for patient_dir in tqdm(test_patient_dirs, desc=f"Test Pred ({t_str}) 提取进度"):
            patient_id = patient_dir.name
            img_path = patient_dir / f"{patient_id}_Image.nii.gz"
            pred_path = patient_dir / f"{patient_id}_Pred_{t_str}.nii.gz"

            if not (img_path.exists() and pred_path.exists()):
                continue

            try:
                sitk_img = sitk.ReadImage(str(img_path))
                sitk_pred = sitk.ReadImage(str(pred_path))
                sitk_img_norm = apply_pure_zscore(sitk_img)
                pred_array = sitk.GetArrayFromImage(sitk_pred)

                # 寻找对应的 GT 以确保格式对齐
                gt_template = next((item for item in gt_features_list if item["Patient_ID"] == patient_id), None)
                if not gt_template:
                    continue

                patient_feat_pred = {"Patient_ID": patient_id}

                # 原始逻辑：空掩码的 0 值惩罚
                if np.sum(pred_array) == 0:
                    for key in gt_template.keys():
                        if key == "Patient_ID":
                            patient_feat_pred[key] = patient_id
                        else:
                            patient_feat_pred[key] = 0.0
                else:
                    features_pred = extractor.execute(sitk_img_norm, sitk_pred)
                    for key, value in features_pred.items():
                        if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                            patient_feat_pred[key] = value

                    for key in gt_template.keys():
                        if key not in patient_feat_pred:
                            patient_feat_pred[key] = 0.0

                pred_features_list.append(patient_feat_pred)

            except Exception as e:
                pass

        if len(pred_features_list) > 0:
            df_pred = pd.DataFrame(pred_features_list)
            df_pred.to_csv(OUTPUT_TEST_PRED, index=False)
            print(f"💾 阈值 {t:.2f} 的特征已存入: {OUTPUT_TEST_PRED}")

    print("\n🎉 完美收官！所有阈值特征提取完毕！")


if __name__ == "__main__":
    extract_run7_radiomics()