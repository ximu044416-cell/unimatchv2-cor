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
    """
    只针对真实组织(大于0的像素)进行 Z-score 归一化，
    并完美保留原始医学图像的物理空间坐标！
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


def extract_run7_radiomics():
    # ================= 1. 路径准备 =================
    WORK_DIR = Path(r"/RUN7/radiomics")
    TRAIN_DIR = WORK_DIR / "train"
    TEST_DIR = WORK_DIR / "test"

    # 这里用你现成的 yaml；如果你已经复制到 RUN7，也可以改成 RUN7 下的绝对路径
    YAML_PATH = Path(r"/downstreamtasks/radiomics/radiomics_features.yaml")

    OUTPUT_TRAIN_GT = WORK_DIR / "Train_GT_Features.csv"
    OUTPUT_TEST_GT = WORK_DIR / "Test_GT_Features.csv"
    OUTPUT_TEST_PRED = WORK_DIR / "Test_Pred_Features.csv"

    print("🚀 启动 RUN7 Step2: Train GT + Test(GT & Pred) 一体化组学提取引擎...")

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
            print(f"⚠️ 警告: 患者 {patient_id} 缺少 Image 或 GT 文件，跳过。")
            continue

        try:
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_gt = sitk.ReadImage(str(gt_path))

            gt_array = sitk.GetArrayFromImage(sitk_gt)
            if np.sum(gt_array) == 0:
                print(f"⚠️ 警告: 患者 {patient_id} 的 GT 为全空(无水肿)，跳过组学提取。")
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
        print(f"   shape = {df_train.shape}")
    else:
        print("\n⚠️ 未成功提取任何 Train GT 特征。")

    # =========================================================
    # Part B: Test GT + Pred
    # =========================================================
    test_patient_dirs = [d for d in TEST_DIR.iterdir() if d.is_dir()]
    print(f"\n📊 在 Test 文件夹中共发现 {len(test_patient_dirs)} 名患者。准备执行双轨开采...\n")

    gt_features_list = []
    pred_features_list = []

    for patient_dir in tqdm(test_patient_dirs, desc="Test 双轨特征提取进度"):
        patient_id = patient_dir.name

        img_path = patient_dir / f"{patient_id}_Image.nii.gz"
        gt_path = patient_dir / f"{patient_id}_GT.nii.gz"
        pred_path = patient_dir / f"{patient_id}_Pred.nii.gz"

        if not (img_path.exists() and gt_path.exists() and pred_path.exists()):
            print(f"\n⚠️ 警告: 患者 {patient_id} 缺少 Image、GT 或 Pred 文件，跳过。")
            continue

        try:
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_gt = sitk.ReadImage(str(gt_path))
            sitk_pred = sitk.ReadImage(str(pred_path))

            gt_array = sitk.GetArrayFromImage(sitk_gt)
            if np.sum(gt_array) == 0:
                print(f"\n⚠️ 警告: 患者 {patient_id} 的 GT 为全空(无水肿)，跳过。")
                continue

            sitk_img_norm = apply_pure_zscore(sitk_img)

            # ----- 轨道 A：GT -----
            features_gt = extractor.execute(sitk_img_norm, sitk_gt)
            patient_feat_gt = {"Patient_ID": patient_id}
            for key, value in features_gt.items():
                if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                    patient_feat_gt[key] = value
            gt_features_list.append(patient_feat_gt)

            # ----- 轨道 B：Pred -----
            patient_feat_pred = {"Patient_ID": patient_id}
            pred_array = sitk.GetArrayFromImage(sitk_pred)

            if np.sum(pred_array) == 0:
                # 原始逻辑：Pred 全空时，所有特征填 0
                print(f"\n   -> 🤖 AI 漏诊通报: 患者 {patient_id} 的 Pred 为空。已作 0 值惩罚填补。")
                for key in patient_feat_gt.keys():
                    if key == "Patient_ID":
                        patient_feat_pred[key] = patient_id
                    else:
                        patient_feat_pred[key] = 0.0
            else:
                features_pred = extractor.execute(sitk_img_norm, sitk_pred)
                for key, value in features_pred.items():
                    if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                        patient_feat_pred[key] = value

                # 防止某些 key 缺失
                for key in patient_feat_gt.keys():
                    if key not in patient_feat_pred:
                        patient_feat_pred[key] = 0.0

            pred_features_list.append(patient_feat_pred)

        except Exception as e:
            print(f"\n❌ 患者 {patient_id} 提取失败，原因: {e}")

    if len(gt_features_list) > 0 and len(pred_features_list) > 0:
        df_gt = pd.DataFrame(gt_features_list)
        df_pred = pd.DataFrame(pred_features_list)

        df_gt.to_csv(OUTPUT_TEST_GT, index=False)
        df_pred.to_csv(OUTPUT_TEST_PRED, index=False)

        print(f"\n🎉 完美收官！成功提取 {len(df_gt)} 名测试集患者的双轨特征！")
        print(f"💾 Test GT 特征已存入: {OUTPUT_TEST_GT}")
        print(f"💾 Test Pred 特征已存入: {OUTPUT_TEST_PRED}")
    else:
        print("\n⚠️ 未成功提取任何 Test 双轨特征，请检查测试集数据。")


if __name__ == "__main__":
    extract_run7_radiomics()