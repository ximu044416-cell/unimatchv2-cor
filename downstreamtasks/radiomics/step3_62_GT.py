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


def extract_test_dual_track_radiomics():
    # ================= 1. 路径准备 =================
    # ⚠️ 务必确认这里的路径是你存放测试集病人数据的文件夹！
    TEST_DIR = Path(r"F:\radiomics\test")
    YAML_PATH = Path(r"radiomics_features.yaml")

    # 最终生成的两份特征表
    OUTPUT_GT_CSV = Path(r"F:\radiomics\Test_GT_Features_62.csv")
    OUTPUT_PRED_CSV = Path(r"F:\radiomics\Test_Pred_Features_62.csv")

    print(f"🚀 启动 Test 测试集双轨 (GT & Pred) 组学榨取引擎...")

    if not YAML_PATH.exists():
        print(f"❌ 错误：找不到配置文件 {YAML_PATH}！")
        return

    # ================= 2. 初始化提取器 =================
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(str(YAML_PATH))
        print("✅ PyRadiomics 提取器初始化成功。")
    except Exception as e:
        print(f"❌ 提取器初始化失败: {e}")
        return

    patient_dirs = [d for d in TEST_DIR.iterdir() if d.is_dir()]
    print(f"📊 在 Test 文件夹中共发现 {len(patient_dirs)} 名患者。准备执行双轨开采...\n")

    gt_features_list = []
    pred_features_list = []

    # ================= 3. 逐患者双轨开采 =================
    for patient_dir in tqdm(patient_dirs, desc="双轨特征榨取进度"):
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

            # 检查医生金标准(GT)是否为空。如果医生都认为没水肿，直接跳过。
            gt_array = sitk.GetArrayFromImage(sitk_gt)
            if np.sum(gt_array) == 0:
                print(f"\n⚠️ 警告: 患者 {patient_id} 的 GT 为全空(无水肿)，跳过。")
                continue

            sitk_img_norm = apply_pure_zscore(sitk_img)

            # ----- 轨道 A：提取 GT 特征 -----
            features_gt = extractor.execute(sitk_img_norm, sitk_gt)
            patient_feat_gt = {"Patient_ID": patient_id}
            for key, value in features_gt.items():
                if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                    patient_feat_gt[key] = value
            gt_features_list.append(patient_feat_gt)

            # ----- 轨道 B：提取 Pred 特征 -----
            patient_feat_pred = {"Patient_ID": patient_id}
            pred_array = sitk.GetArrayFromImage(sitk_pred)

            if np.sum(pred_array) == 0:
                # 🔥 AI 漏诊防崩机制：如果 AI 掩码全黑，给所有特征填 0
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

            pred_features_list.append(patient_feat_pred)

        except Exception as e:
            print(f"\n❌ 患者 {patient_id} 提取失败，原因: {e}")

    # ================= 4. 分别保存两份 CSV =================
    if len(gt_features_list) > 0 and len(pred_features_list) > 0:
        df_gt = pd.DataFrame(gt_features_list)
        df_pred = pd.DataFrame(pred_features_list)

        df_gt.to_csv(OUTPUT_GT_CSV, index=False)
        df_pred.to_csv(OUTPUT_PRED_CSV, index=False)

        print(f"\n🎉 完美收官！成功提取 {len(df_gt)} 名测试集患者的双轨特征！")
        print(f"💾 金标准特征已存入: {OUTPUT_GT_CSV}")
        print(f"💾 AI预测特征已存入: {OUTPUT_PRED_CSV}")
        print("👉 弹药已就绪！现在你可以直接去运行你刚才的 Step 6 终极审判代码了！")
    else:
        print("\n⚠️ 未成功提取任何患者的特征，请检查测试集数据。")


if __name__ == "__main__":
    extract_test_dual_track_radiomics()