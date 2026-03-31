import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
from tqdm import tqdm


def apply_pure_zscore(sitk_image):
    """纯净 Z-score 归一化，完美保留物理空间坐标"""
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


def extract_test_gt_radiomics():
    # ================= 1. 路径准备 =================
    # 我们直接去 check\test 文件夹里读测试集病人
    TEST_DIR = Path(r"F:\check\test")
    YAML_PATH = Path(r"../../../downstreamtasks/radiomics/radiomics_features.yaml")
    # 🔥 核心：输出到报错提示找不到的那个路径！
    OUTPUT_CSV = Path(r"F:\radiomics\Test_GT_Features.csv")

    print(f"🚀 启动补课: 测试集金标准 (Test GT) 组学榨取引擎...")

    if not YAML_PATH.exists():
        print(f"❌ 错误：找不到配置文件 {YAML_PATH}！")
        return

    # ================= 2. 初始化提取器 =================
    extractor = featureextractor.RadiomicsFeatureExtractor(str(YAML_PATH))

    # ================= 3. 搜刮测试集病人 =================
    patient_dirs = [d for d in TEST_DIR.iterdir() if d.is_dir()]
    print(f"📊 在 Test 文件夹中共发现 {len(patient_dirs)} 名患者。准备开采...\n")

    all_features_list = []

    # ================= 4. 逐患者开采 =================
    for patient_dir in tqdm(patient_dirs, desc="Test GT 特征榨取进度"):
        patient_id = patient_dir.name

        img_path = patient_dir / f"{patient_id}_Image.nii.gz"
        gt_path = patient_dir / f"{patient_id}_GT.nii.gz"

        if not (img_path.exists() and gt_path.exists()):
            continue

        try:
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_gt = sitk.ReadImage(str(gt_path))

            # 检查 GT 是否为空
            gt_array = sitk.GetArrayFromImage(sitk_gt)
            if np.sum(gt_array) == 0:
                continue

            sitk_img_norm = apply_pure_zscore(sitk_img)
            features = extractor.execute(sitk_img_norm, sitk_gt)

            patient_features = {"Patient_ID": patient_id}
            for key, value in features.items():
                if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                    patient_features[key] = value

            all_features_list.append(patient_features)

        except Exception as e:
            print(f"❌ 患者 {patient_id} 提取失败，原因: {e}")

    # ================= 5. 保存 =================
    if len(all_features_list) > 0:
        df = pd.DataFrame(all_features_list)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 完美收官！成功提取 {len(all_features_list)} 名测试集患者的 GT 组学特征！")
        print(f"💾 特征表已存入: {OUTPUT_CSV}")
    else:
        print("\n⚠️ 未提取到特征。")


if __name__ == "__main__":
    extract_test_gt_radiomics()