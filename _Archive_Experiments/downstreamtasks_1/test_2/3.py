import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from pathlib import Path
from tqdm import tqdm


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


def extract_multi_threshold_radiomics():
    # ================= 1. 路径准备 =================
    # 🔥 指向刚才生成的包含多阈值掩码的测试集文件夹
    DATA_DIR = Path(r"F:\check\test")
    YAML_PATH = Path(r"../../../downstreamtasks/radiomics/radiomics_features.yaml")
    # 🔥 所有生成的 CSV 统一输出到这个总控制台文件夹
    OUTPUT_DIR = Path(r"F:\check")

    # 你指定的三个黄金对比阈值
    target_thresholds = ["065", "070", "075"]

    print(f"🚀 启动 Step 3: 多阈值测试集 (Pred) 组学并行榨取引擎...")

    if not YAML_PATH.exists():
        print(f"❌ 错误：找不到配置文件 {YAML_PATH}！")
        return

    # ================= 2. 初始化提取器 =================
    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(str(YAML_PATH))
        print("✅ PyRadiomics 提取器初始化成功，高阶滤波器已挂载。")
    except Exception as e:
        print(f"❌ 提取器初始化失败: {e}")
        return

    # ================= 3. 搜刮病人列表 =================
    if not DATA_DIR.exists():
        print(f"❌ 错误：找不到数据文件夹 {DATA_DIR}！")
        return

    patient_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"📊 在 Test 文件夹中共发现 {len(patient_dirs)} 名患者。准备开采...\n")

    # 创建一个字典，为每个阈值准备一个独立的列表，用来存放各自的特征
    all_features_dict = {t: [] for t in target_thresholds}

    # ================= 4. 逐患者开采 (极致性能优化版) =================
    for patient_dir in tqdm(patient_dirs, desc="多阈值特征榨取进度"):
        patient_id = patient_dir.name
        img_path = patient_dir / f"{patient_id}_Image.nii.gz"

        if not img_path.exists():
            print(f"⚠️ 警告: 患者 {patient_id} 缺少 Image 文件，跳过。")
            continue

        try:
            # 💡 性能优化点：原图只读 1 次，Z-score 只算 1 次！
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_img_norm = apply_pure_zscore(sitk_img)

            # 针对该病人的原图，连续套用 3 个阈值的掩码
            for t in target_thresholds:
                pred_path = patient_dir / f"{patient_id}_Pred_{t}.nii.gz"

                if not pred_path.exists():
                    continue

                sitk_pred = sitk.ReadImage(str(pred_path))
                pred_array = sitk.GetArrayFromImage(sitk_pred)

                # 防爆机制：如果这个高阈值导致水肿全部消失(全为0)，必须跳过
                if np.sum(pred_array) == 0:
                    print(f"\n⚠️ 警告: 患者 {patient_id} 在阈值 {t} 下预测出的水肿体积为 0，跳过该阈值提取。")
                    continue

                # 核心提取动作！
                features = extractor.execute(sitk_img_norm, sitk_pred)

                # 整理结果字典
                patient_features = {"Patient_ID": patient_id}
                for key, value in features.items():
                    if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                        patient_features[key] = value

                # 装入对应阈值的专属列表中
                all_features_dict[t].append(patient_features)

        except Exception as e:
            print(f"❌ 患者 {patient_id} 提取失败，原因: {e}")

    # ================= 5. 批量保存为 CSV =================
    print("\n================ 榨取报告 ================")
    for t in target_thresholds:
        features_list = all_features_dict[t]
        if len(features_list) > 0:
            df = pd.DataFrame(features_list)
            output_csv = OUTPUT_DIR / f"Test_Pred_Features_{t}.csv"
            df.to_csv(output_csv, index=False)
            print(f"🎉 阈值 {t} 收官！成功提取 {len(features_list)} 名患者，存入 -> {output_csv.name}")
        else:
            print(f"⚠️ 阈值 {t} 未提取到任何有效特征。")

    print("👉 所有文件已统一输出至 F:\\check 目录，随时准备进行 Step 6 对比测试！")


if __name__ == "__main__":
    extract_multi_threshold_radiomics()