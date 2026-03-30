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
    # 1. 转换为 Numpy 数组
    img_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

    # 2. 找到肉体像素进行纯净统计
    valid_pixels = img_array[img_array > 0]

    if len(valid_pixels) == 0:
        return sitk_image  # 如果全黑，直接返回原图

    mean_val = np.mean(valid_pixels)
    std_val = np.std(valid_pixels)

    if std_val < 1e-5:
        std_val = 1e-5

    # 3. Z-score 归一化
    img_normalized = (img_array - mean_val) / std_val

    # 将原本纯黑的背景（或者极低噪声）强行压回一个极低的值（比如最小值的稍微小一点），
    # 保证背景不会被提升导致纹理计算错误
    img_normalized[img_array <= 0] = img_normalized.min() - 1.0

    # 4. 转回 SimpleITK 对象
    new_sitk_image = sitk.GetImageFromArray(img_normalized)

    # 🔥 核心防错锁：完美继承原始图像的空间几何信息 (否则 PyRadiomics 必崩)
    new_sitk_image.CopyInformation(sitk_image)

    return new_sitk_image


def extract_train_gt_radiomics():
    # ================= 1. 路径准备 =================
    TRAIN_DIR = Path(r"F:\radiomics\train")  # 你的 142 个病人所在的目录
    YAML_PATH = Path(r"radiomics_features.yaml")  # 刚才保存的完美配置文件
    OUTPUT_CSV = Path(r"F:\radiomics\Train_pred_Features.csv")  # 最终生成的特征表

    print(f"🚀 启动 Step 3: 训练集金标准 (GT) 组学榨取引擎...")

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
    patient_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    print(f"📊 在 Train 文件夹中共发现 {len(patient_dirs)} 名患者。准备开采 1200+ 维特征...\n")

    all_features_list = []

    # ================= 4. 逐患者开采 =================
    for patient_dir in tqdm(patient_dirs, desc="特征榨取进度"):
        patient_id = patient_dir.name

        img_path = patient_dir / f"{patient_id}_Image.nii.gz"
        pred_path = patient_dir / f"{patient_id}_Pred.nii.gz"

        if not (img_path.exists() and pred_path.exists()):
            print(f"⚠️ 警告: 患者 {patient_id} 缺少 Image 或 GT 文件，跳过。")
            continue

        try:
            # 1. 读取 NIfTI
            sitk_img = sitk.ReadImage(str(img_path))
            sitk_pred = sitk.ReadImage(str(pred_path))

            # 检查 GT 是否为空 (如果没有水肿，PyRadiomics 会报错，必须跳过或标记)
            pred_array = sitk.GetArrayFromImage(sitk_pred)
            if np.sum(pred_array) == 0:
                print(f"⚠️ 警告: 患者 {patient_id} 的 GT 为全空(无水肿)，跳过组学提取。")
                continue

            # 2. 纯净 Z-score 处理，完美保留物理坐标
            sitk_img_norm = apply_pure_zscore(sitk_img)

            # 3. 核心提取动作！
            # 注意：传入的是归一化后的原图 和 pred！
            features = extractor.execute(sitk_img_norm, sitk_pred)

            # 4. 整理结果字典
            # PyRadiomics 返回的字典里包含了一些 general_info（如版本号、提取时间等），我们只保留实际特征
            patient_features = {"Patient_ID": patient_id}
            for key, value in features.items():
                if key.startswith("original_") or key.startswith("log-") or key.startswith("wavelet-"):
                    patient_features[key] = value

            all_features_list.append(patient_features)

        except Exception as e:
            print(f"❌ 患者 {patient_id} 提取失败，原因: {e}")

    # ================= 5. 保存为 CSV =================
    if len(all_features_list) > 0:
        df = pd.DataFrame(all_features_list)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 极其完美的收官！成功提取 {len(all_features_list)} 名患者的组学特征！")
        print(f"💎 特征矩阵维度: {df.shape[0]} 行 × {df.shape[1] - 1} 列 (特征数)")
        print(f"💾 特征表已存入: {OUTPUT_CSV}")
        print("👉 随时准备开启 Step 4：LASSO 降维与 Rad-score 构建！")
    else:
        print("\n⚠️ 未成功提取任何患者的特征，请检查输入数据。")


if __name__ == "__main__":
    extract_train_gt_radiomics()