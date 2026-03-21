import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. 核心调度配置区 (请核对路径)
# ==========================================
# [名单来源] 上次训练切好的文件夹，用于提取 train/val/unlabel 的分类名单
REFERENCE_DIR = Path(r"F:\Dinov2_data\final_slices_cor")

# [数据来源 1] 原始配准好的 NIfTI 文件夹
SOURCE_NII_DIR = Path(r"F:\Dinov2_data\ALL_COR_FINAL")

# [数据来源 2] 包含 YOLO 绝对坐标的 TXT 文件夹
SOURCE_TXT_DIR = Path(r"F:\YOLO\Inference_Annotated_2")

# [最终目标] 聚合后的全新大本营
OUTPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_YOLO")

# 你原版脚本里的命名规范
NII_FILES = ["t1.nii.gz", "water.nii.gz", "fat.nii.gz", "label.nii.gz"]
# 如果你的文件是大写开头，也可以加进这个备用列表，脚本会自动适配
ALT_NII_FILES = ["T1.nii.gz", "Water.nii.gz", "Fat.nii.gz", "Mask.nii.gz"]


# ==========================================

def move_patient_data():
    if not REFERENCE_DIR.exists():
        print(f"❌ 找不到名单目录: {REFERENCE_DIR}")
        return

    splits = ['train', 'val', 'unlabel']
    total_patients_moved = 0

    print("\n" + "=" * 50)
    print("🚀 启动数据乾坤大挪移：物理聚合流水线")
    print("=" * 50 + "\n")

    for split in splits:
        split_ref_dir = REFERENCE_DIR / split
        if not split_ref_dir.exists():
            print(f"⚠️ 找不到分类目录 {split}，跳过...")
            continue

        # 提取当前集下的所有病人 ID
        patient_ids = [d.name for d in split_ref_dir.iterdir() if d.is_dir()]
        print(f"\n📦 正在聚合 [{split.upper()}] 组，共发现 {len(patient_ids)} 名患者...")

        for pid in tqdm(patient_ids, desc=f"搬运 {split} 数据"):
            # 1. 创建目标目录
            target_patient_dir = OUTPUT_DIR / split / pid
            target_patient_dir.mkdir(parents=True, exist_ok=True)

            # 2. 创建专属的 YOLO 坐标收纳盒
            target_txt_dir = target_patient_dir / "yolo_boxes"
            target_txt_dir.mkdir(exist_ok=True)

            # 3. 搬运 NIfTI 核心四件套 (T1, Water, Fat, Label)
            src_nii_patient = SOURCE_NII_DIR / pid
            for nii_name, alt_name in zip(NII_FILES, ALT_NII_FILES):
                # 兼容大小写文件名
                src_nii = src_nii_patient / nii_name
                if not src_nii.exists():
                    src_nii = src_nii_patient / alt_name

                if src_nii.exists():
                    # shutil.copy2 会保留原始文件的创建时间等元数据
                    shutil.copy2(src_nii, target_patient_dir / nii_name)
                else:
                    print(f"\n⚠️ 警告: 病人 {pid} 缺失序列文件 {nii_name}！")

            # 4. 搬运该病人的所有 YOLO TXT 坐标
            src_txt_patient = SOURCE_TXT_DIR / pid
            if src_txt_patient.exists():
                txt_files = list(src_txt_patient.glob("*.txt"))
                for txt_file in txt_files:
                    # 把所有的 txt 统统复制进 yolo_boxes 文件夹
                    shutil.copy2(txt_file, target_txt_dir / txt_file.name)
            else:
                print(f"\n⚠️ 警告: 病人 {pid} 找不到 YOLO TXT 文件夹！")

            total_patients_moved += 1

    print("\n" + "=" * 50)
    print("🎉 物理分拣与聚合彻底完工！")
    print(f"   - 🎯 总计完美迁移患者: {total_patients_moved} 例")
    print(f"   - 📂 请前往 {OUTPUT_DIR} 查阅全新基地！")
    print("==================================================")


if __name__ == "__main__":
    move_patient_data()