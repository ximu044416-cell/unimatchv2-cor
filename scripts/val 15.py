import os
import shutil
import random
from pathlib import Path

# ================= 配置区域 =================
BASE_DIR = Path(r"F:\Dinov2_data\final_slices_cor")
LABEL_DIR = BASE_DIR / "label"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

VAL_SIZE = 15
SEED = 42  # 锁定随机种子，确保每次分出来的都是同一批人
# ===========================================

def main():
    if not LABEL_DIR.exists():
        print(f"❌ 找不到标签目录: {LABEL_DIR}")
        return

    # 获取所有病人的文件夹
    cases = [d for d in LABEL_DIR.iterdir() if d.is_dir()]
    total_cases = len(cases)
    print(f"📦 发现总计有标签病例: {total_cases} 例")

    if total_cases < VAL_SIZE:
        print("❌ 病例数不足，无法划分！")
        return

    # 设置随机种子并打乱
    random.seed(SEED)
    random.shuffle(cases)

    # 划分
    val_cases = cases[:VAL_SIZE]
    train_cases = cases[VAL_SIZE:]

    # 创建目标目录
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # 物理移动文件 (速度极快，不占用额外硬盘空间)
    print(f"🚚 正在将 {VAL_SIZE} 例移动到 val 验证集...")
    for case in val_cases:
        shutil.move(str(case), str(VAL_DIR / case.name))

    print(f"🚚 正在将剩余 {len(train_cases)} 例移动到 train 训练集...")
    for case in train_cases:
        shutil.move(str(case), str(TRAIN_DIR / case.name))

    # 清理现场
    if not any(LABEL_DIR.iterdir()):
        LABEL_DIR.rmdir()
        print("🗑️ 原 label 文件夹已清空并安全移除。")

    print("\n" + "=" * 50)
    print("🎉 物理分流大功告成！")
    print(f"   - 🏋️ 训练集 (train): {len(train_cases)} 例")
    print(f"   - 🧪 验证集 (val): {VAL_SIZE} 例")
    print(f"   - 👻 无标签集 (unlabel): 原封不动保留")
    print("==================================================")

if __name__ == "__main__":
    main()