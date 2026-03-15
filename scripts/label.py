import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import gc

# ================= 配置 =================
DATA_DIR = Path(r"F:\Dinov2_data\ALL_COR_N4")


# =======================================

def main():
    print("🚀 启动全体 Label 物理坐标强制修复程序 (完美侦察版)...")

    if not DATA_DIR.exists():
        print(f"❌ 找不到目录: {DATA_DIR}")
        return

    case_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    print(f"📂 共发现 {len(case_dirs)} 个病例，开始修复...\n")

    fixed_count = 0
    skip_count = 0
    empty_warn_count = 0

    for case_dir in tqdm(case_dirs, desc="Fixing Labels"):
        path_w = case_dir / "water.nii.gz"
        path_label = case_dir / "label.nii.gz"
        path_backup = case_dir / "label_backup.nii.gz"

        if not path_w.exists() or not path_label.exists():
            skip_count += 1
            continue

        # 鲁棒备份：只 Copy，不 Move
        if not path_backup.exists():
            shutil.copy2(str(path_label), str(path_backup))

        source_label_path = path_backup

        # 读取图像
        img_w = sitk.ReadImage(str(path_w), sitk.sitkFloat32)
        img_label = sitk.ReadImage(str(source_label_path), sitk.sitkUInt8)

        # 核心修复：重采样
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_w)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        try:
            fixed_label = resampler.Execute(img_label)
        except Exception as e:
            print(f"\n❌ {case_dir.name} 重采样失败: {e}")
            continue

        # --- 融入你的高级微调逻辑 ---
        label_view = sitk.GetArrayViewFromImage(fixed_label)
        unique_labels = np.unique(label_view)

        if len(unique_labels) == 1 and unique_labels[0] == 0:
            # 只有背景 0，说明病灶丢了
            origin_w = img_w.GetOrigin()
            origin_l = img_label.GetOrigin()
            print(f"\n⚠️ 警告: {case_dir.name} 物理坐标不匹配，标签全空！")
            print(f"   👉 Water Origin: {[round(x, 2) for x in origin_w]}")
            print(f"   👉 Label Origin: {[round(x, 2) for x in origin_l]}")
            empty_warn_count += 1
        # -----------------------------

        # 保存修复后的 Label
        sitk.WriteImage(fixed_label, str(path_label))
        fixed_count += 1

        # 内存防爆盾
        del img_w, img_label, fixed_label, label_view, resampler
        if fixed_count % 10 == 0:
            gc.collect()

    print("\n" + "=" * 40)
    print("🎉 全体 Label 修复圆满完成！")
    print(f"✅ 成功对齐: {fixed_count} 例")
    print(f"⏭️ 跳过(无Label): {skip_count} 例")
    if empty_warn_count > 0:
        print(f"🚨 警告: 发现 {empty_warn_count} 个空标签，请看上方日志！")
    else:
        print("🌟 完美！没有任何标签丢失，全部精准落盘！")
    print("\n👉 下一步：请立刻去跑 step0_global_geometry_check.py 验证战果！")


if __name__ == "__main__":
    main()