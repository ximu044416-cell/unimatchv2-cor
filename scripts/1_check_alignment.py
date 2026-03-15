import SimpleITK as sitk
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你标注好的黄金数据集目录
INPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_FINAL")

# 2. 排查报告的保存位置
OUTPUT_CSV = Path(r"F:\Dinov2_data\alignment_check_report_1.csv")

# 3. 容差 (1e-5，小数点后五位)
TOLERANCE = 1e-5


# ===========================================

def check_geometry(img_ref, img_test):
    """严格校验核心物理参数，返回 (是否通过, 错误详情)"""
    # 1. 检查尺寸
    if img_ref.GetSize() != img_test.GetSize():
        return False, f"Size 不匹配: {img_ref.GetSize()} vs {img_test.GetSize()}"

    # 2. 检查分辨率 (Spacing)
    if not np.allclose(img_ref.GetSpacing(), img_test.GetSpacing(), atol=TOLERANCE):
        return False, "Spacing 不匹配"

    # 3. 检查原点坐标 (Origin)
    if not np.allclose(img_ref.GetOrigin(), img_test.GetOrigin(), atol=TOLERANCE):
        return False, "Origin 不匹配 (空间位置错位)"

    # 4. 检查扫描方向 (Direction)
    if not np.allclose(img_ref.GetDirection(), img_test.GetDirection(), atol=TOLERANCE):
        return False, "Direction 不匹配 (扫描方向不同)"

    return True, "✅ 完美对齐"


def main():
    if not INPUT_DIR.exists():
        print(f"❌ 找不到目录: {INPUT_DIR}")
        return

    case_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    print(f"🔍 开始对 {len(case_dirs)} 个病例进行严格的空间对齐排查...")

    report_data = []

    for case_dir in tqdm(case_dirs):
        case_id = case_dir.name

        path_w = case_dir / "water.nii.gz"
        path_f = case_dir / "fat.nii.gz"
        path_t1 = case_dir / "t1.nii.gz"

        # 记录默认状态
        case_status = {
            "Patient_ID": case_id,
            "T1_Alignment": "Missing",
            "Fat_Alignment": "Missing",
            "Overall_Status": "⚠️ 缺文件"
        }

        if path_w.exists():
            try:
                # 只读取元数据信息，不加载全部像素，速度极快
                img_w = sitk.ReadImage(str(path_w))

                # 检查 T1
                if path_t1.exists():
                    img_t1 = sitk.ReadImage(str(path_t1))
                    t1_ok, t1_msg = check_geometry(img_w, img_t1)
                    case_status["T1_Alignment"] = t1_msg

                # 检查 Fat
                if path_f.exists():
                    img_f = sitk.ReadImage(str(path_f))
                    fat_ok, fat_msg = check_geometry(img_w, img_f)
                    case_status["Fat_Alignment"] = fat_msg

                # 综合评判
                if "✅" in case_status["T1_Alignment"] and "✅" in case_status["Fat_Alignment"]:
                    case_status["Overall_Status"] = "✅ 全部通过"
                else:
                    case_status["Overall_Status"] = "❌ 需要配准"

            except Exception as e:
                case_status["Overall_Status"] = f"读取报错: {e}"

        report_data.append(case_status)

    # 生成 DataFrame 并保存
    df = pd.DataFrame(report_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 在控制台打印简报
    passed_count = len(df[df["Overall_Status"] == "✅ 全部通过"])
    failed_count = len(df[df["Overall_Status"] == "❌ 需要配准"])

    print("\n" + "=" * 40)
    print("📋 排查完成报告")
    print("=" * 40)
    print(f"总病例数: {len(case_dirs)}")
    print(f"✅ 完美对齐的病例: {passed_count}")
    print(f"❌ 空间错位(需配准): {failed_count}")
    print(f"\n👉 详细报告已生成，请用 Excel 打开查看: ")
    print(f"   {OUTPUT_CSV}")
    print("\n💡 接下来：请查看 Excel 表格。如果有打 ❌ 的，我们再跑你的那套配准代码把它修正过来！")


if __name__ == "__main__":
    main()