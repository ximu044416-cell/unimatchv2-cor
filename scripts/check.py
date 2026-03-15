import pandas as pd
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = Path(r"F:\Dinov2_data\ALL_COR_N4")
OUTPUT_CSV = Path(r"F:\Dinov2_data\global_geometry_report1.csv")

# 容差值 (对于浮点数坐标，允许极微小的计算误差)
TOLERANCE = 1e-3


# ===========================================

def get_image_metadata(filepath):
    """只读取头文件信息，极其快速，不占内存"""
    if not filepath or not filepath.exists():
        return None

    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(filepath))
        reader.ReadImageInformation()

        return {
            'Size': reader.GetSize(),
            'Spacing': np.array(reader.GetSpacing()),
            'Origin': np.array(reader.GetOrigin()),
            'Direction': np.array(reader.GetDirection())
        }
    except Exception:
        return None


def compare_geometry(ref_meta, target_meta):
    """严苛比对，并返回具体的切实位置/差异信息"""
    if ref_meta is None:
        return "Base Missing"
    if target_meta is None:
        return "Missing"

    # 1. 检查 Size
    if ref_meta['Size'] != target_meta['Size']:
        return f"Size Diff: {ref_meta['Size']} vs {target_meta['Size']}"

    # 2. 检查 Spacing
    if not np.allclose(ref_meta['Spacing'], target_meta['Spacing'], atol=TOLERANCE):
        s1 = tuple(np.round(ref_meta['Spacing'], 2))
        s2 = tuple(np.round(target_meta['Spacing'], 2))
        return f"Spacing Diff: {s1} vs {s2}"

    # 3. 检查 Origin (这是最关键的物理位置)
    if not np.allclose(ref_meta['Origin'], target_meta['Origin'], atol=TOLERANCE):
        o1 = tuple(np.round(ref_meta['Origin'], 2))
        o2 = tuple(np.round(target_meta['Origin'], 2))
        return f"Origin Diff: {o1} vs {o2}"

    # 4. 检查 Direction
    if not np.allclose(ref_meta['Direction'], target_meta['Direction'], atol=TOLERANCE):
        return "Direction Mismatch"

    return "True"


def format_tuple(arr):
    """格式化数值用于 Excel 显示"""
    if arr is None: return "N/A"
    return str(tuple(np.round(arr, 2)))


def main():
    print("🚀 启动全局 NIfTI 物理坐标系详细普查...")

    if not INPUT_DIR.exists():
        print(f"❌ 找不到输入目录: {INPUT_DIR}")
        return

    case_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    print(f"📂 共发现 {len(case_dirs)} 个病例文件夹。")

    report_data = []

    for case_dir in tqdm(case_dirs, desc="Scanning metadata"):
        case_id = case_dir.name

        # 定义文件路径
        paths = {
            'water': case_dir / "water.nii.gz",
            't1': case_dir / "t1.nii.gz",
            'fat': case_dir / "fat.nii.gz",
            'label': case_dir / "label.nii.gz"
        }

        # 提取所有元数据
        metas = {k: get_image_metadata(v) for k, v in paths.items()}
        meta_w = metas['water']

        # 对比逻辑
        res_label = compare_geometry(meta_w, metas['label'])
        res_t1 = compare_geometry(meta_w, metas['t1'])
        res_fat = compare_geometry(meta_w, metas['fat'])

        # 记录到字典中
        report_data.append({
            "Patient_ID": case_id,
            "Label_Match?": res_label,
            "T1_Match?": res_t1,
            "Fat_Match?": res_fat,
            "Water_Origin": format_tuple(meta_w['Origin']) if meta_w else "Missing",
            "Label_Origin": format_tuple(metas['label']['Origin']) if metas['label'] else "Missing",
            "Water_Spacing": format_tuple(meta_w['Spacing']) if meta_w else "Missing",
            "Label_Spacing": format_tuple(metas['label']['Spacing']) if metas['label'] else "Missing",
            "Water_Size": str(meta_w['Size']) if meta_w else "Missing",
            "Label_Size": str(metas['label']['Size']) if metas['label'] else "Missing",
        })

    # 保存为 CSV
    df = pd.DataFrame(report_data)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 统计汇总
    error_cases = df[df["Label_Match?"].str.contains("Diff|Mismatch", na=False)]

    print("\n" + "=" * 40)
    print("🎉 普查结束！详细报告已生成。")
    print(f"📊 报告位置: {OUTPUT_CSV}")
    print(f"🚨 发现 Label 物理位置异常: {len(error_cases)} / {len(df)}")
    if len(error_cases) > 0:
        print("💡 请在 CSV 的 'Label_Match?' 列查看具体的数值差异。")
    print("=" * 40)


if __name__ == "__main__":
    main()