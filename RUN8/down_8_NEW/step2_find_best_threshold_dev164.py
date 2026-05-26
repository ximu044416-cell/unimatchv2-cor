import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================================================
# 路径
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # F:\cor\RUN8
OUT_ROOT = Path(r"F:\cor\RUN8\down_8_NEW\down_2_threshold")
RECON_DIR = OUT_ROOT / "reconstructed_dev164"

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

PER_CASE_DIR = OUT_ROOT / "metrics_per_case"
PER_CASE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_ROOT / "threshold_summary_dev164.csv"
SUMMARY_JSON = OUT_ROOT / "threshold_summary_dev164.json"
SUMMARY_TXT = OUT_ROOT / "best_threshold_summary.txt"
PLOT_PATH = OUT_ROOT / "Threshold_Optimization_Curve_RUN8_dev164.png"


def calculate_3d_dice(pred_np, gt_np):
    intersection = np.logical_and(pred_np == 1, gt_np == 1).sum()
    sum_masks = (pred_np == 1).sum() + (gt_np == 1).sum()
    if sum_masks == 0:
        return 1.0
    return 2.0 * intersection / sum_masks


def rve_percent(pred_np, gt_np):
    pred_vol = (pred_np == 1).sum()
    gt_vol = (gt_np == 1).sum()

    if gt_vol == 0:
        return np.nan
    return 100.0 * (pred_vol - gt_vol) / (gt_vol + 1e-8)


def main():
    patient_dirs = sorted([p for p in RECON_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if len(patient_dirs) == 0:
        raise RuntimeError(f"❌ 重建目录为空：{RECON_DIR}，请先跑 step1。")

    print("🚀 启动 RUN8 dev164 阈值校准（只用开发集，不碰 test40）")
    print(f"📂 重建目录: {RECON_DIR}")
    print(f"📊 患者数: {len(patient_dirs)}")
    print(f"🎯 阈值列表: {THRESHOLDS}")

    summary_rows = []

    for thr in THRESHOLDS:
        t_str = f"{int(round(thr * 100)):03d}"
        case_rows = []

        for patient_dir in tqdm(patient_dirs, desc=f"Threshold {thr:.2f}"):
            patient_id = patient_dir.name
            gt_path = patient_dir / f"{patient_id}_GT.nii.gz"
            pred_path = patient_dir / f"{patient_id}_Pred_{t_str}.nii.gz"

            if not gt_path.exists() or not pred_path.exists():
                continue

            gt_np = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.uint8)
            pred_np = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(np.uint8)

            gt_vol = int((gt_np == 1).sum())
            pred_vol = int((pred_np == 1).sum())

            dice = calculate_3d_dice(pred_np, gt_np)
            rve = rve_percent(pred_np, gt_np)
            abs_rve = np.abs(rve) if not np.isnan(rve) else np.nan

            case_rows.append({
                "Patient_ID": patient_id,
                "Threshold": thr,
                "GT_Volume": gt_vol,
                "Pred_Volume": pred_vol,
                "Dice_3D": dice,
                "RVE_percent": rve,
                "Abs_RVE_percent": abs_rve,
                "GT_Empty": int(gt_vol == 0),
                "Pred_Empty": int(pred_vol == 0)
            })

        df_case = pd.DataFrame(case_rows)
        per_case_csv = PER_CASE_DIR / f"per_case_threshold_{t_str}.csv"
        df_case.to_csv(per_case_csv, index=False, encoding="utf-8-sig")

        df_nonempty = df_case[df_case["GT_Empty"] == 0].copy()

        mean_dice = float(df_case["Dice_3D"].mean()) if len(df_case) > 0 else np.nan
        median_dice = float(df_case["Dice_3D"].median()) if len(df_case) > 0 else np.nan

        mean_rve = float(df_nonempty["RVE_percent"].mean()) if len(df_nonempty) > 0 else np.nan
        median_rve = float(df_nonempty["RVE_percent"].median()) if len(df_nonempty) > 0 else np.nan
        mean_abs_rve = float(df_nonempty["Abs_RVE_percent"].mean()) if len(df_nonempty) > 0 else np.nan
        median_abs_rve = float(df_nonempty["Abs_RVE_percent"].median()) if len(df_nonempty) > 0 else np.nan

        pred_empty_rate = float(df_case["Pred_Empty"].mean()) if len(df_case) > 0 else np.nan

        summary_rows.append({
            "Threshold": thr,
            "Num_Cases": int(len(df_case)),
            "Num_GT_Empty_Cases": int(df_case["GT_Empty"].sum()),
            "Num_Pred_Empty_Cases": int(df_case["Pred_Empty"].sum()),
            "Pred_Empty_Rate": pred_empty_rate,
            "Mean_3D_Dice": mean_dice,
            "Median_3D_Dice": median_dice,
            "Mean_RVE_percent": mean_rve,
            "Median_RVE_percent": median_rve,
            "Mean_Abs_RVE_percent": mean_abs_rve,
            "Median_Abs_RVE_percent": median_abs_rve,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    # =========================
    # 自动找两个“最佳阈值”
    # =========================
    valid_dice = df_summary["Mean_3D_Dice"].astype(float)
    best_dice_idx = valid_dice.idxmax()
    best_dice_thr = float(df_summary.loc[best_dice_idx, "Threshold"])
    best_dice_val = float(df_summary.loc[best_dice_idx, "Mean_3D_Dice"])

    valid_abs_rve = df_summary["Mean_Abs_RVE_percent"].astype(float)
    best_rve_idx = valid_abs_rve.idxmin()
    best_rve_thr = float(df_summary.loc[best_rve_idx, "Threshold"])
    best_rve_val = float(df_summary.loc[best_rve_idx, "Mean_Abs_RVE_percent"])

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("RUN8 dev164 Threshold Calibration Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best threshold by Mean 3D Dice      : {best_dice_thr:.2f} (Dice={best_dice_val:.4f})\n")
        f.write(f"Best threshold by Mean Abs RVE      : {best_rve_thr:.2f} (AbsRVE={best_rve_val:.4f}%)\n")
        f.write("\n")
        f.write("Reminder: final downstream deployment threshold should be decided on dev164 only,\n")
        f.write("and then frozen before evaluation on test40.\n")

    # =========================
    # 画图：3D Dice 曲线
    # =========================
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(
        df_summary["Threshold"],
        df_summary["Mean_3D_Dice"],
        marker="o",
        linestyle="-",
        linewidth=2
    )
    plt.plot(
        best_dice_thr,
        best_dice_val,
        marker="*",
        markersize=14,
        label=f"Best Dice: {best_dice_thr:.2f} ({best_dice_val:.3f})"
    )
    plt.xlabel("Probability Threshold")
    plt.ylabel("Mean 3D Dice")
    plt.title("Threshold Optimization via 3D Dice (RUN8 dev164)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    print("\n✅ RUN8 dev164 阈值校准完成。")
    print(f"📄 summary csv : {SUMMARY_CSV}")
    print(f"📄 summary txt : {SUMMARY_TXT}")
    print(f"📈 curve png   : {PLOT_PATH}")
    print(f"📁 per-case dir: {PER_CASE_DIR}")


if __name__ == "__main__":
    main()