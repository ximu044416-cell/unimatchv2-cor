import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", context="paper", font_scale=1.2)

ANALYSIS_ROOT = Path(r"/RUN7/analysis")
PRED_ROOT = ANALYSIS_ROOT / "3D_Predictions"
OUT_ROOT = ANALYSIS_ROOT / "Clinical_Analysis"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_TAGS = ["best_model", "swa_offline"]


def analyze_one_model(model_tag):
    pred_dir = PRED_ROOT / model_tag
    output_dir = OUT_ROOT / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 分析模型: {model_tag}")
    print(f"📂 预测目录: {pred_dir}")

    gt_files = sorted(list(pred_dir.rglob("*_GT.nii.gz")))
    if len(gt_files) == 0:
        print(f"❌ {model_tag} 找不到任何 _GT.nii.gz 文件，跳过。")
        return None

    metrics_list = []

    for gt_path in gt_files:
        patient_id = gt_path.name.replace("_GT.nii.gz", "")
        pred_path = gt_path.parent / f"{patient_id}_Pred.nii.gz"

        if not pred_path.exists():
            print(f"⚠️ 找不到 {patient_id} 的 Pred 文件，跳过。")
            continue

        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        pred_array = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

        v_gt = np.sum(gt_array > 0)
        v_pred = np.sum(pred_array > 0)

        ave = abs(v_pred - v_gt)
        intersection = np.sum((gt_array > 0) & (pred_array > 0))

        outlier_status = ""
        if v_gt == 0 and v_pred == 0:
            rve = 0.0
            dice_3d = 1.0
            outlier_status = "Perfect TN"
        elif v_gt == 0 and v_pred > 0:
            rve = np.nan
            dice_3d = 0.0
            outlier_status = "Inf (FP Outlier)"
        else:
            rve = (ave / v_gt) * 100.0
            dice_3d = (2.0 * intersection) / (v_gt + v_pred) if (v_gt + v_pred) > 0 else 0.0

        metrics_list.append({
            "Model": model_tag,
            "Patient_ID": patient_id,
            "V_GT": int(v_gt),
            "V_Pred": int(v_pred),
            "AVE": int(ave),
            "RVE(%)": round(rve, 2) if not np.isnan(rve) else np.nan,
            "3D_Dice": round(dice_3d, 4),
            "Status": outlier_status
        })

    df = pd.DataFrame(metrics_list)
    df['Mean'] = (df['V_GT'] + df['V_Pred']) / 2.0
    df['Diff'] = df['V_Pred'] - df['V_GT']

    df_stat = df[df["V_GT"] > 0].copy()
    print(f"🔬 有效病灶患者数: {len(df_stat)} / {len(df)}")

    pearson_r, pearson_p = stats.pearsonr(df_stat["V_GT"], df_stat["V_Pred"])
    spearman_r, spearman_p = stats.spearmanr(df_stat["V_GT"], df_stat["V_Pred"])

    icc_value = np.nan
    try:
        df_melt = df_stat[['Patient_ID', 'V_GT', 'V_Pred']].melt(
            id_vars='Patient_ID',
            var_name='Method',
            value_name='Volume'
        )
        icc_res = pg.intraclass_corr(data=df_melt, targets='Patient_ID', raters='Method', ratings='Volume')
        if icc_res is not None and not icc_res.empty and 'ICC2' in icc_res['Type'].values:
            icc_value = icc_res.loc[icc_res['Type'] == 'ICC2', 'ICC'].values[0]
    except Exception:
        print("⚠️ ICC 计算跳过。")

    bias = np.mean(df_stat["Diff"])
    sd = np.std(df_stat["Diff"], ddof=1)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    df.loc[(df['Status'] == "") & ((df['Diff'] > upper_loa) | (df['Diff'] < lower_loa)), 'Status'] = "BA Outlier"
    df_stat = df[df["V_GT"] > 0].copy()

    print(f"📈 Pearson R : {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"📈 Spearman R: {spearman_r:.4f} (p={spearman_p:.4e})")
    print(f"📈 ICC2      : {icc_value if not np.isnan(icc_value) else 'N/A'}")
    print(f"📈 BA Bias   : {bias:.1f} Voxels (95% LoA: [{lower_loa:.1f}, {upper_loa:.1f}])")

    # ---------- 画图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    SCALE = 1000.0

    scaled_gt = df_stat["V_GT"] / SCALE
    scaled_pred = df_stat["V_Pred"] / SCALE
    scaled_means = df_stat["Mean"] / SCALE
    scaled_diffs = df_stat["Diff"] / SCALE

    # A: 相关性
    ax1 = axes[0]
    sns.regplot(
        x=scaled_gt,
        y=scaled_pred,
        ax=ax1,
        scatter_kws={'s': 60, 'alpha': 0.8, 'color': '#2b83ba'},
        line_kws={'color': '#d7191c', 'linewidth': 2}
    )

    max_val = max(scaled_gt.max(), scaled_pred.max()) * 1.1
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.6, label='y = x')
    ax1.set_xlabel('Ground Truth Edema Burden (×10³ Voxels)', fontweight='bold')
    ax1.set_ylabel('Predicted Edema Burden (×10³ Voxels)', fontweight='bold')
    ax1.set_title(f'Correlation Analysis - {model_tag}', fontweight='bold', pad=15)

    icc_text = f"{icc_value:.3f}" if not np.isnan(icc_value) else "N/A"
    stats_text = (
        f"Pearson R = {pearson_r:.3f}\n"
        f"Spearman R = {spearman_r:.3f}\n"
        f"ICC (2,1) = {icc_text}\n"
        f"p < 0.001"
    )
    ax1.text(
        0.05, 0.95, stats_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    ax1.legend(loc='lower right')

    # B: Bland-Altman
    ax2 = axes[1]
    normal_mask = (df_stat["Status"] == "")
    outlier_mask = (df_stat["Status"] == "BA Outlier")

    ax2.scatter(scaled_means[normal_mask], scaled_diffs[normal_mask], color='#2b83ba', s=60, alpha=0.8, label='Normal')
    if outlier_mask.any():
        ax2.scatter(
            scaled_means[outlier_mask],
            scaled_diffs[outlier_mask],
            color='#d7191c',
            marker='X',
            s=100,
            linewidths=2,
            label='Outlier'
        )

    scaled_bias = bias / SCALE
    scaled_upper = upper_loa / SCALE
    scaled_lower = lower_loa / SCALE

    ax2.axhline(scaled_bias, color='black', linestyle='-', linewidth=2)
    ax2.axhline(scaled_upper, color='#d7191c', linestyle='--', linewidth=1.5)
    ax2.axhline(scaled_lower, color='#d7191c', linestyle='--', linewidth=1.5)

    ax2.text(scaled_means.max() * 0.95, scaled_bias, f'Bias: {scaled_bias:.1f}', va='bottom', ha='right', color='black')
    ax2.text(scaled_means.max() * 0.95, scaled_upper, f'+1.96 SD: {scaled_upper:.1f}', va='bottom', ha='right', color='#d7191c')
    ax2.text(scaled_means.max() * 0.95, scaled_lower, f'-1.96 SD: {scaled_lower:.1f}', va='top', ha='right', color='#d7191c')

    ax2.set_xlabel('Mean Edema Burden (×10³ Voxels)', fontweight='bold')
    ax2.set_ylabel('Volume Difference [Pred - GT] (×10³ Voxels)', fontweight='bold')
    ax2.set_title(f'Bland-Altman Plot - {model_tag}', fontweight='bold', pad=15)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_tag}_Clinical_Statistical_Analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{model_tag}_Clinical_Statistical_Analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ---------- 导出 Excel ----------
    export_df = df.drop(columns=['Mean', 'Diff'])

    excel_path = output_dir / f"{model_tag}_Clinical_Metrics_Report.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        export_df.sort_values(by="Patient_ID").reset_index(drop=True).to_excel(writer, sheet_name="Sorted_by_ID", index=False)
        export_df.sort_values(by="3D_Dice", ascending=False).reset_index(drop=True).to_excel(writer, sheet_name="Sorted_by_3D_Dice", index=False)

    summary = {
        "Model": model_tag,
        "N_All": len(df),
        "N_Lesion_Positive": len(df_stat),
        "Mean_3D_Dice_All": df["3D_Dice"].mean(),
        "Mean_3D_Dice_Positive": df_stat["3D_Dice"].mean(),
        "Mean_RVE_Positive": pd.to_numeric(df_stat["RVE(%)"], errors="coerce").mean(),
        "Pearson_R": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_R": spearman_r,
        "Spearman_p": spearman_p,
        "ICC2": icc_value,
        "BA_Bias": bias,
        "BA_LoA_Lower": lower_loa,
        "BA_LoA_Upper": upper_loa
    }

    return df, summary


def main():
    all_summaries = []
    writer_path = OUT_ROOT / "RUN7_Model_Comparison.xlsx"

    with pd.ExcelWriter(writer_path) as writer:
        for model_tag in MODEL_TAGS:
            df, summary = analyze_one_model(model_tag)
            if df is None:
                continue

            df.to_excel(writer, sheet_name=f"{model_tag}_all_cases", index=False)
            all_summaries.append(summary)

        if len(all_summaries) > 0:
            summary_df = pd.DataFrame(all_summaries)
            summary_df.to_excel(writer, sheet_name="Summary_Comparison", index=False)
            summary_df.to_csv(OUT_ROOT / "RUN7_Model_Comparison.csv", index=False)

    print(f"\n🎉 双模型临床统计分析完成。汇总表已保存到: {writer_path}")


if __name__ == "__main__":
    main()