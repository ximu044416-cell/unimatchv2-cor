import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局绘图风格，极简学术风
sns.set_theme(style="ticks", context="paper", font_scale=1.2)


def analyze_clinical_metrics():
    # ================= 1. 路径与配置 =================
    PRED_DIR = Path(r"F:\downstreamtasks\analyze_model\3D_Predictions")
    OUTPUT_DIR = Path(r"F:\downstreamtasks\analyze_model\Clinical_Analysis")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("🚀 启动自动化临床统计学大脑...")

    # 使用 rglob 递归搜索所有子文件夹中的 GT 文件
    gt_files = sorted(list(PRED_DIR.rglob("*_GT.nii.gz")))
    if len(gt_files) == 0:
        print("❌ 错误：找不到任何 _GT.nii.gz 文件！请检查 3D_Predictions 文件夹。")
        return

    print(f"📥 成功追踪到 {len(gt_files)} 名患者的 3D 预测数据。\n")

    metrics_list = []

    # ================= 2. 核心指标提取与极端边界处理 =================
    for gt_path in gt_files:
        patient_id = gt_path.name.replace("_GT.nii.gz", "")
        # 在 GT 文件所在的同一个子文件夹 (gt_path.parent) 中寻找 Pred 文件
        pred_path = gt_path.parent / f"{patient_id}_Pred.nii.gz"

        if not pred_path.exists():
            print(f"⚠️ 警告: 找不到 {patient_id} 对应的预测文件，跳过。")
            continue

        # 读取 3D 矩阵
        gt_array = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
        pred_array = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

        # 计算体素计数 (Voxel Count)
        v_gt = np.sum(gt_array > 0)
        v_pred = np.sum(pred_array > 0)

        ave = abs(v_pred - v_gt)
        intersection = np.sum((gt_array > 0) & (pred_array > 0))

        # 🛡️ 导师级异常拦截 (Zero-Division Logic)
        outlier_status = ""
        if v_gt == 0 and v_pred == 0:
            # 完美真阴性
            rve = 0.0
            dice_3d = 1.0
            outlier_status = "Perfect TN"
        elif v_gt == 0 and v_pred > 0:
            # 假阳性无中生有
            rve = np.nan
            dice_3d = 0.0
            outlier_status = "Inf (FP Outlier)"
        else:
            # 正常情况
            rve = (ave / v_gt) * 100.0
            dice_3d = (2.0 * intersection) / (v_gt + v_pred) if (v_gt + v_pred) > 0 else 0.0

        metrics_list.append({
            "Patient_ID": patient_id,
            "V_GT": int(v_gt),
            "V_Pred": int(v_pred),
            "AVE": int(ave),
            "RVE(%)": round(rve, 2) if not np.isnan(rve) else "NaN",
            "3D_Dice": round(dice_3d, 4),
            "Status": outlier_status
        })

    df = pd.DataFrame(metrics_list)

    # 提前计算所有患者的 Mean 和 Diff
    df['Mean'] = (df['V_GT'] + df['V_Pred']) / 2.0
    df['Diff'] = df['V_Pred'] - df['V_GT']

    # ================= 3. 顶级医学统计分析 (Pearson, Spearman, ICC, BA) =================
    # 过滤掉 V_GT=0 的数据，仅针对有病灶的群体做纯净的统计分析
    df_stat = df[df["V_GT"] > 0].copy()

    print(f"🔬 参与统计学严苛检验的有效病灶患者数: {len(df_stat)} / {len(df)}")

    # 相关性
    pearson_r, pearson_p = stats.pearsonr(df_stat["V_GT"], df_stat["V_Pred"])
    spearman_r, spearman_p = stats.spearmanr(df_stat["V_GT"], df_stat["V_Pred"])

    # 🔥 修复致命报错：加装 ICC 统计学安全气囊 (Try-Except Fallback)
    icc_value = np.nan
    try:
        df_melt = df_stat[['Patient_ID', 'V_GT', 'V_Pred']].melt(id_vars='Patient_ID', var_name='Method',
                                                                 value_name='Volume')
        icc_res = pg.intraclass_corr(data=df_melt, targets='Patient_ID', raters='Method', ratings='Volume')
        if icc_res is not None and not icc_res.empty and 'ICC2' in icc_res['Type'].values:
            icc_value = icc_res.loc[icc_res['Type'] == 'ICC2', 'ICC'].values[0]
    except Exception as e:
        print(f"⚠️ ICC 绝对一致性计算跳过 (原因: 样本量过少或方差分布异常导致数学不可解)。")

    # Bland-Altman 计算 (绝对纯净，不受 0,0 干扰)
    bias = np.mean(df_stat["Diff"])
    sd = np.std(df_stat["Diff"], ddof=1)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    # 狸猫换太子。只对状态干净的正常病灶病例进行 BA Outlier 判定！
    df.loc[(df['Status'] == "") & ((df['Diff'] > upper_loa) | (df['Diff'] < lower_loa)), 'Status'] = "BA Outlier"

    # 刷新 df_stat，让它继承最新的 Status，为接下来的纯净画图做准备
    df_stat = df[df["V_GT"] > 0].copy()

    icc_print_str = f"{icc_value:.4f} 🥇" if not np.isnan(icc_value) else "N/A (样本分布受限)"

    print("📈 统计学检验结果：")
    print(f"   - Pearson R : {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"   - Spearman R: {spearman_r:.4f} (p={spearman_p:.4e})")
    print(f"   - 绝杀 ICC2 : {icc_print_str}")
    print(f"   - BA Bias   : {bias:.1f} Voxels (95% LoA: [{lower_loa:.1f}, {upper_loa:.1f}])\n")

    # ================= 4. 画神图 (极简学术风 + 科学倍数轴) =================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 比例缩放常数
    SCALE = 1000.0

    # 图 A 和 图 B 绝对统一，只使用 df_stat (排除 V_GT=0)
    scaled_gt = df_stat["V_GT"] / SCALE
    scaled_pred = df_stat["V_Pred"] / SCALE
    scaled_means = df_stat["Mean"] / SCALE
    scaled_diffs = df_stat["Diff"] / SCALE

    # ------------------ 图 A: 相关性散点图 ------------------
    ax1 = axes[0]
    sns.regplot(x=scaled_gt, y=scaled_pred, ax=ax1, scatter_kws={'s': 60, 'alpha': 0.8, 'color': '#2b83ba'},
                line_kws={'color': '#d7191c', 'linewidth': 2})

    # 画 y=x 理想基准线
    max_val = max(scaled_gt.max(), scaled_pred.max()) * 1.1
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.6, label='y = x')

    ax1.set_xlabel('Ground Truth Edema Burden (×10³ Voxels)', fontweight='bold')
    ax1.set_ylabel('Predicted Edema Burden (×10³ Voxels)', fontweight='bold')
    ax1.set_title('Correlation Analysis', fontweight='bold', pad=15)

    # 🔥 修复：安全格式化 ICC 输出，防止 NaN 崩溃
    icc_text = f"{icc_value:.3f}" if not np.isnan(icc_value) else "N/A"
    stats_text = (f"Pearson R = {pearson_r:.3f}\n"
                  f"Spearman R = {spearman_r:.3f}\n"
                  f"ICC (2,1) = {icc_text}\n"
                  f"p < 0.001")
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax1.legend(loc='lower right')

    # ------------------ 图 B: Bland-Altman 图 ------------------
    ax2 = axes[1]

    # 分离正常点和 Outliers 进行靶向绘制
    normal_mask = (df_stat["Status"] == "")
    outlier_mask = (df_stat["Status"] == "BA Outlier")

    ax2.scatter(scaled_means[normal_mask], scaled_diffs[normal_mask], color='#2b83ba', s=60, alpha=0.8, label='Normal')
    if outlier_mask.any():
        ax2.scatter(scaled_means[outlier_mask], scaled_diffs[outlier_mask], color='#d7191c', marker='X', s=100,
                    linewidths=2, label='Outlier')

    # 绘制统计线
    scaled_bias = bias / SCALE
    scaled_upper = upper_loa / SCALE
    scaled_lower = lower_loa / SCALE

    ax2.axhline(scaled_bias, color='black', linestyle='-', linewidth=2)
    ax2.axhline(scaled_upper, color='#d7191c', linestyle='--', linewidth=1.5)
    ax2.axhline(scaled_lower, color='#d7191c', linestyle='--', linewidth=1.5)

    # 添加标注文字
    ax2.text(scaled_means.max() * 0.95, scaled_bias, f'Bias: {scaled_bias:.1f}', va='bottom', ha='right', color='black')
    ax2.text(scaled_means.max() * 0.95, scaled_upper, f'+1.96 SD: {scaled_upper:.1f}', va='bottom', ha='right',
             color='#d7191c')
    ax2.text(scaled_means.max() * 0.95, scaled_lower, f'-1.96 SD: {scaled_lower:.1f}', va='top', ha='right',
             color='#d7191c')

    ax2.set_xlabel('Mean Edema Burden (×10³ Voxels)', fontweight='bold')
    ax2.set_ylabel('Volume Difference [Pred - GT] (×10³ Voxels)', fontweight='bold')
    ax2.set_title('Bland-Altman Plot', fontweight='bold', pad=15)
    ax2.legend(loc='upper right')

    plt.tight_layout()

    # 保存高分辨率图像
    fig_path_pdf = OUTPUT_DIR / "Clinical_Statistical_Analysis.pdf"
    fig_path_png = OUTPUT_DIR / "Clinical_Statistical_Analysis.png"
    plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    print(f"🎨 高清统计学神图已生成: {fig_path_pdf} (及 PNG)")

    # ================= 5. 一键导出 Excel (双重排序) =================
    export_df = df.drop(columns=['Mean', 'Diff'])

    excel_path = OUTPUT_DIR / "Clinical_Metrics_Report.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df_by_id = export_df.sort_values(by="Patient_ID").reset_index(drop=True)
        df_by_id.to_excel(writer, sheet_name="Sorted_by_ID", index=False)

        df_by_dice = export_df.sort_values(by="3D_Dice", ascending=False).reset_index(drop=True)
        df_by_dice.to_excel(writer, sheet_name="Sorted_by_3D_Dice", index=False)

    print(f"💾 论文素材库 (Excel 双榜单) 已导出至: {excel_path}")

    # ================= 6. 终端揪出内鬼 =================
    # 豁免 "Perfect TN"，只抓真正的异常！
    outliers = df[(df["Status"] != "") & (df["Status"] != "Perfect TN")]
    if len(outliers) > 0:
        print("\n🚨 【内鬼追踪报告】发现以下异常 / 离群病例，请在下一步定性分析时重点关照：")
        for _, row in outliers.iterrows():
            print(
                f"   ► Patient: {row['Patient_ID']} | Status: {row['Status']} | Dice: {row['3D_Dice']} | RVE: {row['RVE(%)']}%")
    else:
        print("\n🌟 完美！没有任何病例落在一致性界限外，你的模型无可挑剔！")


if __name__ == "__main__":
    analyze_clinical_metrics()