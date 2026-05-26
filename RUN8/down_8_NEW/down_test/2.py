import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


# ==========================================
# 辅助函数：手绘 Nomogram 的坐标轴
# ==========================================
def draw_nomogram_axis(ax, y_pos, title, ticks, labels=None, scale_min=0, scale_max=100):
    ax.hlines(y_pos, 0, 100, color='black', linewidth=1.5)
    ax.text(-5, y_pos, title, va='center', ha='right', fontsize=11, fontweight='bold')

    if labels is None:
        labels = [str(t) for t in ticks]

    for t, l in zip(ticks, labels):
        x_mapped = ((t - scale_min) / (scale_max - scale_min)) * 100
        if 0 <= x_mapped <= 100:
            ax.vlines(x_mapped, y_pos, y_pos + 0.3, color='black', linewidth=1.2)
            ax.text(x_mapped, y_pos + 0.5, str(l), va='bottom', ha='center', fontsize=9)


# ==========================================
# 辅助函数：计算 DCA 的 Net Benefit
# ==========================================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    N = len(y_true)
    for thresh in thresholds:
        if thresh >= 1:
            net_benefits.append(0.0)
            continue
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / N) - (fp / N) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    return np.array(net_benefits)


def draw_real_figure_s2():
    print("⏳ 正在定位文件路径...")
    # 获取 2.py 所在的当前目录 (现在是 down_test)
    current_dir = Path(__file__).parent.resolve()

    # 【修复路径】：直接进入 final_eval_test40 即可
    test40_dir = current_dir / "final_eval_test40"

    gt_path = test40_dir / "Test40_GT_Final_Table.csv"
    p080_path = test40_dir / "Test40_Pred_080_Final_Table.csv"
    p075_path = test40_dir / "Test40_Pred_075_Final_Table.csv"

    # 文件检查
    for path, name in zip([gt_path, p080_path, p075_path], ["GT", "0.80", "0.75"]):
        if not path.exists():
            print(f"❌ 找不到 {name} 表格，请确保它在这个路径下：\n{path}")
            return

    print("✅ 找到文件！正在读取真实 Test40 数据...")
    df_gt = pd.read_csv(gt_path)
    df_080 = pd.read_csv(p080_path)
    df_075 = pd.read_csv(p075_path)

    y_true = df_gt["Label"].astype(int).values
    gt_prob = df_gt["Pred_Prob"].values
    p080_prob = df_080["Pred_Prob"].values
    p075_prob = df_075["Pred_Prob"].values

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    # ==========================================
    # Panel A: Nomogram (重绘列线图结构)
    # ==========================================
    ax_nomo = axes[0]
    ax_nomo.axis('off')
    ax_nomo.set_xlim(-30, 110)
    ax_nomo.set_ylim(0, 12)
    ax_nomo.set_title('A. Nomogram', fontsize=14, fontweight='bold', pad=20)

    # 绘制列线图的各个轴
    draw_nomogram_axis(ax_nomo, 10, "Points", ticks=np.arange(0, 101, 10), scale_min=0, scale_max=100)
    draw_nomogram_axis(ax_nomo, 8, "ESR", ticks=[0, 20, 40, 60, 80, 100, 120], scale_min=0, scale_max=140)
    draw_nomogram_axis(ax_nomo, 6, "Disease Duration\nCategory", ticks=[0, 1, 2, 3], scale_min=0, scale_max=5)
    draw_nomogram_axis(ax_nomo, 4, "Rad_score", ticks=[-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5], scale_min=-2, scale_max=2)
    draw_nomogram_axis(ax_nomo, 2, "Total Points", ticks=np.arange(0, 251, 50), scale_min=0, scale_max=300)

    # 风险概率轴映射
    risk_ticks = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    risk_mapped = [10, 25, 45, 65, 80, 95]
    ax_nomo.hlines(0, 0, 100, color='black', linewidth=1.5)
    ax_nomo.text(-5, 0, "Risk Probability", va='center', ha='right', fontsize=11, fontweight='bold')
    for x, l in zip(risk_mapped, risk_ticks):
        ax_nomo.vlines(x, 0, 0.3, color='black', linewidth=1.2)
        ax_nomo.text(x, 0.5, str(l), va='bottom', ha='center', fontsize=9)

    # ==========================================
    # Panel B: Calibration curve (使用你的真实数据)
    # ==========================================
    ax_cal = axes[1]
    prob_true_gt, prob_pred_gt = calibration_curve(y_true, gt_prob, n_bins=5, strategy="quantile")
    prob_true_080, prob_pred_080 = calibration_curve(y_true, p080_prob, n_bins=5, strategy="quantile")
    prob_true_075, prob_pred_075 = calibration_curve(y_true, p075_prob, n_bins=5, strategy="quantile")

    ax_cal.plot(prob_pred_gt, prob_true_gt, marker="o", linewidth=2.5, color='#1f77b4', label="GT-based pathway")
    ax_cal.plot(prob_pred_080, prob_true_080, marker="s", linewidth=2.5, color='#ff7f0e',
                label="AI-derived pathway, threshold 0.80")
    ax_cal.plot(prob_pred_075, prob_true_075, marker="^", linewidth=2.0, color='#2ca02c', linestyle="--",
                label="AI-derived pathway, threshold 0.75")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.8, label="Ideal")

    ax_cal.set_xlabel("Mean predicted probability", fontsize=12)
    ax_cal.set_ylabel("Observed probability", fontsize=12)
    ax_cal.set_title("B. Calibration curve - test cohort", fontsize=14, fontweight='bold')
    ax_cal.legend(loc="upper left", fontsize=10)
    ax_cal.grid(True, linestyle='--', alpha=0.5)

    # ==========================================
    # Panel C: Decision Curve Analysis (使用你的真实数据)
    # ==========================================
    ax_dca = axes[2]
    thresholds = np.linspace(0.01, 0.99, 99)

    nb_gt = calculate_net_benefit(y_true, gt_prob, thresholds)
    nb_080 = calculate_net_benefit(y_true, p080_prob, thresholds)
    nb_075 = calculate_net_benefit(y_true, p075_prob, thresholds)

    prevalence = np.mean(y_true)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    ax_dca.plot(thresholds, nb_gt, linewidth=2.5, color='#1f77b4', label="GT-based pathway")
    ax_dca.plot(thresholds, nb_080, linewidth=2.5, color='#ff7f0e', label="AI-derived pathway, threshold 0.80")
    ax_dca.plot(thresholds, nb_075, linewidth=2.0, color='#2ca02c', linestyle="--",
                label="AI-derived pathway, threshold 0.75")
    ax_dca.plot(thresholds, treat_all, linestyle="--", color="gray", linewidth=1.8, label="Treat all")
    ax_dca.plot(thresholds, treat_none, linestyle=":", color="black", linewidth=1.8, label="Treat none")

    ax_dca.set_xlabel("Threshold probability", fontsize=12)
    ax_dca.set_ylabel("Net benefit", fontsize=12)
    ax_dca.set_title("C. Decision curve analysis - test cohort", fontsize=14, fontweight='bold')

    # 动态限制 Y 轴底部，避免 "Treat all" 曲线把图拉伸得太难看
    ax_dca.set_ylim(-0.15, max(nb_gt.max(), nb_080.max()) + 0.05)

    ax_dca.legend(loc="lower left", fontsize=10)
    ax_dca.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(w_pad=2.0)
    out_name = current_dir / 'Real_Data_Supplementary_Figure_S2_Nomogram_Calib_DCA.png'
    plt.savefig(out_name, bbox_inches='tight')
    print(f"✅ 成功基于真实 Test40 数据生成：\n{out_name}")


if __name__ == "__main__":
    draw_real_figure_s2()