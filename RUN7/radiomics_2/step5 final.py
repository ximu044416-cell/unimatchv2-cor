import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc

# =========================
# 0. Paths
# =========================
WORK_DIR = r"F:\cor\RUN7\radiomics_2"
EVAL_DIR = os.path.join(WORK_DIR, "evaluation")
SAVE_DIR = r"E:\PyCharm 2023.3.6\project\pythonProject\unimatch\finger4"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 1. Global style
# =========================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "axes.linewidth": 1.0,
})
sns.set_style("whitegrid")

# =========================
# 2. Colors
# =========================
COLOR_TRAIN = "#7F7F7F"
COLOR_GT = "#2B83BA"
COLOR_PRED = "#D7191C"
COLOR_SCATTER = "#2B83BA"
COLOR_REG = "#C0392B"
COLOR_HIST = "#2B83BA"
GRID_COLOR = "#D9D9D9"

# =========================
# 3. Helper: rebuild train ROC from Train_Radscore_Table
# =========================
def rebuild_train_roc(work_dir):
    import statsmodels.api as sm
    from sklearn.metrics import roc_curve, auc

    train_table = os.path.join(work_dir, "Train_Radscore_Table.csv")
    df_train = pd.read_csv(train_table)

    required_cols = ["Label", "ESR", "Disease_Duration_Category", "Rad_score"]
    missing_cols = [c for c in required_cols if c not in df_train.columns]
    if missing_cols:
        raise ValueError(f"Train_Radscore_Table.csv 缺少必要列: {missing_cols}")

    df_train = df_train.copy()
    for col in required_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

    df_train = df_train.dropna(subset=required_cols)

    y_train = df_train["Label"].astype(int).values
    X_train = sm.add_constant(df_train[["ESR", "Disease_Duration_Category", "Rad_score"]], has_constant="add")

    # 直接重建训练端 logistic model
    train_model = sm.Logit(y_train, X_train).fit(disp=False)
    prob_train = train_model.predict(X_train)

    fpr_train, tpr_train, _ = roc_curve(y_train, prob_train)
    auc_train = auc(fpr_train, tpr_train)

    return fpr_train, tpr_train, auc_train

# =========================
# 4. Helper: load final test ROC
# =========================
def load_final_test_roc(eval_dir):
    gt_file = os.path.join(eval_dir, "GT_Test_Reference_Pathway.csv")
    ai_file = os.path.join(eval_dir, "Patientwise_primary_rescue_thr_070.csv")

    df_gt = pd.read_csv(gt_file)
    df_ai = pd.read_csv(ai_file)

    df_gt["Patient_ID"] = df_gt["Patient_ID"].astype(str).str.strip()
    df_ai["Patient_ID"] = df_ai["Patient_ID"].astype(str).str.strip()

    merged = pd.merge(
        df_ai[["Patient_ID", "Label", "PredProb"]],
        df_gt[["Patient_ID", "PredProb"]],
        on="Patient_ID",
        how="inner",
        suffixes=("_AI", "_GT")
    )

    y_test = merged["Label"].astype(int).values
    prob_gt = merged["PredProb_GT"].values
    prob_ai = merged["PredProb_AI"].values

    fpr_gt, tpr_gt, _ = roc_curve(y_test, prob_gt)
    fpr_ai, tpr_ai, _ = roc_curve(y_test, prob_ai)

    auc_gt = auc(fpr_gt, tpr_gt)
    auc_ai = auc(fpr_ai, tpr_ai)

    return merged, y_test, prob_gt, prob_ai, fpr_gt, tpr_gt, auc_gt, fpr_ai, tpr_ai, auc_ai

# =========================
# 5. Helper: load final Rad-score vs SPARCC from test cohort
# =========================
def load_final_radscore_sparcc(eval_dir):
    ai_file = os.path.join(eval_dir, "Patientwise_primary_rescue_thr_070.csv")
    df = pd.read_csv(ai_file)

    valid_df = df.dropna(subset=["Rad_score", "sparcc"]).copy()
    r, p_val = stats.spearmanr(valid_df["Rad_score"], valid_df["sparcc"])

    return valid_df, r, p_val

# =========================
# 6. Build Figure 4
# =========================
def make_figure4():
    # train ROC
    fpr_train, tpr_train, auc_train = rebuild_train_roc(WORK_DIR)

    # final test ROC
    merged, y_test, prob_gt, prob_ai, fpr_gt, tpr_gt, auc_gt, fpr_ai, tpr_ai, auc_ai = load_final_test_roc(EVAL_DIR)

    # fixed final NI text
    delong_p = 1.0000
    delta_auc = -0.0002
    ci_low = -0.0477
    ci_high = 0.0470
    ni_margin = -0.05

    # final test construct validity
    valid_df, spearman_r, spearman_p = load_final_radscore_sparcc(EVAL_DIR)

    fig = plt.figure(figsize=(15.5, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.06], wspace=0.22)

    # ==================================================
    # Panel A: ROC-based non-inferiority
    # ==================================================
    ax1 = fig.add_subplot(outer[0, 0])

    ax1.plot(
        fpr_train, tpr_train,
        color=COLOR_TRAIN, lw=2.0, linestyle="--",
        label=f"Train AUC = {auc_train:.3f}"
    )
    ax1.plot(
        fpr_gt, tpr_gt,
        color=COLOR_GT, lw=2.6,
        label=f"Test (GT) AUC = {auc_gt:.3f}"
    )
    ax1.plot(
        fpr_ai, tpr_ai,
        color=COLOR_PRED, lw=2.6,
        label=f"Test (AI-Pred) AUC = {auc_ai:.3f}"
    )
    ax1.plot([0, 1], [0, 1], color="black", lw=1.2, linestyle=":")

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("False positive rate", fontweight="bold")
    ax1.set_ylabel("True positive rate", fontweight="bold")
    ax1.set_title("A. ROC-based non-inferiority", pad=10, fontsize=12.5, fontweight="bold")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35, color=GRID_COLOR)

    ax1.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=True,
        framealpha=0.92,
        fontsize=9.3,
        borderpad=0.45
    )

    stat_text = (
        f"ΔAUC (AI − GT) = {delta_auc:.4f}\n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
        f"NI margin = {ni_margin:.2f}\n"
        f"DeLong p = {delong_p:.4f}"
    )

    ax1.text(
        0.63, 0.26,
        stat_text,
        transform=ax1.transAxes,
        fontsize=9.6,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.92,
            edgecolor="#A0A0A0"
        )
    )

    # ==================================================
    # Panel B: Scatter + marginal histograms
    # ==================================================
    gs_right = outer[0, 1].subgridspec(
        2, 2,
        height_ratios=[0.22, 1.0],
        width_ratios=[1.0, 0.22],
        hspace=0.04,
        wspace=0.04
    )

    ax_top = fig.add_subplot(gs_right[0, 0])
    ax_scatter = fig.add_subplot(gs_right[1, 0])
    ax_right = fig.add_subplot(gs_right[1, 1], sharey=ax_scatter)
    ax_empty = fig.add_subplot(gs_right[0, 1])
    ax_empty.axis("off")

    ax_top.hist(
        valid_df["Rad_score"],
        bins=16,
        color=COLOR_HIST,
        alpha=0.35,
        edgecolor="white"
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.set_title("B. Construct validity of AI-derived Rad-score", pad=6, fontsize=12.5, fontweight="bold")

    ax_right.hist(
        valid_df["sparcc"],
        bins=14,
        orientation="horizontal",
        color=COLOR_HIST,
        alpha=0.35,
        edgecolor="white"
    )
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)

    sns.regplot(
        x="Rad_score",
        y="sparcc",
        data=valid_df,
        ax=ax_scatter,
        scatter_kws={
            "alpha": 0.55,
            "s": 42,
            "color": COLOR_SCATTER
        },
        line_kws={
            "color": COLOR_REG,
            "linewidth": 2.5
        },
        ci=95
    )

    ax_scatter.set_xlabel("AI Radiomics Score (Rad-score)", fontweight="bold")
    ax_scatter.set_ylabel("Ground-truth SPARCC score", fontweight="bold")
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)
    ax_scatter.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.25, color=GRID_COLOR)

    stat_text = (
        f"Spearman r = {spearman_r:.3f}\n"
        + ("P < 0.001" if spearman_p < 0.001 else f"P = {spearman_p:.4f}")
    )
    ax_scatter.text(
        0.05, 0.93,
        stat_text,
        transform=ax_scatter.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.92,
            edgecolor="#A0A0A0"
        )
    )

    x_min = float(valid_df["Rad_score"].min())
    x_max = float(valid_df["Rad_score"].max())
    y_min = float(valid_df["sparcc"].min())
    y_max = float(valid_df["sparcc"].max())

    ax_scatter.set_xlim(x_min - 0.05, x_max + 0.05)
    ax_scatter.set_ylim(min(-2, y_min - 2), y_max + 5)

    # =========================
    # 7. Save
    # =========================
    png_path = os.path.join(SAVE_DIR, "Figure4_clinical_validity_combined_updated.png")
    pdf_path = os.path.join(SAVE_DIR, "Figure4_clinical_validity_combined_updated.pdf")
    svg_path = os.path.join(SAVE_DIR, "Figure4_clinical_validity_combined_updated.svg")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.show()

    print("Saved to:")
    print(png_path)
    print(pdf_path)
    print(svg_path)

if __name__ == "__main__":
    make_figure4()