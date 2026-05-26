import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc

# =========================
# 0. Paths
# =========================
EVAL_DIR = Path(r"F:\cor\RUN8\down_8_NEW\down_test\final_eval_test40")
SAVE_DIR = Path(r"F:\cor\RUN8\down_8_NEW\figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

GT_FILE = EVAL_DIR / "Test40_GT_Final_Table.csv"
AI080_FILE = EVAL_DIR / "Test40_Pred_080_Final_Table.csv"

DELONG_P = 0.2407

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

COLOR_GT = "#2B83BA"
COLOR_AI = "#C0392B"
COLOR_SCATTER = "#2B83BA"
COLOR_REG = "#C0392B"
COLOR_HIST = "#2B83BA"
GRID_COLOR = "#D9D9D9"


# =========================
# 2. Helpers
# =========================
def find_col(df, candidates):
    """
    Flexible column finder.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    raise KeyError(
        f"Cannot find any of columns: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def load_test_tables():
    """
    Robustly load GT and AI threshold-0.80 final tables.
    """
    if not GT_FILE.exists():
        raise FileNotFoundError(f"Cannot find GT table: {GT_FILE}")
    if not AI080_FILE.exists():
        raise FileNotFoundError(f"Cannot find AI table: {AI080_FILE}")

    df_gt = pd.read_csv(GT_FILE)
    df_ai = pd.read_csv(AI080_FILE)

    df_gt["Patient_ID"] = df_gt["Patient_ID"].astype(str).str.strip()
    df_ai["Patient_ID"] = df_ai["Patient_ID"].astype(str).str.strip()

    # -------------------------
    # GT columns
    # -------------------------
    gt_label_col = find_col(df_gt, ["Label", "label", "y_true", "Outcome"])
    gt_prob_col = find_col(df_gt, ["Pred_Prob", "PredProb", "Probability", "prob", "Prob"])

    gt_small = df_gt[["Patient_ID", gt_label_col, gt_prob_col]].copy()
    gt_small = gt_small.rename(columns={
        gt_label_col: "Label",
        gt_prob_col: "Prob_GT"
    })

    # -------------------------
    # AI columns
    # -------------------------
    ai_prob_col = find_col(df_ai, ["Pred_Prob", "PredProb", "Probability", "prob", "Prob"])
    ai_rad_col = find_col(df_ai, ["Rad_score", "RadScore", "radscore"])
    ai_sparcc_col = find_col(df_ai, ["sparcc", "SPARCC", "Sparcc"])

    keep_ai_cols = ["Patient_ID", ai_prob_col, ai_rad_col, ai_sparcc_col]

    ai_small = df_ai[keep_ai_cols].copy()
    ai_small = ai_small.rename(columns={
        ai_prob_col: "Prob_AI",
        ai_rad_col: "Rad_score",
        ai_sparcc_col: "sparcc"
    })

    merged = pd.merge(
        gt_small,
        ai_small,
        on="Patient_ID",
        how="inner"
    )

    if len(merged) == 0:
        raise RuntimeError("GT table and AI table have no overlapping Patient_ID values.")

    return merged


# =========================
# 3. Build Figure 4
# =========================
def make_figure4():
    df = load_test_tables()

    y = df["Label"].astype(int).values
    prob_gt = pd.to_numeric(df["Prob_GT"], errors="coerce").values
    prob_ai = pd.to_numeric(df["Prob_AI"], errors="coerce").values

    valid = np.isfinite(prob_gt) & np.isfinite(prob_ai)
    y = y[valid]
    prob_gt = prob_gt[valid]
    prob_ai = prob_ai[valid]

    fpr_gt, tpr_gt, _ = roc_curve(y, prob_gt)
    fpr_ai, tpr_ai, _ = roc_curve(y, prob_ai)

    auc_gt = auc(fpr_gt, tpr_gt)
    auc_ai = auc(fpr_ai, tpr_ai)

    # Construct validity data
    rad_col = find_col(df, ["Rad_score", "RadScore", "radscore"])
    sparcc_col = find_col(df, ["sparcc", "SPARCC", "Sparcc"])

    valid_df = df.copy()
    valid_df[rad_col] = pd.to_numeric(valid_df[rad_col], errors="coerce")
    valid_df[sparcc_col] = pd.to_numeric(valid_df[sparcc_col], errors="coerce")
    valid_df = valid_df.dropna(subset=[rad_col, sparcc_col]).copy()

    spearman_r, spearman_p = stats.spearmanr(valid_df[rad_col], valid_df[sparcc_col])

    # Figure layout
    fig = plt.figure(figsize=(15.2, 6.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.23)

    # -------------------------
    # Panel A: ROC
    # -------------------------
    ax1 = fig.add_subplot(outer[0, 0])

    # Changed to step plot for staircase effect
    ax1.step(
        fpr_gt, tpr_gt, where="post",
        color=COLOR_GT,
        lw=2.8,
        label=f"GT-based pathway AUC = {auc_gt:.4f}"
    )
    ax1.step(
        fpr_ai, tpr_ai, where="post",
        color=COLOR_AI,
        lw=2.8,
        label=f"AI-derived pathway AUC = {auc_ai:.4f}"
    )
    ax1.plot([0, 1], [0, 1], color="black", lw=1.1, linestyle=":")

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("False positive rate", fontweight="bold")
    ax1.set_ylabel("True positive rate", fontweight="bold")

    # Shortened title
    ax1.set_title("A. ROC comparison", pad=10, fontweight="bold")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35, color=GRID_COLOR)

    ax1.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        fontsize=9.3,
        borderpad=0.45
    )

    # Simplified text box and moved to right-bottom (above legend)
    stat_text_roc = (
        f"Frozen AI threshold = 0.80\n"
        f"DeLong p = {DELONG_P:.4f}"
    )

    ax1.text(
        0.96, 0.22,
        stat_text_roc,
        transform=ax1.transAxes,
        fontsize=9.4,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.94,
            edgecolor="#A0A0A0"
        )
    )

    # -------------------------
    # Panel B: Scatter + marginal histograms
    # -------------------------
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

    xvals = valid_df[rad_col].values
    yvals = valid_df[sparcc_col].values

    ax_top.hist(
        xvals,
        bins=14,
        color=COLOR_HIST,
        alpha=0.35,
        edgecolor="white"
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.set_title("B. Construct validity of AI-derived Rad-score", pad=6, fontweight="bold")

    # Lowered alpha for the right histogram to make it less prominent
    ax_right.hist(
        yvals,
        bins=12,
        orientation="horizontal",
        color=COLOR_HIST,
        alpha=0.15,
        edgecolor="white"
    )
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)

    ax_scatter.scatter(
        xvals,
        yvals,
        s=44,
        alpha=0.62,
        color=COLOR_SCATTER,
        edgecolor="white",
        linewidth=0.5
    )

    # Changed regression line to dashed with slight transparency
    if len(valid_df) >= 3:
        coef = np.polyfit(xvals, yvals, deg=1)
        x_line = np.linspace(np.min(xvals), np.max(xvals), 100)
        y_line = coef[0] * x_line + coef[1]
        ax_scatter.plot(x_line, y_line, color=COLOR_REG, linewidth=2.0, linestyle="--", alpha=0.7)

    ax_scatter.set_xlabel("AI-derived Radiomics Score (Rad-score)", fontweight="bold")
    ax_scatter.set_ylabel("SPARCC score", fontweight="bold")
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)
    ax_scatter.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.25, color=GRID_COLOR)

    stat_text_scatter = (
            f"Spearman r = {spearman_r:.3f}\n"
            + ("P < 0.001" if spearman_p < 0.001 else f"P = {spearman_p:.4f}")
    )

    ax_scatter.text(
        0.05,
        0.93,
        stat_text_scatter,
        transform=ax_scatter.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.94,
            edgecolor="#A0A0A0"
        )
    )

    ax_scatter.set_xlim(float(np.min(xvals)) - 0.05, float(np.max(xvals)) + 0.05)
    ax_scatter.set_ylim(min(-2, float(np.min(yvals)) - 2), float(np.max(yvals)) + 5)

    # Save
    png_path = SAVE_DIR / "Figure4_clinical_validity_test40_final1.png"
    pdf_path = SAVE_DIR / "Figure4_clinical_validity_test40_final1.pdf"
    svg_path = SAVE_DIR / "Figure4_clinical_validity_test40_final1.svg"
    tif_path = SAVE_DIR / "Figure4_clinical_validity_test40_final1.tif"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(tif_path, bbox_inches="tight", dpi=600)

    plt.show()

    print("Saved to:")
    print(png_path)
    print(pdf_path)
    print(svg_path)
    print(tif_path)


if __name__ == "__main__":
    make_figure4()