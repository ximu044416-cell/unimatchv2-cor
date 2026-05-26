import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib

from scipy import stats
from sklearn.metrics import roc_curve, auc

# =========================
# 0. Paths
# =========================
WORK_DIR = r"F:\cor\RUN7\radiomics"
SAVE_DIR = r"E:\PyCharm 2023.3.6\project\pythonProject\unimatch\finger6"
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
# 3. Helper: rebuild train ROC
# =========================
def rebuild_train_roc(work_dir):
    clinical_file = os.path.join(work_dir, "clinical_info_train.xlsx")
    features_file = os.path.join(work_dir, "Train_GT_Features.csv")
    scaler_file = os.path.join(work_dir, "train_scaler.pkl")
    lasso_file = os.path.join(work_dir, "lasso_weights_dict.pkl")
    imputer_file = os.path.join(work_dir, "imputer_dict.pkl")

    df_train_clinical = pd.read_excel(clinical_file)
    df_train_clinical = df_train_clinical.rename(columns={"label": "Label"})
    df_train_features = pd.read_csv(features_file)

    for col in ["CRP", "ESR", "HLA-B27", "Disease_Duration_Category"]:
        if col in df_train_clinical.columns:
            df_train_clinical[col] = pd.to_numeric(df_train_clinical[col], errors="coerce")

    imputer_dict = joblib.load(imputer_file)
    for col in imputer_dict.keys():
        if col in df_train_clinical.columns:
            df_train_clinical[col] = df_train_clinical[col].fillna(imputer_dict[col])

    df_train_clinical["Patient_ID"] = df_train_clinical["Patient_ID"].astype(str).str.strip()
    df_train_features["Patient_ID"] = df_train_features["Patient_ID"].astype(str).str.strip()
    df_train = pd.merge(df_train_features, df_train_clinical, on="Patient_ID", how="inner")

    y_train = df_train["Label"].values
    feature_cols_all = [c for c in df_train_features.columns if c != "Patient_ID"]

    scaler = joblib.load(scaler_file)
    X_train_scaled = pd.DataFrame(
        scaler.transform(df_train[feature_cols_all]),
        columns=feature_cols_all
    )

    lasso_weights_dict = joblib.load(lasso_file)
    lasso_intercept = lasso_weights_dict["intercept"]
    selected_features = list(lasso_weights_dict["coefs"].keys())
    lasso_coefs = np.array(list(lasso_weights_dict["coefs"].values()))

    df_train["Rad_score"] = lasso_intercept + np.dot(X_train_scaled[selected_features], lasso_coefs)

    clinical_vars = ["ESR", "Disease_Duration_Category", "Rad_score"]
    X_train_clinical = sm.add_constant(df_train[clinical_vars])
    train_model = sm.Logit(y_train, X_train_clinical).fit(disp=False)

    prob_train = train_model.predict(X_train_clinical)
    fpr_train, tpr_train, _ = roc_curve(y_train, prob_train)
    auc_train = auc(fpr_train, tpr_train)

    return fpr_train, tpr_train, auc_train

# =========================
# 4. Helper: load test ROC
# =========================
def load_test_roc(work_dir):
    results_file = os.path.join(work_dir, "Final_Test_Patientwise_Results.csv")
    summary_file = os.path.join(work_dir, "RUN7_Final_Test_Summary.csv")

    df_test = pd.read_csv(results_file)
    summary_df = pd.read_csv(summary_file)

    y_test = df_test["Label"].values
    prob_gt = df_test["GT_Prob"].values
    prob_pred = df_test["Pred_Prob"].values

    fpr_gt, tpr_gt, _ = roc_curve(y_test, prob_gt)
    fpr_pred, tpr_pred, _ = roc_curve(y_test, prob_pred)

    auc_gt = float(summary_df.loc[0, "Test_GT_AUC"])
    auc_pred = float(summary_df.loc[0, "Test_Pred_AUC"])
    delong_p = float(summary_df.loc[0, "DeLong_p"])

    return fpr_gt, tpr_gt, auc_gt, fpr_pred, tpr_pred, auc_pred, delong_p

# =========================
# 5. Helper: load Rad-score vs SPARCC
# =========================
def load_radscore_sparcc(work_dir):
    table_file = os.path.join(work_dir, "Train_Radscore_Table.csv")
    df = pd.read_csv(table_file)

    sparcc_col = None
    for c in df.columns:
        if c.lower() == "sparcc":
            sparcc_col = c
            break

    if sparcc_col is None:
        raise ValueError("Could not find SPARCC column in Train_Radscore_Table.csv")

    valid_df = df.dropna(subset=["Rad_score", sparcc_col]).copy()
    r, p_val = stats.spearmanr(valid_df["Rad_score"], valid_df[sparcc_col])

    return valid_df, sparcc_col, r, p_val

# =========================
# 6. Build Figure 6
# =========================
def make_figure6():
    fpr_train, tpr_train, auc_train = rebuild_train_roc(WORK_DIR)
    fpr_gt, tpr_gt, auc_gt, fpr_pred, tpr_pred, auc_pred, delong_p = load_test_roc(WORK_DIR)
    valid_df, sparcc_col, spearman_r, spearman_p = load_radscore_sparcc(WORK_DIR)

    fig = plt.figure(figsize=(15.5, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.06], wspace=0.22)

    # ==================================================
    # Panel A: ROC
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
        fpr_pred, tpr_pred,
        color=COLOR_PRED, lw=2.6,
        label=f"Test (AI-Pred) AUC = {auc_pred:.3f}"
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

    delong_text = f"DeLong test\nGT vs AI-Pred\nP = {delong_p:.4f}"
    ax1.legend(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        frameon=True,
        framealpha=0.92,
        fontsize=9.3,
        borderpad=0.45
    )

    delong_text = f"DeLong test\nGT vs AI-Pred\nP = {delong_p:.4f}"
    ax1.text(
        0.73, 0.24,
        delong_text,
        transform=ax1.transAxes,
        fontsize=9.8,
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

    # top histogram
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

    # right histogram
    ax_right.hist(
        valid_df[sparcc_col],
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

    # main scatter + regression
    sns.regplot(
        x="Rad_score",
        y=sparcc_col,
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

    # stats box
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
    y_min = float(valid_df[sparcc_col].min())
    y_max = float(valid_df[sparcc_col].max())

    ax_scatter.set_xlim(x_min - 0.05, x_max + 0.05)
    ax_scatter.set_ylim(min(-2, y_min - 2), y_max + 5)

    # =========================
    # 7. Save
    # =========================
    png_path = os.path.join(SAVE_DIR, "Figure6_clinical_validity_combined_final.png")
    pdf_path = os.path.join(SAVE_DIR, "Figure6_clinical_validity_combined_final.pdf")
    svg_path = os.path.join(SAVE_DIR, "Figure6_clinical_validity_combined_final.svg")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.show()

    print("Saved to:")
    print(png_path)
    print(pdf_path)
    print(svg_path)

if __name__ == "__main__":
    make_figure6()