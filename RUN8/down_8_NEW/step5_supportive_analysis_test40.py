import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")


# =========================================================
# 路径
# =========================================================
BASE_DIR = Path(r"F:\cor\RUN8\down_8_NEW\down_test")
FINAL_DIR = BASE_DIR / "final_eval_test40"
OUT_DIR = BASE_DIR / "supportive_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_TABLE = FINAL_DIR / "Test40_GT_Final_Table.csv"
P080_TABLE = FINAL_DIR / "Test40_Pred_080_Final_Table.csv"
P075_TABLE = FINAL_DIR / "Test40_Pred_075_Final_Table.csv"


# =========================================================
# DeLong
# =========================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, m):
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * (1 - st.norm.cdf(np.abs(z)))[0][0]


def delong_roc_test(y_true, y_prob1, y_prob2):
    y_true = np.asarray(y_true)
    y_prob1 = np.asarray(y_prob1)
    y_prob2 = np.asarray(y_prob2)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    preds = np.array([y_prob1, y_prob2])
    preds_sorted = np.hstack([preds[:, pos_idx], preds[:, neg_idx]])

    aucs, sigma = fastDeLong(preds_sorted, len(pos_idx))
    p_value = calc_pvalue(aucs, sigma)
    return float(p_value)


# =========================================================
# HL
# =========================================================
def hosmer_lemeshow_test(y_true, y_prob, g=10):
    df = pd.DataFrame({
        "y_true": pd.to_numeric(pd.Series(y_true), errors="coerce"),
        "y_prob": pd.to_numeric(pd.Series(y_prob), errors="coerce")
    }).dropna().copy()

    if len(df) < 10 or df["y_true"].nunique() < 2:
        return np.nan, np.nan, 0, pd.DataFrame()

    df["y_prob"] = np.clip(df["y_prob"], 1e-6, 1 - 1e-6)

    try:
        df["bin"] = pd.qcut(df["y_prob"], q=min(g, len(df)), duplicates="drop")
    except Exception:
        return np.nan, np.nan, 0, pd.DataFrame()

    grouped = df.groupby("bin", observed=False).agg(
        n=("y_true", "size"),
        observed_events=("y_true", "sum"),
        expected_events=("y_prob", "sum"),
        mean_pred_prob=("y_prob", "mean")
    ).reset_index()

    grouped = grouped[grouped["n"] > 0].copy()
    if len(grouped) < 3:
        return np.nan, np.nan, len(grouped), grouped

    grouped["observed_nonevents"] = grouped["n"] - grouped["observed_events"]
    grouped["expected_nonevents"] = grouped["n"] - grouped["expected_events"]

    eps = 1e-8
    hl_stat = (
        ((grouped["observed_events"] - grouped["expected_events"]) ** 2) / (grouped["expected_events"] + eps) +
        ((grouped["observed_nonevents"] - grouped["expected_nonevents"]) ** 2) / (grouped["expected_nonevents"] + eps)
    ).sum()

    dof = max(len(grouped) - 2, 1)
    p_value = 1 - st.chi2.cdf(hl_stat, dof)

    return float(hl_stat), float(p_value), int(len(grouped)), grouped


# =========================================================
# DCA
# =========================================================
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


# =========================================================
# 工具
# =========================================================
def load_final_table(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到文件: {path}")

    df = pd.read_csv(path)
    if "label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"label": "Label"})

    need_cols = ["Patient_ID", "Label", "Pred_Prob"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise KeyError(f"❌ {path.name} 缺少列: {missing}")

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df["Pred_Prob"] = pd.to_numeric(df["Pred_Prob"], errors="coerce")

    if "Rad_score" in df.columns:
        df["Rad_score"] = pd.to_numeric(df["Rad_score"], errors="coerce")
    if "sparcc" in df.columns:
        df["sparcc"] = pd.to_numeric(df["sparcc"], errors="coerce")

    df = df.dropna(subset=["Patient_ID", "Label", "Pred_Prob"]).copy()
    return df


def align_tables(df_gt, df_ai):
    common_ids = sorted(set(df_gt["Patient_ID"]) & set(df_ai["Patient_ID"]))
    if len(common_ids) == 0:
        raise RuntimeError("❌ GT 与 AI 表没有共同 Patient_ID")

    gt = df_gt.set_index("Patient_ID").loc[common_ids].reset_index()
    ai = df_ai.set_index("Patient_ID").loc[common_ids].reset_index()
    return gt, ai


def summarize_one_path(name, df):
    y_true = df["Label"].astype(int).values
    y_prob = df["Pred_Prob"].values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr, tpr)

    hl_stat, hl_p, hl_groups, hl_table = hosmer_lemeshow_test(y_true, y_prob, g=10)

    if "Rad_score" in df.columns and "sparcc" in df.columns:
        corr_df = df[["Rad_score", "sparcc"]].dropna().copy()
        if len(corr_df) >= 3:
            sp = st.spearmanr(corr_df["Rad_score"], corr_df["sparcc"])
            sp_r = float(sp.statistic) if sp.statistic is not None else np.nan
            sp_p = float(sp.pvalue) if sp.pvalue is not None else np.nan
        else:
            sp_r, sp_p = np.nan, np.nan
    else:
        sp_r, sp_p = np.nan, np.nan

    return {
        "Path": name,
        "AUC": float(auc_val),
        "HL_stat": hl_stat,
        "HL_p": hl_p,
        "HL_groups": hl_groups,
        "Spearman_RadScore_vs_SPARCC_r": sp_r,
        "Spearman_RadScore_vs_SPARCC_p": sp_p
    }, hl_table


# =========================================================
# 绘图
# =========================================================
def plot_roc(gt_df, p080_df, p075_df, out_path):
    y = gt_df["Label"].astype(int).values
    gt_prob = gt_df["Pred_Prob"].values
    p080_prob = p080_df["Pred_Prob"].values
    p075_prob = p075_df["Pred_Prob"].values

    fpr_gt, tpr_gt, _ = roc_curve(y, gt_prob)
    fpr_080, tpr_080, _ = roc_curve(y, p080_prob)
    fpr_075, tpr_075, _ = roc_curve(y, p075_prob)

    auc_gt = auc(fpr_gt, tpr_gt)
    auc_080 = auc(fpr_080, tpr_080)
    auc_075 = auc(fpr_075, tpr_075)

    delong_080 = delong_roc_test(y, gt_prob, p080_prob)
    delong_075 = delong_roc_test(y, gt_prob, p075_prob)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_gt, tpr_gt, lw=2.5, label=f"GT (AUC={auc_gt:.3f})")
    plt.plot(fpr_080, tpr_080, lw=2.5, label=f"Pred_080 (AUC={auc_080:.3f})")
    plt.plot(fpr_075, tpr_075, lw=2.5, linestyle="--", label=f"Pred_075 (AUC={auc_075:.3f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="black", lw=1.5)

    text = (
        f"DeLong P (080 vs GT) = {delong_080:.4f}\n"
        f"DeLong P (075 vs GT) = {delong_075:.4f}"
    )
    plt.text(
        0.48, 0.16, text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
    )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC comparison on strict test40")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_calibration(gt_df, p080_df, p075_df, out_path):
    y = gt_df["Label"].astype(int).values
    gt_prob = gt_df["Pred_Prob"].values
    p080_prob = p080_df["Pred_Prob"].values
    p075_prob = p075_df["Pred_Prob"].values

    prob_true_gt, prob_pred_gt = calibration_curve(y, gt_prob, n_bins=5, strategy="quantile")
    prob_true_080, prob_pred_080 = calibration_curve(y, p080_prob, n_bins=5, strategy="quantile")
    prob_true_075, prob_pred_075 = calibration_curve(y, p075_prob, n_bins=5, strategy="quantile")

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(prob_pred_gt, prob_true_gt, marker="o", linewidth=2.5, label="GT")
    plt.plot(prob_pred_080, prob_true_080, marker="s", linewidth=2.5, label="Pred_080")
    plt.plot(prob_pred_075, prob_true_075, marker="^", linewidth=2.0, linestyle="--", label="Pred_075")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.8)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed probability")
    plt.title("Calibration curves on strict test40")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_dca(gt_df, p080_df, p075_df, out_path):
    y = gt_df["Label"].astype(int).values
    gt_prob = gt_df["Pred_Prob"].values
    p080_prob = p080_df["Pred_Prob"].values
    p075_prob = p075_df["Pred_Prob"].values

    thresholds = np.linspace(0.01, 0.99, 99)

    nb_gt = calculate_net_benefit(y, gt_prob, thresholds)
    nb_080 = calculate_net_benefit(y, p080_prob, thresholds)
    nb_075 = calculate_net_benefit(y, p075_prob, thresholds)

    prevalence = np.mean(y)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(thresholds, nb_gt, linewidth=2.5, label="GT")
    plt.plot(thresholds, nb_080, linewidth=2.5, label="Pred_080")
    plt.plot(thresholds, nb_075, linewidth=2.0, linestyle="--", label="Pred_075")
    plt.plot(thresholds, treat_all, linestyle="--", color="gray", linewidth=1.8, label="Treat all")
    plt.plot(thresholds, treat_none, linestyle=":", color="black", linewidth=1.8, label="Treat none")

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision curve analysis on strict test40")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# =========================================================
# 主流程
# =========================================================
def main():
    print("=" * 80)
    print("🚀 RUN8 / test40 支持性分析：HL + Calibration + DCA")
    print("=" * 80)
    print(f"GT   = {GT_TABLE}")
    print(f"080  = {P080_TABLE}")
    print(f"075  = {P075_TABLE}")
    print(f"OUT  = {OUT_DIR}")
    print("=" * 80)

    df_gt = load_final_table(GT_TABLE)
    df_080 = load_final_table(P080_TABLE)
    df_075 = load_final_table(P075_TABLE)

    df_gt, df_080 = align_tables(df_gt, df_080)
    df_gt2, df_075 = align_tables(df_gt, df_075)

    # 再确认三者 ID 顺序一致
    common_ids = sorted(set(df_gt["Patient_ID"]) & set(df_080["Patient_ID"]) & set(df_075["Patient_ID"]))
    df_gt = df_gt.set_index("Patient_ID").loc[common_ids].reset_index()
    df_080 = df_080.set_index("Patient_ID").loc[common_ids].reset_index()
    df_075 = df_075.set_index("Patient_ID").loc[common_ids].reset_index()

    gt_summary, gt_hl_table = summarize_one_path("GT", df_gt)
    p080_summary, p080_hl_table = summarize_one_path("Pred_080", df_080)
    p075_summary, p075_hl_table = summarize_one_path("Pred_075", df_075)

    # 保存 HL tables
    gt_hl_table.to_csv(OUT_DIR / "HL_Table_GT.csv", index=False, encoding="utf-8-sig")
    p080_hl_table.to_csv(OUT_DIR / "HL_Table_Pred_080.csv", index=False, encoding="utf-8-sig")
    p075_hl_table.to_csv(OUT_DIR / "HL_Table_Pred_075.csv", index=False, encoding="utf-8-sig")

    # 汇总表
    df_summary = pd.DataFrame([gt_summary, p080_summary, p075_summary])
    df_summary.to_csv(OUT_DIR / "Supportive_Analysis_Summary.csv", index=False, encoding="utf-8-sig")

    # DeLong 再单独算一遍
    y_true = df_gt["Label"].astype(int).values
    delong_080 = delong_roc_test(y_true, df_gt["Pred_Prob"].values, df_080["Pred_Prob"].values)
    delong_075 = delong_roc_test(y_true, df_gt["Pred_Prob"].values, df_075["Pred_Prob"].values)

    # 作图
    plot_roc(df_gt, df_080, df_075, OUT_DIR / "ROC_Comparison_Test40.png")
    plot_calibration(df_gt, df_080, df_075, OUT_DIR / "Calibration_Comparison_Test40.png")
    plot_dca(df_gt, df_080, df_075, OUT_DIR / "DCA_Comparison_Test40.png")

    # 文本总结
    with open(OUT_DIR / "Supportive_Analysis_Summary.txt", "w", encoding="utf-8") as f:
        f.write("RUN8 test40 supportive analysis\n")
        f.write("=" * 72 + "\n\n")

        f.write("Hosmer-Lemeshow results:\n")
        for row in [gt_summary, p080_summary, p075_summary]:
            f.write(
                f"{row['Path']}: "
                f"AUC={row['AUC']:.4f}, "
                f"HL={row['HL_stat']:.4f}, "
                f"P={row['HL_p']:.4f}, "
                f"groups={row['HL_groups']}, "
                f"Spearman_r={row['Spearman_RadScore_vs_SPARCC_r']:.4f}, "
                f"Spearman_p={row['Spearman_RadScore_vs_SPARCC_p']:.4e}\n"
            )

        f.write("\n")
        f.write(f"DeLong P (Pred_080 vs GT) = {delong_080:.6f}\n")
        f.write(f"DeLong P (Pred_075 vs GT) = {delong_075:.6f}\n")

    # 额外 json
    payload = {
        "GT": gt_summary,
        "Pred_080": p080_summary,
        "Pred_075": p075_summary,
        "DeLong_P_080_vs_GT": delong_080,
        "DeLong_P_075_vs_GT": delong_075
    }
    with open(OUT_DIR / "Supportive_Analysis_Summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n🎉 支持性分析完成。")
    print(f"📄 Summary CSV : {OUT_DIR / 'Supportive_Analysis_Summary.csv'}")
    print(f"📄 Summary TXT : {OUT_DIR / 'Supportive_Analysis_Summary.txt'}")
    print(f"📈 ROC         : {OUT_DIR / 'ROC_Comparison_Test40.png'}")
    print(f"📈 Calibration : {OUT_DIR / 'Calibration_Comparison_Test40.png'}")
    print(f"📈 DCA         : {OUT_DIR / 'DCA_Comparison_Test40.png'}")


if __name__ == "__main__":
    main()