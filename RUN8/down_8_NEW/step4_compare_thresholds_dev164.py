import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# =========================================================
# 路径
# =========================================================
OUT_ROOT = Path(r"F:\cor\RUN8\down_8_NEW\down_2_threshold")

CLINICAL_DIR = OUT_ROOT / "clinical"
FEATURE_DIR = OUT_ROOT / "features_dev164"
MODEL_DIR = OUT_ROOT / "models_dev164"
COMPARE_DIR = OUT_ROOT / "threshold_compare_dev164"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_FILE = CLINICAL_DIR / "clinical_info_dev164.xlsx"
GT_FEATURES_FILE = FEATURE_DIR / "GT_Features_Dev164.csv"

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

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
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, label_0_count):
    m = label_1_count
    n = label_0_count
    b = predictions_sorted_transposed
    tx = np.empty([np.shape(b)[0], m], dtype=float)
    ty = np.empty([np.shape(b)[0], n], dtype=float)
    tz = np.empty([np.shape(b)[0], m + n], dtype=float)
    for r in range(np.shape(b)[0]):
        tz[r, :] = compute_midrank(b[r, :])
        tx[r, :] = compute_midrank(b[r, :m])
        ty[r, :] = compute_midrank(b[r, m:])
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
    aucs, sigma = fastDeLong(preds_sorted, len(pos_idx), len(neg_idx))
    p_value = calc_pvalue(aucs, sigma)
    return p_value


def calc_metrics(y_t, prob, locked_cutoff):
    pred_label = (prob >= locked_cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_t, pred_label, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_t, pred_label)
    return sens, spec, acc


def main():
    print("🚀 启动 RUN8 Step4：dev164 组学阈值比较（只用于阈值选择，不用于最终性能报告）")
    print(f"📄 clinical = {CLINICAL_FILE}")
    print(f"📄 GT feat  = {GT_FEATURES_FILE}")

    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(f"❌ 找不到 dev164 临床表：{CLINICAL_FILE}")
    if not GT_FEATURES_FILE.exists():
        raise FileNotFoundError(f"❌ 找不到 GT 特征表：{GT_FEATURES_FILE}")

    # -----------------------------------------------------
    # 1. 读入 dev164 GT 基线
    # -----------------------------------------------------
    df_clinical = pd.read_excel(CLINICAL_FILE)
    df_gt = pd.read_csv(GT_FEATURES_FILE)

    if "label" in df_clinical.columns and "Label" not in df_clinical.columns:
        df_clinical = df_clinical.rename(columns={"label": "Label"})

    df_clinical["Patient_ID"] = df_clinical["Patient_ID"].astype(str).str.strip()
    df_gt["Patient_ID"] = df_gt["Patient_ID"].astype(str).str.strip()

    for col in ["CRP", "ESR", "HLA-B27", "Disease_Duration_Category", "sparcc"]:
        if col in df_clinical.columns:
            df_clinical[col] = pd.to_numeric(df_clinical[col], errors="coerce")

    # -----------------------------------------------------
    # 2. 缺失值填补
    # -----------------------------------------------------
    crp_median = df_clinical["CRP"].median() if "CRP" in df_clinical.columns else np.nan
    esr_median = df_clinical["ESR"].median() if "ESR" in df_clinical.columns else np.nan
    hla_mode = df_clinical["HLA-B27"].mode()[0] if "HLA-B27" in df_clinical.columns else np.nan
    dur_mode = df_clinical["Disease_Duration_Category"].mode()[0] if "Disease_Duration_Category" in df_clinical.columns else np.nan

    imputer_dict = {
        "CRP": crp_median,
        "ESR": esr_median,
        "HLA-B27": hla_mode,
        "Disease_Duration_Category": dur_mode
    }
    joblib.dump(imputer_dict, MODEL_DIR / "imputer_dict.pkl")

    for col, val in imputer_dict.items():
        if col in df_clinical.columns:
            df_clinical[col] = df_clinical[col].fillna(val)

    # -----------------------------------------------------
    # 3. GT 路径建模（在 dev164 内完成）
    # -----------------------------------------------------
    df_merged = pd.merge(df_gt, df_clinical, on="Patient_ID", how="inner")
    if len(df_merged) == 0:
        raise RuntimeError("❌ GT 特征和临床表 merge 后为空")

    y = df_merged["Label"].values
    feature_cols_all = [c for c in df_gt.columns if c != "Patient_ID"]
    X_features = df_merged[feature_cols_all]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=feature_cols_all)
    joblib.dump(scaler, MODEL_DIR / "train_scaler.pkl")

    # MWU
    mwu_selected = []
    for c in feature_cols_all:
        try:
            pval = stats.mannwhitneyu(X_scaled.loc[y == 0, c], X_scaled.loc[y == 1, c]).pvalue
            if pval < 0.05:
                mwu_selected.append(c)
        except Exception:
            pass

    if len(mwu_selected) == 0:
        mwu_selected = feature_cols_all.copy()

    X_mwu = X_scaled[mwu_selected]

    # Spearman 去共线
    corr_matrix = X_mwu.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    spearman_selected = [c for c in mwu_selected if c not in to_drop]
    if len(spearman_selected) == 0:
        spearman_selected = mwu_selected.copy()

    X_spearman = X_mwu[spearman_selected]

    # 纯 CV-optimal LASSO
    pos_n = int(np.sum(y == 1))
    neg_n = int(np.sum(y == 0))
    cv_folds = max(3, min(10, pos_n, neg_n))

    lasso_cv = LassoCV(
        alphas=np.logspace(-3, 1, 100),
        cv=cv_folds,
        random_state=42,
        max_iter=10000
    )
    lasso_cv.fit(X_spearman, y)

    coefs = lasso_cv.coef_
    final_features = np.array(spearman_selected)[coefs != 0]
    final_coefs = coefs[coefs != 0]
    intercept = lasso_cv.intercept_

    if len(final_features) == 0:
        abs_coef_order = np.argsort(np.abs(coefs))[::-1]
        topk = min(5, len(abs_coef_order))
        keep_idx = abs_coef_order[:topk]
        final_features = np.array(spearman_selected)[keep_idx]
        final_coefs = coefs[keep_idx]

    joblib.dump(list(final_features), MODEL_DIR / "selected_features.pkl")
    lasso_weights_dict = {"intercept": float(intercept), "coefs": dict(zip(final_features, final_coefs))}
    joblib.dump(lasso_weights_dict, MODEL_DIR / "lasso_weights_dict.pkl")

    df_merged["Rad_score"] = intercept + np.dot(X_spearman[final_features], final_coefs)

    clinical_vars = ["ESR", "Disease_Duration_Category", "Rad_score"]
    missing_vars = [v for v in clinical_vars if v not in df_merged.columns]
    if len(missing_vars) > 0:
        raise KeyError(f"❌ 临床表缺少列：{missing_vars}")

    X_clinical = sm.add_constant(df_merged[clinical_vars], has_constant="add")[["const"] + clinical_vars]
    result = sm.Logit(y, X_clinical).fit(disp=False)

    prob_gt = result.predict(X_clinical)
    fpr_gt, tpr_gt, thr_gt = roc_curve(y, prob_gt)
    auc_gt = auc(fpr_gt, tpr_gt)

    youden_idx = np.argmax(tpr_gt - fpr_gt)
    optimal_cutoff = thr_gt[youden_idx]
    joblib.dump({"optimal_cutoff": float(optimal_cutoff)}, MODEL_DIR / "optimal_cutoff.pkl")

    healthy_min_radscore = df_merged.loc[df_merged["Label"] == 0, "Rad_score"].min()
    if pd.isna(healthy_min_radscore):
        healthy_min_radscore = df_merged["Rad_score"].min()

    gt_sens, gt_spec, gt_acc = calc_metrics(y, prob_gt, optimal_cutoff)

    df_gt_rad = df_merged[["Patient_ID", "Label", "ESR", "Disease_Duration_Category", "Rad_score"]].copy()
    df_gt_rad["GT_Prob"] = prob_gt
    df_gt_rad.to_csv(MODEL_DIR / "Dev164_GT_Radscore_Table.csv", index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # 4. 各阈值 Pred 路径比较
    # -----------------------------------------------------
    summary_rows = []

    print("\n================== 🌟 dev164 GT 基准表现 ==================")
    print(f"AUC={auc_gt:.3f} | ACC={gt_acc:.3f} | SEN={gt_sens:.3f} | SPE={gt_spec:.3f} | cutoff={optimal_cutoff:.4f}")

    for t in THRESHOLDS:
        t_str = f"{int(t * 100):03d}"
        pred_csv = FEATURE_DIR / f"Pred_Features_{t_str}.csv"

        if not pred_csv.exists():
            print(f"⚠️ 缺少 {pred_csv}，跳过...")
            continue

        df_pred = pd.read_csv(pred_csv)
        df_pred["Patient_ID"] = df_pred["Patient_ID"].astype(str).str.strip()

        # 先按 GT 患者名单对齐
        df_merge_pred = pd.merge(df_gt[["Patient_ID"]], df_pred, on="Patient_ID", how="left")
        df_merge_pred = pd.merge(df_merge_pred, df_clinical, on="Patient_ID", how="left")

        failed_extraction_idx = df_merge_pred[feature_cols_all[0]].isna()
        df_merge_pred[feature_cols_all] = df_merge_pred[feature_cols_all].fillna(0.0)

        X_pred_scaled = pd.DataFrame(
            scaler.transform(df_merge_pred[feature_cols_all]),
            columns=feature_cols_all
        )
        df_merge_pred["Rad_score"] = intercept + np.dot(X_pred_scaled[final_features], final_coefs)

        # rescue：全 0 或提取失败 → healthy_min_radscore
        all_zero_idx = (df_merge_pred[feature_cols_all].abs().sum(axis=1) == 0.0)
        rescue_idx = failed_extraction_idx | all_zero_idx
        df_merge_pred.loc[rescue_idx, "Rad_score"] = healthy_min_radscore

        X_pred_clin = sm.add_constant(df_merge_pred[clinical_vars], has_constant="add")[["const"] + clinical_vars]
        prob_pred = result.predict(X_pred_clin)

        fpr_pred, tpr_pred, _ = roc_curve(y, prob_pred)
        auc_pred = auc(fpr_pred, tpr_pred)
        pred_sens, pred_spec, pred_acc = calc_metrics(y, prob_pred, optimal_cutoff)
        delong_p = delong_roc_test(y, prob_gt, prob_pred)

        df_out = df_merge_pred[["Patient_ID", "Label", "ESR", "Disease_Duration_Category", "Rad_score"]].copy()
        df_out["Pred_Prob"] = prob_pred
        df_out["Failed_Extraction"] = failed_extraction_idx.astype(int)
        df_out["All_Zero_Features"] = all_zero_idx.astype(int)
        df_out["Rescued"] = rescue_idx.astype(int)
        df_out.to_csv(COMPARE_DIR / f"Dev164_Pred_Radscore_Table_{t_str}.csv", index=False, encoding="utf-8-sig")

        summary_rows.append({
            "Threshold": t,
            "GT_AUC": auc_gt,
            "Pred_AUC": auc_pred,
            "GT_ACC": gt_acc,
            "Pred_ACC": pred_acc,
            "GT_SEN": gt_sens,
            "Pred_SEN": pred_sens,
            "GT_SPE": gt_spec,
            "Pred_SPE": pred_spec,
            "DeLong_P_vs_GT": delong_p,
            "Failed_Extraction_N": int(failed_extraction_idx.sum()),
            "All_Zero_Features_N": int(all_zero_idx.sum()),
            "Rescued_N": int(rescue_idx.sum())
        })

        print(
            f"thr={t:.2f} | AUC={auc_pred:.3f} | ACC={pred_acc:.3f} | "
            f"SEN={pred_sens:.3f} | SPE={pred_spec:.3f} | "
            f"DeLong={delong_p:.4f} | Rescue={int(rescue_idx.sum())}"
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = COMPARE_DIR / "Threshold_Sweep_Dev164.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 简单画图
    if len(df_summary) > 0:
        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(df_summary["Threshold"], df_summary["Pred_AUC"], marker="o", label="Pred AUC")
        plt.axhline(y=auc_gt, linestyle="--", label=f"GT AUC={auc_gt:.3f}")
        plt.xlabel("Threshold")
        plt.ylabel("AUC")
        plt.title("RUN8 dev164 Downstream Threshold Comparison")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(COMPARE_DIR / "Threshold_Sweep_Dev164_AUC.png")
        plt.close()

    with open(COMPARE_DIR / "Threshold_Sweep_Dev164_Summary.txt", "w", encoding="utf-8") as f:
        f.write("RUN8 dev164 downstream threshold comparison\n")
        f.write("=" * 70 + "\n")
        f.write(f"GT baseline AUC: {auc_gt:.4f}\n")
        f.write(f"GT baseline ACC: {gt_acc:.4f}\n")
        f.write(f"GT baseline SEN: {gt_sens:.4f}\n")
        f.write(f"GT baseline SPE: {gt_spec:.4f}\n")
        f.write(f"Locked cutoff   : {optimal_cutoff:.6f}\n\n")
        f.write("Threshold sweep results are in Threshold_Sweep_Dev164.csv\n")
        f.write("Use this dev164-only comparison to freeze the final deployment threshold,\n")
        f.write("then evaluate once on test40.\n")

    print("\n🎉 RUN8 Step4 完成。")
    print(f"📄 summary csv = {summary_csv}")
    print(f"📄 model dir   = {MODEL_DIR}")
    print(f"📄 compare dir = {COMPARE_DIR}")


if __name__ == "__main__":
    main()