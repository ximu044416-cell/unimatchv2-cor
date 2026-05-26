import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")


# =========================================================
# 路径
# =========================================================
BASE_ROOT = Path(r"F:\cor\RUN8\down_8_NEW")
DEV_ROOT = BASE_ROOT / "down_2_threshold"
TEST_ROOT = BASE_ROOT / "down_test"

DEV_CLINICAL_FILE = DEV_ROOT / "clinical" / "clinical_info_dev164.xlsx"
TEST_CLINICAL_FILE = DEV_ROOT / "clinical" / "clinical_info_test.xlsx"

DEV_GT_FEATURES_FILE = DEV_ROOT / "features_dev164" / "GT_Features_Dev164.csv"

TEST_FEATURE_DIR = TEST_ROOT / "features_test40"
TEST_GT_FEATURES_FILE = TEST_FEATURE_DIR / "Test40_GT_Features.csv"
TEST_P080_FEATURES_FILE = TEST_FEATURE_DIR / "Test40_Pred_080_Features.csv"
TEST_P075_FEATURES_FILE = TEST_FEATURE_DIR / "Test40_Pred_075_Features.csv"

OUT_DIR = TEST_ROOT / "one_se_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def fast_delong(predictions_sorted_transposed, label_1_count, label_0_count):
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
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * (1 - st.norm.cdf(np.abs(z)))[0][0]


def delong_roc_test(y_true, y_prob1, y_prob2):
    y_true = np.asarray(y_true).astype(int)
    y_prob1 = np.asarray(y_prob1)
    y_prob2 = np.asarray(y_prob2)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    preds = np.array([y_prob1, y_prob2])
    preds_sorted = np.hstack([preds[:, pos_idx], preds[:, neg_idx]])

    aucs, sigma = fast_delong(preds_sorted, len(pos_idx), len(neg_idx))
    return float(calc_pvalue(aucs, sigma))


# =========================================================
# HL
# =========================================================
def hosmer_lemeshow_test(y_true, y_prob, g=10):
    df = pd.DataFrame({
        "y_true": pd.to_numeric(pd.Series(y_true), errors="coerce"),
        "y_prob": pd.to_numeric(pd.Series(y_prob), errors="coerce")
    }).dropna().copy()

    if len(df) < 10 or df["y_true"].nunique() < 2:
        return np.nan, np.nan, 0

    df["y_prob"] = np.clip(df["y_prob"], 1e-6, 1 - 1e-6)

    try:
        df["bin"] = pd.qcut(df["y_prob"], q=min(g, len(df)), duplicates="drop")
    except Exception:
        return np.nan, np.nan, 0

    grouped = df.groupby("bin", observed=False).agg(
        n=("y_true", "size"),
        observed_events=("y_true", "sum"),
        expected_events=("y_prob", "sum")
    ).reset_index()

    grouped = grouped[grouped["n"] > 0].copy()
    if len(grouped) < 3:
        return np.nan, np.nan, len(grouped)

    grouped["observed_nonevents"] = grouped["n"] - grouped["observed_events"]
    grouped["expected_nonevents"] = grouped["n"] - grouped["expected_events"]

    eps = 1e-8
    hl_stat = (
        ((grouped["observed_events"] - grouped["expected_events"]) ** 2) /
        (grouped["expected_events"] + eps)
        +
        ((grouped["observed_nonevents"] - grouped["expected_nonevents"]) ** 2) /
        (grouped["expected_nonevents"] + eps)
    ).sum()

    dof = max(len(grouped) - 2, 1)
    p_value = 1 - st.chi2.cdf(hl_stat, dof)

    return float(hl_stat), float(p_value), int(len(grouped))


# =========================================================
# 工具函数
# =========================================================
def load_clinical(path):
    if not path.exists():
        raise FileNotFoundError(f"❌ 找不到临床表: {path}")

    df = pd.read_excel(path)

    if "label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={"label": "Label"})

    if "Patient_ID" not in df.columns:
        raise KeyError(f"❌ {path.name} 缺少 Patient_ID 列")

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    for col in ["CRP", "ESR", "HLA-B27", "Disease_Duration_Category", "sparcc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def calc_metrics(y_true, prob, cutoff):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob)

    pred = (prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, pred)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc_val = roc_auc_score(y_true, prob)

    return auc_val, acc, sen, spe, pred


def fit_logit(y, X):
    X_const = sm.add_constant(X, has_constant="add")
    X_const = X_const[["const"] + list(X.columns)]

    try:
        model = sm.Logit(y, X_const).fit(disp=False)
    except Exception:
        print("⚠️ statsmodels Logit 常规拟合失败，改用极小惩罚的 fit_regularized。")
        model = sm.Logit(y, X_const).fit_regularized(alpha=1e-6, disp=False)

    return model


def predict_logit(model, X):
    X_const = sm.add_constant(X, has_constant="add")
    X_const = X_const[["const"] + list(X.columns)]
    return np.asarray(model.predict(X_const))


def get_feature_columns(df_features):
    ignore_cols = {
        "Patient_ID",
        "ExtractionFailed",
        "MaskEmpty",
        "Failed_Extraction",
        "All_Zero_Features",
        "Rescued"
    }
    return [c for c in df_features.columns if c not in ignore_cols]


def compute_radscore(df_features, feature_cols_all, scaler, selected_features, lasso_intercept, coef_dict):
    X = df_features[feature_cols_all].copy()

    for c in feature_cols_all:
        if c not in X.columns:
            X[c] = 0.0

    X = X.fillna(0.0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols_all)

    radscore = np.full(len(df_features), float(lasso_intercept), dtype=float)
    for feat in selected_features:
        radscore += X_scaled[feat].values * float(coef_dict[feat])

    return radscore


def evaluate_path(path_name, df_feat, df_test_clin, artifacts):
    feature_cols_all = artifacts["feature_cols_all"]
    scaler = artifacts["scaler"]
    selected_features = artifacts["selected_features"]
    lasso_intercept = artifacts["lasso_intercept"]
    coef_dict = artifacts["coef_dict"]
    logit_model = artifacts["logit_model"]
    locked_cutoff = artifacts["locked_cutoff"]
    healthy_min_radscore = artifacts["healthy_min_radscore"]

    keep_cols = ["Patient_ID", "Label", "ESR", "Disease_Duration_Category"]
    if "sparcc" in df_test_clin.columns:
        keep_cols.append("sparcc")

    df_eval = pd.merge(
        df_test_clin[keep_cols],
        df_feat,
        on="Patient_ID",
        how="left"
    )

    if "ExtractionFailed" not in df_eval.columns:
        df_eval["ExtractionFailed"] = 1
    if "MaskEmpty" not in df_eval.columns:
        df_eval["MaskEmpty"] = 1

    for c in feature_cols_all:
        if c not in df_eval.columns:
            df_eval[c] = 0.0

    failed_idx = df_eval["ExtractionFailed"].fillna(1).astype(int).astype(bool)
    mask_empty_idx = df_eval["MaskEmpty"].fillna(1).astype(int).astype(bool)

    df_eval[feature_cols_all] = df_eval[feature_cols_all].fillna(0.0)
    all_zero_idx = (df_eval[feature_cols_all].abs().sum(axis=1) == 0.0)

    df_eval["Rad_score"] = compute_radscore(
        df_eval,
        feature_cols_all,
        scaler,
        selected_features,
        lasso_intercept,
        coef_dict
    )

    rescue_idx = failed_idx | mask_empty_idx | all_zero_idx
    df_eval["Rescued"] = rescue_idx.astype(int)
    df_eval.loc[rescue_idx, "Rad_score"] = healthy_min_radscore

    clinical_vars = ["ESR", "Disease_Duration_Category", "Rad_score"]
    prob = predict_logit(logit_model, df_eval[clinical_vars])

    y = df_eval["Label"].values.astype(int)
    auc_val, acc, sen, spe, pred = calc_metrics(y, prob, locked_cutoff)

    hl_stat, hl_p, hl_groups = hosmer_lemeshow_test(y, prob, g=10)

    if "sparcc" in df_eval.columns and df_eval["sparcc"].notna().sum() >= 3:
        sp = stats.spearmanr(df_eval["Rad_score"], df_eval["sparcc"], nan_policy="omit")
        sp_r = float(sp.statistic)
        sp_p = float(sp.pvalue)
    else:
        sp_r, sp_p = np.nan, np.nan

    df_eval["Pred_Prob"] = prob
    df_eval["Pred_Label"] = pred
    df_eval["All_Zero_Features"] = all_zero_idx.astype(int)

    summary = {
        "Path": path_name,
        "AUC": auc_val,
        "ACC": acc,
        "SEN": sen,
        "SPE": spe,
        "Rescue_N": int(rescue_idx.sum()),
        "Failed_Extraction_N": int(failed_idx.sum()),
        "MaskEmpty_N": int(mask_empty_idx.sum()),
        "All_Zero_Features_N": int(all_zero_idx.sum()),
        "HL_stat": hl_stat,
        "HL_p": hl_p,
        "HL_groups": hl_groups,
        "Spearman_RadScore_vs_SPARCC_r": sp_r,
        "Spearman_RadScore_vs_SPARCC_p": sp_p
    }

    return df_eval, prob, summary


# =========================================================
# 主流程
# =========================================================
def main():
    print("=" * 90)
    print("🚀 最终协议下 1-SE LASSO 敏感性分析：dev164 建模 + test40 冻结评估")
    print("=" * 90)

    # -----------------------------
    # 0. 检查输入文件
    # -----------------------------
    needed = [
        DEV_CLINICAL_FILE,
        TEST_CLINICAL_FILE,
        DEV_GT_FEATURES_FILE,
        TEST_GT_FEATURES_FILE,
        TEST_P080_FEATURES_FILE,
        TEST_P075_FEATURES_FILE
    ]

    for p in needed:
        if not p.exists():
            raise FileNotFoundError(f"❌ 缺少文件: {p}")

    # -----------------------------
    # 1. 读 dev164 数据
    # -----------------------------
    df_dev_clin = load_clinical(DEV_CLINICAL_FILE)
    df_test_clin = load_clinical(TEST_CLINICAL_FILE)

    df_dev_gt = pd.read_csv(DEV_GT_FEATURES_FILE)
    df_dev_gt["Patient_ID"] = df_dev_gt["Patient_ID"].astype(str).str.strip()

    feature_cols_all = get_feature_columns(df_dev_gt)

    # 缺失值填补器只从 dev164 学
    imputer_dict = {}
    for col in ["CRP", "ESR", "HLA-B27", "Disease_Duration_Category"]:
        if col in df_dev_clin.columns:
            if col in ["CRP", "ESR"]:
                imputer_dict[col] = df_dev_clin[col].median()
            else:
                imputer_dict[col] = df_dev_clin[col].mode()[0]

    for col, val in imputer_dict.items():
        if col in df_dev_clin.columns:
            df_dev_clin[col] = df_dev_clin[col].fillna(val)
        if col in df_test_clin.columns:
            df_test_clin[col] = df_test_clin[col].fillna(val)

    df_dev = pd.merge(df_dev_gt, df_dev_clin, on="Patient_ID", how="inner")
    if len(df_dev) == 0:
        raise RuntimeError("❌ dev164 GT features 与 clinical merge 后为空")

    y_dev = df_dev["Label"].values.astype(int)

    # -----------------------------
    # 2. StandardScaler
    # -----------------------------
    X_raw = df_dev[feature_cols_all].copy().fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols_all)

    # -----------------------------
    # 3. Mann-Whitney U
    # -----------------------------
    print("\n🧪 Mann-Whitney U 筛选...")
    mwu_selected = []
    for c in feature_cols_all:
        try:
            pval = stats.mannwhitneyu(
                X_scaled.loc[y_dev == 0, c],
                X_scaled.loc[y_dev == 1, c],
                alternative="two-sided"
            ).pvalue
            if pval < 0.05:
                mwu_selected.append(c)
        except Exception:
            pass

    if len(mwu_selected) == 0:
        print("⚠️ MWU 未筛出特征，回退使用全部特征。")
        mwu_selected = feature_cols_all.copy()

    print(f"   MWU 后保留: {len(mwu_selected)}")

    X_mwu = X_scaled[mwu_selected]

    # -----------------------------
    # 4. Spearman 去共线
    # -----------------------------
    print("\n🧪 Spearman 去共线...")
    corr = X_mwu.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    spearman_selected = [c for c in mwu_selected if c not in to_drop]

    if len(spearman_selected) == 0:
        print("⚠️ Spearman 后为空，回退 MWU 特征。")
        spearman_selected = mwu_selected.copy()

    print(f"   Spearman 后保留: {len(spearman_selected)}")

    X_spearman = X_scaled[spearman_selected]

    # -----------------------------
    # 5. LASSO CV + 1-SE
    # -----------------------------
    print("\n🪓 LASSO CV + 1-SE 选择...")

    pos_n = int(np.sum(y_dev == 1))
    neg_n = int(np.sum(y_dev == 0))
    cv_folds = max(3, min(10, pos_n, neg_n))

    lasso_cv = LassoCV(
        alphas=np.logspace(-3, 1, 100),
        cv=cv_folds,
        random_state=42,
        max_iter=10000
    )
    lasso_cv.fit(X_spearman, y_dev)

    mse_path = lasso_cv.mse_path_
    mse_mean = mse_path.mean(axis=1)
    mse_se = mse_path.std(axis=1, ddof=1) / np.sqrt(mse_path.shape[1])

    idx_min = int(np.argmin(mse_mean))
    alpha_min = float(lasso_cv.alphas_[idx_min])
    mse_min = float(mse_mean[idx_min])
    se_min = float(mse_se[idx_min])

    eligible_idx = np.where(mse_mean <= mse_min + se_min)[0]
    # sklearn 的 alphas_ 通常是从大到小排列，1-SE 取 eligible 中最大的 alpha
    idx_1se = eligible_idx[np.argmax(lasso_cv.alphas_[eligible_idx])]
    alpha_1se = float(lasso_cv.alphas_[idx_1se])

    lasso_1se = Lasso(alpha=alpha_1se, max_iter=10000)
    lasso_1se.fit(X_spearman, y_dev)

    coefs = lasso_1se.coef_
    selected_features = list(np.array(spearman_selected)[np.abs(coefs) > 1e-8])
    selected_coefs = coefs[np.abs(coefs) > 1e-8]

    # 避免极端情况下 1-SE 过强导致 0 特征
    note = "Standard 1-SE alpha selected."
    if len(selected_features) == 0:
        print("⚠️ 1-SE alpha 未产生非零特征，自动在 eligible alpha 中选择最近的非零解。")
        found = False
        # alpha 从大到小尝试，找到第一个非零解
        for idx in sorted(eligible_idx, key=lambda i: lasso_cv.alphas_[i], reverse=True):
            alpha_try = float(lasso_cv.alphas_[idx])
            lasso_try = Lasso(alpha=alpha_try, max_iter=10000)
            lasso_try.fit(X_spearman, y_dev)
            coefs_try = lasso_try.coef_
            if np.sum(np.abs(coefs_try) > 1e-8) > 0:
                alpha_1se = alpha_try
                lasso_1se = lasso_try
                coefs = coefs_try
                selected_features = list(np.array(spearman_selected)[np.abs(coefs) > 1e-8])
                selected_coefs = coefs[np.abs(coefs) > 1e-8]
                found = True
                note = "Nearest eligible non-zero 1-SE solution selected."
                break
        if not found:
            raise RuntimeError("❌ 1-SE 未能产生任何非零特征，请检查特征矩阵。")

    lasso_intercept = float(lasso_1se.intercept_)
    coef_dict = {f: float(c) for f, c in zip(selected_features, selected_coefs)}

    print(f"   CV-optimal alpha = {alpha_min:.8f}")
    print(f"   1-SE alpha       = {alpha_1se:.8f}")
    print(f"   Final features   = {len(selected_features)}")
    print(f"   Note             = {note}")

    # -----------------------------
    # 6. dev164 Rad-score + Logistic
    # -----------------------------
    df_dev["Rad_score"] = compute_radscore(
        df_dev,
        feature_cols_all,
        scaler,
        selected_features,
        lasso_intercept,
        coef_dict
    )

    healthy_min_radscore = df_dev.loc[df_dev["Label"] == 0, "Rad_score"].min()
    if pd.isna(healthy_min_radscore):
        healthy_min_radscore = df_dev["Rad_score"].min()

    clinical_vars = ["ESR", "Disease_Duration_Category", "Rad_score"]
    missing = [v for v in clinical_vars if v not in df_dev.columns]
    if missing:
        raise KeyError(f"❌ dev164 临床表缺少列: {missing}")

    logit_model = fit_logit(y_dev, df_dev[clinical_vars])
    prob_dev = predict_logit(logit_model, df_dev[clinical_vars])

    fpr_dev, tpr_dev, thr_dev = roc_curve(y_dev, prob_dev)
    auc_dev = auc(fpr_dev, tpr_dev)
    youden_idx = int(np.argmax(tpr_dev - fpr_dev))
    locked_cutoff = float(thr_dev[youden_idx])

    dev_auc, dev_acc, dev_sen, dev_spe, dev_pred = calc_metrics(y_dev, prob_dev, locked_cutoff)

    print(f"\n🔒 dev164 1-SE 模型已冻结 | AUC={dev_auc:.4f} | ACC={dev_acc:.4f} | SEN={dev_sen:.4f} | SPE={dev_spe:.4f}")
    print(f"🔒 locked cutoff = {locked_cutoff:.6f}")
    print(f"🔒 healthy minimum Rad-score = {healthy_min_radscore:.6f}")

    # -----------------------------
    # 7. test40 读取特征并评估
    # -----------------------------
    df_test_gt = pd.read_csv(TEST_GT_FEATURES_FILE)
    df_test_080 = pd.read_csv(TEST_P080_FEATURES_FILE)
    df_test_075 = pd.read_csv(TEST_P075_FEATURES_FILE)

    for df in [df_test_gt, df_test_080, df_test_075]:
        df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    artifacts = {
        "feature_cols_all": feature_cols_all,
        "scaler": scaler,
        "selected_features": selected_features,
        "lasso_intercept": lasso_intercept,
        "coef_dict": coef_dict,
        "logit_model": logit_model,
        "locked_cutoff": locked_cutoff,
        "healthy_min_radscore": healthy_min_radscore
    }

    gt_table, gt_prob, gt_summary = evaluate_path("GT_1SE", df_test_gt, df_test_clin, artifacts)
    p080_table, p080_prob, p080_summary = evaluate_path("Pred_080_1SE", df_test_080, df_test_clin, artifacts)
    p075_table, p075_prob, p075_summary = evaluate_path("Pred_075_1SE", df_test_075, df_test_clin, artifacts)

    y_test = gt_table["Label"].values.astype(int)
    p080_summary["DeLong_P_vs_GT"] = delong_roc_test(y_test, gt_prob, p080_prob)
    p075_summary["DeLong_P_vs_GT"] = delong_roc_test(y_test, gt_prob, p075_prob)
    gt_summary["DeLong_P_vs_GT"] = np.nan

    # -----------------------------
    # 8. 保存结果
    # -----------------------------
    selected_df = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": [coef_dict[f] for f in selected_features]
    })
    selected_df.to_csv(OUT_DIR / "OneSE_Selected_Features.csv", index=False, encoding="utf-8-sig")

    cv_summary = pd.DataFrame({
        "alpha": lasso_cv.alphas_,
        "mse_mean": mse_mean,
        "mse_se": mse_se,
        "is_cv_min": [i == idx_min for i in range(len(lasso_cv.alphas_))],
        "is_1se": [i == idx_1se for i in range(len(lasso_cv.alphas_))]
    })
    cv_summary.to_csv(OUT_DIR / "OneSE_LASSO_CV_Summary.csv", index=False, encoding="utf-8-sig")

    gt_table.to_csv(OUT_DIR / "OneSE_Test40_GT_Table.csv", index=False, encoding="utf-8-sig")
    p080_table.to_csv(OUT_DIR / "OneSE_Test40_Pred_080_Table.csv", index=False, encoding="utf-8-sig")
    p075_table.to_csv(OUT_DIR / "OneSE_Test40_Pred_075_Table.csv", index=False, encoding="utf-8-sig")

    final_summary = pd.DataFrame([gt_summary, p080_summary, p075_summary])
    final_summary.to_csv(OUT_DIR / "OneSE_Test40_Final_Summary.csv", index=False, encoding="utf-8-sig")

    model_payload = {
        "alpha_min": alpha_min,
        "alpha_1se": alpha_1se,
        "mse_min": mse_min,
        "se_min": se_min,
        "n_mwu_features": len(mwu_selected),
        "n_spearman_features": len(spearman_selected),
        "n_final_features": len(selected_features),
        "selected_features": selected_features,
        "coef_dict": coef_dict,
        "lasso_intercept": lasso_intercept,
        "locked_cutoff": locked_cutoff,
        "healthy_min_radscore": healthy_min_radscore,
        "note": note
    }

    with open(OUT_DIR / "OneSE_Model_Metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_payload, f, ensure_ascii=False, indent=2)

    joblib.dump(model_payload, OUT_DIR / "OneSE_Model_Metadata.pkl")

    with open(OUT_DIR / "OneSE_Test40_Final_Summary.txt", "w", encoding="utf-8") as f:
        f.write("1-SE LASSO sensitivity analysis under final strict protocol\n")
        f.write("=" * 80 + "\n")
        f.write(f"CV-optimal alpha: {alpha_min:.8f}\n")
        f.write(f"1-SE alpha      : {alpha_1se:.8f}\n")
        f.write(f"MWU features    : {len(mwu_selected)}\n")
        f.write(f"Spearman features: {len(spearman_selected)}\n")
        f.write(f"Final features  : {len(selected_features)}\n")
        f.write(f"Locked cutoff   : {locked_cutoff:.6f}\n")
        f.write(f"Healthy min Rad : {healthy_min_radscore:.6f}\n")
        f.write(f"Note            : {note}\n\n")
        f.write(final_summary.to_string(index=False))
        f.write("\n\nSelected features:\n")
        for feat in selected_features:
            f.write(f"{feat}\t{coef_dict[feat]:.8f}\n")

    print("\n🎉 1-SE 敏感性分析完成。")
    print(f"📂 输出目录: {OUT_DIR}")
    print(f"📄 结果表: {OUT_DIR / 'OneSE_Test40_Final_Summary.csv'}")
    print(f"📄 特征表: {OUT_DIR / 'OneSE_Selected_Features.csv'}")
    print(f"📄 文本总结: {OUT_DIR / 'OneSE_Test40_Final_Summary.txt'}")


if __name__ == "__main__":
    main()