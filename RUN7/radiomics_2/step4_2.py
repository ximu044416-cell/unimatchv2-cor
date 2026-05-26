import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as st
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
import joblib
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="ticks", context="paper", font_scale=1.15)


# =========================================================
# 基础设置
# =========================================================
WORK_DIR = r"/RUN7/radiomics_2"
TABLE_DIR = os.path.join(WORK_DIR, "radiomics_tables")
EVAL_DIR = os.path.join(WORK_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

FINAL_THRESHOLD_FOR_PLOTS = 0.70
FINAL_MODE_FOR_PLOTS = "primary_rescue"   # strict_impute / primary_rescue / complete_case
BOOTSTRAP_N = 3000
RANDOM_SEED = 42

THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]


# =========================================================
# 工具函数
# =========================================================
def find_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"以下路径都不存在:\n" + "\n".join(candidates))


def load_test_clinical():
    candidates = [
        os.path.join(WORK_DIR, "clinical_info_test.xlsx"),
        r"F:\downstreamtasks\radiomics\clinical_info_test.xlsx"
    ]
    path = find_first_existing(candidates)
    df = pd.read_excel(path)

    if 'label' in df.columns and 'Label' not in df.columns:
        df = df.rename(columns={'label': 'Label'})

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category', 'sparcc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
    return df, path


def load_train_clinical():
    candidates = [
        os.path.join(WORK_DIR, "clinical_info_train.xlsx"),
        r"F:\downstreamtasks\radiomics\clinical_info_train.xlsx"
    ]
    path = find_first_existing(candidates)
    df = pd.read_excel(path)

    if 'label' in df.columns and 'Label' not in df.columns:
        df = df.rename(columns={'label': 'Label'})

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category', 'sparcc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
    return df, path


def custom_scale_selected_features(df_features, scaler, selected_features):
    """
    只对 selected_features 做缩放，避免处理 1000+ 全部特征时的 NaN 兼容问题。
    """
    if hasattr(scaler, "feature_names_in_"):
        feature_order = list(scaler.feature_names_in_)
    else:
        raise RuntimeError("当前 scaler 不包含 feature_names_in_，请检查 sklearn 版本。")

    scaled = pd.DataFrame(index=df_features.index)

    for feat in selected_features:
        if feat not in df_features.columns:
            scaled[feat] = np.nan
            continue

        idx = feature_order.index(feat)
        mean_ = scaler.mean_[idx]
        scale_ = scaler.scale_[idx]
        if scale_ == 0:
            scale_ = 1.0

        scaled[feat] = (pd.to_numeric(df_features[feat], errors='coerce') - mean_) / scale_

    return scaled


def compute_radscore(df_scaled_selected, lasso_weights_dict):
    intercept = float(lasso_weights_dict['intercept'])
    coef_dict = lasso_weights_dict['coefs']

    rad = np.full(len(df_scaled_selected), np.nan, dtype=float)
    for i in range(len(df_scaled_selected)):
        row = df_scaled_selected.iloc[i]
        if row.isna().any():
            rad[i] = np.nan
        else:
            s = intercept
            for feat, coef in coef_dict.items():
                s += row[feat] * coef
            rad[i] = s
    return rad


def fit_final_train_logit():
    train_table_path = os.path.join(WORK_DIR, "Train_Radscore_Table.csv")
    if not os.path.exists(train_table_path):
        raise FileNotFoundError(f"找不到: {train_table_path}")

    df_train = pd.read_csv(train_table_path)
    df_train['Label'] = pd.to_numeric(df_train['Label'], errors='coerce')

    X = sm.add_constant(df_train[['ESR', 'Disease_Duration_Category', 'Rad_score']])
    y = df_train['Label'].values

    result = sm.Logit(y, X).fit(disp=False)
    prob_train = result.predict(X)
    fpr, tpr, thr = roc_curve(y, prob_train)
    roc_auc = auc(fpr, tpr)

    return df_train, result, roc_auc


def get_min_healthy_radscore(df_train):
    healthy_df = df_train[df_train['Label'] == 0].copy()
    if len(healthy_df) == 0:
        raise RuntimeError("训练集中没有 Label=0 的健康组，无法定义最低健康 Rad-score。")
    return float(healthy_df['Rad_score'].min())


def confusion_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spe = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return acc, sen, spe, tn, fp, fn, tp


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
# Paired bootstrap non-inferiority
# =========================================================
def paired_bootstrap_auc_diff(y_true, gt_prob, ai_prob, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    gt_prob = np.asarray(gt_prob)
    ai_prob = np.asarray(ai_prob)

    n = len(y_true)
    diffs = []

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        y_b = y_true[idx]
        gt_b = gt_prob[idx]
        ai_b = ai_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        auc_gt = roc_auc_score_safe(y_b, gt_b)
        auc_ai = roc_auc_score_safe(y_b, ai_b)
        diffs.append(auc_ai - auc_gt)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan

    diff_mean = np.mean(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)
    return float(diff_mean), float(ci_low), float(ci_high)


def roc_auc_score_safe(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return auc(fpr, tpr)


# =========================================================
# DCA
# =========================================================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    N = len(y_true)

    for thresh in thresholds:
        if thresh >= 1:
            net_benefits.append(0)
            continue

        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        nb = (tp / N) - (fp / N) * (thresh / (1 - thresh))
        net_benefits.append(nb)

    return np.array(net_benefits)


# =========================================================
# 准备 GT / AI pathway
# =========================================================
def prepare_pathway_df(df_clinical, df_features, scaler, selected_features, lasso_weights_dict,
                       result_model, healthy_min_radscore, imputer_dict,
                       allow_rescue=False, df_fallback=None):
    """
    统一入口：
    - GT 路径：allow_rescue=False, df_features=GT
    - AI strict：allow_rescue=False, df_features=Pred_thr_xxx
    - AI primary rescue：allow_rescue=True, df_fallback=Pred_thr_050
    """
    df_clin = df_clinical.copy()

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category']:
        if col in df_clin.columns:
            df_clin[col] = pd.to_numeric(df_clin[col], errors='coerce')

    df_clin['CRP'] = df_clin['CRP'].fillna(imputer_dict['CRP'])
    df_clin['ESR'] = df_clin['ESR'].fillna(imputer_dict['ESR'])
    df_clin['HLA-B27'] = df_clin['HLA-B27'].fillna(imputer_dict['HLA-B27'])
    df_clin['Disease_Duration_Category'] = df_clin['Disease_Duration_Category'].fillna(imputer_dict['Disease_Duration_Category'])

    df_feat = df_features.copy()
    df_feat['Patient_ID'] = df_feat['Patient_ID'].astype(str).str.strip()

    merged = pd.merge(df_clin, df_feat, on='Patient_ID', how='left')

    # 原始 zero 判定
    merged['OriginalZeroMask'] = (
        merged['MaskVolume'].isna() |
        (pd.to_numeric(merged['MaskVolume'], errors='coerce').fillna(0) == 0) |
        (pd.to_numeric(merged.get('IsZeroMask', 1), errors='coerce').fillna(1) == 1)
    ).astype(int)

    merged['RescuedFrom050'] = 0
    merged['ImputedHealthyMin'] = 0

    # rescue
    if allow_rescue and df_fallback is not None:
        fb = df_fallback.copy()
        fb['Patient_ID'] = fb['Patient_ID'].astype(str).str.strip()

        rescue_cols = [c for c in fb.columns if c not in ['Patient_ID']]
        fb = fb.rename(columns={c: f"FB__{c}" for c in rescue_cols})

        merged = pd.merge(merged, fb, on='Patient_ID', how='left')

        need_rescue = merged['OriginalZeroMask'] == 1
        fallback_nonzero = (
            (~merged['FB__MaskVolume'].isna()) &
            (pd.to_numeric(merged['FB__MaskVolume'], errors='coerce').fillna(0) > 0) &
            (pd.to_numeric(merged['FB__IsZeroMask'], errors='coerce').fillna(1) == 0)
        )

        can_rescue = need_rescue & fallback_nonzero
        merged.loc[can_rescue, 'RescuedFrom050'] = 1

        # 用 fallback 0.50 的 radiomics 特征覆盖
        feature_candidates = [c for c in merged.columns if c.startswith("original_") or c.startswith("log-") or c.startswith("wavelet-")]
        for feat in feature_candidates:
            fb_col = f"FB__{feat}"
            if fb_col in merged.columns:
                merged.loc[can_rescue, feat] = merged.loc[can_rescue, fb_col]

        merged.loc[can_rescue, 'MaskVolume'] = merged.loc[can_rescue, 'FB__MaskVolume']
        merged.loc[can_rescue, 'IsZeroMask'] = 0

    # 计算 selected features 的缩放值
    df_scaled = custom_scale_selected_features(merged, scaler, selected_features)
    merged['Rad_score'] = compute_radscore(df_scaled, lasso_weights_dict)

    # 剩余 zero / missing -> 健康最低 Rad_score
    unresolved = merged['Rad_score'].isna()
    merged.loc[unresolved, 'Rad_score'] = healthy_min_radscore
    merged.loc[unresolved, 'ImputedHealthyMin'] = 1

    # 预测概率
    X = sm.add_constant(merged[['ESR', 'Disease_Duration_Category', 'Rad_score']], has_constant='add')
    merged['PredProb'] = result_model.predict(X)

    return merged


# =========================================================
# 绘图
# =========================================================
def plot_final_roc(y_true, gt_prob, ai_prob, out_path, delong_p):
    fpr_gt, tpr_gt, _ = roc_curve(y_true, gt_prob)
    fpr_ai, tpr_ai, _ = roc_curve(y_true, ai_prob)

    auc_gt = auc(fpr_gt, tpr_gt)
    auc_ai = auc(fpr_ai, tpr_ai)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_gt, tpr_gt, color='#2980b9', lw=2.5, label=f'GT AUC = {auc_gt:.3f}')
    plt.plot(fpr_ai, tpr_ai, color='#e74c3c', lw=2.5, label=f'AI-Pred AUC = {auc_ai:.3f}')
    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle=':')

    text = f"DeLong test (GT vs AI-Pred)\nP = {delong_p:.4f}"
    plt.text(
        0.52, 0.18, text,
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    plt.xlabel("False positive rate", fontweight='bold')
    plt.ylabel("True positive rate", fontweight='bold')
    plt.title("ROC-based non-inferiority", fontweight='bold', fontsize=14)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_final_calibration(y_true, gt_prob, ai_prob, out_path):
    prob_true_gt, prob_pred_gt = calibration_curve(y_true, gt_prob, n_bins=5, strategy='quantile')
    prob_true_ai, prob_pred_ai = calibration_curve(y_true, ai_prob, n_bins=5, strategy='quantile')

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(prob_pred_gt, prob_true_gt, marker='o', linewidth=2.5, color='#2980b9', label='GT contour')
    plt.plot(prob_pred_ai, prob_true_ai, marker='s', linewidth=2.5, color='#e67e22', label='AI contour')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

    plt.xlabel("Mean predicted probability", fontweight='bold')
    plt.ylabel("Observed probability", fontweight='bold')
    plt.title("Calibration curve - test cohort", fontweight='bold', fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_final_dca(y_true, gt_prob, ai_prob, out_path):
    thresholds = np.linspace(0.01, 0.99, 99)

    nb_gt = calculate_net_benefit(y_true, gt_prob, thresholds)
    nb_ai = calculate_net_benefit(y_true, ai_prob, thresholds)
    prevalence = np.mean(y_true)

    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_none = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(thresholds, nb_gt, color='#2980b9', linewidth=2.5, label='GT contour')
    plt.plot(thresholds, nb_ai, color='#e67e22', linewidth=2.5, label='AI contour')
    plt.plot(thresholds, treat_all, color='gray', linestyle='--', linewidth=2, label='Treat all')
    plt.plot(thresholds, treat_none, color='black', linestyle=':', linewidth=2, label='Treat none')

    plt.xlabel("Threshold probability", fontweight='bold')
    plt.ylabel("Net benefit", fontweight='bold')
    plt.title("Decision curve analysis - test cohort", fontweight='bold', fontsize=14)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


# =========================================================
# 主流程
# =========================================================
def main():
    print("🚀 启动 RUN7 Step4（radiomics_2 测试端正式评估）")

    np.random.seed(RANDOM_SEED)

    # -----------------------------
    # 读临床
    # -----------------------------
    df_test_clinical, test_clin_path = load_test_clinical()
    df_train_clinical, train_clin_path = load_train_clinical()
    print(f"📥 Test clinical:  {test_clin_path}")
    print(f"📥 Train clinical: {train_clin_path}")
    print(f"📊 Test cohort size = {len(df_test_clinical)}")

    # -----------------------------
    # 读冻结权重
    # -----------------------------
    scaler = joblib.load(os.path.join(WORK_DIR, "train_scaler.pkl"))
    selected_features = joblib.load(os.path.join(WORK_DIR, "selected_features.pkl"))
    lasso_weights_dict = joblib.load(os.path.join(WORK_DIR, "lasso_weights_dict.pkl"))
    cutoff_dict = joblib.load(os.path.join(WORK_DIR, "optimal_cutoff.pkl"))
    imputer_dict = joblib.load(os.path.join(WORK_DIR, "imputer_dict.pkl"))
    locked_cutoff = cutoff_dict['optimal_cutoff']

    # -----------------------------
    # 训练端 logit 重建
    # -----------------------------
    df_train_table, result_model, train_auc = fit_final_train_logit()
    healthy_min_radscore = get_min_healthy_radscore(df_train_table)

    print(f"🔒 训练端模型已重建 | Train AUC = {train_auc:.3f}")
    print(f"🔒 Frozen cutoff = {locked_cutoff:.4f}")
    print(f"🔒 Healthy minimum Rad_score = {healthy_min_radscore:.6f}")

    # -----------------------------
    # 构建 GT reference
    # -----------------------------
    gt_features_path = os.path.join(TABLE_DIR, "Test_GT_Features.csv")
    if not os.path.exists(gt_features_path):
        raise FileNotFoundError(f"找不到: {gt_features_path}")
    df_test_gt = pd.read_csv(gt_features_path)

    df_gt_pathway = prepare_pathway_df(
        df_clinical=df_test_clinical,
        df_features=df_test_gt,
        scaler=scaler,
        selected_features=selected_features,
        lasso_weights_dict=lasso_weights_dict,
        result_model=result_model,
        healthy_min_radscore=healthy_min_radscore,
        imputer_dict=imputer_dict,
        allow_rescue=False,
        df_fallback=None
    )

    y_true_full = df_gt_pathway['Label'].astype(int).values
    gt_prob_full = df_gt_pathway['PredProb'].values
    gt_pred_full = (gt_prob_full >= locked_cutoff).astype(int)

    auc_gt = roc_auc_score_safe(y_true_full, gt_prob_full)
    acc_gt, sen_gt, spe_gt, _, _, _, _ = confusion_metrics(y_true_full, gt_pred_full)

    df_gt_pathway.to_csv(os.path.join(EVAL_DIR, "GT_Test_Reference_Pathway.csv"), index=False, encoding='utf-8-sig')

    print(f"\n📌 GT-based reference:")
    print(f"   AUC = {auc_gt:.3f} | ACC = {acc_gt:.3f} | SEN = {sen_gt:.3f} | SPE = {spe_gt:.3f}")

    # -----------------------------
    # 读 0.50 fallback 表
    # -----------------------------
    fallback050_path = os.path.join(TABLE_DIR, "Test_Pred_Features_thr_050.csv")
    if not os.path.exists(fallback050_path):
        raise FileNotFoundError(f"找不到 fallback 文件: {fallback050_path}")
    df_fallback_050 = pd.read_csv(fallback050_path)

    # -----------------------------
    # threshold sweep
    # -----------------------------
    all_rows = []

    for thr in THRESHOLDS:
        thr_tag = f"thr_{int(thr * 100):03d}"
        pred_path = os.path.join(TABLE_DIR, f"Test_Pred_Features_{thr_tag}.csv")
        if not os.path.exists(pred_path):
            print(f"⚠️ 缺少: {pred_path}，跳过")
            continue

        df_pred = pd.read_csv(pred_path)

        mode_dict = {
            "strict_impute": {"allow_rescue": False},
            "primary_rescue": {"allow_rescue": True},
            "complete_case": {"allow_rescue": False}
        }

        for mode_name, mode_cfg in mode_dict.items():
            df_ai = prepare_pathway_df(
                df_clinical=df_test_clinical,
                df_features=df_pred,
                scaler=scaler,
                selected_features=selected_features,
                lasso_weights_dict=lasso_weights_dict,
                result_model=result_model,
                healthy_min_radscore=healthy_min_radscore,
                imputer_dict=imputer_dict,
                allow_rescue=mode_cfg["allow_rescue"],
                df_fallback=df_fallback_050 if mode_cfg["allow_rescue"] else None
            )

            original_zero = int(df_ai['OriginalZeroMask'].sum())
            rescued = int(df_ai['RescuedFrom050'].sum())
            imputed = int(df_ai['ImputedHealthyMin'].sum())

            if mode_name == "complete_case":
                eval_df = df_ai[df_ai['OriginalZeroMask'] == 0].copy()
                gt_eval_df = df_gt_pathway[df_gt_pathway['Patient_ID'].isin(eval_df['Patient_ID'])].copy()
            else:
                eval_df = df_ai.copy()
                gt_eval_df = df_gt_pathway.copy()

            y_true = eval_df['Label'].astype(int).values
            ai_prob = eval_df['PredProb'].values
            ai_pred = (ai_prob >= locked_cutoff).astype(int)

            gt_prob = gt_eval_df.set_index('Patient_ID').loc[eval_df['Patient_ID'], 'PredProb'].values
            y_true_gt = gt_eval_df.set_index('Patient_ID').loc[eval_df['Patient_ID'], 'Label'].astype(int).values

            auc_ai = roc_auc_score_safe(y_true, ai_prob)
            acc_ai, sen_ai, spe_ai, tn, fp, fn, tp = confusion_metrics(y_true, ai_pred)

            delong_p = delong_roc_test(y_true_gt, gt_prob, ai_prob)
            diff_mean, ci_low, ci_high = paired_bootstrap_auc_diff(
                y_true_gt, gt_prob, ai_prob,
                n_boot=BOOTSTRAP_N,
                seed=RANDOM_SEED
            )

            ni_005 = int(ci_low > -0.05) if not np.isnan(ci_low) else 0
            ni_003 = int(ci_low > -0.03) if not np.isnan(ci_low) else 0

            # SPARCC 相关性（AI pathway）
            if 'sparcc' in eval_df.columns:
                corr_df = eval_df[['Rad_score', 'sparcc']].dropna().copy()
                if len(corr_df) >= 3:
                    r_ai, p_ai = stats.spearmanr(corr_df['Rad_score'], corr_df['sparcc'])
                else:
                    r_ai, p_ai = np.nan, np.nan
            else:
                r_ai, p_ai = np.nan, np.nan

            # 保存 patientwise
            out_patientwise = os.path.join(EVAL_DIR, f"Patientwise_{mode_name}_{thr_tag}.csv")
            eval_df.to_csv(out_patientwise, index=False, encoding='utf-8-sig')

            all_rows.append({
                "Threshold": thr,
                "Mode": mode_name,
                "N_eval": len(eval_df),
                "OriginalZeroMaskCases": original_zero,
                "RescuedFrom050": rescued,
                "ImputedHealthyMin": imputed,
                "AUC_GT_reference": auc_gt if mode_name != "complete_case" else roc_auc_score_safe(y_true_gt, gt_prob),
                "AUC_AI": auc_ai,
                "DeltaAUC_AI_minus_GT": auc_ai - (auc_gt if mode_name != "complete_case" else roc_auc_score_safe(y_true_gt, gt_prob)),
                "DeltaAUC_boot_mean": diff_mean,
                "DeltaAUC_CI_low": ci_low,
                "DeltaAUC_CI_high": ci_high,
                "NI_margin_0.05_pass": ni_005,
                "NI_margin_0.03_pass": ni_003,
                "DeLong_p": delong_p,
                "ACC_AI": acc_ai,
                "SEN_AI": sen_ai,
                "SPE_AI": spe_ai,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "AI_Spearman_r_vs_SPARCC": r_ai,
                "AI_Spearman_p": p_ai,
                "PatientwiseCSV": out_patientwise
            })

            print(f"thr={thr:.2f} | mode={mode_name:15s} | AUC={auc_ai:.3f} | Zero={original_zero} | Rescue={rescued} | Impute={imputed} | Delong={delong_p:.4f} | NI0.05={ni_005}")

    df_summary = pd.DataFrame(all_rows)
    df_summary.to_csv(os.path.join(EVAL_DIR, "Threshold_Sweep_Downstream_AllModes.csv"), index=False, encoding='utf-8-sig')

    # primary summary 单独保存
    df_primary = df_summary[df_summary['Mode'] == 'primary_rescue'].copy()
    df_primary.to_csv(os.path.join(EVAL_DIR, "Threshold_Sweep_PrimaryRescue.csv"), index=False, encoding='utf-8-sig')

    # -----------------------------
    # 最终图：默认 0.70 + primary_rescue
    # -----------------------------
    final_tag = f"thr_{int(FINAL_THRESHOLD_FOR_PLOTS * 100):03d}"
    final_patientwise = os.path.join(EVAL_DIR, f"Patientwise_{FINAL_MODE_FOR_PLOTS}_{final_tag}.csv")
    if os.path.exists(final_patientwise):
        df_final = pd.read_csv(final_patientwise)
        y_final = df_final['Label'].astype(int).values
        ai_final_prob = df_final['PredProb'].values
        gt_final = df_gt_pathway[df_gt_pathway['Patient_ID'].isin(df_final['Patient_ID'])].copy()
        gt_final_prob = gt_final.set_index('Patient_ID').loc[df_final['Patient_ID'], 'PredProb'].values
        y_gt_final = gt_final.set_index('Patient_ID').loc[df_final['Patient_ID'], 'Label'].astype(int).values

        final_delong = delong_roc_test(y_gt_final, gt_final_prob, ai_final_prob)

        plot_final_roc(
            y_true=y_gt_final,
            gt_prob=gt_final_prob,
            ai_prob=ai_final_prob,
            out_path=os.path.join(EVAL_DIR, f"Final_{FINAL_MODE_FOR_PLOTS}_{final_tag}_ROC.png"),
            delong_p=final_delong
        )

        plot_final_calibration(
            y_true=y_gt_final,
            gt_prob=gt_final_prob,
            ai_prob=ai_final_prob,
            out_path=os.path.join(EVAL_DIR, f"Final_{FINAL_MODE_FOR_PLOTS}_{final_tag}_Calibration.png")
        )

        plot_final_dca(
            y_true=y_gt_final,
            gt_prob=gt_final_prob,
            ai_prob=ai_final_prob,
            out_path=os.path.join(EVAL_DIR, f"Final_{FINAL_MODE_FOR_PLOTS}_{final_tag}_DCA.png")
        )

    # -----------------------------
    # 推荐文本
    # -----------------------------
    with open(os.path.join(EVAL_DIR, "Step4_KeyFindings.txt"), "w", encoding="utf-8") as f:
        f.write("RUN7 radiomics_2 Step4 summary\n")
        f.write("=====================================\n\n")
        f.write(f"GT reference AUC = {auc_gt:.4f}\n")
        f.write(f"GT reference ACC = {acc_gt:.4f}\n")
        f.write(f"GT reference SEN = {sen_gt:.4f}\n")
        f.write(f"GT reference SPE = {spe_gt:.4f}\n\n")

        f.write("Primary rescue mode summary by threshold:\n")
        f.write(df_primary.to_string(index=False))
        f.write("\n")

    print("\n🎉 Step4 完成。")
    print(f"📂 结果目录: {EVAL_DIR}")
    print(f"📄 主 summary: {os.path.join(EVAL_DIR, 'Threshold_Sweep_Downstream_AllModes.csv')}")


if __name__ == "__main__":
    main()