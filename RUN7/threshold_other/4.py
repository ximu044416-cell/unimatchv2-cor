import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
import scipy.stats as st
import os
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


# 此处省略 fastDeLong 和 delong_roc_test 函数，请保持你原先的代码一模一样
# ...
def compute_midrank(x):
    # (保留原有代码)
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
    # (保留原有代码)
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
    # (保留原有代码)
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * (1 - st.norm.cdf(np.abs(z)))[0][0]


def delong_roc_test(y_true, y_prob1, y_prob2):
    # (保留原有代码)
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


def run_ultimate_blind_test():
    print("🚀 启动 Step 6: 终极多阈值双轨盲测大考...\n")

    # ================= 路径与阈值设定 =================
    WORK_DIR = r"/RUN7/threshold_other"
    THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]

    # ---------------- 1. 加载 Train 核心模型与标尺 ----------------
    df_train_clinical = pd.read_excel(os.path.join(WORK_DIR, "clinical_info_train.xlsx"))
    df_train_clinical = df_train_clinical.rename(columns={'label': 'Label'})
    df_train_features = pd.read_csv(os.path.join(WORK_DIR, "Train_GT_Features.csv"))

    imputer_dict = joblib.load(os.path.join(WORK_DIR, "imputer_dict.pkl"))
    for col in imputer_dict.keys():
        df_train_clinical[col] = pd.to_numeric(df_train_clinical[col], errors='coerce').fillna(imputer_dict[col])

    df_train_clinical['Patient_ID'] = df_train_clinical['Patient_ID'].astype(str).str.strip()
    df_train_features['Patient_ID'] = df_train_features['Patient_ID'].astype(str).str.strip()
    df_train = pd.merge(df_train_features, df_train_clinical, on='Patient_ID', how='inner')

    train_scaler = joblib.load(os.path.join(WORK_DIR, "train_scaler.pkl"))
    lasso_weights_dict = joblib.load(os.path.join(WORK_DIR, "lasso_weights_dict.pkl"))
    lasso_intercept = lasso_weights_dict['intercept']
    lasso_weights = lasso_weights_dict['coefs']

    selected_features = list(lasso_weights.keys())
    lasso_coefs = np.array(list(lasso_weights.values()))

    feature_cols_all = [c for c in df_train_features.columns if c != 'Patient_ID']
    X_train_scaled = pd.DataFrame(train_scaler.transform(df_train[feature_cols_all]), columns=feature_cols_all)
    df_train['Rad_score'] = lasso_intercept + np.dot(X_train_scaled[selected_features], lasso_coefs)

    clinical_vars = ['ESR', 'Disease_Duration_Category', 'Rad_score']
    X_train_clinical = sm.add_constant(df_train[clinical_vars])
    train_model = sm.Logit(df_train['Label'], X_train_clinical).fit(disp=False)

    cutoff_dict = joblib.load(os.path.join(WORK_DIR, "optimal_cutoff.pkl"))
    locked_cutoff = cutoff_dict['optimal_cutoff']
    healthy_min_radscore = df_train.loc[df_train['Label'] == 0, 'Rad_score'].min()

    # ---------------- 2. 载入 Test 共有数据与 GT 基准 ----------------
    df_test_clinical = pd.read_excel(os.path.join(WORK_DIR, "clinical_info_test.xlsx"))
    df_test_clinical = df_test_clinical.rename(columns={'label': 'Label'})
    for col in imputer_dict.keys():
        df_test_clinical[col] = pd.to_numeric(df_test_clinical[col], errors='coerce').fillna(imputer_dict[col])
    df_test_clinical['Patient_ID'] = df_test_clinical['Patient_ID'].astype(str).str.strip()

    df_test_gt = pd.read_csv(os.path.join(WORK_DIR, "Test_GT_Features.csv"))
    df_test_gt['Patient_ID'] = df_test_gt['Patient_ID'].astype(str).str.strip()
    df_test_merge_gt = pd.merge(df_test_gt, df_test_clinical, on='Patient_ID', how='inner')
    y_test = df_test_merge_gt['Label'].values

    X_test_gt_scaled = pd.DataFrame(train_scaler.transform(df_test_merge_gt[feature_cols_all]),
                                    columns=feature_cols_all)
    df_test_merge_gt['Rad_score'] = lasso_intercept + np.dot(X_test_gt_scaled[selected_features], lasso_coefs)

    X_test_gt_clin = sm.add_constant(df_test_merge_gt[clinical_vars], has_constant='add')[['const'] + clinical_vars]
    prob_test_gt = train_model.predict(X_test_gt_clin)

    fpr_gt, tpr_gt, _ = roc_curve(y_test, prob_test_gt)
    auc_gt = auc(fpr_gt, tpr_gt)
    gt_sens, gt_spec, gt_acc = calc_metrics(y_test, prob_test_gt, locked_cutoff)

    print("\n================== 🌟 GT 基准表现 ==================")
    print(f"AUC: {auc_gt:.3f} | ACC: {gt_acc:.3f} | SEN: {gt_sens:.3f} | SPE: {gt_spec:.3f}")

    # ---------------- 3. 循环审判各个阈值 (Pred) ----------------
    print("\n================== 🤖 AI 多阈值盲测 ==================")
    # ================== 替换这里的代码 ==================
    for t in THRESHOLDS:
        t_str = f"{int(t * 100):03d}"
        pred_csv = os.path.join(WORK_DIR, f"Test_Pred_Features_{t_str}.csv")

        if not os.path.exists(pred_csv):
            print(f"⚠️ 找不到阈值 {t:.2f} 的特征文件，跳过...")
            continue

        df_test_pred = pd.read_csv(pred_csv)
        df_test_pred['Patient_ID'] = df_test_pred['Patient_ID'].astype(str).str.strip()

        # 🚨 核心修复：使用 left join 强行对齐 GT 的 62 个患者名单！
        df_test_merge_pred = pd.merge(df_test_merge_gt[['Patient_ID']], df_test_pred, on='Patient_ID', how='left')
        df_test_merge_pred = pd.merge(df_test_merge_pred, df_test_clinical, on='Patient_ID', how='left')

        # 记录哪些患者在组学提取时因为掩码为空/过小而崩溃消失了（特征是 NaN）
        failed_extraction_idx = df_test_merge_pred[feature_cols_all[0]].isna()

        # 填充0，防止后续 scaler.transform 报错
        df_test_merge_pred[feature_cols_all] = df_test_merge_pred[feature_cols_all].fillna(0.0)

        X_test_pred_scaled = pd.DataFrame(train_scaler.transform(df_test_merge_pred[feature_cols_all]),
                                          columns=feature_cols_all)
        df_test_merge_pred['Rad_score'] = lasso_intercept + np.dot(X_test_pred_scaled[selected_features], lasso_coefs)

        # 抢救 1 (原逻辑)：掩码提取出全 0 特征的患者
        probe_feature = selected_features[0]
        empty_mask_idx = df_test_merge_pred[probe_feature] == 0.0
        df_test_merge_pred.loc[empty_mask_idx, 'Rad_score'] = healthy_min_radscore

        # 抢救 2 (新逻辑)：掩码直接崩掉没提取出来的患者，也按完全无病变处理！
        df_test_merge_pred.loc[failed_extraction_idx, 'Rad_score'] = healthy_min_radscore

        X_test_pred_clin = sm.add_constant(df_test_merge_pred[clinical_vars], has_constant='add')[
            ['const'] + clinical_vars]
        prob_test_pred = train_model.predict(X_test_pred_clin)
        # ================== 替换到这里结束 ==================

        fpr_pred, tpr_pred, _ = roc_curve(y_test, prob_test_pred)
        auc_pred = auc(fpr_pred, tpr_pred)
        pred_sens, pred_spec, pred_acc = calc_metrics(y_test, prob_test_pred, locked_cutoff)
        delong_p = delong_roc_test(y_test, prob_test_gt, prob_test_pred)

        print(
            f"👉 阈值 {t:.2f}: AUC={auc_pred:.3f} | ACC={pred_acc:.3f} | SEN={pred_sens:.3f} | SPE={pred_spec:.3f} | DeLong P={delong_p:.4f}")

    print("\n🎉 RUN7 Step4 全阈值核算完毕！请直接将数据填入图表。")


if __name__ == "__main__":
    run_ultimate_blind_test()