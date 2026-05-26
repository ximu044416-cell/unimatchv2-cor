import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
import joblib
import scipy.stats as st
import os
import warnings

warnings.filterwarnings('ignore')


# ================= 底层核武器：DeLong's Test 算法实现 =================
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


# ================= 底层核武器：DCA 决策曲线净获益计算 =================
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    N = len(y_true)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        if thresh == 1:
            nb = 0
        else:
            nb = (tp / N) - (fp / N) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    return np.array(net_benefits)


# ================= 核心流水线 =================
def run_ultimate_blind_test():
    print("🚀 启动 Step 6: 终极双轨盲测大考 (纯自动化无损读取 + 异常拦截版)...\n")

    WORK_DIR = r"/RUN7/radiomics"

    # ---------------- 1. 自动读取 Train 数据与所有冻结权重 ----------------
    print("🔒 正在读取 Train 数据，并自动加载所有冻结权重...")
    df_train_clinical = pd.read_excel(os.path.join(WORK_DIR, "clinical_info_train.xlsx"))
    df_train_clinical = df_train_clinical.rename(columns={'label': 'Label'})
    df_train_features = pd.read_csv(os.path.join(WORK_DIR, "Train_GT_Features.csv"))

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category']:
        df_train_clinical[col] = pd.to_numeric(df_train_clinical[col], errors='coerce')

    imputer_dict = joblib.load(os.path.join(WORK_DIR, "imputer_dict.pkl"))
    for col in imputer_dict.keys():
        df_train_clinical[col] = df_train_clinical[col].fillna(imputer_dict[col])

    df_train_clinical['Patient_ID'] = df_train_clinical['Patient_ID'].astype(str).str.strip()
    df_train_features['Patient_ID'] = df_train_features['Patient_ID'].astype(str).str.strip()
    df_train = pd.merge(df_train_features, df_train_clinical, on='Patient_ID', how='inner')

    y_train = df_train['Label'].values
    train_scaler = joblib.load(os.path.join(WORK_DIR, "train_scaler.pkl"))

    lasso_weights_dict = joblib.load(os.path.join(WORK_DIR, "lasso_weights_dict.pkl"))
    lasso_intercept = lasso_weights_dict['intercept']
    lasso_weights = lasso_weights_dict['coefs']

    selected_features = list(lasso_weights.keys())
    lasso_coefs = np.array(list(lasso_weights.values()))
    print(f"   => ✅ 成功加载 {len(selected_features)} 个 LASSO 黄金特征！")

    feature_cols_all = [c for c in df_train_features.columns if c != 'Patient_ID']
    X_train_scaled = pd.DataFrame(train_scaler.transform(df_train[feature_cols_all]), columns=feature_cols_all)

    # 严格执行常量乘法计算 Train 的 Rad_score
    df_train['Rad_score'] = lasso_intercept + np.dot(X_train_scaled[selected_features], lasso_coefs)

    clinical_vars = ['ESR', 'Disease_Duration_Category', 'Rad_score']
    X_train_clinical = sm.add_constant(df_train[clinical_vars])
    train_model = sm.Logit(y_train, X_train_clinical).fit(disp=False)

    prob_train = train_model.predict(X_train_clinical)
    fpr_train, tpr_train, thr_train = roc_curve(y_train, prob_train)
    auc_train = auc(fpr_train, tpr_train)

    cutoff_dict = joblib.load(os.path.join(WORK_DIR, "optimal_cutoff.pkl"))
    locked_cutoff = cutoff_dict['optimal_cutoff']

    print(f"   => ⚔️ Train 临床 Logistic 模型重构成功！AUC: {auc_train:.3f}")
    print(f"   => 🛑 成功加载绝对冻结的 Cutoff 截断值: {locked_cutoff:.4f}")

    # ---------------- 2. 测试集数据准备 (绝对隔离) ----------------
    print("\n📦 正在加载并无损处理 Test 数据...")
    df_test_clinical = pd.read_excel(os.path.join(WORK_DIR, "clinical_info_test.xlsx"))
    df_test_clinical = df_test_clinical.rename(columns={'label': 'Label'})

    df_test_gt = pd.read_csv(os.path.join(WORK_DIR, "Test_GT_Features.csv"))
    df_test_pred = pd.read_csv(os.path.join(WORK_DIR, "Test_Pred_Features.csv"))

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category']:
        df_test_clinical[col] = pd.to_numeric(df_test_clinical[col], errors='coerce')

    for col in imputer_dict.keys():
        df_test_clinical[col] = df_test_clinical[col].fillna(imputer_dict[col])

    df_test_clinical['Patient_ID'] = df_test_clinical['Patient_ID'].astype(str).str.strip()
    df_test_gt['Patient_ID'] = df_test_gt['Patient_ID'].astype(str).str.strip()
    df_test_pred['Patient_ID'] = df_test_pred['Patient_ID'].astype(str).str.strip()

    df_test_merge_gt = pd.merge(df_test_gt, df_test_clinical, on='Patient_ID', how='inner')
    df_test_merge_pred = pd.merge(df_test_pred, df_test_clinical, on='Patient_ID', how='inner')
    y_test = df_test_merge_gt['Label'].values
    print(f"   => 匹配到 {len(df_test_merge_gt)} 名有效测试集患者。")

    # ---------------- 3. 计算双轨特征与预测概率 ----------------
    print("\n⚙️ 正在执行双轨模型预测 (金标准 GT vs 全自动 AI Pred)...")
    X_test_gt_scaled = pd.DataFrame(train_scaler.transform(df_test_merge_gt[feature_cols_all]), columns=feature_cols_all)
    X_test_pred_scaled = pd.DataFrame(train_scaler.transform(df_test_merge_pred[feature_cols_all]), columns=feature_cols_all)

    df_test_merge_gt['Rad_score'] = lasso_intercept + np.dot(X_test_gt_scaled[selected_features], lasso_coefs)
    df_test_merge_pred['Rad_score'] = lasso_intercept + np.dot(X_test_pred_scaled[selected_features], lasso_coefs)

    # =========================================================================
    # 原始逻辑：空掩码患者抢救
    probe_feature = selected_features[0]
    empty_mask_idx = df_test_merge_pred[probe_feature] == 0.0

    healthy_min_radscore = df_train.loc[df_train['Label'] == 0, 'Rad_score'].min()
    df_test_merge_pred.loc[empty_mask_idx, 'Rad_score'] = healthy_min_radscore

    if empty_mask_idx.sum() > 0:
        print(f"\n   => 🚑 导师抢救介入：成功拦截 {empty_mask_idx.sum()} 名 AI 判定无水肿的患者！")
        print(f"   => 已将其因 0.0 引起的乱码 Rad_score，重置为绝对健康底线 ({healthy_min_radscore:.4f})！")
    # =========================================================================

    X_test_gt_clin = sm.add_constant(df_test_merge_gt[clinical_vars], has_constant='add')
    X_test_pred_clin = sm.add_constant(df_test_merge_pred[clinical_vars], has_constant='add')

    # 确保列顺序完全一致
    X_test_gt_clin = X_test_gt_clin[['const'] + clinical_vars]
    X_test_pred_clin = X_test_pred_clin[['const'] + clinical_vars]

    prob_test_gt = train_model.predict(X_test_gt_clin)
    prob_test_pred = train_model.predict(X_test_pred_clin)

    # ---------------- 4. 统计学审判 (AUC与DeLong) ----------------
    fpr_gt, tpr_gt, _ = roc_curve(y_test, prob_test_gt)
    auc_gt = auc(fpr_gt, tpr_gt)
    fpr_pred, tpr_pred, _ = roc_curve(y_test, prob_test_pred)
    auc_pred = auc(fpr_pred, tpr_pred)

    delong_p = delong_roc_test(y_test, prob_test_gt, prob_test_pred)

    print(f"\n🏆 ====== 终极战斗报告 ======")
    print(f"   * Test GT AUC (医生金标准): {auc_gt:.3f}")
    print(f"   * Test Pred AUC (AI全自动): {auc_pred:.3f}")
    print(f"   * DeLong Test P-value: {delong_p:.4f}")

    if delong_p > 0.05 and auc_pred >= 0.80:
        print("   ✅ 统计学结论：AI 预测效能极佳 (AUC>=0.8) 且与医生手工分割无显著差异 (P>0.05)，证明可临床替代！")
    elif delong_p > 0.05 and auc_pred < 0.80:
        print("   ⚠️ 统计学结论：虽然无显著差异 (P>0.05)，但 AI 本身预测效能欠佳 (AUC<0.8)，需谨慎解读。")
    else:
        print("   ⚠️ 统计学结论：两组存在显著差异。")

    def calc_metrics(y_t, prob):
        pred_label = (prob >= locked_cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, pred_label, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = accuracy_score(y_t, pred_label)
        return sens, spec, acc

    gt_sens, gt_spec, gt_acc = calc_metrics(y_test, prob_test_gt)
    pred_sens, pred_spec, pred_acc = calc_metrics(y_test, prob_test_pred)

    print(f"\n📌 GT   | ACC={gt_acc:.3f} | SEN={gt_sens:.3f} | SPE={gt_spec:.3f}")
    print(f"📌 Pred | ACC={pred_acc:.3f} | SEN={pred_sens:.3f} | SPE={pred_spec:.3f}")

    # ---------------- 5. 结果导出 ----------------
    result_df = pd.DataFrame({
        "Patient_ID": df_test_merge_gt["Patient_ID"],
        "Label": y_test,
        "GT_Rad_score": df_test_merge_gt["Rad_score"],
        "Pred_Rad_score": df_test_merge_pred["Rad_score"],
        "GT_Prob": prob_test_gt,
        "Pred_Prob": prob_test_pred
    })
    result_df.to_csv(os.path.join(WORK_DIR, "Final_Test_Patientwise_Results.csv"), index=False)

    summary_df = pd.DataFrame([{
        "Train_AUC": auc_train,
        "Test_GT_AUC": auc_gt,
        "Test_Pred_AUC": auc_pred,
        "DeLong_p": delong_p,
        "GT_ACC": gt_acc,
        "GT_SEN": gt_sens,
        "GT_SPE": gt_spec,
        "Pred_ACC": pred_acc,
        "Pred_SEN": pred_sens,
        "Pred_SPE": pred_spec,
        "Cutoff": locked_cutoff
    }])
    summary_df.to_csv(os.path.join(WORK_DIR, "RUN7_Final_Test_Summary.csv"), index=False)

    # ---------------- 6. ROC 曲线 ----------------
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr_gt, tpr_gt, color='#2b83ba', lw=2.5, label=f'GT AUC = {auc_gt:.3f}')
    plt.plot(fpr_pred, tpr_pred, color='#d7191c', lw=2.5, label=f'Pred AUC = {auc_pred:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('ROC Curve - Test Cohort', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(WORK_DIR, "Test_ROC_Curve.png"), bbox_inches='tight')
    plt.close()

    # ---------------- 7. 校准曲线 ----------------
    plt.figure(figsize=(8, 6), dpi=300)
    frac_pos_gt, mean_pred_gt = calibration_curve(y_test, prob_test_gt, n_bins=5)
    frac_pos_pred, mean_pred_pred = calibration_curve(y_test, prob_test_pred, n_bins=5)

    plt.plot(mean_pred_gt, frac_pos_gt, marker='o', linewidth=2, label='GT Contour')
    plt.plot(mean_pred_pred, frac_pos_pred, marker='s', linewidth=2, label='AI Contour')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Mean Predicted Probability', fontweight='bold')
    plt.ylabel('Observed Probability', fontweight='bold')
    plt.title('Calibration Curve - Test Cohort', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "Test_Calibration_Curve.png"), bbox_inches='tight')
    plt.close()

    # ---------------- 8. DCA ----------------
    thresholds = np.linspace(0.01, 0.99, 99)
    nb_gt = calculate_net_benefit(y_test, prob_test_gt, thresholds)
    nb_pred = calculate_net_benefit(y_test, prob_test_pred, thresholds)
    prevalence = np.mean(y_test)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_none = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(thresholds, nb_gt, label='GT Contour', linewidth=2)
    plt.plot(thresholds, nb_pred, label='AI Contour', linewidth=2)
    plt.plot(thresholds, nb_all, '--', color='gray', label='Treat All')
    plt.plot(thresholds, nb_none, ':', color='black', label='Treat None')
    plt.xlabel('Threshold Probability', fontweight='bold')
    plt.ylabel('Net Benefit', fontweight='bold')
    plt.title('Decision Curve Analysis - Test Cohort', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "Test_DCA.png"), bbox_inches='tight')
    plt.close()

    print("\n🎉 RUN7 Step4 已全部完成。")


if __name__ == "__main__":
    run_ultimate_blind_test()