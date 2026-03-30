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
    print("🚀 启动 Step 6: 终极双轨盲测大考 (GT vs Pred)...\n")
    WORK_DIR = r"F:\radiomics"

    # ---------------- 1. 时光回溯：重建 Train 模型并锁定尤登指数 ----------------
    print("🔒 正在读取 Train 数据，重构模型并提取冻结参数...")
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

    # 绝对禁止重新 fit LASSO！直接硬编码确切权重
    lasso_intercept = 0.3521
    lasso_weights = {
        'original_shape_Sphericity': -0.0342,
        'original_shape_SurfaceVolumeRatio': -0.1115,
        'wavelet-LLH_glcm_MCC': -0.1122,
        'wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis': 0.0189,
        'wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis': 0.0052,
        'wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis': 0.0152,
        'wavelet-HHH_firstorder_Maximum': 0.0127
    }
    selected_features = list(lasso_weights.keys())
    lasso_coefs = np.array(list(lasso_weights.values()))

    feature_cols_all = [c for c in df_train_features.columns if c != 'Patient_ID']
    X_train_scaled = pd.DataFrame(train_scaler.transform(df_train[feature_cols_all]), columns=feature_cols_all)

    # 严格执行常量乘法计算 Train 的 Rad_score
    df_train['Rad_score'] = lasso_intercept + np.dot(X_train_scaled[selected_features], lasso_coefs)

    # 极简全明星阵容，剔除阵亡变量
    clinical_vars = ['ESR', 'Disease_Duration_Category', 'Rad_score']
    X_train_clinical = sm.add_constant(df_train[clinical_vars])
    train_model = sm.Logit(y_train, X_train_clinical).fit(disp=False)

    prob_train = train_model.predict(X_train_clinical)
    fpr_train, tpr_train, thr_train = roc_curve(y_train, prob_train)
    auc_train = auc(fpr_train, tpr_train)

    # 冻结截断值 (Youden Index)
    youden_idx = np.argmax(tpr_train - fpr_train)
    locked_cutoff = thr_train[youden_idx]
    print(f"   => ⚔️ Train 模型重构成功！AUC: {auc_train:.3f}")
    print(f"   => 🛑 绝对冻结：最佳临床概率截断值 (Cutoff) 锁定为 {locked_cutoff:.4f}")

    # ---------------- 2. 测试集数据准备 (绝对隔离) ----------------
    print("\n📦 正在加载并无损处理 Test 数据...")
    df_test_clinical = pd.read_excel(os.path.join(WORK_DIR, "clinical_info_test.xlsx"))
    df_test_clinical = df_test_clinical.rename(columns={'label': 'Label'})
    df_test_gt = pd.read_csv(os.path.join(WORK_DIR, "Test_GT_Features_62.csv"))
    df_test_pred = pd.read_csv(os.path.join(WORK_DIR, "Test_Pred_Features_62.csv"))

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
    X_test_gt_scaled = pd.DataFrame(train_scaler.transform(df_test_merge_gt[feature_cols_all]),
                                    columns=feature_cols_all)
    X_test_pred_scaled = pd.DataFrame(train_scaler.transform(df_test_merge_pred[feature_cols_all]),
                                      columns=feature_cols_all)

    # 严格执行常量乘法计算 Test 的 Rad_score
    df_test_merge_gt['Rad_score'] = lasso_intercept + np.dot(X_test_gt_scaled[selected_features], lasso_coefs)
    df_test_merge_pred['Rad_score'] = lasso_intercept + np.dot(X_test_pred_scaled[selected_features], lasso_coefs)

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

    # 🔥 修复 2：加入绝对实力前提校验的胜利宣言
    if delong_p > 0.05 and auc_pred >= 0.80:
        print("   ✅ 统计学结论：AI 预测效能极佳 (AUC>=0.8) 且与医生手工分割无显著差异 (P>0.05)，证明可临床替代！")
    elif delong_p > 0.05 and auc_pred < 0.80:
        print("   ⚠️ 统计学结论：虽然无显著差异 (P>0.05)，但 AI 本身预测效能欠佳 (AUC<0.8)，需谨慎解读。")
    else:
        print("   ⚠️ 统计学结论：两组存在显著差异。")

    # 计算基于锁定 Cutoff 的硬指标
    def calc_metrics(y_t, prob):
        pred_label = (prob >= locked_cutoff).astype(int)
        # 🔥 修复 1：防爆声明 labels=[0, 1] 彻底杜绝小样本解包崩溃
        tn, fp, fn, tp = confusion_matrix(y_t, pred_label, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = accuracy_score(y_t, pred_label)
        return sens, spec, acc

    gt_sens, gt_spec, gt_acc = calc_metrics(y_test, prob_test_gt)
    pred_sens, pred_spec, pred_acc = calc_metrics(y_test, prob_test_pred)

    print("\n📊 基于锁定阈值 (Cutoff={:.3f}) 的硬核指标：".format(locked_cutoff))
    print(f"   * GT 组   - 准确率: {gt_acc:.3f}, 敏感度: {gt_sens:.3f}, 特异度: {gt_spec:.3f}")
    print(f"   * Pred 组 - 准确率: {pred_acc:.3f}, 敏感度: {pred_sens:.3f}, 特异度: {pred_spec:.3f}")

    # ---------------- 5. 顶刊图表三连击 ----------------
    print("\n🎨 正在绘制顶刊三大神图...")

    # 图 1: 三线合一 ROC 曲线
    plt.figure(figsize=(8, 7), dpi=300)
    plt.plot(fpr_train, tpr_train, color='gray', lw=1.5, linestyle='--', label=f'Train AUC = {auc_train:.3f}')
    plt.plot(fpr_gt, tpr_gt, color='#2980b9', lw=2.5, label=f'Test (GT) AUC = {auc_gt:.3f}')
    plt.plot(fpr_pred, tpr_pred, color='#c0392b', lw=2.5, label=f'Test (AI-Pred) AUC = {auc_pred:.3f}')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)

    delong_text = f"DeLong Test (GT vs Pred)\nP-value = {delong_p:.3f}"
    plt.text(0.5, 0.2, delong_text, fontsize=11, ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.savefig(os.path.join(WORK_DIR, "Step6_ROC_Curve.png"), bbox_inches='tight')
    plt.close()

    # 图 2: 双轨校准曲线 (Calibration Curve)
    plt.figure(figsize=(8, 7), dpi=300)
    prob_true_gt, prob_pred_gt = calibration_curve(y_test, prob_test_gt, n_bins=5, strategy='quantile')
    prob_true_pred, prob_pred_pred = calibration_curve(y_test, prob_test_pred, n_bins=5, strategy='quantile')

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred_gt, prob_true_gt, "s-", color='#2980b9', label="Test (GT)")
    plt.plot(prob_pred_pred, prob_true_pred, "o-", color='#c0392b', label="Test (AI-Pred)")
    plt.xlabel('Mean predicted probability', fontweight='bold', fontsize=12)
    plt.ylabel('Fraction of positives', fontweight='bold', fontsize=12)
    plt.title('Calibration Curve', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(WORK_DIR, "Step6_Calibration_Curve.png"), bbox_inches='tight')
    plt.close()

    # 图 3: 双轨 DCA 决策曲线
    plt.figure(figsize=(8, 7), dpi=300)
    thresholds = np.linspace(0.01, 0.99, 100)

    nb_all = calculate_net_benefit(y_test, np.ones_like(y_test), thresholds)
    nb_none = np.zeros_like(thresholds)
    nb_gt = calculate_net_benefit(y_test, prob_test_gt, thresholds)
    nb_pred = calculate_net_benefit(y_test, prob_test_pred, thresholds)

    plt.plot(thresholds, nb_all, color='gray', linestyle=':', label='Treat All')
    plt.plot(thresholds, nb_none, color='black', linestyle='-', label='Treat None')
    plt.plot(thresholds, nb_gt, color='#2980b9', lw=2, label='Test (GT)')
    plt.plot(thresholds, nb_pred, color='#c0392b', lw=2, label='Test (AI-Pred)')

    plt.xlim([0.0, 0.8])
    plt.ylim([-0.05, max(nb_all.max(), nb_gt.max(), nb_pred.max()) + 0.1])
    plt.xlabel('Threshold Probability', fontweight='bold', fontsize=12)
    plt.ylabel('Net Benefit', fontweight='bold', fontsize=12)
    plt.title('Decision Curve Analysis (DCA)', fontweight='bold', fontsize=14)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(WORK_DIR, "Step6_DCA_Curve.png"), bbox_inches='tight')
    plt.close()

    print("🎉 顶刊图表三连击生成完毕！请前往文件夹查看：")
    print("   1. Step6_ROC_Curve.png")
    print("   2. Step6_Calibration_Curve.png")
    print("   3. Step6_DCA_Curve.png")


if __name__ == "__main__":
    run_ultimate_blind_test()