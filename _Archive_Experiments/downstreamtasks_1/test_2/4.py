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
    print("🚀 启动 Step 6: 终极多轨盲测大考 (自适应样本对齐版)...\n")
    WORK_DIR = r"F:\radiomics"
    PRED_DIR = r"F:\check"

    target_thresholds = ["065", "070", "075"]

    # ---------------- 1. 重建 Train 模型并锁定尤登指数 ----------------
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

    df_train['Rad_score'] = lasso_intercept + np.dot(X_train_scaled[selected_features], lasso_coefs)

    clinical_vars = ['ESR', 'Disease_Duration_Category', 'Rad_score']
    X_train_clinical = sm.add_constant(df_train[clinical_vars])
    train_model = sm.Logit(y_train, X_train_clinical).fit(disp=False)

    prob_train = train_model.predict(X_train_clinical)
    fpr_train, tpr_train, thr_train = roc_curve(y_train, prob_train)
    auc_train = auc(fpr_train, tpr_train)

    youden_idx = np.argmax(tpr_train - fpr_train)
    locked_cutoff = thr_train[youden_idx]
    print(f"   => ⚔️ Train 模型重构成功！AUC: {auc_train:.3f}")
    print(f"   => 🛑 绝对冻结：最佳临床概率截断值 (Cutoff) 锁定为 {locked_cutoff:.4f}")

    # ---------------- 2. 测试集 GT 数据准备 ----------------
    print("\n📦 正在加载测试集临床表与 GT 特征...")
    df_test_clinical = pd.read_excel(os.path.join(WORK_DIR, "clinical_info_test.xlsx"))
    df_test_clinical = df_test_clinical.rename(columns={'label': 'Label'})
    df_test_gt = pd.read_csv(os.path.join(PRED_DIR, "Test_GT_Features.csv"))

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category']:
        df_test_clinical[col] = pd.to_numeric(df_test_clinical[col], errors='coerce')
    for col in imputer_dict.keys():
        df_test_clinical[col] = df_test_clinical[col].fillna(imputer_dict[col])

    df_test_clinical['Patient_ID'] = df_test_clinical['Patient_ID'].astype(str).str.strip()
    df_test_gt['Patient_ID'] = df_test_gt['Patient_ID'].astype(str).str.strip()
    df_test_merge_gt = pd.merge(df_test_gt, df_test_clinical, on='Patient_ID', how='inner')
    y_test_gt = df_test_merge_gt['Label'].values

    # 计算 GT 整体概率
    X_test_gt_scaled = pd.DataFrame(train_scaler.transform(df_test_merge_gt[feature_cols_all]),
                                    columns=feature_cols_all)
    df_test_merge_gt['Rad_score'] = lasso_intercept + np.dot(X_test_gt_scaled[selected_features], lasso_coefs)
    X_test_gt_clin = sm.add_constant(df_test_merge_gt[clinical_vars], has_constant='add')[['const'] + clinical_vars]
    prob_test_gt = train_model.predict(X_test_gt_clin)
    fpr_gt, tpr_gt, _ = roc_curve(y_test_gt, prob_test_gt)
    auc_gt = auc(fpr_gt, tpr_gt)

    # ---------------- 3. 批量计算多阈值 Pred (动态对齐样本) ----------------
    print(f"\n⚙️ 正在执行多轨 AI 模型预测对比 (阈值: {target_thresholds})...")

    pred_results = {}

    def calc_metrics(y_t, prob):
        pred_label = (prob >= locked_cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, pred_label, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = accuracy_score(y_t, pred_label)
        return sens, spec, acc

    gt_sens, gt_spec, gt_acc = calc_metrics(y_test_gt, prob_test_gt)

    for t in target_thresholds:
        csv_path = os.path.join(PRED_DIR, f"Test_Pred_Features_{t}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ 找不到文件: {csv_path}，跳过该阈值。")
            continue

        df_pred = pd.read_csv(csv_path)
        df_pred['Patient_ID'] = df_pred['Patient_ID'].astype(str).str.strip()
        df_merge_pred = pd.merge(df_pred, df_test_clinical, on='Patient_ID', how='inner')

        # 🔥 关键修复 1：动态提取当前存活病人的 Label (比如 54 人)
        y_test_p = df_merge_pred['Label'].values

        # 🔥 关键修复 2：为了 DeLong 检验公平，必须提取这 54 人对应的 GT 预测概率！
        df_aligned_gt = pd.merge(df_pred[['Patient_ID']], df_test_merge_gt, on='Patient_ID', how='inner')
        X_aligned_gt_clin = sm.add_constant(df_aligned_gt[clinical_vars], has_constant='add')[['const'] + clinical_vars]
        prob_gt_aligned = train_model.predict(X_aligned_gt_clin)

        # 计算当前阈值下 Pred 的概率
        X_pred_scaled = pd.DataFrame(train_scaler.transform(df_merge_pred[feature_cols_all]), columns=feature_cols_all)
        df_merge_pred['Rad_score'] = lasso_intercept + np.dot(X_pred_scaled[selected_features], lasso_coefs)
        X_pred_clin = sm.add_constant(df_merge_pred[clinical_vars], has_constant='add')[['const'] + clinical_vars]
        prob_pred = train_model.predict(X_pred_clin)

        fpr_p, tpr_p, _ = roc_curve(y_test_p, prob_pred)
        auc_p = auc(fpr_p, tpr_p)

        delong_p = delong_roc_test(y_test_p, prob_gt_aligned, prob_pred)
        sens, spec, acc = calc_metrics(y_test_p, prob_pred)

        # 保存时带上 y_test_p，防止画图时崩溃
        pred_results[t] = {
            'prob': prob_pred, 'fpr': fpr_p, 'tpr': tpr_p, 'auc': auc_p,
            'delong': delong_p, 'sens': sens, 'spec': spec, 'acc': acc,
            'y_test_p': y_test_p, 'num_samples': len(y_test_p)
        }

    # ---------------- 4. 终极排行榜播报 ----------------
    print("\n🏆 ====== 终极战斗排行榜 (Test Set) ======")
    print(
        f"【金标准】GT AUC: {auc_gt:.3f} | 准确率: {gt_acc:.3f} | 敏感度: {gt_sens:.3f} | 特异度: {gt_spec:.3f} (N={len(y_test_gt)})")
    print("-" * 90)
    print(
        f"{'AI 阈值':<8} | {'有效人数':<6} | {'AUC':<7} | {'DeLong P':<10} | {'准确率':<8} | {'敏感度':<8} | {'特异度':<8}")
    print("-" * 90)

    for t in target_thresholds:
        if t in pred_results:
            res = pred_results[t]
            marker = "🌟" if res['auc'] >= 0.8 else "  "
            print(
                f"{marker} 0.{t} | N={res['num_samples']:<4} | {res['auc']:.3f}   | {res['delong']:.4f}     | {res['acc']:.3f}    | {res['sens']:.3f}    | {res['spec']:.3f}")
    print("-" * 90)

    # ---------------- 5. 顶刊图表：五线合一绘制 ----------------
    print("\n🎨 正在绘制顶刊多轨对比三大神图...")
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    # 图 1: 多线 ROC 曲线
    plt.figure(figsize=(8, 7), dpi=300)
    plt.plot(fpr_train, tpr_train, color='gray', lw=1.5, linestyle='--', label=f'Train AUC = {auc_train:.3f}')
    plt.plot(fpr_gt, tpr_gt, color='#2980b9', lw=2.5, label=f'Test (GT) AUC = {auc_gt:.3f}')

    for idx, t in enumerate(target_thresholds):
        if t in pred_results:
            plt.plot(pred_results[t]['fpr'], pred_results[t]['tpr'], color=colors[idx], lw=2,
                     label=f'Test (Pred {t}) AUC = {pred_results[t]["auc"]:.3f}')

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('Receiver Operating Characteristic (Sensitivity Analysis)', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.savefig(os.path.join(PRED_DIR, "Step6_ROC_Curve_Multi.png"), bbox_inches='tight')
    plt.close()

    # 图 2: 多线校准曲线
    plt.figure(figsize=(8, 7), dpi=300)
    prob_true_gt, prob_pred_gt = calibration_curve(y_test_gt, prob_test_gt, n_bins=5, strategy='quantile')
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred_gt, prob_true_gt, "s-", color='#2980b9', label="Test (GT)")

    for idx, t in enumerate(target_thresholds):
        if t in pred_results:
            # 🔥 关键修复 3：画图时也必须用存活的 y_test_p！
            prob_true_p, prob_pred_p = calibration_curve(pred_results[t]['y_test_p'], pred_results[t]['prob'], n_bins=5,
                                                         strategy='quantile')
            plt.plot(prob_pred_p, prob_true_p, "o-", color=colors[idx], alpha=0.8, label=f"Test (Pred {t})")

    plt.xlabel('Mean predicted probability', fontweight='bold', fontsize=12)
    plt.ylabel('Fraction of positives', fontweight='bold', fontsize=12)
    plt.title('Calibration Curve (Sensitivity Analysis)', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PRED_DIR, "Step6_Calibration_Curve_Multi.png"), bbox_inches='tight')
    plt.close()

    # 图 3: 多线 DCA 决策曲线
    plt.figure(figsize=(8, 7), dpi=300)
    thresholds_arr = np.linspace(0.01, 0.99, 100)
    nb_all = calculate_net_benefit(y_test_gt, np.ones_like(y_test_gt), thresholds_arr)
    nb_none = np.zeros_like(thresholds_arr)
    nb_gt = calculate_net_benefit(y_test_gt, prob_test_gt, thresholds_arr)

    plt.plot(thresholds_arr, nb_all, color='gray', linestyle=':', label='Treat All')
    plt.plot(thresholds_arr, nb_none, color='black', linestyle='-', label='Treat None')
    plt.plot(thresholds_arr, nb_gt, color='#2980b9', lw=2.5, label='Test (GT)')

    for idx, t in enumerate(target_thresholds):
        if t in pred_results:
            # 🔥 关键修复 4：画 DCA 时也用对应的 y_test_p
            nb_pred = calculate_net_benefit(pred_results[t]['y_test_p'], pred_results[t]['prob'], thresholds_arr)
            plt.plot(thresholds_arr, nb_pred, color=colors[idx], lw=2, alpha=0.8, label=f'Test (Pred {t})')

    plt.xlim([0.0, 0.8])
    plt.ylim([-0.05, max(nb_all.max(), nb_gt.max()) + 0.1])
    plt.xlabel('Threshold Probability', fontweight='bold', fontsize=12)
    plt.ylabel('Net Benefit', fontweight='bold', fontsize=12)
    plt.title('Decision Curve Analysis (Sensitivity Analysis)', fontweight='bold', fontsize=14)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(PRED_DIR, "Step6_DCA_Curve_Multi.png"), bbox_inches='tight')
    plt.close()

    print("🎉 顶刊对比图表生成完毕！请前往 F:\\check 文件夹查看以 _Multi 结尾的图片。")


if __name__ == "__main__":
    run_ultimate_blind_test()