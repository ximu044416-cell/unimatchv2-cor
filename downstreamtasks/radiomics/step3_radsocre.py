import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="ticks", context="paper", font_scale=1.2)


def draw_handmade_nomogram(result_model, X_data, out_path):
    """动态适应任意数量特征的纯手工顶刊列线图引擎"""
    params = result_model.params
    features = [f for f in params.index if f != 'const']
    scores = {}
    max_score_range = 0

    for feat in features:
        coef = params[feat]
        f_min = X_data[feat].min()
        f_max = X_data[feat].max()
        val_range = np.linspace(f_min, f_max, 5) if len(np.unique(X_data[feat])) > 2 else np.array([f_min, f_max])
        contrib = coef * val_range
        scores[feat] = {
            'min_val': f_min, 'max_val': f_max,
            'min_contrib': min(contrib), 'max_contrib': max(contrib),
            'range': max(contrib) - min(contrib),
            'coef': coef
        }
        if scores[feat]['range'] > max_score_range:
            max_score_range = scores[feat]['range']

    if max_score_range == 0: max_score_range = 1
    scale = 100.0 / max_score_range

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    y_pos = len(features) + 2

    ax.hlines(y_pos, 0, 100, color='black', linewidth=1.5)
    for p in range(0, 101, 10):
        ax.vlines(p, y_pos, y_pos + 0.15, color='black')
        ax.text(p, y_pos + 0.3, str(p), ha='center', va='bottom', fontsize=9)
    ax.text(-5, y_pos, "Points", ha='right', va='center', fontsize=11, fontweight='bold')

    max_total_points = 0
    for idx, feat in enumerate(features):
        y = len(features) - idx
        s = scores[feat]
        pts_range = s['range'] * scale
        max_total_points += pts_range
        ax.hlines(y, 0, pts_range, color='black', linewidth=1.2)

        ticks = np.linspace(s['min_val'], s['max_val'], 5) if len(np.unique(X_data[feat])) > 2 else np.unique(
            X_data[feat])
        for t in ticks:
            pts = (s['coef'] * t - s['min_contrib']) * scale
            ax.vlines(pts, y, y + 0.15, color='black')
            label = f"{t:.1f}" if isinstance(t, float) and len(np.unique(X_data[feat])) > 2 else str(int(t))
            ax.text(pts, y + 0.3, label, ha='center', va='bottom', fontsize=8)
        ax.text(-5, y, feat, ha='right', va='center', fontsize=11, fontweight='bold')

    y_total = -1
    max_total_points = min(max_total_points, 400)
    ax.hlines(y_total, 0, max_total_points, color='black', linewidth=1.5)
    step = max(10, int(max_total_points) // 10)
    for p in range(0, int(max_total_points) + 1, step):
        ax.vlines(p, y_total, y_total + 0.15, color='black')
        ax.text(p, y_total + 0.3, str(p), ha='center', va='bottom', fontsize=9)
    ax.text(-5, y_total, "Total Points", ha='right', va='center', fontsize=11, fontweight='bold')

    y_risk = -2.5
    risk_probs = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    intercept = params['const']
    sum_min_contrib = sum([scores[f]['min_contrib'] for f in features])

    ax.hlines(y_risk, 0, max_total_points, color='black', linewidth=1.5)
    for r in risk_probs:
        pts = (np.log(r / (1 - r)) - intercept - sum_min_contrib) * scale
        if 0 <= pts <= max_total_points:
            ax.vlines(pts, y_risk, y_risk + 0.15, color='black')
            ax.text(pts, y_risk + 0.3, str(r), ha='center', va='bottom', fontsize=9)
    ax.text(-5, y_risk, "Risk Probability", ha='right', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(-25, max_total_points + 10)
    ax.set_ylim(-3.5, y_pos + 1.5)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def build_ultimate_pipeline():
    print("🚀 启动 Step3: 全维度临床+组学 降维建模与验证引擎...\n")

    # ================= 1. 路径修改 =================
    WORK_DIR = r"F:\downstreamtasks\radiomics"  # 🔥 请确保这是你的实际路径
    CLINICAL_FILE = os.path.join(WORK_DIR, "clinical_info_train.xlsx")
    FEATURES_FILE = os.path.join(WORK_DIR, "Train_GT_Features.csv")

    # ================= 2. 数据读取与自动化清洗 =================
    print("🧹 正在读取纯净临床基线表与新提取的 GT 特征...")
    df_clinical = pd.read_excel(CLINICAL_FILE)
    df_features = pd.read_csv(FEATURES_FILE)

    df_clinical = df_clinical.rename(columns={'label': 'Label'})

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category', 'sparcc']:
        if col in df_clinical.columns:
            df_clinical[col] = pd.to_numeric(df_clinical[col], errors='coerce')

    crp_median = df_clinical['CRP'].median()
    esr_median = df_clinical['ESR'].median()
    hla_mode = df_clinical['HLA-B27'].mode()[0]
    dur_mode = df_clinical['Disease_Duration_Category'].mode()[0]

    imputer_dict = {
        'CRP': crp_median,
        'ESR': esr_median,
        'HLA-B27': hla_mode,
        'Disease_Duration_Category': dur_mode
    }
    joblib.dump(imputer_dict, os.path.join(WORK_DIR, "imputer_dict.pkl"))
    print(f"   => 🔒 自动填充法则已生成防穿越字典 imputer_dict.pkl！")

    df_clinical['CRP'] = df_clinical['CRP'].fillna(imputer_dict['CRP'])
    df_clinical['ESR'] = df_clinical['ESR'].fillna(imputer_dict['ESR'])
    df_clinical['HLA-B27'] = df_clinical['HLA-B27'].fillna(imputer_dict['HLA-B27'])
    df_clinical['Disease_Duration_Category'] = df_clinical['Disease_Duration_Category'].fillna(
        imputer_dict['Disease_Duration_Category'])

    df_clinical['Patient_ID'] = df_clinical['Patient_ID'].astype(str).str.strip()
    df_features['Patient_ID'] = df_features['Patient_ID'].astype(str).str.strip()

    df_merged = pd.merge(df_features, df_clinical, on='Patient_ID', how='inner')
    print(f"📊 数据合并完毕！成功匹配 {len(df_merged)} 名患者。")

    y = df_merged['Label'].values
    feature_cols = [col for col in df_features.columns if col != 'Patient_ID']
    X_features = df_merged[feature_cols]

    # ================= 3. 影像特征 Z-score 标准化 =================
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=feature_cols)
    joblib.dump(scaler, os.path.join(WORK_DIR, "train_scaler.pkl"))

    # ================= 4. 降维三板斧 =================
    print("\n🪓 影像特征降维启动 (T检验 -> Spearman -> LASSO)...")
    mwu_selected = [c for c in feature_cols if
                    stats.mannwhitneyu(X_scaled.loc[y == 0, c], X_scaled.loc[y == 1, c]).pvalue < 0.05]
    X_mwu = X_scaled[mwu_selected]

    corr_matrix = X_mwu.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    spearman_selected = [c for c in mwu_selected if c not in to_drop]
    X_spearman = X_mwu[spearman_selected]

    lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 100), cv=10, random_state=42, max_iter=10000)
    lasso_cv.fit(X_spearman, y)
    coefs = lasso_cv.coef_
    optimal_alpha = lasso_cv.alpha_

    final_features = np.array(spearman_selected)[coefs != 0]
    final_coefs = coefs[coefs != 0]
    intercept = lasso_cv.intercept_

    if len(final_features) > 5:
        last_valid_features, last_valid_coefs, last_valid_intercept = final_features, final_coefs, intercept
        for a in np.logspace(np.log10(optimal_alpha), np.log10(optimal_alpha) + 2, 200):
            hard_lasso = Lasso(alpha=a, random_state=42, max_iter=10000)
            hard_lasso.fit(X_spearman, y)
            num_nonzero = np.sum(hard_lasso.coef_ != 0)
            if 0 < num_nonzero <= 7:
                final_features = np.array(spearman_selected)[hard_lasso.coef_ != 0]
                final_coefs = hard_lasso.coef_[hard_lasso.coef_ != 0]
                intercept = hard_lasso.intercept_
                break
            elif num_nonzero == 0:
                final_features, final_coefs, intercept = last_valid_features, last_valid_coefs, last_valid_intercept
                break
            else:
                last_valid_features = np.array(spearman_selected)[hard_lasso.coef_ != 0]
                last_valid_coefs = hard_lasso.coef_[hard_lasso.coef_ != 0]
                last_valid_intercept = hard_lasso.intercept_

    joblib.dump(list(final_features), os.path.join(WORK_DIR, "selected_features.pkl"))

    # 🔥 导师新增补丁：直接把 LASSO 的截距和权重打包成字典，封存进保险箱！
    lasso_weights_dict = {'intercept': float(intercept), 'coefs': dict(zip(final_features, final_coefs))}
    joblib.dump(lasso_weights_dict, os.path.join(WORK_DIR, "lasso_weights_dict.pkl"))
    print("   => 🔒 LASSO 截距与绝对权重已硬化封存至 lasso_weights_dict.pkl，Step 6 可直接无损调用！")

    df_merged['Rad_score'] = intercept + np.dot(X_spearman[final_features], final_coefs)

    print("\n🔬 影像组学分数 (Rad-score) 公式已生成并锁定：")
    formula = f"Rad-score = {intercept:.4f} "
    for feat, coef in zip(final_features, final_coefs):
        sign = "+" if coef > 0 else "-"
        formula += f"\n          {sign} {abs(coef):.4f} * [{feat}]"
    print(formula)

    # ================= 5. 临床多因素模型构建 =================
    print("\n📈 训练全维度 Logistic Regression 模型...")
    # 🔥 核心变量 (可以根据实际情况微调)
    clinical_vars_final = ['ESR', 'Disease_Duration_Category', 'Rad_score']

    X_clinical_final = df_merged[clinical_vars_final]
    X_clinical_final = sm.add_constant(X_clinical_final)

    log_reg = sm.Logit(y, X_clinical_final)
    result = log_reg.fit(disp=False)
    print(result.summary())

    # ================= 6. 绘制 Train ROC 曲线 =================
    print("\n🎨 正在绘制训练集 ROC 曲线...")
    y_pred_prob = result.predict(X_clinical_final)
    fpr, tpr, thr = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # 打印 Youden Index 供参考
    youden_idx = np.argmax(tpr - fpr)
    optimal_cutoff = thr[youden_idx]
    print(f"   => Train AUC: {roc_auc:.3f}")
    print(f"   => 建议锁定最佳 Cutoff (Youden Index): {optimal_cutoff:.4f}")

    # 🔥 顺便把 Cutoff 也存进字典，实现 100% 自动化闭环
    cutoff_dict = {'optimal_cutoff': float(optimal_cutoff)}
    joblib.dump(cutoff_dict, os.path.join(WORK_DIR, "optimal_cutoff.pkl"))

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color='#c0392b', lw=2.5, label=f'Train AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('ROC Curve - Train Cohort', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(os.path.join(WORK_DIR, "Train_ROC_Curve.png"), bbox_inches='tight')
    plt.close()
    print("   => 曲线已保存至 Train_ROC_Curve.png")

    # ================= 7. 绘制手工顶刊 Nomogram =================
    print("\n🎨 正在绘制核心特征列线图 (Nomogram)...")
    draw_handmade_nomogram(result, df_merged[clinical_vars_final], os.path.join(WORK_DIR, "Nomogram_Ultimate.png"))
    print("   => 列线图已保存至 Nomogram_Ultimate.png")

    # ================= 8. 终极杀招：Rad-score vs SPARCC =================
    print("\n🔬 正在绘制 Rad-score 与临床金标准 SPARCC 的结构效度分析...")
    if 'sparcc' in df_merged.columns:
        valid_corr_df = df_merged.dropna(subset=['sparcc'])
        if len(valid_corr_df) > 10:
            r, p_val = stats.spearmanr(valid_corr_df['Rad_score'], valid_corr_df['sparcc'])

            plt.figure(figsize=(8, 6), dpi=300)
            sns.regplot(x='Rad_score', y='sparcc', data=valid_corr_df,
                        scatter_kws={'alpha': 0.6, 'color': '#2980b9', 's': 50},
                        line_kws={'color': '#c0392b', 'linewidth': 2})

            plt.title('Construct Validity: Rad-score vs SPARCC', fontweight='bold', pad=15, fontsize=14)
            plt.xlabel('AI Radiomics Score (Rad-score)', fontweight='bold', fontsize=12)
            plt.ylabel('Ground Truth SPARCC Score', fontweight='bold', fontsize=12)

            text_str = f'Spearman r = {r:.3f}\nP-value < 0.001' if p_val < 0.001 else f'Spearman r = {r:.3f}\nP-value = {p_val:.3f}'
            plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

            plt.tight_layout()
            plt.savefig(os.path.join(WORK_DIR, "RadScore_vs_SPARCC_Spearman.png"))
            print(f"   => 散点图生成！Spearman r = {r:.3f}。已保存至 RadScore_vs_SPARCC_Spearman.png")
        else:
            print("   ⚠️ 警告：具有有效 SPARCC 评分的患者数不足 10 人，跳过绘图。")


if __name__ == "__main__":
    build_ultimate_pipeline()