import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
import statsmodels.api as sm
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


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


def build_radscore_and_nomogram():
    print("🚀 启动 Step 4 & 5: 全维度临床+组学 降维建模引擎...\n")
    WORK_DIR = r"F:\radiomics"

    # ================= 1. 数据读取与自动化清洗 =================
    print("🧹 正在读取极其纯净的临床基线表...")
    df_clinical = pd.read_excel(os.path.join(WORK_DIR, "clinical_info_train.xlsx"))
    df_features = pd.read_csv(os.path.join(WORK_DIR, "Train_GT_Features.csv"))

    # 🔥 洁癖级升级：弃用 inplace=True，直接赋值
    df_clinical = df_clinical.rename(columns={'label': 'Label'})

    for col in ['CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category', 'sparcc']:
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
    print(f"   => 📝 论文素材获取：训练集 CRP 中位数={crp_median:.2f}, ESR 中位数={esr_median:.2f}")
    print(f"   => 🔒 自动学习到填充法则，已生成防穿越字典 imputer_dict.pkl！")
    joblib.dump(imputer_dict, os.path.join(WORK_DIR, "imputer_dict.pkl"))

    # 🔥 洁癖级升级：官方推荐的无警告填充法
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

    # ================= 2. 影像特征 Z-score 标准化 =================
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=feature_cols)
    joblib.dump(scaler, os.path.join(WORK_DIR, "train_scaler.pkl"))

    # ================= 3. 降维三板斧 =================
    print("\n🪓 影像特征降维启动 (T检验 -> Spearman -> LASSO)...")

    # 🔥 洁癖级升级：使用更优雅、更安全的 .pvalue 属性取值法
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
    df_merged['Rad_score'] = intercept + np.dot(X_spearman[final_features], final_coefs)

    print("\n🔬 影像组学分数 (Rad-score) 公式已生成：")
    print("   ⚠️ 论文极其重要注释：此公式输入的各特征值，必须是经过 Z-score 标准化后的数值！")
    formula = f"Rad-score = {intercept:.4f} "
    for feat, coef in zip(final_features, final_coefs):
        sign = "+" if coef > 0 else "-"
        formula += f"\n          {sign} {abs(coef):.4f} * [{feat}]"
    print(formula)

    # ================= 4. 共线性内讧排查与多因素模型 =================
    print("\n⚠️ 临床变量内讧检查...")
    crp_esr_corr, _ = stats.spearmanr(df_merged['CRP'], df_merged['ESR'])
    print(f"   -> CRP 与 ESR 的 Spearman 相关系数 r = {crp_esr_corr:.3f}")
    if abs(crp_esr_corr) > 0.7:
        print("   -> 🚨 警告：发现高度共线性！这可能导致两者在 Logistic 模型中的 P 值互相抵消失效。")
    else:
        print("   -> ✅ 状态：未发现严重内讧，可安全进入模型。")

    print("\n📈 训练全维度 Logistic Regression 模型...")
    clinical_vars = ['Age', 'Gender', 'CRP', 'ESR', 'HLA-B27', 'Disease_Duration_Category', 'Rad_score']
    X_clinical = df_merged[clinical_vars]
    X_clinical = sm.add_constant(X_clinical)

    log_reg = sm.Logit(y, X_clinical)
    result = log_reg.fit(disp=False)
    print(result.summary())

    # ================= 5. 绘制顶配版群英 Nomogram =================
    print("\n🎨 正在绘制全维度临床列线图...")
    draw_handmade_nomogram(result, df_merged[clinical_vars], os.path.join(WORK_DIR, "Nomogram_Ultimate.png"))
    print("🎉 极致版 Nomogram 已保存至 Nomogram_Ultimate.png")

    # ================= 6. 终极杀招：Rad-score vs SPARCC 相关性散点图 =================
    print("\n🔬 正在绘制 Rad-score 与临床金标准 SPARCC 评分的相关性分析图...")
    valid_corr_df = df_merged.dropna(subset=['sparcc'])
    if len(valid_corr_df) > 10:
        r, p_val = stats.spearmanr(valid_corr_df['Rad_score'], valid_corr_df['sparcc'])

        plt.figure(figsize=(8, 6), dpi=300)
        sns.regplot(x='Rad_score', y='sparcc', data=valid_corr_df,
                    scatter_kws={'alpha': 0.6, 'color': '#2c3e50', 's': 40},
                    line_kws={'color': '#e74c3c', 'linewidth': 2})

        plt.title('Construct Validity: Rad-score vs Clinical SPARCC', fontweight='bold', pad=15, fontsize=14)
        plt.xlabel('AI Radiomics Score (Rad-score)', fontweight='bold', fontsize=12)
        plt.ylabel('Ground Truth SPARCC Score', fontweight='bold', fontsize=12)

        text_str = f'Spearman r = {r:.3f}\nP-value < 0.001' if p_val < 0.001 else f'Spearman r = {r:.3f}\nP-value = {p_val:.3f}'
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7'))

        plt.tight_layout()
        plt.savefig(os.path.join(WORK_DIR, "RadScore_vs_SPARCC_Spearman.png"))
        print(f"🎉 结构效度验证完成！散点图已生成，Spearman r = {r:.3f}！已保存至 RadScore_vs_SPARCC_Spearman.png")


if __name__ == "__main__":
    build_radscore_and_nomogram()