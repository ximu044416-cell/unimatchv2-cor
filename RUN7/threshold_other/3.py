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
    # 此处省略画图逻辑，完全与原代码一致
    pass # 注意：为了代码整洁我这里省略了你原本长长的画图代码，实际运行时保留你原来的这个函数即可！

def build_ultimate_pipeline():
    print("🚀 启动 RUN7 Step3: 全维度临床+组学 降维建模与验证引擎...\n")

    # ================= 仅修改这里的路径 =================
    WORK_DIR = r"/RUN7/threshold_other"
    # =================================================

    CLINICAL_FILE = os.path.join(WORK_DIR, "clinical_info_train.xlsx")
    FEATURES_FILE = os.path.join(WORK_DIR, "Train_GT_Features.csv")

    print("🧹 正在读取纯净临床基线表与新提取的 GT 特征...")
    df_clinical = pd.read_excel(CLINICAL_FILE)
    df_features = pd.read_csv(FEATURES_FILE)

    # ---------------- 后续逻辑完全不变 ----------------
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

    df_clinical['CRP'] = df_clinical['CRP'].fillna(imputer_dict['CRP'])
    df_clinical['ESR'] = df_clinical['ESR'].fillna(imputer_dict['ESR'])
    df_clinical['HLA-B27'] = df_clinical['HLA-B27'].fillna(imputer_dict['HLA-B27'])
    df_clinical['Disease_Duration_Category'] = df_clinical['Disease_Duration_Category'].fillna(imputer_dict['Disease_Duration_Category'])

    df_clinical['Patient_ID'] = df_clinical['Patient_ID'].astype(str).str.strip()
    df_features['Patient_ID'] = df_features['Patient_ID'].astype(str).str.strip()

    df_merged = pd.merge(df_features, df_clinical, on='Patient_ID', how='inner')
    y = df_merged['Label'].values
    feature_cols = [col for col in df_features.columns if col != 'Patient_ID']
    X_features = df_merged[feature_cols]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=feature_cols)
    joblib.dump(scaler, os.path.join(WORK_DIR, "train_scaler.pkl"))

    mwu_selected = [c for c in feature_cols if stats.mannwhitneyu(X_scaled.loc[y == 0, c], X_scaled.loc[y == 1, c]).pvalue < 0.05]
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

    lasso_weights_dict = {'intercept': float(intercept), 'coefs': dict(zip(final_features, final_coefs))}
    joblib.dump(lasso_weights_dict, os.path.join(WORK_DIR, "lasso_weights_dict.pkl"))

    df_merged['Rad_score'] = intercept + np.dot(X_spearman[final_features], final_coefs)

    clinical_vars_final = ['ESR', 'Disease_Duration_Category', 'Rad_score']
    X_clinical = sm.add_constant(df_merged[clinical_vars_final])
    result = sm.Logit(y, X_clinical).fit(disp=False)

    prob_train = result.predict(X_clinical)
    fpr, tpr, thr = roc_curve(y, prob_train)
    roc_auc = auc(fpr, tpr)
    youden_idx = np.argmax(tpr - fpr)
    optimal_cutoff = thr[youden_idx]

    cutoff_dict = {'optimal_cutoff': float(optimal_cutoff)}
    joblib.dump(cutoff_dict, os.path.join(WORK_DIR, "optimal_cutoff.pkl"))

    print("\n🎉 RUN7 Step3 完成。")

if __name__ == "__main__":
    build_ultimate_pipeline()