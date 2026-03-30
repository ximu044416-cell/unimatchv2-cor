import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# 导入 simpleNomo
try:
    from simpleNomo import nomogram

    HAS_SIMPLENOMO = True
except ImportError:
    HAS_SIMPLENOMO = False
    print("⚠️ 提示: 未检测到 simpleNomo 包，将跳过列线图绘制。")


def build_radscore_and_nomogram():
    print("🚀 启动 Step 4 & 5: 降维三板斧与临床多因素建模引擎...\n")

    # ================= 1. 数据会师 =================
    WORK_DIR = r"F:\radiomics"
    CLINICAL_FILE = os.path.join(WORK_DIR, "clinical_info_train.xlsx")
    FEATURES_FILE = os.path.join(WORK_DIR, "Train_GT_Features.csv")

    df_clinical = pd.read_excel(CLINICAL_FILE)
    df_features = pd.read_csv(FEATURES_FILE)

    # 强行清除所有列名可能带有的隐藏空格
    df_clinical.columns = df_clinical.columns.str.strip()
    df_features.columns = df_features.columns.str.strip()

    # 确保 Patient_ID 格式一致且没有空格
    df_clinical['Patient_ID'] = df_clinical['Patient_ID'].astype(str).str.strip()
    df_features['Patient_ID'] = df_features['Patient_ID'].astype(str).str.strip()

    df_merged = pd.merge(df_features, df_clinical, on='Patient_ID', how='inner')
    print(f"📊 数据合并成功！共匹配到 {len(df_merged)} 名患者。")

    if 'Label' not in df_merged.columns:
        print(f"\n❌ 致命错误：在 Excel 中找不到名为 'Label' 的列！")
        return

    # 提取 y (标签)
    y = df_merged['Label'].values

    # 🔥 修复：弃用排除法，启用白名单法！
    # 真正的影像组学特征，绝对且只来源于 df_features (只需排除 Patient_ID 即可)
    # 这样就算你的 Excel 里有一百列汉字，也绝对不会污染 LASSO 引擎！
    feature_cols = [col for col in df_features.columns if col != 'Patient_ID']
    X_features = df_merged[feature_cols]

    print(f"💎 初始影像组学特征池维度: {X_features.shape[1]} 维")

    # ================= 2. 铸造 Z-score 冻结尺子 =================
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_features)
    X_scaled = pd.DataFrame(X_scaled_array, columns=feature_cols)

    joblib.dump(scaler, os.path.join(WORK_DIR, "train_scaler.pkl"))
    print("🔒 Z-score 尺子已铸造并冻结保存至: train_scaler.pkl")

    # ================= 3. 降维第一斧：Mann-Whitney U 检验 =================
    print("\n🪓 挥出第一斧：单变量统计学检验 (P < 0.05)...")
    mwu_selected = []
    for col in feature_cols:
        group0 = X_scaled.loc[y == 0, col]
        group1 = X_scaled.loc[y == 1, col]
        stat, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
        if p_val < 0.05:
            mwu_selected.append(col)

    X_mwu = X_scaled[mwu_selected]
    print(f"   => 第一斧砍完，剩余极具潜力的特征: {len(mwu_selected)} 个")

    # ================= 4. 降维第二斧：Spearman 共线性剔除 (红线 0.9) =================
    print(f"🪓 挥出第二斧：Spearman 相关性剔除 (阈值 r > 0.9)...")
    corr_matrix = X_mwu.corr(method='spearman').abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

    spearman_selected = [col for col in mwu_selected if col not in to_drop]
    X_spearman = X_mwu[spearman_selected]
    print(f"   => 第二斧砍完，去伪存真后剩余特征: {len(spearman_selected)} 个")

    # ================= 5. 降维第三斧：LASSO (带断崖缓冲机制) =================
    print("\n🪓 挥出第三斧：10-Fold LASSO 交叉验证...")
    alphas = np.logspace(-3, 1, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=10, random_state=42, max_iter=10000)
    lasso_cv.fit(X_spearman, y)

    coefs = lasso_cv.coef_
    optimal_alpha = lasso_cv.alpha_
    selected_mask = coefs != 0
    final_features = np.array(spearman_selected)[selected_mask]
    final_coefs = coefs[selected_mask]
    intercept = lasso_cv.intercept_

    # 🛡️ 导师级防御：修复“断崖式归零”漏洞
    if len(final_features) > 5:
        print(f"   ⚠️ LASSO 选出了 {len(final_features)} 个特征，超出了 EPV 承载极限！")
        print("   => 正在启动强行压缩机制 (带断崖缓冲)...")

        # 记录上一轮的有效特征，以防下一次循环直接归零
        last_valid_features = final_features
        last_valid_coefs = final_coefs
        last_valid_intercept = intercept

        for a in np.logspace(np.log10(optimal_alpha), np.log10(optimal_alpha) + 2, 200):
            hard_lasso = Lasso(alpha=a, random_state=42, max_iter=10000)
            hard_lasso.fit(X_spearman, y)
            num_nonzero = np.sum(hard_lasso.coef_ != 0)

            # 🔥 放宽边界到 6 或 7，并且防止断崖归零
            if 0 < num_nonzero <= 7:
                coefs = hard_lasso.coef_
                selected_mask = coefs != 0
                final_features = np.array(spearman_selected)[selected_mask]
                final_coefs = coefs[selected_mask]
                intercept = hard_lasso.intercept_
                break
            elif num_nonzero == 0:
                # 触发断崖，直接使用断崖前保存的最后一套有效特征（即使稍微多了一两个）
                final_features = last_valid_features
                final_coefs = last_valid_coefs
                intercept = last_valid_intercept
                print("   !! 触发断崖式归零，已启用最后一次有效压缩边界。")
                break
            else:
                # 还没压够，保存当前状态，继续加大惩罚
                last_valid_features = np.array(spearman_selected)[hard_lasso.coef_ != 0]
                last_valid_coefs = hard_lasso.coef_[hard_lasso.coef_ != 0]
                last_valid_intercept = hard_lasso.intercept_

    print(f"   => 第三斧绝杀！最终锁定 {len(final_features)} 个黄金影像组学特征！")
    joblib.dump(list(final_features), os.path.join(WORK_DIR, "selected_features.pkl"))

    # ================= 6. Rad-score 构建与展示 =================
    print("\n🔬 影像组学分数 (Rad-score) 公式已生成：")
    formula = f"Rad-score = {intercept:.4f} "
    for feat, coef in zip(final_features, final_coefs):
        sign = "+" if coef > 0 else "-"
        formula += f"\n          {sign} {abs(coef):.4f} * [{feat}]"
    print(formula)

    # 计算 Rad-score
    df_merged['Rad_score'] = intercept + np.dot(X_spearman[final_features], final_coefs)

    # ================= 7. 临床多因素 Logistic Regression =================
    print("\n📈 训练多因素 Logistic Regression 模型 (Age + Gender + Rad-score)...")
    X_clinical = df_merged[['Age', 'Gender', 'Rad_score']]
    X_clinical = sm.add_constant(X_clinical)

    log_reg = sm.Logit(y, X_clinical)
    result = log_reg.fit(disp=False)
    print(result.summary())

    # 预测并画 ROC
    y_pred_prob = result.predict(X_clinical)
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Train AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curve - Clinical & Radiomics Nomogram Model', fontweight='bold')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(WORK_DIR, "Train_ROC_Curve.png"))
    plt.close()  # 🔥 修复：清空画布，防止与 simpleNomo 发生灾难性串图交叉！
    print(f"\n🎨 ROC 曲线已保存至: Train_ROC_Curve.png (AUC: {roc_auc:.3f})")

    # ================= 8. 绘制 Nomogram (列线图) =================
    if HAS_SIMPLENOMO:
        try:
            print("\n🎨 正在调用 simpleNomo 绘制顶刊列线图...")
            nomo_df = df_merged[['Label', 'Age', 'Gender', 'Rad_score']].copy()
            nomogram_plot = nomogram(
                data=nomo_df,
                y_name='Label',
                x_names=['Age', 'Gender', 'Rad_score'],
            )
            nomogram_plot.draw()
            nomogram_plot.save(os.path.join(WORK_DIR, "Nomogram.png"))
            print("🎉 列线图绘制成功！已保存至 Nomogram.png")
        except Exception as e:
            print(f"⚠️ simpleNomo 绘制过程出现异常: {e}")


if __name__ == "__main__":
    build_radscore_and_nomogram()