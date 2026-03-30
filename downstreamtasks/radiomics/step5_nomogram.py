import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings('ignore')


def draw_beautiful_nomogram():
    print("🎨 启动独立的 Nomogram (列线图) 绘制引擎...")

    WORK_DIR = r"F:\radiomics"
    CLINICAL_FILE = os.path.join(WORK_DIR, "clinical_info_train.xlsx")
    FEATURES_FILE = os.path.join(WORK_DIR, "Train_GT_Features.csv")

    # ---------------- 1. 极速复刻数据与 Rad-score ----------------
    print("   -> 正在读取并组装数据...")
    df_clinical = pd.read_excel(CLINICAL_FILE)
    df_features = pd.read_csv(FEATURES_FILE)

    df_clinical.columns = df_clinical.columns.str.strip()
    df_features.columns = df_features.columns.str.strip()
    df_clinical['Patient_ID'] = df_clinical['Patient_ID'].astype(str).str.strip()
    df_features['Patient_ID'] = df_features['Patient_ID'].astype(str).str.strip()

    df_merged = pd.merge(df_features, df_clinical, on='Patient_ID', how='inner')
    y = df_merged['Label'].values

    feature_cols = [col for col in df_features.columns if col != 'Patient_ID']
    X_features = df_merged[feature_cols]

    # 用之前的逻辑还原 Z-score 和 LASSO
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=feature_cols)

    # 直接使用之前锁定的 7 个特征（为了极速，这里写死上次提取出的名单）
    # 如果以后变了，可以改成动态读取
    final_features = [
        'original_shape_Sphericity',
        'original_shape_SurfaceVolumeRatio',
        'wavelet-LLH_glcm_MCC',
        'wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis',
        'wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis',
        'wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis',
        'wavelet-HHH_firstorder_Maximum'
    ]

    # 复现回归系数 (极其快速)
    lasso = LassoCV(alphas=np.logspace(-3, 1, 100), cv=10, random_state=42, max_iter=10000)
    lasso.fit(X_scaled[final_features], y)
    df_merged['Rad_score'] = lasso.intercept_ + np.dot(X_scaled[final_features], lasso.coef_)

    # ---------------- 2. 拟合 Logistic Regression ----------------
    print("   -> 正在拟合最终的临床多因素模型...")
    X_clinical = df_merged[['Age', 'Gender', 'Rad_score']]
    X_clinical = sm.add_constant(X_clinical)
    result = sm.Logit(y, X_clinical).fit(disp=False)

    # ---------------- 3. 纯手工顶级列线图绘制引擎 ----------------
    print("   -> 正在启动【纯手工 Matplotlib 列线图绘制引擎】...")

    params = result.params
    features_to_plot = ['Age', 'Gender', 'Rad_score']
    scores = {}
    max_score_range = 0

    for feat in features_to_plot:
        coef = params[feat]
        f_min = df_merged[feat].min()
        f_max = df_merged[feat].max()

        # 提取用于计算打分的极值
        val_range = np.linspace(f_min, f_max, 5) if len(np.unique(df_merged[feat])) > 2 else np.array([f_min, f_max])
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
    scale = 100.0 / max_score_range  # 设定贡献最大的特征为 100 分满分

    # 创建高清画布
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    y_pos = len(features_to_plot) + 2

    # --- 画最顶部的打分尺 (Points) ---
    ax.hlines(y_pos, 0, 100, color='black', linewidth=1.5)
    for p in range(0, 101, 10):
        ax.vlines(p, y_pos, y_pos + 0.15, color='black')
        ax.text(p, y_pos + 0.3, str(p), ha='center', va='bottom', fontsize=10)
    ax.text(-5, y_pos, "Points", ha='right', va='center', fontsize=12, fontweight='bold')

    max_total_points = 0
    # --- 画每个特征的刻度 ---
    for idx, feat in enumerate(features_to_plot):
        y = len(features_to_plot) - idx
        s = scores[feat]
        pts_range = s['range'] * scale
        max_total_points += pts_range
        ax.hlines(y, 0, pts_range, color='black', linewidth=1.2)

        ticks = np.linspace(s['min_val'], s['max_val'], 5) if len(np.unique(df_merged[feat])) > 2 else np.unique(
            df_merged[feat])
        for t in ticks:
            pts = (s['coef'] * t - s['min_contrib']) * scale
            ax.vlines(pts, y, y + 0.15, color='black')
            label = f"{t:.2f}" if isinstance(t, float) and len(np.unique(df_merged[feat])) > 2 else str(int(t))
            ax.text(pts, y + 0.3, label, ha='center', va='bottom', fontsize=9)
        ax.text(-5, y, feat, ha='right', va='center', fontsize=12, fontweight='bold')

    # --- 画总分尺 (Total Points) ---
    y_total = -1
    max_total_points = min(max_total_points, 300)
    ax.hlines(y_total, 0, max_total_points, color='black', linewidth=1.5)
    step = max(10, int(max_total_points) // 10)
    for p in range(0, int(max_total_points) + 1, step):
        ax.vlines(p, y_total, y_total + 0.15, color='black')
        ax.text(p, y_total + 0.3, str(p), ha='center', va='bottom', fontsize=10)
    ax.text(-5, y_total, "Total Points", ha='right', va='center', fontsize=12, fontweight='bold')

    # --- 画风险概率尺 (Risk Probability) ---
    y_risk = -2.5
    # 典型的严重程度概率梯队
    risk_probs = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    intercept = params['const']
    sum_min_contrib = sum([scores[f]['min_contrib'] for f in features_to_plot])

    ax.hlines(y_risk, 0, max_total_points, color='black', linewidth=1.5)
    for r in risk_probs:
        # Logistic 反函数的数学推导：从概率反推总得分
        pts = (np.log(r / (1 - r)) - intercept - sum_min_contrib) * scale
        if 0 <= pts <= max_total_points:
            ax.vlines(pts, y_risk, y_risk + 0.15, color='black')
            ax.text(pts, y_risk + 0.3, str(r), ha='center', va='bottom', fontsize=10)
    ax.text(-5, y_risk, "Risk Probability", ha='right', va='center', fontsize=12, fontweight='bold')

    ax.set_xlim(-25, max_total_points + 10)
    ax.set_ylim(-3.5, y_pos + 1.5)
    ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(WORK_DIR, "Nomogram_Final.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"🎉 成功！一张绝美的顶刊级 Nomogram 列线图已生成！")
    print(f"📁 请前往查看: {out_path}")


if __name__ == "__main__":
    draw_beautiful_nomogram()