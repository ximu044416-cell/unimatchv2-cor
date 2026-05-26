import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.preprocessing import StandardScaler


def draw_real_figure_s1():
    print("⏳ 正在定位文件路径...")

    # 自动获取当前 1.py 所在的绝对路径文件夹
    current_dir = Path(__file__).parent.resolve()

    # 告诉代码去子文件夹里找文件
    feat_path = current_dir / "features_dev164" / "GT_Features_Dev164.csv"
    label_path = current_dir / "models_dev164" / "Dev164_GT_Radscore_Table.csv"

    # 检查文件到底在不在
    if not feat_path.exists():
        print(f"❌ 找不到特征文件，请确保它在这个路径下：\n{feat_path}")
        return
    if not label_path.exists():
        print(f"❌ 找不到标签文件，请确保它在这个路径下：\n{label_path}")
        return

    print("✅ 找到文件！正在读取真实数据...")
    df_feat = pd.read_csv(feat_path)
    df_label = pd.read_csv(label_path)

    # 按 Patient_ID 对齐，确保特征和标签一一对应
    df_merged = pd.merge(df_feat, df_label[['Patient_ID', 'Label']], on='Patient_ID', how='inner')

    y = df_merged['Label'].values
    feat_cols = [c for c in df_feat.columns if c != 'Patient_ID']

    # 【修复重点】：将组学特征里提取失败的缺失值 (NaN) 填补为 0.0，防止 LASSO 报错！
    X_raw = df_merged[feat_cols].fillna(0.0)

    # 标准化
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=feat_cols)

    # 你的核心 7 个特征 (严格对应 Supplementary Table S1)
    core_features = [
        "original_shape_SurfaceVolumeRatio",
        "wavelet-LLH_glcm_MCC",
        "original_shape_Sphericity",
        "wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis",
        "wavelet-HHH_firstorder_Maximum",
        "wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis",
        "wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis"
    ]

    # 保留核心特征，并随机抽取 80 个其他特征作为背景轨迹 (避免图太乱)
    other_features = [c for c in feat_cols if c not in core_features]
    np.random.seed(42)
    sampled_others = np.random.choice(other_features, size=80, replace=False).tolist()
    plot_features = core_features + sampled_others
    X_plot = X_scaled[plot_features]

    print("⏳ 正在基于真实数据计算 LASSO 路径与交叉验证误差...")
    # 运行真实的 10 折交叉验证 LASSO
    lasso_cv = LassoCV(cv=10, random_state=42, max_iter=10000)
    lasso_cv.fit(X_plot, y)

    alphas = lasso_cv.alphas_
    log_alphas = np.log10(alphas)

    # 提取真实的 MSE 误差曲线数据
    mse_path = lasso_cv.mse_path_
    mean_mse = mse_path.mean(axis=1)
    std_mse = mse_path.std(axis=1) / np.sqrt(10)

    # 计算客观准则点
    cv_optimal_alpha = lasso_cv.alpha_
    cv_opt_idx = np.where(alphas == cv_optimal_alpha)[0][0]

    target_mse = mean_mse[cv_opt_idx] + std_mse[cv_opt_idx]
    valid_idx = np.where(mean_mse <= target_mse)[0]
    one_se_idx = valid_idx.min()
    one_se_alpha = alphas[one_se_idx]

    log_cv_opt = np.log10(cv_optimal_alpha)
    log_one_se = np.log10(one_se_alpha)

    # ================= 开始画图 =================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    # ------------------------------------------------
    # Panel A: 真实 LASSO 轨迹
    # ------------------------------------------------
    ax1 = axes[0]
    _, coefs_lasso, _ = lasso_path(X_plot, y, alphas=alphas, max_iter=10000)
    colors = plt.cm.tab20(np.linspace(0, 1, coefs_lasso.shape[0]))
    for i in range(coefs_lasso.shape[0]):
        # 核心特征画深色一点，背景特征透明度调低
        alpha_val = 0.85 if plot_features[i] in core_features else 0.3
        ax1.plot(log_alphas, coefs_lasso[i], color=colors[i], alpha=alpha_val)

    ax1.axvline(x=log_cv_opt, color='gray', linestyle='--', label='CV-optimal alpha')
    ax1.axvline(x=log_one_se, color='darkred', linestyle=':', label='1-SE alpha')
    ax1.set_xlabel('Log10(Alpha)', fontsize=12)
    ax1.set_ylabel('LASSO Coefficients', fontsize=12)
    ax1.set_title('A. LASSO coefficient path', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=10)

    # ------------------------------------------------
    # Panel B: 真实 10-fold CV Error
    # ------------------------------------------------
    ax2 = axes[1]
    ax2.plot(log_alphas, mean_mse, color='#1f77b4', linewidth=2)
    ax2.fill_between(log_alphas, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2, color='#1f77b4')
    ax2.axvline(x=log_cv_opt, color='gray', linestyle='--', label='CV-optimal alpha')
    ax2.axvline(x=log_one_se, color='darkred', linestyle=':', label='1-SE alpha')
    ax2.set_xlabel('Log10(Alpha)', fontsize=12)
    ax2.set_ylabel('Mean CV Error', fontsize=12)
    ax2.set_title('B. 10-fold CV error curve', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper center', fontsize=10)

    # ------------------------------------------------
    # Panel C: 你精确调整过的 7 个特征及系数
    # ------------------------------------------------
    ax3 = axes[2]
    features_s1 = [
        "original_shape_SurfaceVolumeRatio",
        "wavelet-LLH_glcm_MCC",
        "original_shape_Sphericity",
        "wavelet-LHL_gldm_LargeDependenceHighGrayLevelEmphasis",
        "wavelet-HHH_firstorder_Maximum",
        "wavelet-HLL_gldm_LargeDependenceHighGrayLevelEmphasis",
        "wavelet-LLH_glszm_LargeAreaHighGrayLevelEmphasis"
    ]
    coefs_s1 = [-0.11147, -0.11224, -0.03423, 0.00518, 0.01272, 0.01515, 0.01892]

    sorted_indices = np.argsort(coefs_s1)
    features_sorted = [features_s1[i] for i in sorted_indices]
    coefs_sorted = [coefs_s1[i] for i in sorted_indices]

    bar_colors = ['#d62728' if c > 0 else '#1f77b4' for c in coefs_sorted]
    bars = ax3.barh(features_sorted, coefs_sorted, color=bar_colors, alpha=0.85)
    ax3.axvline(x=0, color='black', linewidth=1)

    # 【排版修改点 1】：强行固定横坐标的范围，给左边的负数标签留出足够的空间
    ax3.set_xlim(-0.16, 0.04)

    for bar, coef in zip(bars, coefs_sorted):
        x_offset = 0.005 if coef > 0 else -0.005
        ha = 'left' if coef > 0 else 'right'
        ax3.text(coef + x_offset, bar.get_y() + bar.get_height() / 2, f'{coef:.4f}',
                 va='center', ha=ha, fontsize=10)

    # 【排版修改点 2】：缩小左边特征名字的字号，防止太拥挤
    ax3.tick_params(axis='y', labelsize=8)

    ax3.set_xlabel('LASSO Coefficient', fontsize=12)
    ax3.set_title('C. CV-optimal retained feature weights', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='x', linestyle='--', alpha=0.5)

    # 【排版修改点 3】：增加子图之间的间距 (w_pad) 避免文字穿透到中间的图
    plt.tight_layout(w_pad=2.0)

    # 图片也会保存在 1.py 同一个文件夹下
    out_name = current_dir / 'Real_Data_Supplementary_Figure_S1_LASSO.png'
    plt.savefig(out_name, bbox_inches='tight')
    print(f"✅ 成功基于真数据生成：\n{out_name}")


if __name__ == "__main__":
    draw_real_figure_s1()