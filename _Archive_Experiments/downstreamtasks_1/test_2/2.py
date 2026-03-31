import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def calculate_3d_dice(pred_np, gt_np):
    """严谨的 3D Dice 计算公式"""
    intersection = np.logical_and(pred_np == 1, gt_np == 1).sum()
    sum_masks = (pred_np == 1).sum() + (gt_np == 1).sum()
    if sum_masks == 0:
        return 1.0  # 完美避开分母为0的数学崩溃（两者都没水肿）
    return 2.0 * intersection / sum_masks


def find_best_threshold():
    print("🚀 启动 3D Dice 全自动寻优雷达 (Threshold Optimizer)...\n")

    BASE_DIR = r"F:\check"
    # 我们遍历 check 下的 train 和 test (通常寻优只看验证/训练集，但为了全局视角，我们全算)
    splits = ["train", "test"]
    thresholds = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75]

    # 记录每个阈值下的所有病人的 Dice
    results_dict = {t: [] for t in thresholds}
    volume_dict = {t: [] for t in thresholds}  # 记录预测出的平均体积
    gt_volumes = []  # 记录真实平均体积

    total_patients = 0

    for split in splits:
        split_dir = os.path.join(BASE_DIR, split)
        if not os.path.exists(split_dir):
            continue

        patients = [p for p in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, p))]

        for patient_id in tqdm(patients, desc=f"阅卷进度 ({split})"):
            patient_dir = os.path.join(split_dir, patient_id)
            gt_path = os.path.join(patient_dir, f"{patient_id}_GT.nii.gz")

            if not os.path.exists(gt_path):
                continue

            # 读取 GT 金标准
            gt_sitk = sitk.ReadImage(gt_path)
            gt_np = sitk.GetArrayFromImage(gt_sitk)
            gt_volumes.append((gt_np == 1).sum())

            total_patients += 1

            # 遍历该病人的所有阈值掩码
            for t in thresholds:
                t_str = f"{int(t * 100):03d}"
                pred_path = os.path.join(patient_dir, f"{patient_id}_Pred_{t_str}.nii.gz")

                if not os.path.exists(pred_path):
                    results_dict[t].append(0.0)
                    volume_dict[t].append(0)
                    continue

                pred_sitk = sitk.ReadImage(pred_path)
                pred_np = sitk.GetArrayFromImage(pred_sitk)

                # 计算 Dice 和体积
                dice = calculate_3d_dice(pred_np, gt_np)
                vol = (pred_np == 1).sum()

                results_dict[t].append(dice)
                volume_dict[t].append(vol)

    if total_patients == 0:
        print("❌ 未在目录中找到任何有效患者，请检查路径！")
        return

    # ================= 成绩统计与播报 =================
    print(f"\n📊 阅卷完毕！共批改 {total_patients} 名患者试卷。成绩汇总如下：")
    print("-" * 65)
    print(f"{'阈值 (Threshold)':<18} | {'平均 Dice':<12} | {'平均预测体积':<15} | {'体积相对误差'}")
    print("-" * 65)

    avg_gt_vol = np.mean(gt_volumes)

    avg_dices = []
    for t in thresholds:
        mean_dice = np.mean(results_dict[t])
        mean_vol = np.mean(volume_dict[t])
        rve = (mean_vol - avg_gt_vol) / (avg_gt_vol + 1e-5) * 100  # 相对体积误差
        avg_dices.append(mean_dice)

        # 标出当前的体积比 GT 大还是小
        vol_trend = f"({'+' if rve > 0 else ''}{rve:.1f}%)"

        print(f"Prob > {t:.2f} ({int(t * 100):03d})  | {mean_dice:.4f}       | {mean_vol:^11.1f}   | {vol_trend}")

    print("-" * 65)
    print(f"✅ 医生金标准 (GT) 平均真实体积: {avg_gt_vol:.1f}")

    # 找到冠军阈值
    best_idx = np.argmax(avg_dices)
    best_threshold = thresholds[best_idx]
    best_dice = avg_dices[best_idx]
    print(f"\n🏆 ====== 终极判决 ======")
    print(f"🥇 最佳黄金阈值: {best_threshold:.2f}")
    print(f"📈 最高平均 Dice: {best_dice:.4f}")
    print(f"👉 下一步指令: 请使用 _Pred_{int(best_threshold * 100):03d}.nii.gz 重新提取 PyRadiomics 组学特征！")

    # ================= 绘制顶刊寻优曲线图 =================
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(thresholds, avg_dices, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=8)

    # 标出最高点
    plt.plot(best_threshold, best_dice, marker='*', color='#e74c3c', markersize=15,
             label=f'Optimal Cutoff: {best_threshold:.2f} (Dice: {best_dice:.3f})')

    plt.title('Threshold Optimization via 3D Dice Score', fontweight='bold', fontsize=14)
    plt.xlabel('Probability Threshold', fontweight='bold', fontsize=12)
    plt.ylabel('Mean 3D Dice Coefficient', fontweight='bold', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower center", fontsize=11)

    save_path = os.path.join(BASE_DIR, "Threshold_Optimization_Curve.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n🎨 寻优曲线图已保存至: {save_path}")


if __name__ == "__main__":
    find_best_threshold()