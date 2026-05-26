import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate_3d_dice(pred_np, gt_np):
    intersection = np.logical_and(pred_np == 1, gt_np == 1).sum()
    sum_masks = (pred_np == 1).sum() + (gt_np == 1).sum()
    if sum_masks == 0:
        return 1.0
    return 2.0 * intersection / sum_masks


def find_best_threshold():
    print("🚀 启动 3D Dice 全自动寻优雷达 (Threshold Optimizer)...\n")

    # 🔥 核心修改：读取 Step 1 生成的 "阈值" 文件夹
    BASE_DIR = r"F:\cor\RUN7\Thresholds"
    splits = ["train", "test"]
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    results_dict = {t: [] for t in thresholds}
    volume_dict = {t: [] for t in thresholds}
    gt_volumes = []

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

            gt_sitk = sitk.ReadImage(gt_path)
            gt_np = sitk.GetArrayFromImage(gt_sitk)
            gt_volumes.append((gt_np == 1).sum())

            total_patients += 1

            for t in thresholds:
                t_str = f"{int(t * 100):03d}"
                pred_path = os.path.join(patient_dir, f"{patient_id}_Pred_{t_str}.nii.gz")

                if not os.path.exists(pred_path):
                    results_dict[t].append(0.0)
                    volume_dict[t].append(0)
                    continue

                pred_sitk = sitk.ReadImage(pred_path)
                pred_np = sitk.GetArrayFromImage(pred_sitk)

                dice = calculate_3d_dice(pred_np, gt_np)
                vol = (pred_np == 1).sum()

                results_dict[t].append(dice)
                volume_dict[t].append(vol)

    if total_patients == 0:
        print("❌ 未在目录中找到任何有效患者，请检查路径！")
        return

    print(f"\n📊 阅卷完毕！共批改 {total_patients} 名患者试卷。成绩汇总如下：")
    print("-" * 65)
    print(f"{'阈值 (Threshold)':<18} | {'平均 Dice':<12} | {'平均预测体积':<15} | {'体积相对误差'}")
    print("-" * 65)

    avg_gt_vol = np.mean(gt_volumes)
    avg_dices = []

    for t in thresholds:
        mean_dice = np.mean(results_dict[t])
        mean_vol = np.mean(volume_dict[t])
        rve = (mean_vol - avg_gt_vol) / (avg_gt_vol + 1e-5) * 100
        avg_dices.append(mean_dice)
        vol_trend = f"({'+' if rve > 0 else ''}{rve:.1f}%)"
        print(f"Prob > {t:.2f} ({int(t * 100):03d})  | {mean_dice:.4f}       | {mean_vol:^11.1f}   | {vol_trend}")

    print("-" * 65)
    print(f"✅ 医生金标准 (GT) 平均真实体积: {avg_gt_vol:.1f}")

    best_idx = np.argmax(avg_dices)
    best_threshold = thresholds[best_idx]
    best_dice = avg_dices[best_idx]

    print(f"\n🏆 ====== 终极判决 ======")
    print(f"🥇 最佳黄金阈值: {best_threshold:.2f}")
    print(f"📈 最高平均 3D Dice: {best_dice:.4f}")
    print(f"👉 下一步指令: 请在 F:\\cor\\RUN7\\Thresholds 中使用 _Pred_{int(best_threshold * 100):03d}.nii.gz 进行组学提取！")

    # 绘制顶刊图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(thresholds, avg_dices, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=8)
    plt.plot(best_threshold, best_dice, marker='*', color='#e74c3c', markersize=15,
             label=f'Optimal Cutoff: {best_threshold:.2f} (Dice: {best_dice:.3f})')
    plt.title('Threshold Optimization via 3D Dice Score (Run 7)', fontweight='bold', fontsize=14)
    plt.xlabel('Probability Threshold', fontweight='bold', fontsize=12)
    plt.ylabel('Mean 3D Dice Coefficient', fontweight='bold', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower center", fontsize=11)

    save_path = os.path.join(BASE_DIR, "Threshold_Optimization_Curve_Run7.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n🎨 寻优曲线图已保存至: {save_path}")


if __name__ == "__main__":
    find_best_threshold()