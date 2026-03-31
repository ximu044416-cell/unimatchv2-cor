import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(gt_array, pred_array):
    """计算 3D 级别的高精度 Dice 和相对体积误差 (RVE)"""
    # 🔥 终极防爆补丁：强制转化为大于 0 的布尔矩阵，并转为 float64 防止加法溢出！
    gt_b = (gt_array > 0).astype(np.float64)
    pred_b = (pred_array > 0).astype(np.float64)

    intersection = np.sum(gt_b * pred_b)
    v_gt = np.sum(gt_b)
    v_pred = np.sum(pred_b)

    # 防止分母为0
    if v_gt + v_pred == 0:
        dice = 1.0
    else:
        dice = (2. * intersection) / (v_gt + v_pred)

    rve = abs(v_pred - v_gt) / v_gt if v_gt > 0 else 0.0
    return dice, v_gt, v_pred, rve


def evaluate_thresholds():
    # ================= 1. 绝对防御：只在 Train 组赛马 =================
    TRAIN_DIR = r"F:\check\train"  # 你的训练集 140 人阵地

    thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    print("🏁 启动 3D 分割最佳阈值赛马场 (仅在 Train 集进行，绝对防泄露)...")

    patient_dirs = [os.path.join(TRAIN_DIR, d) for d in os.listdir(TRAIN_DIR) if
                    os.path.isdir(os.path.join(TRAIN_DIR, d))]
    print(f"📊 参赛选手就位：共计 {len(patient_dirs)} 名在册 Train 患者。")

    # 结果收集器
    results = {t: {'dice': [], 'rve': []} for t in thresholds}

    # ================= 2. 逐患者比对 =================
    for p_dir in tqdm(patient_dirs, desc="阈值大乱斗进度"):
        p_id = os.path.basename(p_dir)
        gt_path = os.path.join(p_dir, f"{p_id}_GT.nii.gz")

        if not os.path.exists(gt_path):
            continue

        gt_img = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(gt_img)

        # 排除掉医生都没画水肿的极品罕见病例，以免干扰 Dice 平均值
        if np.sum(gt_array) == 0:
            continue

        for t in thresholds:
            t_str = f"{t:03d}"
            pred_path = os.path.join(p_dir, f"{p_id}_Pred_{t_str}.nii.gz")

            if not os.path.exists(pred_path):
                continue

            pred_img = sitk.ReadImage(pred_path)
            pred_array = sitk.GetArrayFromImage(pred_img)

            dice, _, _, rve = calculate_metrics(gt_array, pred_array)
            results[t]['dice'].append(dice)
            results[t]['rve'].append(rve)

    # ================= 3. 统计颁奖 =================
    print("\n🏆 ====== 终极战报：最佳阈值排行榜 ======")
    print(f"{'阈值':<8} | {'平均 Dice ↑':<12} | {'平均 RVE ↓ (越小体积越准)':<15}")
    print("-" * 45)

    best_dice = 0
    best_t = 0

    for t in thresholds:
        mean_dice = np.mean(results[t]['dice']) if len(results[t]['dice']) > 0 else 0
        mean_rve = np.mean(results[t]['rve']) if len(results[t]['rve']) > 0 else 0

        # 加个小星星标出冠军
        star = "⭐" if mean_dice > best_dice else ""
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_t = t

        print(f"0.{t:<6d} | {mean_dice:.4f}       | {mean_rve:.4f} {star}")

    print("-" * 45)
    print(f"🎉 绝对裁决：最佳切割阈值是 0.{best_t} (Dice: {best_dice:.4f})")
    print("👉 下一步指示：请在你的脑海中、以及后续提取组学特征的代码里，死死锁定这个阈值对应的 _Pred.nii.gz！")


if __name__ == "__main__":
    evaluate_thresholds()