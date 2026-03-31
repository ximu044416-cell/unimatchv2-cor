import os
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from torch.utils.data import DataLoader

# 导入你的核心配置与模型
from configs import config
from models.dinov2_unet import DINOUNet
from data.dataset import UniMatchDataset


def reconstruct_3d_volumes_greedy_strict():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # ================= 1. 核心参数与路径准备 =================
    WEIGHT_PATH = os.path.join(config.OUTPUT_DIR, "best_model.pth")

    # 新的输出阵地
    OUTPUT_BASE_DIR = r"F:\check"
    # 🔥 绝对金标准：你之前手工划分的阵地
    REFERENCE_DIR = r"F:\radiomics"

    os.makedirs(os.path.join(OUTPUT_BASE_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, "test"), exist_ok=True)

    # 网格化阈值 (0.50 ~ 0.95, 间隔 0.05)
    thresholds = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75]

    print(f"🚀 启动 3D 贪婪多阈值重建引擎 (严格镜像模式)...")

    # ================= 2. 🔥 建立绝对隔离花名册 (Roster) =================
    # 直接读取你之前手工划分的文件夹，获取病人 ID 列表
    train_ref_path = os.path.join(REFERENCE_DIR, "train")
    test_ref_path = os.path.join(REFERENCE_DIR, "test")

    if not os.path.exists(train_ref_path) or not os.path.exists(test_ref_path):
        raise FileNotFoundError(f"❌ 致命错误: 找不到参考目录 {train_ref_path} 或 {test_ref_path}，无法执行镜像匹配！")

    train_roster = set(os.listdir(train_ref_path))
    test_roster = set(os.listdir(test_ref_path))

    print(
        f"📋 查阅历史档案: 在 F:\\radiomics 中发现 {len(train_roster)} 名 Train 患者, {len(test_roster)} 名 Test 患者。")

    # ================= 3. 实例化神盾基座 =================
    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device), strict=True)
    model.eval()

    # ================= 4. 搜刮原始切片并按花名册过滤 =================
    target_dirs = [Path(config.TRAIN_DIR), Path(config.UNLABEL_DIR), Path(config.VAL_DIR)]
    patient_dict = {}

    for split_dir in target_dirs:
        if not split_dir.exists():
            continue
        for case_dir in split_dir.iterdir():
            if case_dir.is_dir() and not case_dir.name.startswith('.'):
                patient_id = case_dir.name
                # 🔥 严酷的门卫逻辑：不在花名册里的病人，直接拒之门外！
                if patient_id not in train_roster and patient_id not in test_roster:
                    continue

                slices = sorted([p.absolute() for p in case_dir.glob("*_data.npy")])
                if len(slices) > 0:
                    patient_dict[patient_id] = slices

    print(f"📊 搜刮与过滤完毕！共准备对 {len(patient_dict)} 名在册患者进行降维打击...\n")

    # ================= 5. 逐患者进行 3D 拼装推理 =================
    with torch.no_grad():
        for patient_id, slice_list in tqdm(patient_dict.items(), desc="全队列重建总进度"):

            # 🔥 严格对号入座
            if patient_id in train_roster:
                split_folder = "train"
            else:
                split_folder = "test"

            patient_output_dir = os.path.join(OUTPUT_BASE_DIR, split_folder, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)

            patient_dataset = UniMatchDataset(slice_list, mode='val')
            patient_loader = DataLoader(patient_dataset, batch_size=1, shuffle=False, num_workers=0)

            patient_images = []
            patient_gts = []
            patient_preds_dict = {t: [] for t in thresholds}

            for (data_tuple, slice_path_str) in zip(patient_loader, slice_list):
                img_tensor, _ = data_tuple

                orig_img = np.load(slice_path_str)
                orig_h, orig_w = orig_img.shape[:2]
                patient_images.append(orig_img[..., 0].astype(np.float32))

                tensor_h, tensor_w = img_tensor.shape[-2:]
                img_tensor = img_tensor.to(device)

                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    img_tensor = F.interpolate(img_tensor, size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear',
                                               align_corners=False)

                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1:2, :, :]

                # 仅对连续概率图进行插值还原
                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    probs = F.interpolate(probs, size=(tensor_h, tensor_w), mode='bilinear', align_corners=False)

                prob_map_np = probs.cpu().numpy().astype(np.float32)[0, 0]

                # 中心裁剪还原
                pad_h = max(0, config.IMG_SIZE - orig_h)
                pad_w = max(0, config.IMG_SIZE - orig_w)
                if pad_h > 0 or pad_w > 0:
                    pad_top = pad_h // 2 if pad_h > 0 else 0
                    pad_left = pad_w // 2 if pad_w > 0 else 0
                    prob_map_np = prob_map_np[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]

                # 批量生成多阈值掩码 (矩阵运算，极速)
                for t in thresholds:
                    mask_t = (prob_map_np > t).astype(np.uint8)
                    patient_preds_dict[t].append(mask_t)

                label_path = str(slice_path_str).replace('_data.npy', '_label.npy')
                if os.path.exists(label_path):
                    gt_mask_np = np.load(label_path).astype(np.uint8)
                else:
                    gt_mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
                patient_gts.append(gt_mask_np)

            # Z 轴堆叠并保存
            vol_img = np.stack(patient_images, axis=0)
            vol_gt = np.stack(patient_gts, axis=0)

            sitk_img = sitk.GetImageFromArray(vol_img)
            sitk_gt = sitk.GetImageFromArray(vol_gt)
            sitk_img.SetSpacing((1.0, 1.0, 1.0))
            sitk_gt.SetSpacing((1.0, 1.0, 1.0))

            sitk.WriteImage(sitk_img, os.path.join(patient_output_dir, f"{patient_id}_Image.nii.gz"))
            sitk.WriteImage(sitk_gt, os.path.join(patient_output_dir, f"{patient_id}_GT.nii.gz"))

            for t in thresholds:
                vol_pred_t = np.stack(patient_preds_dict[t], axis=0)
                sitk_pred_t = sitk.GetImageFromArray(vol_pred_t)
                sitk_pred_t.SetSpacing((1.0, 1.0, 1.0))

                t_str = f"{int(t * 100):03d}"
                pred_file = os.path.join(patient_output_dir, f"{patient_id}_Pred_{t_str}.nii.gz")
                sitk.WriteImage(sitk_pred_t, pred_file)

    print(f"\n🎉 惊人的成就！严格镜像引擎执行完毕！")
    print(f"📁 弹药库已封存至: {OUTPUT_BASE_DIR}")
    print(f"👉 划分与 F:\\radiomics 完全一致，零数据泄露风险！")


if __name__ == "__main__":
    reconstruct_3d_volumes_greedy_strict()