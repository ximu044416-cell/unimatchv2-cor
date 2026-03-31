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


def reconstruct_3d_volumes():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # ================= 1. 核心参数与路径准备 =================
    # 坚决使用巅峰权重
    WEIGHT_PATH = os.path.join(config.OUTPUT_DIR, "best_model.pth")

    # 🔥 根据你的要求，将输出路径改为 F:\Dinov2_data\3D_Predictions
    OUTPUT_3D_DIR = r"F:\Dinov2_data\3D_Predictions"
    os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

    THRESHOLD = 0.50

    print(f"🚀 启动 3D 临床视角重建与 ROI 级对齐引擎...")
    print(f"📥 正在加载终极 SOTA 权重: {os.path.basename(WEIGHT_PATH)}")
    print(f"📂 输出目录设定为: {OUTPUT_3D_DIR}")

    # ================= 2. 实例化神盾基座 =================
    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device), strict=True)
    model.eval()

    # ================= 3. 数据集分组与物流调度 =================
    val_dir_path = Path(config.VAL_DIR)
    patient_dict = {}

    for case_dir in val_dir_path.iterdir():
        if case_dir.is_dir() and not case_dir.name.startswith('.'):
            patient_id = case_dir.name
            slices = sorted([p.absolute() for p in case_dir.glob("*_data.npy")])
            if len(slices) > 0:
                patient_dict[patient_id] = slices

    print(f"📊 在测试集中共发现 {len(patient_dict)} 名独立患者。准备进行像素级重建...\n")

    # ================= 4. 逐患者进行 3D 拼装推理 =================
    with torch.no_grad():
        for patient_id, slice_list in patient_dict.items():
            print(f"🩺 正在诊断患者: [{patient_id}] (包含 {len(slice_list)} 张切片)")

            patient_dataset = UniMatchDataset(slice_list, mode='val')
            patient_loader = DataLoader(patient_dataset, batch_size=1, shuffle=False, num_workers=0)

            patient_images = []
            patient_preds = []
            patient_gts = []

            for (data_tuple, slice_path_str) in tqdm(zip(patient_loader, slice_list), total=len(slice_list),
                                                     desc=f"   - 推理与坐标还原", leave=False):
                img_tensor, _ = data_tuple

                # 1. 从硬盘直接读取原图
                orig_img = np.load(slice_path_str)
                orig_h, orig_w = orig_img.shape[:2]

                patient_images.append(orig_img[..., 0].astype(np.float32))

                # 2. 记录进入网络前的张量尺寸
                tensor_h, tensor_w = img_tensor.shape[-2:]
                img_tensor = img_tensor.to(device)

                # 安全的 4D 张量插值
                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    img_tensor = F.interpolate(img_tensor, size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear',
                                               align_corners=False)

                # 前向传播
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[:, 1:2, :, :]
                pred_mask = (probs > THRESHOLD).float()

                # 步骤 A：撤销 Interpolate 缩放
                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    pred_mask = F.interpolate(pred_mask, size=(tensor_h, tensor_w), mode='nearest')

                pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)[0, 0]

                # 🔥 修复致命 Bug 1：撤销中心 Pad 时完美规避奇数像素截断偏差
                # 步骤 B：撤销 Albumentations 的中心 Pad
                pad_h = max(0, config.IMG_SIZE - orig_h)
                pad_w = max(0, config.IMG_SIZE - orig_w)

                if pad_h > 0 or pad_w > 0:
                    # 🔥 修复致命切片 Bug：只有真加了 Pad 的维度才需要中心切除，否则起点就是 0
                    pad_top = pad_h // 2 if pad_h > 0 else 0
                    pad_left = pad_w // 2 if pad_w > 0 else 0
                    pred_mask_np = pred_mask_np[pad_top: pad_top + orig_h, pad_left: pad_left + orig_w]

                # 3. 直接从硬盘加载 Ground Truth
                label_path = str(slice_path_str).replace('_data.npy', '_label.npy')
                if os.path.exists(label_path):
                    gt_mask_np = np.load(label_path).astype(np.uint8)
                else:
                    gt_mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)

                assert pred_mask_np.shape == gt_mask_np.shape == (orig_h, orig_w), \
                    f"尺寸毁灭性异常！预测:{pred_mask_np.shape}, GT:{gt_mask_np.shape}"

                patient_preds.append(pred_mask_np)
                patient_gts.append(gt_mask_np)

            # 5. Z 轴堆叠
            vol_img = np.stack(patient_images, axis=0)
            vol_pred = np.stack(patient_preds, axis=0)
            vol_gt = np.stack(patient_gts, axis=0)

            # 6. 转换为 NIfTI 标准并保存
            sitk_img = sitk.GetImageFromArray(vol_img)
            sitk_pred = sitk.GetImageFromArray(vol_pred)
            sitk_gt = sitk.GetImageFromArray(vol_gt)

            # 统一赋予 Spacing，保证物理空间重叠
            sitk_img.SetSpacing((1.0, 1.0, 1.0))
            sitk_pred.SetSpacing((1.0, 1.0, 1.0))
            sitk_gt.SetSpacing((1.0, 1.0, 1.0))

            img_file = os.path.join(OUTPUT_3D_DIR, f"{patient_id}_Image.nii.gz")
            pred_file = os.path.join(OUTPUT_3D_DIR, f"{patient_id}_Pred.nii.gz")
            gt_file = os.path.join(OUTPUT_3D_DIR, f"{patient_id}_GT.nii.gz")

            sitk.WriteImage(sitk_img, img_file)
            sitk.WriteImage(sitk_pred, pred_file)
            sitk.WriteImage(sitk_gt, gt_file)

    print(f"\n🎉 所有患者的 3D 重建与 ROI 级对齐大功告成！")
    print(f"📁 NIfTI 文件已安全存放在: {OUTPUT_3D_DIR}")
    print(f"👉 这三套文件 (Image, Pred, GT) 已经完美对齐，随时可以提取组学特征！")


if __name__ == "__main__":
    reconstruct_3d_volumes()