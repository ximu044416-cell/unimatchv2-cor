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


def reconstruct_3d_volumes_strict_split():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # ================= 1. 核心参数与路径准备 =================
    # 坚决使用巅峰权重
    WEIGHT_PATH = os.path.join(config.OUTPUT_DIR, "best_model.pth")

    # 🔥 目标输出总目录
    OUTPUT_3D_DIR = r"F:\downstreamtasks\radiomics"

    # 🔥 绝对金标准：你之前手工划分的阵地，用于提取 Train/Test 花名册
    REFERENCE_DIR = r"F:\radiomics"

    # 提前在输出目录中建好物理隔离的护城河
    os.makedirs(os.path.join(OUTPUT_3D_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_3D_DIR, "test"), exist_ok=True)

    # 🏆 黄金阈值锁定：0.70 (压制假阳性，保证组学纯净)
    THRESHOLD = 0.70

    print(f"🚀 启动 3D 临床视角【纯净队列】重建引擎 (严格物理隔离版)...")
    print(f"📥 正在加载终极 SOTA 权重: {os.path.basename(WEIGHT_PATH)}")

    # ================= 2. 🔥 建立绝对隔离花名册 (Roster) =================
    # 直接读取你之前手工划分的文件夹，获取病人 ID 列表
    train_ref_path = os.path.join(REFERENCE_DIR, "train")
    test_ref_path = os.path.join(REFERENCE_DIR, "test")

    if not os.path.exists(train_ref_path) or not os.path.exists(test_ref_path):
        raise FileNotFoundError(f"❌ 致命错误: 找不到参考目录 {train_ref_path} 或 {test_ref_path}，无法执行镜像匹配！")

    train_roster = set(os.listdir(train_ref_path))
    test_roster = set(os.listdir(test_ref_path))

    print(
        f"📋 查阅历史档案: 在 {REFERENCE_DIR} 中发现 {len(train_roster)} 名 Train 患者, {len(test_roster)} 名 Test 患者。")

    # ================= 3. 实例化神盾基座 =================
    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device), strict=True)
    model.eval()

    # ================= 4. 搜刮原始切片并按花名册过滤 =================
    target_dirs = [Path(config.TRAIN_DIR), Path(config.UNLABEL_DIR), Path(config.VAL_DIR)]
    patient_dict = {}

    for split_dir in target_dirs:
        if not split_dir.exists():
            print(f"⚠️ 警告: 找不到数据文件夹 {split_dir}，跳过。")
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

    print(f"📊 搜刮与过滤完毕！共准备对 {len(patient_dict)} 名在册患者进行像素级重建...\n")

    # ================= 5. 逐患者进行 3D 拼装推理 =================
    with torch.no_grad():
        for patient_id, slice_list in tqdm(patient_dict.items(), desc="全队列重建总进度"):

            # 🔥 严格对号入座，决定输出的子文件夹
            if patient_id in train_roster:
                split_folder = "train"
            else:
                split_folder = "test"

            # 最终的专属存放路径，例如：F:\downstreamtasks\radiomics\train\Patient_XX
            patient_output_dir = os.path.join(OUTPUT_3D_DIR, split_folder, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)

            patient_dataset = UniMatchDataset(slice_list, mode='val')
            patient_loader = DataLoader(patient_dataset, batch_size=1, shuffle=False, num_workers=0)

            patient_images = []
            patient_preds = []
            patient_gts = []

            for (data_tuple, slice_path_str) in zip(patient_loader, slice_list):
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

                # 截断执行！
                pred_mask = (probs > THRESHOLD).float()

                # 步骤 A：撤销 Interpolate 缩放
                if tensor_h != config.IMG_SIZE or tensor_w != config.IMG_SIZE:
                    pred_mask = F.interpolate(pred_mask, size=(tensor_h, tensor_w), mode='nearest')

                pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)[0, 0]

                # 步骤 B：撤销 Albumentations 的中心 Pad
                pad_h = max(0, config.IMG_SIZE - orig_h)
                pad_w = max(0, config.IMG_SIZE - orig_w)

                if pad_h > 0 or pad_w > 0:
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

            img_file = os.path.join(patient_output_dir, f"{patient_id}_Image.nii.gz")
            pred_file = os.path.join(patient_output_dir, f"{patient_id}_Pred.nii.gz")
            gt_file = os.path.join(patient_output_dir, f"{patient_id}_GT.nii.gz")

            sitk.WriteImage(sitk_img, img_file)
            sitk.WriteImage(sitk_pred, pred_file)
            sitk.WriteImage(sitk_gt, gt_file)

    print(f"\n🎉 惊人的成就！全队列共 {len(patient_dict)} 名患者的 3D 重建与 ROI 级对齐大功告成！")
    print(f"📁 弹药库已严格按照 Train/Test 物理隔离，安全封存于: {OUTPUT_3D_DIR}")
    print(f"👉 随时准备开启第二阶段：高通量影像组学榨取！")


if __name__ == "__main__":
    reconstruct_3d_volumes_strict_split()