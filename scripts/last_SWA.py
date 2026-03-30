import os
import torch
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入自定义模块
from configs import config
from models.dinov2_unet import DINOUNet
from data.dataset import UniMatchDataset, get_split_indices
from train import validate_metrics_full


def run_post_training_swa():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print("🚀 启动自定义 SWA 终极融合引擎 (纯净版)...")

    # ================= 1. 确认要融合的权重文件 =================
    ckpt_names = [
        "epoch_900.pth",
        "epoch_920.pth",
        "epoch_940.pth",
        "epoch_960.pth",
        "epoch_980.pth",
        "epoch_1000.pth",
        "best_model.pth"  # 974 轮的最高点
    ]

    valid_ckpts = []
    for name in ckpt_names:
        path = os.path.join(config.OUTPUT_DIR, name)
        if os.path.exists(path):
            valid_ckpts.append(path)
        else:
            print(f"⚠️ 警告: 找不到权重文件 {name}，将跳过。")

    if len(valid_ckpts) < 2:
        print("❌ 错误: 至少需要 2 个权重文件才能进行 SWA 融合！")
        return

    print(f"📥 找到 {len(valid_ckpts)} 个巅峰期权重，准备进行维度融合...")

    # ================= 2. 初始化累加器 =================
    base_model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    base_model.load_state_dict(torch.load(valid_ckpts[0], map_location=device))

    swa_state_dict = copy.deepcopy(base_model.state_dict())

    for i in range(1, len(valid_ckpts)):
        print(f"   ➕ 正在融合: {os.path.basename(valid_ckpts[i])}")
        ckpt_state_dict = torch.load(valid_ckpts[i], map_location=device)
        for key in swa_state_dict.keys():
            swa_state_dict[key] += ckpt_state_dict[key]

    # 🔥 导师级排雷成功：使用 torch.is_floating_point 确保 100% 绝对安全
    num_models = len(valid_ckpts)
    for key in swa_state_dict.keys():
        if torch.is_floating_point(swa_state_dict[key]):
            swa_state_dict[key] = swa_state_dict[key] / float(num_models)
        else:
            # 对于 LongTensor (如 num_batches_tracked)，使用整除
            swa_state_dict[key] = swa_state_dict[key] // num_models

    print("✅ 权重数学平均完成！")

    # ================= 3. 极其致命的步骤：手工校准 BN 统计量 =================
    base_model.load_state_dict(swa_state_dict, strict=True)

    # 使用 mode='val' 加载训练数据，剥离所有随机增强，保证统计量纯净！
    train_l_files, _, _ = get_split_indices()
    dl_l_clean = DataLoader(UniMatchDataset(train_l_files, mode='val'),
                            batch_size=1, shuffle=True,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    print("🌟 正在执行纯净版 BN 统计量校准... 这可能需要几分钟...")

    # 手动重置所有 BN 层的统计状态
    for module in base_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            # 🔥 惊艳的物理锁：将 momentum 设为 None，强制 BN 使用全局累加平均
            module.momentum = None
            module.num_batches_tracked *= 0

    base_model.train()  # 必须在 train 模式下才能更新 BN

    # 手写等效且绝对安全的 update_bn 前向传播循环
    with torch.no_grad():
        for img, _ in tqdm(dl_l_clean, desc="Updating BN Stats"):
            img = img.to(device)

            # 🔥 终极保险锁：强制确保输入图像永远是 config.IMG_SIZE (518)
            # 因为 mode='val' 下的 PadIfNeeded 不会裁剪大于 518 的图
            if img.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
                img = torch.nn.functional.interpolate(
                    img,
                    size=(config.IMG_SIZE, config.IMG_SIZE),
                    mode='bilinear',
                    align_corners=False
                )

            # 纯粹前向传播，让 BN 层自动收集统计特征
            base_model(img)

    print("✅ BN 层均值和方差更新完毕！网络状态已完美校准！")

    # ================= 4. 终极大考：验证测试 =================
    print("🏆 开始进行 SWA 终极大考...")
    _, _, val_files = get_split_indices()
    dl_val = DataLoader(UniMatchDataset(val_files, mode='val'), batch_size=1, shuffle=False, num_workers=2)

    val_metrics_50 = validate_metrics_full(base_model, dl_val, device, thresh=0.50)
    val_metrics_60 = validate_metrics_full(base_model, dl_val, device, thresh=0.60)
    val_metrics_65 = validate_metrics_full(base_model, dl_val, device, thresh=0.65)

    print("\n" + "=" * 50)
    print(
        f"🏆 SWA 终极成绩 (Th=0.50): Dice {val_metrics_50['dice']:.4f}, Rec {val_metrics_50['recall']:.4f}, Prec {val_metrics_50['precision']:.4f}, AUC {val_metrics_50['auc']:.4f}")
    print(
        f"🏆 SWA 终极成绩 (Th=0.60): Dice {val_metrics_60['dice']:.4f}, Rec {val_metrics_60['recall']:.4f}, Prec {val_metrics_60['precision']:.4f}, AUC {val_metrics_60['auc']:.4f}")
    print(
        f"🏆 SWA 终极成绩 (Th=0.65): Dice {val_metrics_65['dice']:.4f}, Rec {val_metrics_65['recall']:.4f}, Prec {val_metrics_65['precision']:.4f}, AUC {val_metrics_65['auc']:.4f}")
    print("=" * 50 + "\n")

    # ================= 5. 保存终极兵器 =================
    save_path = os.path.join(config.OUTPUT_DIR, "best_custom_swa.pth")
    torch.save(base_model.state_dict(), save_path)
    print(f"💾 SWA 终极模型已保存至: {save_path}")


if __name__ == "__main__":
    run_post_training_swa()