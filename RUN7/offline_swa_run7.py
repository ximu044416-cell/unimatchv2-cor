import os
import sys
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import update_bn

# ===== 让 RUN7 根目录进入模块搜索路径 =====
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_2 import config
from models.dinov2_unet import DINOUNet
from data.dataset_2 import UniMatchDataset2, get_split_indices_2


# =========================================================
# 1. 你只需要改这里：要参与平均的 checkpoint 列表
# =========================================================
CKPT_PATHS = [
    r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\epoch_380.pth",
    r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\epoch_400.pth",
    r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\best_model.pth",
    r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\epoch_420.pth",
    r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\epoch_440.pth",
]

# 输出文件名
OUT_PATH = r"F:\cor\RUN7\logs\UniMatch_Cor_Run7_NegPrior\best_model_SWA_offline_380_400_best_420_440.pth"

# 是否在平均后重新跑 BN 统计
# 如果模型里没有 BatchNorm，这一步基本没有影响；有的话建议保留
DO_UPDATE_BN = True

# BN 统计用的 batch size
BN_BATCH_SIZE = 4


def load_state_dict_safe(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        # 常见情况：直接就是 state_dict
        return ckpt
    raise ValueError(f"不支持的 checkpoint 格式: {type(ckpt)} | {path}")


def average_state_dicts(paths):
    if len(paths) == 0:
        raise ValueError("CKPT_PATHS 为空")

    state_dicts = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到 checkpoint: {p}")
        state_dicts.append(load_state_dict_safe(p))

    ref_keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if list(sd.keys()) != ref_keys:
            raise ValueError(f"第 {i} 个 checkpoint 的 key 与第一个不一致，不能直接平均")

    avg_sd = OrderedDict()

    for k in ref_keys:
        vals = [sd[k] for sd in state_dicts]

        # 浮点权重：取均值
        if torch.is_floating_point(vals[0]):
            stacked = torch.stack([v.float() for v in vals], dim=0)
            avg_sd[k] = stacked.mean(dim=0)
        else:
            # 非浮点（例如整型计数器）直接用第一个
            avg_sd[k] = vals[0]

    return avg_sd


@torch.no_grad()
def run_bn_update(model, device):
    print("🔄 开始 update_bn ...")
    train_l_files, _, _ = get_split_indices_2()

    dl_bn = DataLoader(
        UniMatchDataset2(train_l_files, mode='labeled'),
        batch_size=BN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # update_bn 会自动取 batch 的第一个元素作为输入
    update_bn(dl_bn, model, device=device)
    print("✅ update_bn 完成")


def main():
    print("📦 即将进行离线 SWA / checkpoint averaging")
    for i, p in enumerate(CKPT_PATHS, 1):
        print(f"  [{i}] {p}")

    avg_sd = average_state_dicts(CKPT_PATHS)

    device = torch.device(config.DEVICE)
    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(avg_sd, strict=True)
    model.eval()

    if DO_UPDATE_BN:
        run_bn_update(model, device)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUT_PATH)
    print(f"💾 离线 SWA 权重已保存到:\n{OUT_PATH}")


if __name__ == "__main__":
    main()