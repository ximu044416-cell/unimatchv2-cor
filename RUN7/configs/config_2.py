import os
from pathlib import Path
import torch

# ================= 1. 路径设置 (Paths) =================
# 假设本文件位于: F:\cor\run7\configs\config_2.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始数据根目录保持不变
DATA_ROOT = Path(r"F:\Dinov2_data\final_slices_YOLO_cor")

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
UNLABEL_DIR = DATA_ROOT / "unlabel"

# DINOv2 预训练权重
# 如果你已经把权重复制到 F:\cor\run7\checkpoints\ 下面，这行不用改
# 否则改成你当前实际的绝对路径
PRETRAINED_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "dinov2_vitb14_pretrain.pth")

# Run7 输出目录
RUN_NAME = "UniMatch_Cor_Ru7"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs", RUN_NAME)

# ================= 2. 数据参数 (Data) =================
IMG_SIZE = 518
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EMBED_DIM = 768

# ================= 3. 训练参数 (Training) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 4
NUM_WORKERS = 4
TOTAL_EPOCHS = 1500
PATIENCE = 500

WEIGHT_DECAY = 0.05
LR_HEAD = 2e-4
LR_BACKBONE = 5e-6
LLRD_DECAY = 0.65

# ================= 4. UniMatch 核心参数 =================
CONF_THRESH = 0.75
UNLABELED_LOSS_WEIGHT = 1.0
EMA_DECAY = 0.99

# ================= 5. Run7: 负向约束开关 =================
USE_NEG_PRIOR_LOSS = True
USE_NEG_ON_LABELED = True
USE_NEG_ON_UNLABELED = True

# ================= 6. Run7: 负向约束来源 =================
# 直接复用现有的 *_medsam2_prior.npy
NEG_PRIOR_SOURCE = "same_as_medsam_npy"

# ================= 7. Run7: 负向约束 Loss 超参数 =================
# soft overlap penalty
NEG_LOSS_TYPE = "soft_overlap"

# 轻微软截断阈值
NEG_TAU = 0.20
NEG_EPS = 1e-6

# 有标签 / 无标签最大权重
NEG_LAMBDA_SUP_MAX = 0.20
NEG_LAMBDA_UNSUP_MAX = 0.10

# 前多少个 epoch 线性 warm-up
NEG_WARMUP_EPOCHS = 20

# 只有存在负向区域时才计算
NEG_ONLY_WHEN_MASK_EXISTS = True

# clamp 到 [0,1]
NEG_PRIOR_CLAMP = True

# 如果后面想做硬掩膜实验，再改 True
NEG_USE_HARD_MASK = False
NEG_HARD_THRESHOLD = 0.50

# 日志
LOG_NEG_LOSS = True
LOG_NEG_RATIO = True

# ================= 8. 兼容性辅助 =================
import sys
config = sys.modules[__name__]