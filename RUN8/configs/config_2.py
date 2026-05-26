import os
from pathlib import Path
import torch

# ================= 1. 路径设置 (Paths) =================
# 假设本文件位于: F:\cor\RUN8\configs\config_2.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RUN8 新数据根目录
DATA_ROOT = Path(r"F:\cor\RUN8\data\final_slices_YOLO_cor")

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
UNLABEL_DIR = DATA_ROOT / "unlabel"
TEST_DIR = DATA_ROOT / "test"   # 严格独立测试集，仅用于后续最终评估，不参与训练

# DINOv2 预训练权重
PRETRAINED_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "dinov2_vitb14_pretrain.pth")

# RUN8 输出目录
RUN_NAME = "UniMatch_Cor_RUN8_new"
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

# ================= 5. 负向约束开关 =================
USE_NEG_PRIOR_LOSS = True
USE_NEG_ON_LABELED = True
USE_NEG_ON_UNLABELED = True

# ================= 6. 负向约束来源 =================
NEG_PRIOR_SOURCE = "same_as_medsam_npy"

# ================= 7. 负向约束 Loss 超参数 =================
NEG_LOSS_TYPE = "soft_overlap"

NEG_TAU = 0.20
NEG_EPS = 1e-6

NEG_LAMBDA_SUP_MAX = 0.20
NEG_LAMBDA_UNSUP_MAX = 0.10

NEG_WARMUP_EPOCHS = 20

NEG_ONLY_WHEN_MASK_EXISTS = True
NEG_PRIOR_CLAMP = True

NEG_USE_HARD_MASK = False
NEG_HARD_THRESHOLD = 0.50

LOG_NEG_LOSS = True
LOG_NEG_RATIO = True

# ================= 8. 兼容性辅助 =================
import sys
config = sys.modules[__name__]