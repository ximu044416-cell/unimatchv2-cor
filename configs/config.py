import os
from pathlib import Path
import torch

# ================= 1. 路径设置 (Paths) =================
# 项目根目录 (自动获取 config.py 的上上级目录)
PROJECT_ROOT = Path(__file__).parent.parent

# 🔥 [新架构路径] 你的双轨制大本营
DATA_ROOT = Path(r"F:\Dinov2_data\final_slices_cor")

# 新增三个维度的具体路径，方便 dataset 极简调用
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
UNLABEL_DIR = DATA_ROOT / "unlabel"

# 权重路径
PRETRAINED_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "dinov2_vits14.pth")

# 输出路径 (Run 11: 冠状面大一统战役)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs", "UniMatch_Cor_Run3")

# ================= 2. 数据参数 (Data) =================
# 保持 518，兼容性最强，极个别超大框将进行中心裁剪
IMG_SIZE = 518
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ================= 3. 训练参数 (Training) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 300
WEIGHT_DECAY = 1e-4

# ================= 4. UniMatch 核心参数 =================
CONF_THRESH = 0.75
UNLABELED_LOSS_WEIGHT = 1.0
EMA_DECAY = 0.99  # Teacher 模型更新速率

# ================= 5. 兼容性辅助 =================
import sys
config = sys.modules[__name__]