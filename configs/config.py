import os
from pathlib import Path
import torch

# ================= 1. 路径设置 (Paths) =================
# 项目根目录 (自动获取 config.py 的上上级目录)
PROJECT_ROOT = Path(__file__).parent.parent

# 🔥 [Run 4 终极纯净大本营] 切换为 YOLO 裁剪后的数据集
DATA_ROOT = Path(r"F:\Dinov2_data\final_slices_YOLO_cor")

# 新增三个维度的具体路径，方便 dataset 极简调用
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"
UNLABEL_DIR = DATA_ROOT / "unlabel"

# 🔥 [引擎升级] 换装 330MB 的 DINOv2-Base 预训练权重
PRETRAINED_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "dinov2_vitb14_pretrain.pth")

# 输出路径 (Run 4: YOLO 靶区纯净版 + Base 巨兽)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs", "UniMatch_Cor_Run6")

# ================= 2. 数据参数 (Data) =================
# 保持 518，兼容性最强，结合 dataset.py 会自动进行黑边 Letterbox 填充
IMG_SIZE = 518
NUM_CLASSES = 2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 🔥 [模型排量] 新增 Base 模型的输出维度参数 (极其重要)
EMBED_DIM = 768  # Base 是 768 (之前 Small 是 384)

# ================= 3. 训练参数 (Training) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 4
NUM_WORKERS = 4
TOTAL_EPOCHS = 1500  # 从 train.py 移入并统一管理
PATIENCE = 500       # 🔥 早停容忍度，移入 config 统一管理

# 🔥 [战术修改 1] 强力正则化，防 Base 模型过拟合 (1e-4 -> 0.05)
WEIGHT_DECAY = 0.05

# 🔥 [战术修改 2] 学习率大劈叉 (Extreme Differential LR)
LR_HEAD = 2e-4       # 给解码器充足火力解析 768 维特征
LR_BACKBONE = 5e-6   # 主干网络保持极低学习率

# 🔥 [战术修改 3] LLRD 逐层衰减率断崖式下调 (0.90 -> 0.65)
LLRD_DECAY = 0.65    # 保护浅层边缘提取特征不被破坏

# ================= 4. UniMatch 核心参数 =================
CONF_THRESH = 0.75
UNLABELED_LOSS_WEIGHT = 1.0
EMA_DECAY = 0.99  # Teacher 模型更新速率

# ================= 5. 兼容性辅助 =================
import sys
config = sys.modules[__name__]