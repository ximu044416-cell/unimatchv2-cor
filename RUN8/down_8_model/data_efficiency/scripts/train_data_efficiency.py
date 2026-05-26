import os
import sys
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# =========================================================
# 路径
# =========================================================
RUN8_ROOT = Path(r"F:\cor\RUN8")
ABLATION_ROOT = RUN8_ROOT / "down_8_model" / "data_efficiency"

sys.path.insert(0, str(RUN8_ROOT))

from configs.config_2 import config
from models.dinov2_unet import DINOUNet
from data.dataset_2 import UniMatchDataset2
from utils.losses_2 import (
    FocalTverskyLoss,
    BoundaryDoULoss,
    NegativePriorPenaltyLoss,
    get_neg_lambda
)


# =========================================================
# 参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_labeled",
        type=int,
        required=True,
        choices=[11, 22, 55],
        help="Number of labeled training patients: 11, 22, or 55."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["semi", "sup"],
        help="semi = semi-supervised; sup = supervised-only."
    )

    parser.add_argument(
        "--debug_epochs",
        type=int,
        default=0,
        help="Set >0 only for debugging. Default 0 means use config.TOTAL_EPOCHS."
    )

    return parser.parse_args()


# =========================================================
# 数据列表
# =========================================================
def get_case_dirs(dir_path: Path):
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_dir() and not p.name.startswith(".")])


def get_case_names(dir_path: Path):
    return set([p.name for p in get_case_dirs(dir_path)])


def get_slices_from_case_dirs(case_dirs):
    files = []
    for cd in case_dirs:
        files.extend(sorted(list(cd.glob("*_data.npy"))))
    return files


def assert_no_overlap(name_a, set_a, name_b, set_b):
    overlap = set_a & set_b
    if overlap:
        preview = sorted(list(overlap))[:10]
        raise RuntimeError(
            f"❌ 数据划分重叠: {name_a} 与 {name_b} 存在 {len(overlap)} 个重复病例, 例如 {preview}"
        )


def load_roster(n_labeled: int):
    roster_path = ABLATION_ROOT / "rosters" / f"train_{n_labeled}_roster.txt"
    if not roster_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到 roster 文件: {roster_path}\n"
            f"请先运行 make_data_efficiency_rosters.py"
        )

    with open(roster_path, "r", encoding="utf-8") as f:
        names = [x.strip() for x in f.readlines() if x.strip()]

    return set(names), roster_path


def get_data_efficiency_split(n_labeled: int, semi_supervised: bool):
    """
    数据敏感性实验协议：

    - labeled branch:
        只使用 train_n_roster.txt 中的 n 个病例标签。

    - unlabeled branch:
        semi 模式下使用：
        1) unlabel 文件夹 94 例
        2) train 文件夹全部 55 例 MRI，作为 image-only unlabeled pool
           其中 selected labeled cases 同时也可作为 unlabeled consistency data；
           未入选 labeled subset 的 train cases 不使用 GT，只作为 unlabeled MRI。
        sup 模式下不使用 unlabeled branch。

    - val:
        固定使用最终协议中的 val 文件夹 15 例。

    - test:
        完全不参与训练。
    """
    train_dir = config.TRAIN_DIR
    val_dir = config.VAL_DIR
    unlabel_dir = config.UNLABEL_DIR
    test_dir = config.TEST_DIR

    train_names = get_case_names(train_dir)
    val_names = get_case_names(val_dir)
    unlabel_names = get_case_names(unlabel_dir)
    test_names = get_case_names(test_dir)

    assert_no_overlap("train", train_names, "val", val_names)
    assert_no_overlap("train", train_names, "unlabel", unlabel_names)
    assert_no_overlap("train", train_names, "test", test_names)
    assert_no_overlap("val", val_names, "unlabel", unlabel_names)
    assert_no_overlap("val", val_names, "test", test_names)
    assert_no_overlap("unlabel", unlabel_names, "test", test_names)

    allowed_names, roster_path = load_roster(n_labeled)

    missing = allowed_names - train_names
    if missing:
        raise RuntimeError(
            f"❌ roster 中有 {len(missing)} 个病例不在 train 文件夹中，例如: {sorted(list(missing))[:10]}"
        )

    all_train_case_dirs = get_case_dirs(train_dir)
    selected_train_case_dirs = [p for p in all_train_case_dirs if p.name in allowed_names]
    unlabel_case_dirs = get_case_dirs(unlabel_dir)
    val_case_dirs = get_case_dirs(val_dir)
    test_case_dirs = get_case_dirs(test_dir)

    train_l_files = get_slices_from_case_dirs(selected_train_case_dirs)
    val_files = get_slices_from_case_dirs(val_case_dirs)
    test_files = get_slices_from_case_dirs(test_case_dirs)

    if semi_supervised:
        # 半监督：无标签池 = 原 unlabel + 全部 train MRI
        train_u_files = get_slices_from_case_dirs(unlabel_case_dirs) + get_slices_from_case_dirs(all_train_case_dirs)
    else:
        train_u_files = []

    print("=" * 90)
    print("📊 Data efficiency split")
    print(f"   Roster file: {roster_path}")
    print(f"   Mode: {'semi-supervised' if semi_supervised else 'supervised-only'}")
    print(f"   Labeled patients used: {len(selected_train_case_dirs)} / {len(all_train_case_dirs)}")
    print(f"   Labeled train slices:  {len(train_l_files)}")
    print(f"   Unlabeled slices:      {len(train_u_files)}")
    print(f"   Validation cases:      {len(val_case_dirs)}")
    print(f"   Validation slices:     {len(val_files)}")
    print(f"   Hold-out test cases:   {len(test_case_dirs)}  不参与训练")
    print(f"   Hold-out test slices:  {len(test_files)}      不参与训练")
    print("=" * 90)

    return train_l_files, train_u_files, val_files


# =========================================================
# 损失权重与训练策略
# =========================================================
def get_current_unsup_weight(epoch):
    warmup_epochs = 50
    max_weight = config.UNLABELED_LOSS_WEIGHT
    if epoch < warmup_epochs:
        return max_weight * (epoch / warmup_epochs)
    return max_weight


def get_cosine_threshold(epoch, total_epochs, max_thresh=0.85, min_thresh=0.55, start_decay_epoch=150):
    if epoch < start_decay_epoch:
        return max_thresh

    progress = (epoch - start_decay_epoch) / max(1, total_epochs - start_decay_epoch)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_thresh + (max_thresh - min_thresh) * cosine_decay


def get_ema_alpha(epoch):
    base_alpha = 0.99
    target_alpha = 0.999
    warmup_epochs = 150

    if epoch < warmup_epochs:
        return base_alpha + (target_alpha - base_alpha) * (epoch / warmup_epochs)
    return target_alpha


# =========================================================
# CutMix
# =========================================================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix_with_aux(data, target_main, target_aux, probability=0.5, beta=1.0):
    if np.random.rand() > probability:
        return data, target_main, target_aux

    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(data.size(0)).to(data.device)

    data_b = data[rand_index]
    main_b = target_main[rand_index]
    aux_b = target_aux[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = data_b[:, :, bbx1:bbx2, bby1:bby2]

    if target_main.ndim == 3:
        target_main[:, bbx1:bbx2, bby1:bby2] = main_b[:, bbx1:bbx2, bby1:bby2]
    elif target_main.ndim == 4:
        target_main[:, :, bbx1:bbx2, bby1:bby2] = main_b[:, :, bbx1:bbx2, bby1:bby2]

    if target_aux.ndim == 3:
        target_aux[:, bbx1:bbx2, bby1:bby2] = aux_b[:, bbx1:bbx2, bby1:bby2]
    elif target_aux.ndim == 4:
        target_aux[:, :, bbx1:bbx2, bby1:bby2] = aux_b[:, :, bbx1:bbx2, bby1:bby2]

    return data, target_main, target_aux


# =========================================================
# 验证
# =========================================================
def sliding_window_inference_tta(model, image, window_size=(518, 518), overlap=0.5):
    b, c, h, w = image.shape
    tile_h, tile_w = window_size
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))

    def get_gaussian(size, sigma_scale=1.0 / 8):
        center_coords = [i // 2 for i in size]
        sigmas = [i * sigma_scale for i in size]
        k_h, k_w = size
        y, x = np.ogrid[:k_h, :k_w]
        h_norm = (y - center_coords[0]) / sigmas[0]
        w_norm = (x - center_coords[1]) / sigmas[1]
        g = np.exp(-(h_norm ** 2 + w_norm ** 2) / 2)
        return torch.from_numpy(g).float().to(image.device)

    gaussian_weight = get_gaussian(window_size)
    output_sum = torch.zeros((b, config.NUM_CLASSES, h, w), device=image.device)
    weight_sum = torch.zeros((b, config.NUM_CLASSES, h, w), device=image.device)

    h_steps = math.ceil((h - tile_h) / stride_h) + 1
    w_steps = math.ceil((w - tile_w) / stride_w) + 1

    def predict_patch_tta(patch):
        pred = torch.softmax(model(patch), dim=1)

        pred_h = torch.softmax(model(torch.flip(patch, [3])), dim=1)
        pred = pred + torch.flip(pred_h, [3])

        pred_v = torch.softmax(model(torch.flip(patch, [2])), dim=1)
        pred = pred + torch.flip(pred_v, [2])

        return pred / 3.0

    for i in range(h_steps):
        for j in range(w_steps):
            h_start = min(i * stride_h, h - tile_h)
            w_start = min(j * stride_w, w - tile_w)

            h_end = h_start + tile_h
            w_end = w_start + tile_w

            patch = image[:, :, h_start:h_end, w_start:w_end]
            prob_patch = predict_patch_tta(patch)

            output_sum[:, :, h_start:h_end, w_start:w_end] += prob_patch * gaussian_weight
            weight_sum[:, :, h_start:h_end, w_start:w_end] += gaussian_weight

    return output_sum / weight_sum.clamp_min(1e-8)


def calculate_metrics_manual(pred_mask, gt_mask, pred_prob):
    smooth = 1e-5

    pred = pred_mask.float()
    gt = gt_mask.float()

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)

    gt_flat = gt.view(-1).detach().cpu().numpy()
    prob_flat = pred_prob.view(-1).detach().cpu().numpy()

    if len(np.unique(gt_flat)) == 1:
        auc = np.nan
    else:
        auc = roc_auc_score(gt_flat, prob_flat)

    return dice.item(), recall.item(), precision.item(), auc


def validate_metrics_full(model, dataloader, device, thresh=0.70):
    model.eval()

    total = {
        "dice": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "auc": 0.0
    }

    count = 0
    valid_auc_count = 0

    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc=f"Val(Th={thresh})", leave=False):
            img = img.to(device)
            mask = mask.to(device)

            with autocast("cuda", enabled=(device.type == "cuda")):
                prob_map = sliding_window_inference_tta(
                    model,
                    img,
                    window_size=(config.IMG_SIZE, config.IMG_SIZE),
                    overlap=0.5
                )

            pred_prob = prob_map[:, 1, :, :]
            pred_mask = (pred_prob > thresh).float()

            d, r, p, auc = calculate_metrics_manual(pred_mask, mask, pred_prob)

            total["dice"] += d
            total["recall"] += r
            total["precision"] += p

            if not np.isnan(auc):
                total["auc"] += auc
                valid_auc_count += 1

            count += 1

    return {
        "dice": total["dice"] / max(count, 1),
        "recall": total["recall"] / max(count, 1),
        "precision": total["precision"] / max(count, 1),
        "auc": total["auc"] / valid_auc_count if valid_auc_count > 0 else 0.0
    }


# =========================================================
# Optimizer helpers
# =========================================================
def get_llrd_params(model, lr_backbone, lr_head, weight_decay, decay_rate=0.90):
    groups = []
    head_params = []

    for name, param in model.named_parameters():
        if "encoder" not in name and param.requires_grad:
            head_params.append(param)

    groups.append({
        "params": head_params,
        "lr": lr_head,
        "weight_decay": weight_decay,
        "target_lr": lr_head
    })

    layer_params = [[] for _ in range(12)]
    embed_params = []
    norm_params = []

    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue

        if "blocks" in name:
            try:
                parts = name.split("blocks.")
                layer_id = int(parts[1].split(".")[0])
                if layer_id < 12:
                    layer_params[layer_id].append(param)
                else:
                    layer_params[11].append(param)
            except Exception:
                embed_params.append(param)
        elif "norm" in name and "blocks" not in name:
            norm_params.append(param)
        else:
            embed_params.append(param)

    if len(norm_params) > 0:
        groups.append({
            "params": norm_params,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
            "target_lr": lr_backbone
        })

    for i in range(11, -1, -1):
        if len(layer_params[i]) > 0:
            scale = decay_rate ** (11 - i)
            groups.append({
                "params": layer_params[i],
                "lr": lr_backbone * scale,
                "weight_decay": weight_decay,
                "target_lr": lr_backbone * scale
            })

    if len(embed_params) > 0:
        groups.append({
            "params": embed_params,
            "lr": lr_backbone * (decay_rate ** 12),
            "weight_decay": weight_decay,
            "target_lr": lr_backbone * (decay_rate ** 12)
        })

    return groups


class EarlyStopping:
    def __init__(self, patience=20, save_path="best_model.pth"):
        self.patience = patience
        self.counter = 0
        self.best_dice = 0.0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, current_dice, model, logging_func):
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            logging_func(f"💾 New Best Saved! Dice={self.best_dice:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


@torch.no_grad()
def update_ema(student, teacher, alpha=0.999):
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.mul_(alpha).add_(param, alpha=1 - alpha)


# =========================================================
# 训练主函数
# =========================================================
def train_one_experiment(n_labeled: int, mode: str, debug_epochs: int = 0):
    semi_supervised = mode == "semi"
    mode_name = f"{mode}_{n_labeled}"

    out_root = ABLATION_ROOT / mode_name
    log_dir = out_root / "logs"
    ckpt_dir = out_root / "checkpoints"
    tb_dir = out_root / "tensorboard_logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = debug_epochs if debug_epochs > 0 else config.TOTAL_EPOCHS

    # -----------------------------
    # logging
    # -----------------------------
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=str(log_dir / "train_log.txt"),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S"))
    logging.getLogger("").addHandler(console)

    device = torch.device(config.DEVICE)
    amp_enabled = device.type == "cuda"

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    logging.info("=" * 90)
    logging.info(f"🚀 Data efficiency experiment started: mode={mode}, n_labeled={n_labeled}")
    logging.info("Backbone: DINOv2")
    logging.info("Prior input: enabled")
    logging.info("Negative-prior penalty: enabled")
    logging.info(f"Output: {out_root}")
    logging.info("=" * 90)

    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    ACCUM_STEPS = 2
    physical_batch_size = max(1, config.BATCH_SIZE // ACCUM_STEPS)

    train_l_files, train_u_files, val_files = get_data_efficiency_split(
        n_labeled=n_labeled,
        semi_supervised=semi_supervised
    )

    dl_l = DataLoader(
        UniMatchDataset2(train_l_files, mode="labeled"),
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    if semi_supervised:
        dl_u = DataLoader(
            UniMatchDataset2(train_u_files, mode="unlabeled"),
            batch_size=physical_batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        iter_u = iter(dl_u)
    else:
        dl_u = None
        iter_u = None

    dl_val = DataLoader(
        UniMatchDataset2(val_files, mode="val"),
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)

    if semi_supervised:
        teacher_model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
        teacher_model.load_state_dict(model.state_dict())
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
    else:
        teacher_model = None

    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_model_path = ckpt_dir / "best_model.pth"
    early_stopping = EarlyStopping(patience=config.PATIENCE, save_path=str(best_model_path))

    criterion_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_tversky = FocalTverskyLoss(
        n_classes=config.NUM_CLASSES,
        alpha=0.3,
        beta=0.7,
        gamma=1.33,
        dynamic_beta=False
    )
    criterion_bdou = BoundaryDoULoss(kernel_size=3)
    criterion_u_ce = nn.CrossEntropyLoss(reduction="none")

    criterion_neg = NegativePriorPenaltyLoss(
        tau=config.NEG_TAU,
        eps=config.NEG_EPS,
        only_when_mask_exists=config.NEG_ONLY_WHEN_MASK_EXISTS
    )

    # stage 1: freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY
    )

    min_lr = config.LR_HEAD * 0.55
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=min_lr
    )

    global_prob_val = torch.ones(1, device=device) * 0.5
    current_stage = 1

    swa_model = AveragedModel(model)
    swa_start_epoch = total_epochs - 50

    epoch_rows = []

    for epoch in range(total_epochs):
        # -----------------------------
        # stage scheduling
        # -----------------------------
        if epoch < 150:
            model.train()
            model.encoder.eval()
            if teacher_model is not None:
                teacher_model.eval()

            if epoch < 100:
                w_ce, w_ft, w_bd = 0.5, 1.0, 0.0
            else:
                w_ce, w_ft, w_bd = 0.2, 0.8, 0.5

        else:
            if current_stage == 1:
                logging.info(f"🔓 [Epoch {epoch}] Unfreezing DINOv2 encoder")
                for p in model.encoder.parameters():
                    p.requires_grad = True

                optimizer = optim.AdamW(
                    get_llrd_params(
                        model,
                        config.LR_BACKBONE,
                        config.LR_HEAD,
                        config.WEIGHT_DECAY,
                        decay_rate=config.LLRD_DECAY
                    )
                )

                early_stopping.best_dice = 0.0
                early_stopping.counter = 0
                logging.info("♻️ EarlyStopping reset after encoder unfreezing")
                current_stage = 2

            model.train()
            if teacher_model is not None:
                teacher_model.eval()

            if epoch < 300:
                w_ce, w_ft, w_bd = 0.2, 0.8, 0.5
            else:
                w_ce, w_ft, w_bd = 0.1, 0.5, 0.2

            if 150 <= epoch < 165:
                warmup_ratio = (epoch - 150 + 1) / 15.0
                for group in optimizer.param_groups:
                    target_lr = group.get("target_lr", group["lr"])
                    group["lr"] = 1e-8 + warmup_ratio * (target_lr - 1e-8)

                if epoch == 164:
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, total_epochs - 165),
                        eta_min=config.LR_BACKBONE * 0.55
                    )
                    logging.info("🚀 Warm-up finished; switched to CosineAnnealingLR")

        current_unsup_weight = get_current_unsup_weight(epoch) if semi_supervised else 0.0
        current_ema_alpha = get_ema_alpha(epoch)

        cosine_base_thresh = get_cosine_threshold(
            epoch,
            total_epochs,
            max_thresh=0.85,
            min_thresh=0.55
        )

        current_thresh = max(global_prob_val.item(), cosine_base_thresh)

        lambda_neg_sup = get_neg_lambda(
            epoch,
            config.NEG_LAMBDA_SUP_MAX,
            config.NEG_WARMUP_EPOCHS
        )

        lambda_neg_unsup = get_neg_lambda(
            epoch,
            config.NEG_LAMBDA_UNSUP_MAX,
            config.NEG_WARMUP_EPOCHS
        ) if semi_supervised else 0.0

        metrics_meter = {
            "loss": 0.0,
            "sup": 0.0,
            "unsup": 0.0,
            "neg_sup": 0.0,
            "neg_unsup": 0.0
        }

        pbar = tqdm(dl_l, total=len(dl_l), desc=f"{mode_name} Ep {epoch + 1}/{total_epochs}")
        optimizer.zero_grad()

        for step, batch_l in enumerate(pbar):
            img_l, mask_l, neg_l = batch_l
            img_l = img_l.to(device)
            mask_l = mask_l.to(device)
            neg_l = neg_l.to(device)

            img_l, mask_l, neg_l = apply_cutmix_with_aux(
                img_l,
                mask_l,
                neg_l,
                probability=0.5,
                beta=1.0
            )

            with autocast("cuda", enabled=amp_enabled):
                pred_l = model(img_l)
                probs_l = torch.softmax(pred_l, dim=1)

                loss_ce = criterion_ce(pred_l, mask_l)
                loss_tversky = criterion_tversky(probs_l, mask_l, current_epoch=epoch)
                loss_bdou = criterion_bdou(probs_l, mask_l)

                loss_sup = w_ce * loss_ce + w_ft * loss_tversky + w_bd * loss_bdou

                loss_neg_sup = torch.tensor(0.0, device=device)
                if config.USE_NEG_PRIOR_LOSS and config.USE_NEG_ON_LABELED and lambda_neg_sup > 0:
                    loss_neg_sup = criterion_neg(probs_l, neg_l)

                loss_unsup = torch.tensor(0.0, device=device)
                loss_neg_unsup = torch.tensor(0.0, device=device)

                if semi_supervised and current_unsup_weight > 0:
                    try:
                        batch_u = next(iter_u)
                    except StopIteration:
                        iter_u = iter(dl_u)
                        batch_u = next(iter_u)

                    img_u_w, img_u_s1, img_u_s2, neg_u = batch_u
                    img_u_w = img_u_w.to(device)
                    img_u_s1 = img_u_s1.to(device)
                    img_u_s2 = img_u_s2.to(device)
                    neg_u = neg_u.to(device)

                    with torch.no_grad():
                        pred_u_w = teacher_model(img_u_w)
                        probs_u_w = torch.softmax(pred_u_w, dim=1)

                        max_probs, pseudo_label = torch.max(probs_u_w, dim=1)

                        prob_edema = probs_u_w[:, 1, :, :]
                        mask_edema = (pseudo_label == 1)

                        if mask_edema.any():
                            current_batch_conf = prob_edema[mask_edema].mean()
                            global_prob_val = global_prob_val * 0.99 + current_batch_conf * 0.01

                        mask_conf = max_probs.ge(current_thresh).float()

                    img_u_s_cat = torch.cat((img_u_s1, img_u_s2), dim=0)
                    pred_u_s_cat = model(img_u_s_cat)
                    pred_u_s1, pred_u_s2 = torch.chunk(pred_u_s_cat, 2, dim=0)

                    loss_u_s1 = (
                        criterion_u_ce(pred_u_s1, pseudo_label) * mask_conf
                    ).sum() / (mask_conf.sum() + 1e-5)

                    loss_u_s2 = (
                        criterion_u_ce(pred_u_s2, pseudo_label) * mask_conf
                    ).sum() / (mask_conf.sum() + 1e-5)

                    loss_unsup = (loss_u_s1 + loss_u_s2) / 2.0

                    if config.USE_NEG_PRIOR_LOSS and config.USE_NEG_ON_UNLABELED and lambda_neg_unsup > 0:
                        probs_u_s1 = torch.softmax(pred_u_s1, dim=1)
                        probs_u_s2 = torch.softmax(pred_u_s2, dim=1)

                        loss_neg_u1 = criterion_neg(probs_u_s1, neg_u)
                        loss_neg_u2 = criterion_neg(probs_u_s2, neg_u)

                        loss_neg_unsup = (loss_neg_u1 + loss_neg_u2) / 2.0

                total_loss = (
                    loss_sup
                    + lambda_neg_sup * loss_neg_sup
                    + current_unsup_weight * loss_unsup
                    + lambda_neg_unsup * loss_neg_unsup
                )

                loss_to_backward = total_loss / ACCUM_STEPS

            scaler.scale(loss_to_backward).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(dl_l):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if semi_supervised:
                    update_ema(model, teacher_model, alpha=current_ema_alpha)

            metrics_meter["loss"] += float(total_loss.item())
            metrics_meter["sup"] += float(loss_sup.item())
            metrics_meter["unsup"] += float(loss_unsup.item())
            metrics_meter["neg_sup"] += float(loss_neg_sup.item())
            metrics_meter["neg_unsup"] += float(loss_neg_unsup.item())

            pbar.set_postfix({
                "L": f"{total_loss.item():.3f}",
                "U_W": f"{current_unsup_weight:.3f}",
                "NegS": f"{lambda_neg_sup:.3f}",
                "NegU": f"{lambda_neg_unsup:.3f}",
                "Th": f"{current_thresh:.3f}"
            })

        # -----------------------------
        # scheduler
        # -----------------------------
        if epoch < 150:
            scheduler.step()
        elif 150 <= epoch < 165:
            pass
        else:
            scheduler.step()

        avg_loss = metrics_meter["loss"] / max(len(dl_l), 1)
        avg_sup = metrics_meter["sup"] / max(len(dl_l), 1)
        avg_unsup = metrics_meter["unsup"] / max(len(dl_l), 1)
        avg_neg_sup = metrics_meter["neg_sup"] / max(len(dl_l), 1)
        avg_neg_unsup = metrics_meter["neg_unsup"] / max(len(dl_l), 1)

        tb_writer.add_scalar("Train/Loss_Total", avg_loss, epoch)
        tb_writer.add_scalar("Train/Loss_Sup", avg_sup, epoch)
        tb_writer.add_scalar("Train/Loss_Unsup", avg_unsup, epoch)
        tb_writer.add_scalar("Train/Loss_NegSup", avg_neg_sup, epoch)
        tb_writer.add_scalar("Train/Loss_NegUnsup", avg_neg_unsup, epoch)

        tb_writer.add_scalar("Sys/Unsup_Weight", current_unsup_weight, epoch)
        tb_writer.add_scalar("Sys/EMA_Alpha", current_ema_alpha, epoch)
        tb_writer.add_scalar("Sys/Dynamic_Thresh", current_thresh, epoch)
        tb_writer.add_scalar("Sys/Neg_Lambda_Sup", lambda_neg_sup, epoch)
        tb_writer.add_scalar("Sys/Neg_Lambda_Unsup", lambda_neg_unsup, epoch)

        # -----------------------------
        # validation
        # -----------------------------
        val_metrics_50 = validate_metrics_full(model, dl_val, device, thresh=0.50)
        val_metrics_60 = validate_metrics_full(model, dl_val, device, thresh=0.60)
        val_metrics_65 = validate_metrics_full(model, dl_val, device, thresh=0.65)

        val_results = {
            "0.50": val_metrics_50,
            "0.60": val_metrics_60,
            "0.65": val_metrics_65
        }

        best_th_str = max(val_results, key=lambda k: val_results[k]["dice"])
        best = val_results[best_th_str]

        for th, met in val_results.items():
            tb_writer.add_scalar(f"Val_{th}/Dice", met["dice"], epoch)
            tb_writer.add_scalar(f"Val_{th}/Recall", met["recall"], epoch)
            tb_writer.add_scalar(f"Val_{th}/Precision", met["precision"], epoch)
            tb_writer.add_scalar(f"Val_{th}/AUC", met["auc"], epoch)

        tb_writer.add_scalar("Val_Best/Dice", best["dice"], epoch)

        early_stopping(best["dice"], model, logging.info)

        row = {
            "epoch": epoch + 1,
            "mode": mode,
            "n_labeled": n_labeled,
            "loss": avg_loss,
            "loss_sup": avg_sup,
            "loss_unsup": avg_unsup,
            "loss_neg_sup": avg_neg_sup,
            "loss_neg_unsup": avg_neg_unsup,
            "best_threshold": best_th_str,
            "val_dice": best["dice"],
            "val_recall": best["recall"],
            "val_precision": best["precision"],
            "val_auc": best["auc"],
            "patience_counter": early_stopping.counter
        }
        epoch_rows.append(row)

        pd.DataFrame(epoch_rows).to_csv(
            out_root / "epoch_metrics.csv",
            index=False,
            encoding="utf-8-sig"
        )

        logging.info(
            f"Epoch {epoch + 1:03d} | "
            f"Loss {avg_loss:.4f} | Sup {avg_sup:.4f} | Unsup {avg_unsup:.4f} | "
            f"NegSup {avg_neg_sup:.4f} | NegUnsup {avg_neg_unsup:.4f} | "
            f"Best(Th={best_th_str}) Dice {best['dice']:.4f}, "
            f"Rec {best['recall']:.4f}, Prec {best['precision']:.4f}, AUC {best['auc']:.4f} | "
            f"Patience {early_stopping.counter}/{config.PATIENCE}"
        )

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            logging.info(f"🔄 SWA captured at epoch {epoch}")

            if epoch == total_epochs - 1:
                logging.info("🌟 Final SWA validation: updating BN statistics...")
                update_bn(dl_l, swa_model, device=device)

                swa_val_metrics = validate_metrics_full(swa_model.module, dl_val, device, thresh=0.50)
                logging.info(f"🏆 SWA final result: Dice {swa_val_metrics['dice']:.4f}")

                torch.save(
                    swa_model.module.state_dict(),
                    ckpt_dir / "best_model_SWA.pth"
                )

        if early_stopping.early_stop:
            logging.info("🛑 Early stopping triggered.")
            if epoch >= swa_start_epoch:
                logging.info("🌟 SWA validation triggered by early stopping.")
                update_bn(dl_l, swa_model, device=device)

                swa_val_metrics = validate_metrics_full(swa_model.module, dl_val, device, thresh=0.50)
                logging.info(f"🏆 SWA result: Dice {swa_val_metrics['dice']:.4f}")

                torch.save(
                    swa_model.module.state_dict(),
                    ckpt_dir / "best_model_SWA.pth"
                )
            break

        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                ckpt_dir / f"epoch_{epoch + 1}.pth"
            )

    tb_writer.close()

    # 最终 summary
    best_row = max(epoch_rows, key=lambda r: r["val_dice"])
    pd.DataFrame([best_row]).to_csv(
        out_root / "best_epoch_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    logging.info("=" * 90)
    logging.info(f"✅ Finished {mode_name}")
    logging.info(f"Best Dice = {best_row['val_dice']:.4f} at epoch {best_row['epoch']}, threshold {best_row['best_threshold']}")
    logging.info(f"Output = {out_root}")
    logging.info("=" * 90)


def main():
    args = parse_args()
    train_one_experiment(
        n_labeled=args.n_labeled,
        mode=args.mode,
        debug_epochs=args.debug_epochs
    )


if __name__ == "__main__":
    main()