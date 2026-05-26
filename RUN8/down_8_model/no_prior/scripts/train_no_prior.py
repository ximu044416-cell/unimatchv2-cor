import os
import sys
import math
import logging
from pathlib import Path

import numpy as np
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
# Path setup
# =========================================================
THIS_DIR = Path(__file__).resolve().parent
ABLATION_ROOT = THIS_DIR.parent
RUN8_ROOT = Path(r"F:\cor\RUN8")

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(RUN8_ROOT))

from dinov2_unet_no_prior import DINOUNetNoPrior
from configs.config_2 import config
from data.dataset_2 import UniMatchDataset2, get_split_indices_2
from utils.losses_2 import FocalTverskyLoss, BoundaryDoULoss


# =========================================================
# Output directories
# =========================================================
OUTPUT_DIR = ABLATION_ROOT / "logs"
CHECKPOINT_DIR = ABLATION_ROOT / "checkpoints"
TENSORBOARD_DIR = ABLATION_ROOT / "tensorboard_logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_EPOCHS = config.TOTAL_EPOCHS
PATIENCE = config.PATIENCE


# =========================================================
# Schedule helpers
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
# Channel helper
# =========================================================
def to_no_prior_input(x):
    """
    dataset_2 returns 4-channel tensors:
    [MRI channel 1, MRI channel 2, MRI channel 3, MedSAM prior]

    no_prior ablation removes the 4th prior channel and keeps only MRI channels.
    """
    if x.shape[1] >= 3:
        return x[:, :3, :, :]
    raise ValueError(f"Expected at least 3 channels, but got {x.shape[1]}")


# =========================================================
# CutMix
# =========================================================
def rand_bbox(size, lam):
    w = size[2]
    h = size[3]

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix(data, target_main, probability=0.5, beta=1.0):
    if np.random.rand() > probability:
        return data, target_main

    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(data.size(0)).to(data.device)

    data_b = data[rand_index]
    main_b = target_main[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = data_b[:, :, bbx1:bbx2, bby1:bby2]

    if target_main.ndim == 3:
        target_main[:, bbx1:bbx2, bby1:bby2] = main_b[:, bbx1:bbx2, bby1:bby2]
    elif target_main.ndim == 4:
        target_main[:, :, bbx1:bbx2, bby1:bby2] = main_b[:, :, bbx1:bbx2, bby1:bby2]

    return data, target_main


# =========================================================
# Inference and metrics
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
            img = to_no_prior_input(img)

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
# Logging
# =========================================================
logging.basicConfig(
    filename=str(OUTPUT_DIR / "train_log.txt"),
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%m-%d %H:%M:%S"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S"))
logging.getLogger("").addHandler(console)


# =========================================================
# Main training
# =========================================================
def train():
    device = torch.device(config.DEVICE)
    amp_enabled = device.type == "cuda"

    tb_writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))

    logging.info("=" * 90)
    logging.info("🚀 no_prior ablation started")
    logging.info("Backbone: DINOv2")
    logging.info("Prior input: DISABLED")
    logging.info("Input channels: 3 MRI channels only")
    logging.info("Negative-prior penalty: DISABLED")
    logging.info("Strict hold-out test folder is excluded from training")
    logging.info(f"Output dir: {ABLATION_ROOT}")
    logging.info("=" * 90)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    ACCUM_STEPS = 2
    physical_batch_size = max(1, config.BATCH_SIZE // ACCUM_STEPS)

    train_l_files, train_u_files, val_files = get_split_indices_2()

    dl_l = DataLoader(
        UniMatchDataset2(train_l_files, mode="labeled"),
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    dl_u = DataLoader(
        UniMatchDataset2(train_u_files, mode="unlabeled"),
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    dl_val = DataLoader(
        UniMatchDataset2(val_files, mode="val"),
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    model = DINOUNetNoPrior(
        local_path=config.PRETRAINED_PATH,
        num_classes=config.NUM_CLASSES
    ).to(device)

    teacher_model = DINOUNetNoPrior(
        local_path=config.PRETRAINED_PATH,
        num_classes=config.NUM_CLASSES
    ).to(device)

    teacher_model.load_state_dict(model.state_dict())

    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_model.eval()

    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_model_path = CHECKPOINT_DIR / "best_model.pth"
    early_stopping = EarlyStopping(patience=PATIENCE, save_path=str(best_model_path))

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

    # Stage 1: freeze encoder
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
        T_max=TOTAL_EPOCHS,
        eta_min=min_lr
    )

    iter_u = iter(dl_u)
    global_prob_val = torch.ones(1, device=device) * 0.5
    current_stage = 1

    swa_model = AveragedModel(model)
    swa_start_epoch = TOTAL_EPOCHS - 50

    for epoch in range(TOTAL_EPOCHS):
        if epoch < 150:
            model.train()
            model.encoder.eval()
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
                        T_max=max(1, TOTAL_EPOCHS - 165),
                        eta_min=config.LR_BACKBONE * 0.55
                    )
                    logging.info("🚀 Warm-up finished; switched to CosineAnnealingLR")

        current_unsup_weight = get_current_unsup_weight(epoch)
        current_ema_alpha = get_ema_alpha(epoch)

        cosine_base_thresh = get_cosine_threshold(
            epoch,
            TOTAL_EPOCHS,
            max_thresh=0.85,
            min_thresh=0.55
        )

        current_thresh = max(global_prob_val.item(), cosine_base_thresh)

        metrics_meter = {
            "loss": 0.0,
            "sup": 0.0,
            "unsup": 0.0
        }

        pbar = tqdm(dl_l, total=len(dl_l), desc=f"NoPrior Ep {epoch + 1}/{TOTAL_EPOCHS}")
        optimizer.zero_grad()

        for step, batch_l in enumerate(pbar):
            img_l, mask_l, _neg_l = batch_l

            img_l = img_l.to(device)
            img_l = to_no_prior_input(img_l)

            mask_l = mask_l.to(device)

            img_l, mask_l = apply_cutmix(
                img_l,
                mask_l,
                probability=0.5,
                beta=1.0
            )

            try:
                batch_u = next(iter_u)
            except StopIteration:
                iter_u = iter(dl_u)
                batch_u = next(iter_u)

            img_u_w, img_u_s1, img_u_s2, _neg_u = batch_u

            img_u_w = to_no_prior_input(img_u_w.to(device))
            img_u_s1 = to_no_prior_input(img_u_s1.to(device))
            img_u_s2 = to_no_prior_input(img_u_s2.to(device))

            with autocast("cuda", enabled=amp_enabled):
                # supervised branch
                pred_l = model(img_l)
                probs_l = torch.softmax(pred_l, dim=1)

                loss_ce = criterion_ce(pred_l, mask_l)
                loss_tversky = criterion_tversky(probs_l, mask_l, current_epoch=epoch)
                loss_bdou = criterion_bdou(probs_l, mask_l)

                loss_sup = w_ce * loss_ce + w_ft * loss_tversky + w_bd * loss_bdou

                # unlabeled branch
                loss_unsup = torch.tensor(0.0, device=device)

                if current_unsup_weight > 0:
                    with torch.no_grad():
                        pred_u_w = teacher_model(img_u_w)
                        probs_u_w = torch.softmax(pred_u_w, dim=1)

                        max_probs, pseudo_label = torch.max(probs_u_w, dim=1)

                        prob_edema = probs_u_w[:, 1, :, :]
                        mask_edema = pseudo_label == 1

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

                total_loss = loss_sup + current_unsup_weight * loss_unsup
                loss_to_backward = total_loss / ACCUM_STEPS

            scaler.scale(loss_to_backward).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(dl_l):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                update_ema(model, teacher_model, alpha=current_ema_alpha)

            metrics_meter["loss"] += float(total_loss.item())
            metrics_meter["sup"] += float(loss_sup.item())
            metrics_meter["unsup"] += float(loss_unsup.item())

            pbar.set_postfix({
                "L": f"{total_loss.item():.3f}",
                "U_W": f"{current_unsup_weight:.3f}",
                "Prior": "OFF",
                "Neg": "OFF",
                "Th": f"{current_thresh:.3f}",
                "EMA": f"{current_ema_alpha:.4f}"
            })

        # scheduler
        if epoch < 150:
            scheduler.step()
        elif 150 <= epoch < 165:
            pass
        else:
            scheduler.step()

        avg_loss = metrics_meter["loss"] / max(len(dl_l), 1)
        avg_sup = metrics_meter["sup"] / max(len(dl_l), 1)
        avg_unsup = metrics_meter["unsup"] / max(len(dl_l), 1)

        tb_writer.add_scalar("Train/Loss_Total", avg_loss, epoch)
        tb_writer.add_scalar("Train/Loss_Sup", avg_sup, epoch)
        tb_writer.add_scalar("Train/Loss_Unsup", avg_unsup, epoch)
        tb_writer.add_scalar("Sys/Unsup_Weight", current_unsup_weight, epoch)
        tb_writer.add_scalar("Sys/EMA_Alpha", current_ema_alpha, epoch)
        tb_writer.add_scalar("Sys/Dynamic_Thresh", current_thresh, epoch)

        # validation
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
        tb_writer.add_scalar("Val_Best/Recall", best["recall"], epoch)
        tb_writer.add_scalar("Val_Best/Precision", best["precision"], epoch)
        tb_writer.add_scalar("Val_Best/AUC", best["auc"], epoch)

        early_stopping(best["dice"], model, logging.info)

        logging.info(
            f"Epoch {epoch + 1:03d} | "
            f"Loss {avg_loss:.4f} | Sup {avg_sup:.4f} | Unsup {avg_unsup:.4f} | "
            f"Best(Th={best_th_str}) Dice {best['dice']:.4f}, "
            f"Rec {best['recall']:.4f}, Prec {best['precision']:.4f}, AUC {best['auc']:.4f} | "
            f"Patience {early_stopping.counter}/{PATIENCE}"
        )

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            logging.info(f"🔄 SWA captured at epoch {epoch}")

            if epoch == TOTAL_EPOCHS - 1:
                logging.info("🌟 Final SWA validation: updating BN statistics...")
                update_bn(dl_l, swa_model, device=device)

                swa_val_metrics = validate_metrics_full(swa_model.module, dl_val, device, thresh=0.50)
                logging.info(f"🏆 SWA final result: Dice {swa_val_metrics['dice']:.4f}")

                torch.save(
                    swa_model.module.state_dict(),
                    CHECKPOINT_DIR / "best_model_SWA.pth"
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
                    CHECKPOINT_DIR / "best_model_SWA.pth"
                )

            break

        if (epoch + 1) % 20 == 0:
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / f"epoch_{epoch + 1}.pth"
            )

    tb_writer.close()
    logging.info("✅ no_prior ablation finished.")


if __name__ == "__main__":
    train()