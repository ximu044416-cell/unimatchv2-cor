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

from resnet50_unet import ResNet50_UNet
from configs.config_2 import config
from data.dataset_2 import UniMatchDataset2, get_split_indices_2
from utils.losses_2 import (
    FocalTverskyLoss,
    BoundaryDoULoss,
    NegativePriorPenaltyLoss,
    get_neg_lambda
)


# =========================================================
# Override output directory for this ablation
# =========================================================
OUTPUT_DIR = ABLATION_ROOT / "logs"
CHECKPOINT_DIR = ABLATION_ROOT / "checkpoints"
TENSORBOARD_DIR = ABLATION_ROOT / "tensorboard_logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Optional: shorten only if you deliberately want quick debugging.
# Otherwise keep identical to final model.
TOTAL_EPOCHS = config.TOTAL_EPOCHS
PATIENCE = config.PATIENCE


# =========================================================
# Strategy helpers
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
    """
    Apply the same CutMix operation to image, segmentation mask, and negative-prior map.
    """
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
    total = {"dice": 0.0, "recall": 0.0, "precision": 0.0, "auc": 0.0}
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
def freeze_resnet_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_resnet_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = True


def get_resnet_param_groups(model, lr_encoder, lr_head, weight_decay):
    """
    ResNet-specific parameter groups.
    Head includes prior projection + decoder + final conv.
    Encoder uses a lower LR.
    """
    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if len(head_params) > 0:
        groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
            "target_lr": lr_head
        })

    if len(encoder_params) > 0:
        groups.append({
            "params": encoder_params,
            "lr": lr_encoder,
            "weight_decay": weight_decay,
            "target_lr": lr_encoder
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
# Main train
# =========================================================
def train():
    device = torch.device(config.DEVICE)
    amp_enabled = device.type == "cuda"

    tb_writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))

    logging.info("=" * 90)
    logging.info("🚀 ResNet50 baseline ablation started")
    logging.info("Backbone: ResNet50-UNet")
    logging.info("Prior input: enabled as 4th channel")
    logging.info("Negative-prior penalty: enabled")
    logging.info("Strict hold-out test folder is excluded from training")
    logging.info(f"Output dir: {OUTPUT_DIR}")
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

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = ResNet50_UNet(num_classes=config.NUM_CLASSES, pretrained=True).to(device)

    teacher_model = ResNet50_UNet(num_classes=config.NUM_CLASSES, pretrained=True).to(device)
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

    criterion_neg = NegativePriorPenaltyLoss(
        tau=config.NEG_TAU,
        eps=config.NEG_EPS,
        only_when_mask_exists=config.NEG_ONLY_WHEN_MASK_EXISTS
    )

    # Stage 1: freeze ResNet encoder
    freeze_resnet_encoder(model)

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

        # -------------------------------------------------
        # Stage scheduling
        # -------------------------------------------------
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
                logging.info(f"🔓 [Epoch {epoch}] Unfreezing ResNet50 encoder")
                unfreeze_resnet_encoder(model)

                optimizer = optim.AdamW(
                    get_resnet_param_groups(
                        model,
                        lr_encoder=config.LR_BACKBONE,
                        lr_head=config.LR_HEAD,
                        weight_decay=config.WEIGHT_DECAY
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
                    logging.info(f"🚀 [Epoch {epoch}] Warm-up finished; switched to CosineAnnealingLR")

        current_unsup_weight = get_current_unsup_weight(epoch)
        current_ema_alpha = get_ema_alpha(epoch)

        cosine_base_thresh = get_cosine_threshold(
            epoch,
            TOTAL_EPOCHS,
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
        )

        metrics_meter = {
            "loss": 0.0,
            "sup": 0.0,
            "unsup": 0.0,
            "neg_sup": 0.0,
            "neg_unsup": 0.0
        }

        pbar = tqdm(dl_l, total=len(dl_l), desc=f"ResNet50 Ep {epoch + 1}/{TOTAL_EPOCHS}")
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

            with autocast("cuda", enabled=amp_enabled):
                # supervised branch
                pred_l = model(img_l)
                probs_l = torch.softmax(pred_l, dim=1)

                loss_ce = criterion_ce(pred_l, mask_l)
                loss_tversky = criterion_tversky(probs_l, mask_l, current_epoch=epoch)
                loss_bdou = criterion_bdou(probs_l, mask_l)

                loss_sup = w_ce * loss_ce + w_ft * loss_tversky + w_bd * loss_bdou

                loss_neg_sup = torch.tensor(0.0, device=device)
                if config.USE_NEG_PRIOR_LOSS and config.USE_NEG_ON_LABELED and lambda_neg_sup > 0:
                    loss_neg_sup = criterion_neg(probs_l, neg_l)

                # unlabeled branch
                loss_unsup = torch.tensor(0.0, device=device)
                loss_neg_unsup = torch.tensor(0.0, device=device)

                if current_unsup_weight > 0:
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
                "Th": f"{current_thresh:.3f}",
                "EMA": f"{current_ema_alpha:.4f}"
            })

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------
        if epoch < 150:
            scheduler.step()
        elif 150 <= epoch < 165:
            pass
        else:
            scheduler.step()

        avg_loss = metrics_meter["loss"] / len(dl_l)
        avg_sup = metrics_meter["sup"] / len(dl_l)
        avg_unsup = metrics_meter["unsup"] / len(dl_l)
        avg_neg_sup = metrics_meter["neg_sup"] / len(dl_l)
        avg_neg_unsup = metrics_meter["neg_unsup"] / len(dl_l)

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

        # -------------------------------------------------
        # Validation at fixed internal thresholds
        # -------------------------------------------------
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
            f"NegSup {avg_neg_sup:.4f} | NegUnsup {avg_neg_unsup:.4f} | "
            f"Best(Th={best_th_str}) Dice {best['dice']:.4f}, "
            f"Rec {best['recall']:.4f}, Prec {best['precision']:.4f}, AUC {best['auc']:.4f} | "
            f"Patience {early_stopping.counter}/{early_stopping.patience}"
        )

        # -------------------------------------------------
        # SWA, normally not reached if early stopping occurs early
        # -------------------------------------------------
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            logging.info(f"🔄 SWA captured at epoch {epoch}")

            if epoch == TOTAL_EPOCHS - 1:
                logging.info("🌟 Final SWA validation: updating BN statistics...")
                update_bn(dl_l, swa_model, device=device)

                swa_val_metrics = validate_metrics_full(swa_model.module, dl_val, device, thresh=0.50)
                logging.info(
                    f"🏆 SWA final result: Dice {swa_val_metrics['dice']:.4f}, "
                    f"Recall {swa_val_metrics['recall']:.4f}, Precision {swa_val_metrics['precision']:.4f}"
                )

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
    logging.info("✅ ResNet50 baseline ablation finished.")


if __name__ == "__main__":
    train()