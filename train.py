import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
import datetime
import math
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

# ================= 导入自定义模块 =================
from configs import config
from models.dinov2_unet import DINOUNet
from data.dataset import UniMatchDataset, get_split_indices
from utils.losses import FocalTverskyLoss, BoundaryDoULoss


# ================= 1. 策略控制 =================
def get_current_unsup_weight(epoch):
    warmup_epochs = 50
    max_weight = config.UNLABELED_LOSS_WEIGHT
    if epoch < warmup_epochs:
        return max_weight * (epoch / warmup_epochs)
    return max_weight


def get_ema_alpha(epoch):
    # 🔥 [修复] 收紧 EMA 动量门槛，保护教师模型
    base_alpha = 0.99
    target_alpha = 0.999
    warmup_epochs = 150
    if epoch < warmup_epochs:
        return base_alpha + (target_alpha - base_alpha) * (epoch / warmup_epochs)
    return target_alpha


# ================= 2. CutMix 核心函数 =================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix(data, targets, probability=0.5, beta=1.0):
    if np.random.rand() > probability:
        return data, targets

    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(data.size()[0]).to(data.device)

    target_b = targets[rand_index]
    data_b = data[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = data_b[:, :, bbx1:bbx2, bby1:bby2]

    if targets.ndim == 3:
        targets[:, bbx1:bbx2, bby1:bby2] = target_b[:, bbx1:bbx2, bby1:bby2]
    elif targets.ndim == 4:
        targets[:, :, bbx1:bbx2, bby1:bby2] = target_b[:, :, bbx1:bbx2, bby1:bby2]

    return data, targets


# ================= 3. 辅助工具 =================
def sliding_window_inference_tta(model, image, window_size=(518, 518), overlap=0.5):
    b, c, h, w = image.shape
    tile_h, tile_w = window_size
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))

    def get_gaussian(size, sigma_scale=1.0 / 8):
        tmp = np.zeros(size)
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
        pred += torch.flip(pred_h, [3])
        pred_v = torch.softmax(model(torch.flip(patch, [2])), dim=1)
        pred += torch.flip(pred_v, [2])
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

    return output_sum / weight_sum


def calculate_metrics_manual(pred_mask, gt_mask, pred_prob):
    smooth = 1e-5
    pred = pred_mask.float()
    gt = gt_mask.float()

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()

    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)

    gt_flat = gt.view(-1).cpu().numpy()
    prob_flat = pred_prob.view(-1).cpu().numpy()
    if len(np.unique(gt_flat)) == 1:
        auc = np.nan
    else:
        auc = roc_auc_score(gt_flat, prob_flat)

    return dice.item(), recall.item(), precision.item(), auc


def get_llrd_params(model, lr_backbone, lr_head, weight_decay, decay_rate=0.90):
    groups = []
    head_params = []
    for name, param in model.named_parameters():
        if 'encoder' not in name and param.requires_grad:
            head_params.append(param)
    groups.append({'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay})

    layer_params = [[] for _ in range(12)]
    embed_params = []
    norm_params = []
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad: continue
        if "blocks" in name:
            try:
                parts = name.split("blocks.")
                layer_id = int(parts[1].split(".")[0])
                if layer_id < 12:
                    layer_params[layer_id].append(param)
                else:
                    layer_params[11].append(param)
            except:
                embed_params.append(param)
        elif "norm" in name and "blocks" not in name:
            norm_params.append(param)
        else:
            embed_params.append(param)
    if len(norm_params) > 0: groups.append({'params': norm_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
    for i in range(11, -1, -1):
        if len(layer_params[i]) > 0:
            scale = decay_rate ** (11 - i)
            groups.append({'params': layer_params[i], 'lr': lr_backbone * scale, 'weight_decay': weight_decay})
    if len(embed_params) > 0:
        groups.append({'params': embed_params, 'lr': lr_backbone * (decay_rate ** 12), 'weight_decay': weight_decay})

    # 🔥 [终极修复 1] 将计算好的目标 LR 作为 'target_lr' 写入字典！杜绝后续索引反推越界！
    for group in groups:
        group['target_lr'] = group['lr']

    return groups


class EarlyStopping:
    def __init__(self, patience=20, save_path='best_model.pth'):
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
            logging_func(f"💾 New Best Saved! (Dice: {self.best_dice:.4f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


@torch.no_grad()
def update_ema(student, teacher, alpha=0.999):
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.mul_(alpha).add_(param, alpha=1 - alpha)


def validate_metrics_full(model, dataloader, device, thresh=0.70):
    model.eval()
    total = {'dice': 0.0, 'recall': 0.0, 'precision': 0.0, 'auc': 0.0}
    count = 0
    valid_auc_count = 0
    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc=f"Val(Th={thresh})", leave=False):
            img = img.to(device)
            mask = mask.to(device)

            with autocast('cuda'):
                prob_map = sliding_window_inference_tta(model, img, window_size=(518, 518), overlap=0.5)

            pred_prob = prob_map[:, 1, :, :]
            pred_mask = (pred_prob > thresh).float()

            d, r, p, auc = calculate_metrics_manual(pred_mask, mask, pred_prob)
            total['dice'] += d
            total['recall'] += r
            total['precision'] += p

            if not np.isnan(auc):
                total['auc'] += auc
                valid_auc_count += 1
            count += 1

    avg_auc = total['auc'] / valid_auc_count if valid_auc_count > 0 else 0.0
    return {
        'dice': total['dice'] / count,
        'recall': total['recall'] / count,
        'precision': total['precision'] / count,
        'auc': avg_auc
    }


# ================= 4. Main =================
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(config.OUTPUT_DIR, 'train_log.txt'), level=logging.INFO,
                    format='[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S'))
logging.getLogger('').addHandler(console)


def train():
    device = torch.device(config.DEVICE)
    tb_writer = SummaryWriter(log_dir=os.path.join(config.OUTPUT_DIR, 'tensorboard_logs'))

    logging.info(f"🚀 启动终极之战: Base 巨兽 | 纯净数据集 | 动静分离架构")
    logging.info(f"⚙️ Config: 15轮极缓预热 | EMA 0.99 | LLRD {config.LLRD_DECAY} | 梯度累加保护")

    # 🔥 [终极修复 2] 梯度累加，防止 Base 模型 OOM
    ACCUM_STEPS = 2
    # 物理 batch size 减半（如 config 里是 4，这里物理就是 2），跑 2 步合并 1 次梯度
    physical_batch_size = max(1, config.BATCH_SIZE // ACCUM_STEPS)

    train_l_files, train_u_files, val_files = get_split_indices()

    dl_l = DataLoader(UniMatchDataset(train_l_files, mode='labeled'),
                      batch_size=physical_batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True,
                      drop_last=True)

    dl_u = DataLoader(UniMatchDataset(train_u_files, mode='unlabeled'),
                      batch_size=physical_batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True,
                      drop_last=True)

    dl_val = DataLoader(UniMatchDataset(val_files, mode='val'),
                        batch_size=1, shuffle=False, num_workers=2)

    logging.info(f"📥 Loading Pretrained DINOv2 Weights: {config.PRETRAINED_PATH}")
    model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)

    teacher_model = DINOUNet(local_path=config.PRETRAINED_PATH, num_classes=config.NUM_CLASSES).to(device)
    teacher_model.load_state_dict(model.state_dict())

    for param in teacher_model.parameters(): param.requires_grad = False
    teacher_model.eval()

    scaler = GradScaler('cuda')
    best_model_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    early_stopping = EarlyStopping(patience=config.PATIENCE, save_path=best_model_path)

    criterion_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_tversky = FocalTverskyLoss(n_classes=config.NUM_CLASSES, alpha=0.3, beta=0.7, gamma=1.33,
                                         dynamic_beta=False)
    criterion_bdou = BoundaryDoULoss(kernel_size=3)
    criterion_u_ce = nn.CrossEntropyLoss(reduction='none')

    for param in model.encoder.parameters(): param.requires_grad = False

    # 第一阶段：用 config.LR_HEAD 给解码器极其充足的火力
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_HEAD, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TOTAL_EPOCHS, eta_min=1e-8)

    iter_u = iter(dl_u)
    global_prob_val = torch.ones(1, device=device) * 0.5
    current_stage = 1

    for epoch in range(config.TOTAL_EPOCHS):

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
                logging.info(f"🔓 [Epoch {epoch}] 除颤启动：释放 Base Backbone！启动 15 轮线性极缓预热！")
                for param in model.encoder.parameters(): param.requires_grad = True

                param_groups = get_llrd_params(model, config.LR_BACKBONE, config.LR_HEAD, config.WEIGHT_DECAY,
                                               decay_rate=config.LLRD_DECAY)
                optimizer = optim.AdamW(param_groups)

                early_stopping.best_dice = 0.0
                early_stopping.counter = 0
                logging.info("♻️ EarlyStopping 已清零！给 Base 留出充足打磨空间！")
                current_stage = 2

            model.train()
            teacher_model.eval()

            if epoch < 300:
                w_ce, w_ft, w_bd = 0.2, 0.8, 0.5
            else:
                w_ce, w_ft, w_bd = 0.1, 0.5, 1.0  # B-DoU 温和化

            # 🔥 [终极修复 3] 预热逻辑提拔到 Batch 循环之前！绝不在循环中途乱降学习率！
            if 150 <= epoch < 165:
                warmup_ratio = (epoch - 150 + 1) / 15.0
                for param_group in optimizer.param_groups:
                    # 安全读取我们挂在字典里的 target_lr
                    target_lr = param_group.get('target_lr', param_group['lr'])
                    param_group['lr'] = 1e-8 + warmup_ratio * (target_lr - 1e-8)

                if epoch == 164:
                    # 预热的最后一天，挂载余弦退火，移交指挥权
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.TOTAL_EPOCHS - 165),
                                                                     eta_min=1e-8)
                    logging.info(f"🚀 [Epoch {epoch}] 15 轮极缓预热结束！平稳移交 CosineAnnealingLR！")

        current_unsup_weight = get_current_unsup_weight(epoch)
        current_ema_alpha = get_ema_alpha(epoch)

        # 紧缩双重门槛
        current_thresh = max(global_prob_val.item(), 0.60)

        metrics_meter = {'loss': 0, 'sup': 0, 'unsup': 0}
        pbar = tqdm(dl_l, total=len(dl_l), desc=f"Ep {epoch + 1}/{config.TOTAL_EPOCHS}")

        # 梯度累加初始清零
        optimizer.zero_grad()

        for step, batch_l in enumerate(pbar):
            img_l, mask_l = batch_l
            img_l, mask_l = img_l.to(device), mask_l.to(device)

            img_l, mask_l = apply_cutmix(img_l, mask_l, probability=0.5, beta=1.0)

            try:
                batch_u = next(iter_u)
            except StopIteration:
                iter_u = iter(dl_u)
                batch_u = next(iter_u)
            img_u_w, img_u_s1, img_u_s2 = batch_u
            img_u_w, img_u_s1, img_u_s2 = img_u_w.to(device), img_u_s1.to(device), img_u_s2.to(device)

            with autocast('cuda'):
                pred_l = model(img_l)
                probs_l = torch.softmax(pred_l, dim=1)

                loss_ce = criterion_ce(pred_l, mask_l)
                loss_tversky = criterion_tversky(probs_l, mask_l)
                loss_bdou = criterion_bdou(probs_l, mask_l)

                loss_sup = w_ce * loss_ce + w_ft * loss_tversky + w_bd * loss_bdou

                loss_unsup = torch.tensor(0.0, device=device)
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

                    loss_u_s1 = (criterion_u_ce(pred_u_s1, pseudo_label) * mask_conf).sum() / (mask_conf.sum() + 1e-5)
                    loss_u_s2 = (criterion_u_ce(pred_u_s2, pseudo_label) * mask_conf).sum() / (mask_conf.sum() + 1e-5)
                    loss_unsup = (loss_u_s1 + loss_u_s2) / 2.0

                total_loss = loss_sup + current_unsup_weight * loss_unsup
                # 累加缩放
                loss_to_backward = total_loss / ACCUM_STEPS

            # 反向传播 (缩放后)
            scaler.scale(loss_to_backward).backward()

            # 🔥 梯度累加步数到达，或者达到整个 Epoch 的最后一个 Step
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(dl_l):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                update_ema(model, teacher_model, alpha=current_ema_alpha)

            metrics_meter['loss'] += total_loss.item()
            metrics_meter['sup'] += loss_sup.item()
            metrics_meter['unsup'] += loss_unsup.item()

            pbar.set_postfix({
                'L': f"{total_loss.item():.3f}",
                'U_W': f"{current_unsup_weight:.3f}",
                'Th': f"{current_thresh:.2f}"
            })

        # 🔥 步进逻辑精细分离
        if epoch < 150:
            scheduler.step()
        elif 150 <= epoch < 165:
            pass  # 处于手动预热期，绝对不调用 scheduler
        else:
            scheduler.step()  # 165轮及以后，正常余弦退火

        avg_loss = metrics_meter['loss'] / len(dl_l)
        avg_sup = metrics_meter['sup'] / len(dl_l)
        avg_unsup = metrics_meter['unsup'] / len(dl_l)

        tb_writer.add_scalar('Train/Loss_Total', avg_loss, epoch)
        tb_writer.add_scalar('Train/Loss_Sup', avg_sup, epoch)
        tb_writer.add_scalar('Train/Loss_Unsup', avg_unsup, epoch)

        tb_writer.add_scalar('Sys/Unsup_Weight', current_unsup_weight, epoch)
        tb_writer.add_scalar('Sys/EMA_Alpha', current_ema_alpha, epoch)
        tb_writer.add_scalar('Sys/Dynamic_Thresh', current_thresh, epoch)

        val_metrics_50 = validate_metrics_full(model, dl_val, device, thresh=0.50)
        val_metrics_60 = validate_metrics_full(model, dl_val, device, thresh=0.60)
        val_metrics_65 = validate_metrics_full(model, dl_val, device, thresh=0.65)

        val_results = {
            "0.50": val_metrics_50,
            "0.60": val_metrics_60,
            "0.65": val_metrics_65
        }
        best_th_str = max(val_results, key=lambda k: val_results[k]['dice'])
        best = val_results[best_th_str]

        tb_writer.add_scalar('Val_0.50/Dice', val_metrics_50['dice'], epoch)
        tb_writer.add_scalar('Val_0.50/Recall', val_metrics_50['recall'], epoch)
        tb_writer.add_scalar('Val_0.50/Precision', val_metrics_50['precision'], epoch)
        tb_writer.add_scalar('Val_0.50/AUC', val_metrics_50['auc'], epoch)

        tb_writer.add_scalar('Val_0.60/Dice', val_metrics_60['dice'], epoch)
        tb_writer.add_scalar('Val_0.60/Recall', val_metrics_60['recall'], epoch)
        tb_writer.add_scalar('Val_0.60/Precision', val_metrics_60['precision'], epoch)
        tb_writer.add_scalar('Val_0.60/AUC', val_metrics_60['auc'], epoch)

        tb_writer.add_scalar('Val_0.65/Dice', val_metrics_65['dice'], epoch)
        tb_writer.add_scalar('Val_0.65/Recall', val_metrics_65['recall'], epoch)
        tb_writer.add_scalar('Val_0.65/Precision', val_metrics_65['precision'], epoch)
        tb_writer.add_scalar('Val_0.65/AUC', val_metrics_65['auc'], epoch)

        tb_writer.add_scalar('Val_Best/Dice', best['dice'], epoch)
        tb_writer.add_scalar('Val_Best/Recall', best['recall'], epoch)
        tb_writer.add_scalar('Val_Best/Precision', best['precision'], epoch)
        tb_writer.add_scalar('Val_Best/AUC', best['auc'], epoch)

        early_stopping(best['dice'], model, logging.info)
        patience_status = f"{early_stopping.counter}/{early_stopping.patience}"

        logging.info(
            f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} | "
            f"Best(Th={best_th_str}): Dice {best['dice']:.4f}, Rec {best['recall']:.4f}, Prec {best['precision']:.4f}, AUC {best['auc']:.4f} | "
            f"Patience: {patience_status}"
        )

        if early_stopping.early_stop:
            logging.info("🛑 Early stopping triggered!")
            break

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, f"epoch_{epoch + 1}.pth"))

    tb_writer.close()


if __name__ == "__main__":
    train()