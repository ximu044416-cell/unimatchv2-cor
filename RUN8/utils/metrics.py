import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(preds, masks, smooth=1e-5):
    """
    计算二分类的核心指标：Dice, Recall, Precision, AUC
    """
    if preds.shape[1] > 1:
        probs = torch.softmax(preds, dim=1)[:, 1, :, :]  # 提取预测为正类的概率 (B, H, W)
        pred_labels = torch.argmax(preds, dim=1)
    else:
        probs = torch.sigmoid(preds)
        pred_labels = (probs > 0.5).long()

    batch_size = preds.shape[0]

    dice_list = []
    recall_list = []
    precision_list = []
    auc_list = []

    for i in range(batch_size):
        pred_flat = pred_labels[i].view(-1)
        mask_flat = masks[i].view(-1)
        prob_flat = probs[i].view(-1).cpu().detach().numpy()  # AUC需要概率值，不能用0/1标签
        mask_flat_np = mask_flat.cpu().numpy()

        TP = (pred_flat * mask_flat).sum().item()
        FP = (pred_flat * (1 - mask_flat)).sum().item()
        FN = ((1 - pred_flat) * mask_flat).sum().item()

        # Dice, Recall, Precision 计算保持不变
        dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)

        if (TP + FN) == 0:
            recall = 1.0
        else:
            recall = (TP + smooth) / (TP + FN + smooth)

        if (TP + FP) == 0:
            precision = 1.0 if (TP + FN) == 0 else 0.0
        else:
            precision = (TP + smooth) / (TP + FP + smooth)

        # --- D. AUC 计算 ---
        # 如果整张图全是背景或者全是病灶，AUC 无法计算，设为 NaN 并在求均值时忽略
        if len(np.unique(mask_flat_np)) == 1:
            auc = np.nan
        else:
            auc = roc_auc_score(mask_flat_np, prob_flat)

        dice_list.append(dice)
        recall_list.append(recall)
        precision_list.append(precision)
        auc_list.append(auc)

    return {
        "dice": np.mean(dice_list),
        "recall": np.mean(recall_list),
        "precision": np.mean(precision_list),
        "auc": np.nanmean(auc_list)  # np.nanmean 会自动忽略那些因为只有背景而导致无法计算AUC的切片
    }