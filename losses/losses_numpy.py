import numpy as np
import torch

def masked_mse_np(preds_np, labels_np, null_val=np.nan):
    if np.isnan(null_val):
        mask_np = ~np.isnan(labels_np)
    else:
        mask_np = (labels_np != null_val)
    mask_np = mask_np.astype(float)
    mask_np /= np.mean((mask_np))
    mask_np = np.where(np.isnan(mask_np), np.zeros_like(mask_np), mask_np)
    loss_np = (preds_np-labels_np)**2
    loss_np = loss_np * mask_np
    loss_np = np.where(np.isnan(loss_np), np.zeros_like(loss_np), loss_np)
    return np.mean(loss_np)

def masked_rmse_np(preds_np, labels_np, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds_np=preds_np, labels_np=labels_np, null_val=null_val))

def masked_mae_loss_np(y_pred_np, y_true_np):
    mask_np = (y_true_np != 0).float()
    mask_np /= mask_np.mean()
    loss_np = np.abs(y_pred_np - y_true_np)
    loss_np = loss_np * mask_np
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss_np[loss_np != loss_np] = 0
    return loss_np.mean()

def masked_mae_np(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.astype(float)
    mask /=  np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds-labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

def masked_mape_np(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.astype(float)
    mask /=  np.mean((mask))
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds-labels)/labels
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

def metric_np(pred, real):
    mae = masked_mae_np(pred,real,0.0).item()
    mape = masked_mape_np(pred,real,0.0).item()
    rmse = masked_rmse_np(pred,real,0.0).item()
    return mae,mape,rmse

if __name__ == "__main__":
    data1 = torch.randn(16, 207, 12)
    data2 = torch.randn(16, 207, 12)
    data1_np = data1.numpy()
    data2_np = data2.numpy()
    r1 = metric_np(data1_np, data2_np)