import torch
import numpy as np

# ============== RMSE ================= #
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


# ============== MAE ================= #
def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


# ============== MAPE ================== #
def masked_mape(preds, labels, null_val=np.nan):
    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    labels = torch.where(labels < 1e-2, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def metric_multi_part(pred, real, part=5):
    B, L_P, N = pred.shape
    assert B % part == 0
    par = int(B / part)
    mae_list = []
    mape_list = []
    rmse_list = []

    for i in range(1, part+1):
        start = (i-1)*par
        end = i * par
        preds_part = pred[start:end, ...]
        labels_part = real[start:end, ...]
        mae, mape, rmse = metric(preds_part, labels_part)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
    mae = sum(mae_list) / part
    mape = sum(mape_list) / part
    rmse = sum(rmse_list) / part
    return mae, mape, rmse


if __name__ == "__main__":
    preds = torch.randn(6850, 1296, 207)
    reals = torch.randn(6850, 1296, 207)
    mae1, mape1, rmse1 = metric(preds, reals)
    mae2, mape2, rmse2 = metric_multi_part(preds, reals)
