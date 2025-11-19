
import torch
from torch import Tensor
import torch.nn.functional as F

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(y_true, y_pred)


def accuracy(y_true: Tensor, y_pred: Tensor):
    if y_pred.ndim > y_true.ndim:
        y_pred = y_pred.argmax(dim=-1)
    return (y_pred == y_true).float().mean()
