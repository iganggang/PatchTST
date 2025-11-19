
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
    """Classification accuracy."""
    if y_true.ndim > 1:
        y_true = y_true.argmax(dim=-1)
    preds = y_pred.argmax(dim=-1)
    return (preds == y_true).float().mean()
