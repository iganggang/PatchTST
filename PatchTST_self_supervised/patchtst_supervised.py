

import argparse
import os

import numpy as np
import torch.nn as nn

from src.models.patchTST import PatchTST
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls


parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset', type=str, default='HandMovementDirection', help='UEA dataset name')
parser.add_argument('--root_path', type=str, default='./data/UEA', help='root directory of the dataset archive')
parser.add_argument('--context_points', type=int, default=512, help='sequence length (after padding/cropping)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio of the training split used for validation')
parser.add_argument('--normalize', type=int, default=1, help='if true, apply per-sample z-normalization')
parser.add_argument('--split_seed', type=int, default=42, help='random seed for the train/val split')
# Patch
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride between patches')
# RevIN
parser.add_argument('--revin', type=int, default=0, help='reversible instance normalization (input only)')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Transformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--aux_weight', type=float, default=0.01, help='weight for auxiliary load balancing loss')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='classification', help='sub-folder used for checkpoints')
# training
parser.add_argument('--is_train', type=int, default=1, help='training the model')


args = parser.parse_args()
print('args:', args)
args.save_model_name = (
    'patchtst_classification'
    + f"_{args.dset}_sl{args.context_points}_patch{args.patch_len}_stride{args.stride}_epochs{args.n_epochs}_model{args.model_id}"
)
args.save_path = f"saved_models/{args.dset}/patchtst_classification/{args.model_type}/"
os.makedirs(args.save_path, exist_ok=True)


def get_model(c_in, n_classes, args):
    """
    c_in: number of input variables
    """
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    model = PatchTST(
        c_in=c_in,
        target_dim=n_classes,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type='classification',
        res_attention=False,
    )
    return model


def _default_cbs(dls):
    cbs = []
    if args.revin:
        cbs.append(RevInCB(dls.vars, denorm=False))
    cbs.append(PatchCB(patch_len=args.patch_len, stride=args.stride))
    return cbs


def find_lr():
    dls = get_dls(args)
    model = get_model(dls.vars, dls.c, args)
    loss_func = nn.CrossEntropyLoss()
    learn = Learner(dls, model, loss_func, cbs=_default_cbs(dls), aux_weight=args.aux_weight)
    return learn.lr_finder()


def train_func(lr=args.lr):
    dls = get_dls(args)
    print('in out', dls.vars, dls.c, dls.len)

    model = get_model(dls.vars, dls.c, args)

    loss_func = nn.CrossEntropyLoss()

    cbs = _default_cbs(dls)
    monitor_metric = 'valid_accuracy' if dls.valid else 'train_loss'
    comp_fn = np.less if monitor_metric == 'train_loss' else np.greater
    cbs.append(
        SaveModelCB(
            monitor=monitor_metric,
            fname=args.save_model_name,
            path=args.save_path,
            comp=comp_fn,
        )
    )

    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[accuracy],
        aux_weight=args.aux_weight,
    )

    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    weight_path = args.save_path + args.save_model_name + '.pth'
    dls = get_dls(args)
    model = get_model(dls.vars, dls.c, args)
    learn = Learner(dls, model, cbs=_default_cbs(dls), aux_weight=args.aux_weight)
    out = learn.test(dls.test, weight_path=weight_path, scores=[accuracy])
    return out


if __name__ == '__main__':

    if args.is_train:   # training mode
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:   # testing mode
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
   
    print('----------- Complete! -----------')


