import os

import torch.nn as nn

from src.callback.core import *
from src.callback.patch_mask import *
from src.callback.scheduler import *
from src.callback.tracking import *
from src.callback.transforms import *
from src.learner import Learner
from src.metrics import accuracy
from src.models.patchTST import PatchTST
from datautils import get_dls

import argparse


parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--dset', type=str, default='BasicMotions', help='dataset name')
parser.add_argument('--root_path', type=str, default='./data/UEA', help='root directory for datasets')
parser.add_argument('--data_path', type=str, default=None, help='optional dataset file name')
parser.add_argument('--val_ratio', type=float, default=0.2, help='validation split ratio')
parser.add_argument('--scale', type=int, default=0, help='apply z-normalization based on the training split')
parser.add_argument('--context_points', type=int, default=None, help='optional override of sequence length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')

# Patch
parser.add_argument('--patch_len', type=int, default=32, help='patch length')
parser.add_argument('--stride', type=int, default=16, help='stride between patches')

# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Transformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')

# Optimisation
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--aux_weight', type=float, default=0.01, help='weight for auxiliary load balancing loss')

# Saving and training control
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='model variant identifier')
parser.add_argument('--is_train', type=int, default=1, help='training flag (1 for training, 0 for evaluation)')


args = parser.parse_args()
print('args:', args)

args.save_path = os.path.join('saved_models', args.dset, 'patchtst_supervised', args.model_type)
os.makedirs(args.save_path, exist_ok=True)


def _prepare_runtime_config(dls):
    seq_len = args.context_points if args.context_points else dls.len
    args.context_points = seq_len
    args.save_model_name = (
        f'patchtst_supervised_cls_cw{seq_len}_patch{args.patch_len}_stride{args.stride}'
        f'_epochs{args.n_epochs}_model{args.model_id}'
    )
    return seq_len


def get_model(c_in, num_classes, seq_len):
    num_patch = (max(seq_len, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    model = PatchTST(
        c_in=c_in,
        target_dim=num_classes,
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


def find_lr():
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    model = get_model(dls.vars, dls.c, seq_len)
    loss_func = nn.CrossEntropyLoss()
    cbs = [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, loss_func, cbs=cbs, aux_weight=args.aux_weight)
    return learn.lr_finder()


def train_func(lr=args.lr):
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    print('inputs', dls.vars, 'classes', dls.c, 'seq_len', seq_len)

    model = get_model(dls.vars, dls.c, seq_len)
    loss_func = nn.CrossEntropyLoss()
    cbs = [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_accuracy', fname=args.save_model_name, path=args.save_path),
    ]

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
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    weight_path = os.path.join(args.save_path, args.save_model_name + '.pth')

    model = get_model(dls.vars, dls.c, seq_len)
    cbs = [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, cbs=cbs, aux_weight=args.aux_weight)
    out = learn.test(dls.test, weight_path=weight_path, scores=[accuracy])
    return out


if __name__ == '__main__':
    if args.is_train:
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:
        out = test_func()
        print('score:', out[2])
        print('logits shape:', out[0].shape)

    print('----------- Complete! -----------')
