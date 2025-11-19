

import numpy as np
import pandas as pd
import os
import torch
import os

import os

import pandas as pd
from torch import nn

from src.basics import set_device
from src.callback.patch_mask import *
from src.callback.tracking import *
from src.callback.transforms import *
from src.learner import Learner
from src.metrics import *
from src.models.patchTST import PatchTST
from datautils import get_dls


import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset_pretrain', type=str, default='BasicMotions', help='dataset name')
parser.add_argument('--root_path', type=str, default='./data/UEA', help='root directory for datasets')
parser.add_argument('--data_path', type=str, default=None, help='optional dataset file name')
parser.add_argument('--val_ratio', type=float, default=0.2, help='validation split ratio')
parser.add_argument('--scale', type=int, default=0, help='apply z-normalization based on the training split')
parser.add_argument('--context_points', type=int, default=None, help='sequence length (optional override)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--aux_weight', type=float, default=0.01, help='weight for auxiliary load balancing loss')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='model variant identifier')


args = parser.parse_args()
print('args:', args)
args.save_path = os.path.join('saved_models', args.dset_pretrain, 'masked_patchtst', args.model_type)
os.makedirs(args.save_path, exist_ok=True)


def _prepare_runtime_config(dls):
    seq_len = args.context_points if args.context_points else dls.len
    args.context_points = seq_len
    args.save_pretrained_model = (
        f'patchtst_pretrained_cls_cw{seq_len}_patch{args.patch_len}_stride{args.stride}'
        f'_epochs-pretrain{args.n_epochs_pretrain}_mask{args.mask_ratio}_model{args.pretrained_model_id}'
    )
    return seq_len


# get available GPU devide
set_device()


def get_model(c_in, seq_len):
    num_patch = (max(seq_len, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    model = PatchTST(
        c_in=c_in,
        target_dim=args.patch_len,
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
        head_type='pretrain',
        res_attention=False,
    )
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    model = get_model(dls.vars, seq_len)
    loss_func = nn.HuberLoss()
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]

    learn = Learner(
        dls,
        model,
        loss_func,
        lr=args.lr,
        cbs=cbs,
        aux_weight=args.aux_weight,
    )
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    model = get_model(dls.vars, seq_len)
    loss_func = nn.HuberLoss()
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
        SaveModelCB(monitor='valid_loss', fname=args.save_pretrained_model, path=args.save_path),
    ]
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        aux_weight=args.aux_weight,
    )
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(
        os.path.join(args.save_path, args.save_pretrained_model + '_losses.csv'),
        float_format='%.6f',
        index=False,
    )


if __name__ == '__main__':
    
    args.dset = args.dset_pretrain
    suggested_lr = find_lr()
    # Pretrain
    pretrain_func(suggested_lr)
    print('pretraining completed')
    

