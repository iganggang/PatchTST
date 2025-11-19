import os

import pandas as pd
import torch
import torch.nn as nn

from src.basics import set_device
from src.callback.core import *
from src.callback.patch_mask import *
from src.callback.tracking import *
from src.callback.transforms import *
from src.learner import Learner, transfer_weights
from src.metrics import accuracy
from src.models.patchTST import PatchTST
from datautils import get_dls

import argparse


parser = argparse.ArgumentParser()

# Training modes
parser.add_argument('--is_finetune', type=int, default=0, help='perform full finetuning')
parser.add_argument('--is_linear_probe', type=int, default=0, help='perform linear probing (freeze backbone)')

# Dataset configuration
parser.add_argument('--dset_finetune', type=str, default='BasicMotions', help='dataset name for finetuning')
parser.add_argument('--root_path', type=str, default='./data/UEA', help='root directory for datasets')
parser.add_argument('--data_path', type=str, default=None, help='optional dataset file name')
parser.add_argument('--val_ratio', type=float, default=0.2, help='validation split ratio')
parser.add_argument('--scale', type=int, default=0, help='apply z-normalization based on the training split')
parser.add_argument('--context_points', type=int, default=None, help='optional override of sequence length')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of DataLoader workers')

# Patch configuration
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patches')

# Model configuration
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Transformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')

# Optimisation
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--aux_weight', type=float, default=0.01, help='weight for auxiliary load balancing loss')

# Pretrained model
parser.add_argument('--pretrained_model', type=str, default=None, help='path to pretrained backbone weights (.pth)')

# Saving configuration
parser.add_argument('--finetuned_model_id', type=int, default=1, help='identifier for saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='model variant identifier')


args = parser.parse_args()
print('args:', args)

args.save_path = os.path.join('saved_models', args.dset_finetune, 'masked_patchtst', args.model_type)
os.makedirs(args.save_path, exist_ok=True)


def _prepare_runtime_config(dls):
    seq_len = args.context_points if args.context_points else dls.len
    args.context_points = seq_len
    suffix = (
        f'_cw{seq_len}_patch{args.patch_len}_stride{args.stride}'
        f'_epochs{args.n_epochs_finetune}_model{args.finetuned_model_id}'
    )
    if args.is_linear_probe:
        prefix = f'{args.dset_finetune}_patchtst_linear-probe_cls'
    else:
        prefix = f'{args.dset_finetune}_patchtst_finetuned_cls'
    args.save_finetuned_model = prefix + suffix
    return seq_len


def get_model(c_in, num_classes, seq_len, head_type='classification', weight_path=None):
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
        head_type=head_type,
        res_attention=False,
    )
    if weight_path:
        model = transfer_weights(weight_path, model)
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    model = get_model(dls.vars, dls.c, seq_len, head_type='classification', weight_path=args.pretrained_model)
    loss_func = nn.CrossEntropyLoss()
    cbs = [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs, aux_weight=args.aux_weight)
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    records = {
        'train_loss': learn.recorder.get('train_loss', []),
        'valid_loss': learn.recorder.get('valid_loss', []),
    }
    if 'train_accuracy' in learn.recorder:
        records['train_accuracy'] = learn.recorder['train_accuracy']
    if 'valid_accuracy' in learn.recorder:
        records['valid_accuracy'] = learn.recorder['valid_accuracy']
    df = pd.DataFrame(records)
    df.to_csv(
        os.path.join(args.save_path, args.save_finetuned_model + '_metrics.csv'),
        float_format='%.6f',
        index=False,
    )


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    model = get_model(dls.vars, dls.c, seq_len, head_type='classification', weight_path=args.pretrained_model)
    loss_func = nn.CrossEntropyLoss()
    cbs = [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_accuracy', fname=args.save_finetuned_model, path=args.save_path),
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
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10)
    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    print('linear probing')
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    model = get_model(dls.vars, dls.c, seq_len, head_type='classification', weight_path=args.pretrained_model)
    loss_func = nn.CrossEntropyLoss()
    cbs = [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_accuracy', fname=args.save_finetuned_model, path=args.save_path),
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
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    save_recorders(learn)


def test_func(weight_path):
    dls = get_dls(args)
    seq_len = _prepare_runtime_config(dls)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(dls.vars, dls.c, seq_len, head_type='classification').to(device)
    cbs = [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, cbs=cbs, aux_weight=args.aux_weight)
    out = learn.test(dls.test, weight_path=weight_path, scores=[accuracy])
    return out


if __name__ == '__main__':
    args.dset = args.dset_finetune
    set_device()
    suggested_lr = find_lr()

    if args.is_finetune:
        finetune_func(suggested_lr)
    elif args.is_linear_probe:
        linear_probe_func(suggested_lr)

    if args.pretrained_model:
        weight_path = os.path.join(args.save_path, args.save_finetuned_model + '.pth')
        out = test_func(weight_path)
        print('accuracy:', out[2])
        print('logits shape:', out[0].shape)

    print('----------- Complete! -----------')
