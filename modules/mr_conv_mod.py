import torch
import dgl
import math
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import TypedLinear
from modules.conv_mod import CustomLinear


def build_mlp(in_dim: int, out_dim: int, p: float, hid_dim: int=64, final_act: bool=True):
    mlp_list = []

    mlp_list.append(CustomLinear(in_dim, hid_dim, bias=True))
    mlp_list.append(nn.ELU())
    mlp_list.append(nn.Dropout(p=p))
    mlp_list.append(nn.LayerNorm(hid_dim))
    mlp_list.append(CustomLinear(hid_dim, out_dim, bias=True))
    if final_act:
        mlp_list.append(nn.ELU())
        mlp_list.append(nn.Dropout(p=p))

    return nn.Sequential(*mlp_list)
