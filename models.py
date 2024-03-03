from typing import Optional, Tuple, Union
from typing import Optional, Tuple, Union
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import modules.mod_utls as m_utls
import numpy as np
from modules.conv_mod import CustomLinear
from modules.mr_conv_mod import build_mlp


class CustomBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input, update_running_stats: bool=True):
        self.track_running_stats = update_running_stats
        return super(CustomBatchNorm1d, self).forward(input)
    

class MySimpleConv_MR_test(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, e_types: list, drop_rate:float=0.0, 
                 mlp3_dim: int=64, bn_type: int=0):
        super(MySimpleConv_MR_test, self).__init__()
        self.e_types = e_types
        self.mlp3_dim = mlp3_dim
        self.bn_type = bn_type
        self.multi_relation = len(self.e_types) > 1
        
        self.proj_edges = nn.ModuleDict()
        for e_t in self.e_types:
            self.proj_edges[e_t] = build_mlp(in_feats * 2, out_feats, drop_rate, hid_dim=self.mlp3_dim)
        
        self.proj_out = CustomLinear(out_feats, out_feats, bias=True)
        if in_feats != out_feats:
            self.proj_skip = CustomLinear(in_feats, out_feats, bias=True)
        else:
            self.proj_skip = nn.Identity()
            
        if self.bn_type in [2, 3]:
            self.edge_bn = nn.ModuleDict()
            for e_t in self.e_types:
                self.edge_bn[e_t] = CustomBatchNorm1d(out_feats)
        
    def udf_edges(self, e_t: str):
        assert e_t in self.e_types, 'Invalid edge types!'
        tmp_fn = self.proj_edges[e_t]
            
        def fnc(edges):
            msg = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
            msg = tmp_fn(msg)
            return {'msg': msg}
        return fnc
        
    def forward(self, g, features, update_bn: bool=True):
        with g.local_scope():
            src_feats = dst_feats = features
            if g.is_block:
                dst_feats = src_feats[:g.num_dst_nodes()]
            g.srcdata['h'] = src_feats
            g.dstdata['h'] = dst_feats
            
            for e_t in g.etypes:
                g.apply_edges(self.udf_edges(e_t), etype=e_t)
                
            if self.bn_type in [2, 3]:
                if not self.multi_relation:
                    g.edata['msg'] = self.edge_bn[self.e_types[0]](g.edata['msg'], update_running_stats=update_bn)
                else:
                    for e_t in g.canonical_etypes:
                        g.edata['msg'][e_t] = self.edge_bn[e_t[1]](g.edata['msg'][e_t], update_running_stats=update_bn)
            
            etype_dict = {}
            for e_t in g.etypes:
                etype_dict[e_t] = (fn.copy_e('msg', 'msg'), fn.sum('msg', 'out'))
            g.multi_update_all(etype_dict=etype_dict, cross_reducer='stack')
            
            out = g.dstdata.pop('out')
            out = torch.sum(out, dim=1)
            out = self.proj_out(out) + self.proj_skip(dst_feats)
    
            return out


class simpleGNN_MR(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int, num_layers: int, e_types: list,
                 input_drop: float, hidden_drop: float, mlp_drop: float, mlp12_dim: int, 
                 mlp3_dim: int, bn_type: int):
        super(simpleGNN_MR, self).__init__()
        self.gnn_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.num_layers = num_layers
        self.input_drop = input_drop 
        self.hidden_drop = hidden_drop
        self.mlp_drop = mlp_drop
        self.mlp12_dim = mlp12_dim
        self.mlp3_dim = mlp3_dim
        self.bn_type = bn_type
        
        self.proj_in = build_mlp(in_feats, hidden_feats, self.mlp_drop, hid_dim=self.mlp12_dim)
        in_feats = hidden_feats
        
        self.in_bn = None
        if self.bn_type in [1, 3]:
            self.in_bn = CustomBatchNorm1d(hidden_feats)
        
        for i in range(num_layers):
            in_dim = in_feats if i==0 else hidden_feats
            
            self.gnn_list.append(
                MySimpleConv_MR_test(in_feats=in_dim, out_feats=hidden_feats, 
                                e_types=e_types, drop_rate=self.mlp_drop, 
                                mlp3_dim=self.mlp3_dim, bn_type=self.bn_type))
            
            self.bn_list.append(CustomBatchNorm1d(hidden_feats))
        
        self.proj_out = build_mlp(hidden_feats*(num_layers+1), out_feats, self.mlp_drop, 
                                  hid_dim=self.mlp12_dim, final_act=False)
        
        self.dropout = nn.Dropout(p=self.hidden_drop)
        self.dropout_in = nn.Dropout(p=self.input_drop)
        self.activation = F.selu
        
    def forward(self, blocks: list, update_bn: bool=True, return_logits: bool=False):
        final_num = blocks[-1].num_dst_nodes()
        h = blocks[0].srcdata['feature']
        h = self.dropout_in(h)
        
        inter_results = []
        h = self.proj_in(h)
        
        if self.in_bn is not None:
            h = self.in_bn(h, update_running_stats=update_bn)
        
        inter_results.append(h[:final_num])
        for block, gnn, bn in zip(blocks, self.gnn_list, self.bn_list):
            h = gnn(block, h, update_bn)
            h = bn(h, update_running_stats=update_bn)
            h = self.activation(h)
            h = self.dropout(h)
            
            inter_results.append(h[:final_num])
        
        if return_logits:
            return inter_results
        else:
            h = torch.stack(inter_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            h = self.proj_out(h)
            return h.log_softmax(dim=-1)
    
    