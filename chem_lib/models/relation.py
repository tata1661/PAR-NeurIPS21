
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers,batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
            if l < num_layers - 1:
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential(layer_list)
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out

class Attention(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """
    def __init__(self, dim, num_heads=1, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class ContextMLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers,pre_fc=0,batch_norm=False, dropout=0.,ctx_head=1,):
        super(ContextMLP, self).__init__()
        self.pre_fc = pre_fc #0, 1
        in_dim = inp_dim
        out_dim = hidden_dim
        if self.pre_fc:
            hidden_dim=int(hidden_dim//2)  
            self.attn_layer = Attention(hidden_dim,num_heads=ctx_head,attention_dropout=dropout)        
            self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)
        else:
            self.attn_layer = Attention(inp_dim)
            inp_dim=int(inp_dim*2)
            self.mlp_proj = MLP(inp_dim=inp_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                batch_norm=batch_norm, dropout=dropout)

    def forward(self, s_emb, q_emb):
        if self.pre_fc:
            s_emb = self.mlp_proj(s_emb)
            q_emb = self.mlp_proj(q_emb)
        n_support = s_emb.size(0)
        n_query = q_emb.size(0)

        s_emb_rep = s_emb.unsqueeze(0).repeat(n_query, 1, 1)
        q_emb_rep = q_emb.unsqueeze(1)
        all_emb = torch.cat((s_emb_rep, q_emb_rep), 1)
        orig_all_emb =  all_emb

        n_shot=int(n_support//2)
        neg_proto_emb = all_emb[:,:n_shot].mean(1).unsqueeze(1).repeat(1, n_support + 1, 1)
        pos_proto_emb = all_emb[:,n_shot:2*n_shot].mean(1).unsqueeze(1).repeat(1, n_support + 1, 1)
        all_emb =torch.stack((all_emb, neg_proto_emb,pos_proto_emb), -2)
        
        q,s,n, d = all_emb.shape
        x=all_emb.reshape((q*s,n,d))
        attn_x =self.attn_layer(x)
        attn_x=attn_x.reshape((q,s,n, d))
        all_emb = attn_x[:,:,0,]

        all_emb = torch.cat([all_emb, orig_all_emb],dim=-1)

        if not self.pre_fc:
            all_emb = self.mlp_proj(all_emb)

        return all_emb, None

class NodeUpdateNetwork(nn.Module):
    def __init__(self, inp_dim, out_dim, n_layer=2, edge_dim=2, batch_norm=False, dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.edge_dim = edge_dim
        num_dims_list = [out_dim] * n_layer  # [num_features * r for r in ratio]
        if n_layer > 1:
            num_dims_list[0] = 2 * out_dim

        # layers
        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_dims_list[l - 1] if l > 0 else (self.edge_dim + 1) * inp_dim,
                out_channels=num_dims_list[l],
                kernel_size=1,
                bias=False)
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_dims_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0 and l == (len(num_dims_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, self.edge_dim, 1, 1).to(node_feat.device)

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), self.edge_dim).squeeze(1), node_feat)

        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, n_layer=3, top_k=-1,
                 edge_dim=2, batch_norm=False, dropout=0.0, adj_type='dist', activation='softmax'):
        super(EdgeUpdateNetwork, self).__init__()
        self.top_k = top_k
        self.adj_type = adj_type
        self.edge_dim = edge_dim
        self.activation = activation

        num_dims_list = [hidden_features] * n_layer  # [num_features * r for r in ratio]
        if n_layer > 1:
            num_dims_list[0] = 2 * hidden_features
        if n_layer > 3:
            num_dims_list[1] = 2 * hidden_features
        # layers
        layer_list = OrderedDict()
        for l in range(len(num_dims_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=num_dims_list[l - 1] if l > 0 else in_features,
                                                       out_channels=num_dims_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            if batch_norm:
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_dims_list[l], )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=num_dims_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def softmax_with_mask(self, adj, mask=None):
        if mask is not None:
            adj_new = adj - (1 - mask.expand_as(adj)) * 1e8
        else:
            adj_new = adj
        n_q, n_edge, n1, n2 = adj_new.size()
        adj_new = adj_new.reshape(n_q * n_edge * n1, n2)
        adj_new = F.softmax(adj_new, dim=-1)
        adj_new = adj_new.reshape((n_q, n_edge, n1, n2))
        return adj_new

    def forward(self, node_feat, edge_feat=None):  # x: bs*N*num_feat
        # compute abs(x_i, x_j)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)  # size: bs x fs X N x N  (2,128,11,11)
        if self.adj_type == 'sim':
            x_ij = torch.exp(-x_ij)

        sim_val = self.sim_network(x_ij)
        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(
            node_feat.device)
        if self.activation == 'softmax':
            sim_val = self.softmax_with_mask(sim_val, diag_mask)
        elif self.activation == 'sigmoid':
            sim_val = torch.sigmoid(sim_val) * diag_mask
        else:
            sim_val = sim_val * diag_mask

        if self.edge_dim == 2:
            if self.activation == 'softmax':
                dsim_val = self.softmax_with_mask(1 - sim_val, diag_mask)
            else:
                dsim_val = (1 - sim_val) * diag_mask
            adj_val = torch.cat([sim_val, dsim_val], 1)
        else:
            adj_val = sim_val

        if self.top_k > 0:
            n_q, n_edge, n1, n2 = adj_val.size()
            k=min(self.top_k,n1)
            adj_temp = adj_val.reshape(n_q*n_edge*n1,n2)
            topk, indices = torch.topk(adj_temp, k)
            mask = torch.zeros_like(adj_temp)
            mask = mask.scatter(1, indices, 1)
            mask = mask.reshape((n_q, n_edge, n1, n2))
            mask = ((mask + mask.transpose(2,3)) > 0).type(torch.float32)
            if self.activation == 'softmax':
                adj_val = self.softmax_with_mask(adj_val, mask)
            else:
                adj_val = adj_val * mask

        return adj_val, edge_feat


class TaskAwareRelation(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, edge_n_layer, num_class=2,
                res_alpha=0., top_k=-1, node_concat=True, batch_norm=False, dropout=0.0,
                 edge_dim=2, adj_type='sim', activation='softmax',pre_dropout=0.0):
        super(TaskAwareRelation, self).__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_concat = node_concat
        self.res_alpha = res_alpha
        self.dropout_rate = dropout
        self.pre_dropout = pre_dropout
        self.adj_type=adj_type
        node_n_layer = max(1, min(int(edge_n_layer // 2), 2))
        gnn_inp_dim = self.inp_dim
        if self.pre_dropout>0:
            self.predrop1 = nn.Dropout(p=self.pre_dropout)
        for i in range(self.num_layers):
            module_w = EdgeUpdateNetwork(in_features=gnn_inp_dim, hidden_features=hidden_dim, n_layer=edge_n_layer,
                                         top_k=top_k,
                                         edge_dim=edge_dim, batch_norm=batch_norm, adj_type=adj_type,
                                         activation=activation, dropout=dropout if i < self.num_layers - 1 else 0.0)
            module_l = NodeUpdateNetwork(inp_dim=gnn_inp_dim, out_dim=hidden_dim, n_layer=node_n_layer,
                                         edge_dim=edge_dim, batch_norm=batch_norm,
                                         dropout=dropout if i < self.num_layers - 1 else 0.0)
            self.add_module('edge_layer{}'.format(i), module_w)
            self.add_module('node_layer{}'.format(i), module_l)

            if self.node_concat:
                gnn_inp_dim = gnn_inp_dim + hidden_dim
            else:
                gnn_inp_dim = hidden_dim

        self.fc1 = nn.Sequential(nn.Linear(gnn_inp_dim, inp_dim), nn.LeakyReLU())
        if self.pre_dropout>0:
            self.predrop2 = nn.Dropout(p=self.pre_dropout)
        self.fc2 = nn.Linear(inp_dim, num_class)

        assert 0 <= res_alpha <= 1

    def forward(self, all_emb, q_emb=None, return_adj=False, return_emb=False):
        node_feat=all_emb
        if self.pre_dropout>0:
            node_feat=self.predrop1(node_feat)
        edge_feat_list = []
        if return_adj:
            x_i = node_feat.unsqueeze(2)
            x_j = torch.transpose(x_i, 1, 2)
            init_adj = torch.abs(x_i - x_j)
            init_adj = torch.transpose(init_adj, 1, 3)  # size: bs x fs X N x N  (2,128,11,11)
            if self.adj_type == 'sim':
                init_adj = torch.exp(-init_adj)
            diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).to(
                node_feat.device)
            init_adj = init_adj*diag_mask
            edge_feat_list.append(init_adj)
        
        for i in range(self.num_layers):
            adj, _ = self._modules['edge_layer{}'.format(i)](node_feat)
            node_feat_new = self._modules['node_layer{}'.format(i)](node_feat, adj)
            if self.node_concat:
                node_feat = torch.cat([node_feat, node_feat_new], 2)
            else:
                node_feat = node_feat_new
            edge_feat_list.append(adj)
        if self.pre_dropout>0:
            node_feat=self.predrop2(node_feat)
        node_feat = self.fc1(node_feat)
        node_feat = self.res_alpha * all_emb +  node_feat

        s_feat = node_feat[:, :-1, :].mean(0)
        q_feat = node_feat[:, -1, :]

        s_logits = self.fc2(s_feat)
        q_logits = self.fc2(q_feat)
        if return_emb:
            return s_logits, q_logits, edge_feat_list, s_feat,q_feat
        else:
            return s_logits, q_logits, edge_feat_list
