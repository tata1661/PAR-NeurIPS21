import torch
import torch.nn as nn

from .encoder import GNN_Encoder
from .relation import MLP,ContextMLP, TaskAwareRelation


class attention(nn.Module):
    def __init__(self, dim):
        super(attention, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = self.layers(x)
        x = self.softmax(torch.transpose(x, 1, 0))
        return x


class ContextAwareRelationNet(nn.Module):
    def __init__(self, args):
        super(ContextAwareRelationNet, self).__init__()
        self.mol_relation_type = args.rel_type
        self.rel_layer = args.rel_layer
        self.edge_type = args.rel_adj
        self.edge_activation = args.rel_act
        self.cuda_num = args.cuda

        self.mol_encoder = GNN_Encoder(num_layer=args.enc_layer, emb_dim=args.emb_dim, JK=args.JK,
                                       drop_ratio=args.dropout, graph_pooling=args.enc_pooling, gnn_type=args.enc_gnn,
                                       batch_norm = args.enc_batch_norm)
        if args.pretrained:
            model_file = args.pretrained_weight_path
            if args.enc_gnn != 'gin':
                temp = model_file.split('/')
                model_file = '/'.join(temp[:-1]) +'/'+args.enc_gnn +'_'+ temp[-1]
            print('load pretrained model from', model_file)
            self.mol_encoder.from_pretrained(model_file, self.cuda_num)

        if self.mol_relation_type not in ['par']:
            self.encode_projection = MLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                     batch_norm=args.batch_norm,dropout=args.map_dropout)
        else:
            self.encode_projection = ContextMLP(inp_dim=args.emb_dim, hidden_dim=args.map_dim, num_layers=args.map_layer,
                                     batch_norm=args.batch_norm,dropout=args.map_dropout,
                                     pre_fc=args.map_pre_fc,ctx_head=args.ctx_head)

        if self.mol_relation_type == 'par':
            inp_dim = args.map_dim
            self.adapt_relation = TaskAwareRelation(inp_dim=inp_dim, hidden_dim=args.rel_hidden_dim,
                                                    num_layers=args.rel_layer, edge_n_layer=args.rel_edge_layer,
                                                    top_k=args.rel_k, res_alpha=args.rel_res,
                                                    batch_norm=args.batch_norm, adj_type=args.rel_adj,
                                                    activation=args.rel_act, node_concat=args.rel_node_concat,dropout=args.rel_dropout,
                                                    pre_dropout=args.rel_dropout2)
        else:
            self.adapt_relation = MLP(inp_dim=args.map_dim, hidden_dim=2, num_layers=1)

    def to_one_hot(self,class_idx, num_classes=2):
        return torch.eye(num_classes)[class_idx].to(class_idx.device)

    def label2edge(self, label, mask_diag=True):
        # get size
        num_samples = label.size(1)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        # compute edge
        edge = torch.eq(label_i, label_j).float().to(label.device)

        # expand
        edge = edge.unsqueeze(1)
        if self.edge_type == 'dist':
            edge = 1 - edge

        if mask_diag:
            diag_mask = 1.0 - torch.eye(edge.size(2)).unsqueeze(0).unsqueeze(0).repeat(edge.size(0), 1, 1, 1).to(edge.device)
            edge=edge*diag_mask
        if self.edge_activation == 'softmax':
            edge = edge / edge.sum(-1).unsqueeze(-1)
        return edge

    def relation_forward(self, s_emb, q_emb, s_label=None, q_pred_adj=False,return_adj=False,return_emb=False):
        if self.mol_relation_type == 'par':
            if not return_emb:
                s_logits, q_logits, adj = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
            else:
                s_logits, q_logits, adj, s_rel_emb, q_rel_emb = self.adapt_relation(s_emb, q_emb,return_adj=return_adj,return_emb=return_emb)
            if q_pred_adj:
                q_sim = adj[-1][:, 0, -1, :-1]
                q_logits = q_sim @ self.to_one_hot(s_label)
        else:
            s_logits = self.adapt_relation(s_emb)
            q_logits = self.adapt_relation(q_emb)
            adj = None
        if not return_emb:
            return s_logits, q_logits, adj
        else:
            return s_logits, q_logits, adj, s_rel_emb, q_rel_emb

    def forward(self, s_data, q_data, s_label=None, q_pred_adj=False):
        s_emb, s_node_emb = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb, q_node_emb = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)

        s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)

        s_logits, q_logits, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)

        return s_logits, q_logits, adj, s_node_emb

    def forward_query_list(self, s_data, q_data_list, s_label=None, q_pred_adj=False):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        q_emb_list = [self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)[0] for q_data in
                      q_data_list]

        if self.mol_relation_type!='par':
            s_emb_map = self.encode_projection(s_emb)
        else:
            s_emb_map = None

        q_logits_list, adj_list = [], []
        for q_emb in q_emb_list:
            if self.mol_relation_type!='par':
                q_emb_map = self.encode_projection(q_emb)
            else:
                s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit.detach())
            if adj is not None:
                sim_adj = adj[-1][:,0].detach()
                q_adj = sim_adj[:,-1]
                adj_list.append(q_adj)

        q_logits = torch.cat(q_logits_list, 0)
        adj_list = torch.cat(adj_list, 0)
        return s_logit.detach(),q_logits, adj_list

    def forward_query_loader(self, s_data, q_loader, s_label=None, q_pred_adj=False):
        s_emb, _ = self.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
        if self.mol_relation_type!='par':
            s_emb_map = self.encode_projection(s_emb)
        else:
            s_emb_map = None
        y_true_list=[]
        q_logits_list, adj_list = [], []
        for q_data in q_loader:
            q_data = q_data.to(s_emb.device)
            y_true_list.append(q_data.y)
            q_emb,_ = self.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            if self.mol_relation_type!='par':
                q_emb_map = self.encode_projection(q_emb)
            else:
                s_emb_map,q_emb_map = self.encode_projection(s_emb,q_emb)
            s_logit, q_logit, adj = self.relation_forward(s_emb_map, q_emb_map, s_label, q_pred_adj=q_pred_adj)
            q_logits_list.append(q_logit)
            if adj is not None and self.mol_relation_type == 'par':
                sim_adj = adj[-1].detach()
                adj_list.append(sim_adj)

        q_logits = torch.cat(q_logits_list, 0)
        y_true = torch.cat(y_true_list, 0)
        sup_labels={'support':s_data.y,'query':y_true_list}
        return s_logit, q_logits, y_true,adj_list,sup_labels