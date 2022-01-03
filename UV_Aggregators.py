import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r):
        # print("\n\nFORWARD")
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        # print("************nodes=", nodes)
        # print("************history_uv=",history_uv)
        # print("************len(history_uv)=",len(history_uv))
        for i in range(len(history_uv)):
            # print("************START FOR :: i", i)
            # print("\n************START FOR")
            history = history_uv[i]
            # print("************history=", history)
            num_histroy_item = len(history)

            # if num_histroy_item > 128 :
            # history = history[:128]

            # print("************num_history_item=", num_histroy_item)
            tmp_label = history_r[i]
            # print("************tmp_label=", tmp_label)
            # if len(tmp_label) > 128 :
            # tmp_label = history[:128]

            ###
            for j in range(len(tmp_label)):
                # print("$%$%$%$%$%$%$%$%$%$%$tmp_label[i]=", tmp_label[i])
                if tmp_label[j] > 0:
                    tmp_label[j] = tmp_label[j] - 1
                if tmp_label[j] < 0:
                    tmp_label[j] = tmp_label[j] + 1
            ###

            # print("************tmp_label=", tmp_label)

            if self.uv == True:
                # user component
                # print("************user component")
                # print("************history=", history)
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes[i]]
                # print("************nodes[i][=", nodes[i])
            else:
                # item component
                # print("************user component")
                # print("************history=", history)
                e_uv = self.u2e.weight[history]
                # print("************e_uv=", e_uv)
                # print("************nodes=", nodes)
                # print("************i=", i)
                # print("************nodes[i]=", nodes[i])
                uv_rep = self.v2e.weight[nodes[i]]
                # print("************uv_rep=", uv_rep)

            # print("************tmp_label=", tmp_label)
            e_r = self.r2e.weight[tmp_label]
            # print("************ e_r complete ************")
            # print("************e_uv.shape=", e_uv.shape)
            # print("************e_r.shape=", e_r.shape)
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
            # print("************END FOR")

        to_feats = embed_matrix
        return to_feats
