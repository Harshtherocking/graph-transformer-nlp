import torch 
from torch import Tensor
import torch.nn as nn

import dgl 
import dgl.function as fn
from dgl import DGLGraph
from dgl.udf import EdgeBatch

eps = 1e-9

dgl.backend_name = "pytorch"

'''
Citation : Below code is modification to the original code by,  
    Title: A Generalization of Transformer Networks to Graphs
    Author: Dwivedi, Vijay Prakash and Bresson, Xavier
    Journal: AAAI Workshop on Deep Learning on Graphs: Methods and Applications
    Link : https://github.com/graphdeeplearning/graphtransformer/tree/main
'''


'''
Utils 
'''
# scaled dot product 
def sdp (q_field, k_field, score_field) : 
    def sdp_(edges : EdgeBatch) -> dict : 
        dk = edges.src[k_field].shape[-1]
        dot_prod = (edges.src[k_field] * edges.dst[q_field]).sum(-1, keepdim = True)
        return {score_field : torch.exp(dot_prod/(dk ** 0.5))}
    return sdp_


'''
Attention Layer
'''

class MultiHeadAttentionLayer (nn.Module) : 
    def __init__(self, in_dim: int, out_dim: int, num_head : int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_head = num_head
        
        self.Q = nn.Linear(in_features=in_dim, out_features=out_dim * num_head)
        self.K = nn.Linear(in_features=in_dim, out_features=out_dim * num_head)
        self.V = nn.Linear(in_features=in_dim, out_features=out_dim * num_head)
        return 
    

    @staticmethod
    def self_attention (g : DGLGraph, q_field : str, k_field : str, v_field : str) -> Tensor: 
        score_field = "score"
        g.apply_edges(sdp(q_field, k_field, score_field))

        eids = g.edges()
        g.send_and_recv(
            eids,
            fn.u_mul_e(v_field, score_field, "wV_"),
            fn.sum("wV_", "wV")
        )

        g.send_and_recv(
            eids,
            fn.copy_e(score_field, "wZ_"), 
            fn.sum("wZ_", "wZ")
        )

        return g.ndata["wV"] / (g.ndata["wZ"] + eps)
        

    def forward (self, g : DGLGraph, field : str) -> Tensor: 
        # projections (nodes, heads, features)
        g.ndata["Q_h"] = self.Q(g.ndata[field]).view(-1, self.num_head, self.out_dim )
        g.ndata["K_h"] = self.K(g.ndata[field]).view(-1, self.num_head, self.out_dim )
        g.ndata["V_h"] = self.V(g.ndata[field]).view(-1, self.num_head, self.out_dim )

        return MultiHeadAttentionLayer.self_attention(g, "Q_h", "K_h", "V_h")
        

        
'''
Graph Transformer Layer 
'''


if __name__ == "__main__" : 
    in_dim = 5
    out_dim = 3
    num_head = 4

    edges = {
        "src": [0, 0, 1, 1, 2],
        "dst": [1, 2, 3, 4, 5], 
    }

    g = dgl.graph((edges["src"], edges["dst"]), num_nodes=6)
    g.ndata["emb"] = torch.randn(6,in_dim)

    mhal  = MultiHeadAttentionLayer(in_dim, out_dim, num_head)
    mhal(g, "emb")
    