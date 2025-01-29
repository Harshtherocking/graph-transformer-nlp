import math
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
# scaled dot product with edge
def scaled_dot_product (q_field, k_field, edge_field, score_field) : 
    def sdp_(edges : EdgeBatch) -> dict : 
        dk = edges.src[k_field].shape[-1]
        # print(edges.data[edge_field].shape)
        dot_prod = (edges.src[k_field] * edges.dst[q_field] * edges.data[edge_field]).sum(-1, keepdim = True)

        # q, k, e are vectors with mean 0 and variance 1 
        # sum(q_i * k_i * e_i) has variance dk 
        return {score_field : torch.exp(dot_prod/math.sqrt(dk))}
    return sdp_



'''
Normalization Layer
'''

class LayerNormalization(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(in_dim))  
        self.beta = nn.Parameter(torch.zeros(in_dim))  

    def forward(self, input1 : Tensor, input2 : Tensor) -> Tensor:
        combined = input1 + input2
        
        mean = combined.mean(dim=-1, keepdim=True)  
        std = combined.std(dim=-1, keepdim=True)
        
        normalized = (combined - mean) / (std + eps)
        
        # Apply learnable parameters
        output = self.gamma * normalized + self.beta
        return output



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

        self.E = nn.Linear(in_features=in_dim, out_features=out_dim * num_head)
        return 

    @staticmethod
    def self_attention (g : DGLGraph, q_field : str, k_field : str, v_field : str, edge_field : str) -> Tensor: 
        score_field = "score"
        g.apply_edges(scaled_dot_product(q_field, k_field, edge_field, score_field))

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

        # eps to avoid division by zero
        return g.ndata["wV"] / (g.ndata["wZ"] + eps)
        

    def forward (self, g : DGLGraph, node_field : str, edge_field : str) -> Tensor: 
        # projections (nodes, heads, features)
        g.ndata["Q_h"] = self.Q(g.ndata[node_field]).view(-1, self.num_head, self.out_dim )
        g.ndata["K_h"] = self.K(g.ndata[node_field]).view(-1, self.num_head, self.out_dim )
        g.ndata["V_h"] = self.V(g.ndata[node_field]).view(-1, self.num_head, self.out_dim )

        # projection for Edge Embeddings
        g.edata["E_h"] = self.E(g.edata[edge_field]).view(-1, self.num_head, self.out_dim )

        return MultiHeadAttentionLayer.self_attention(g, "Q_h", "K_h", "V_h", "E_h")
        


'''
Feed Forward Layer
'''
class FFLayer(nn.Module) : 
    def __init__(self, in_dim : int, ff_dim  : int):
        super().__init__()
        self.l1 = nn.Linear(in_features= in_dim, out_features= ff_dim)
        self.l2 = nn.Linear(in_features= ff_dim, out_features= in_dim)
    
    def forward (self, x : Tensor) -> Tensor : 
        x = self.l1(x)
        return self.l2(x)

        

'''
Graph Encoder Layer 
'''

class GraphEncoderLayer (nn.Module) : 
    def __init__(self, in_dim: int, out_dim: int, num_head : int, ff_dim : int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_head = num_head
        self.ff_dim = ff_dim

        self.attn = MultiHeadAttentionLayer(in_dim, out_dim, num_head)

        self.O = nn.Linear(in_features= num_head * out_dim, out_features= in_dim)
        self.ff = FFLayer(in_dim, ff_dim)

        self.norm1 = LayerNormalization(in_dim)
        self.norm2 = LayerNormalization(in_dim)
        return 
    
    def forward (self, g : DGLGraph, emb_field : str, edge_field :str) -> None: 
        h = g.ndata[emb_field]
        
        # attension layer 
        h0 = self.attn(g, emb_field, edge_field)

        # transform & normalize
        h0 = self.O(h0.view(-1,num_head * out_dim))
        h = self.norm1(h0,h)

        # feed forward
        h0 = self.ff(h)

        g.ndata[emb_field] = self.norm2(h,h0)
        return 


        





if __name__ == "__main__" : 
    in_dim = 5
    out_dim = 3
    num_head = 4

    ff_dim = 9

    edges = {
        "src": [0, 0, 1, 1, 2, 1],
        "dst": [1, 2, 3, 4, 5, 2], 
    }

    g = dgl.graph((edges["src"], edges["dst"]), num_nodes=6)
    g.ndata["emb"] = torch.randn(6, in_dim)
    g.edata["edge_emb"] = torch.randn(g.num_edges(),in_dim)
    encoder = GraphEncoderLayer(in_dim, out_dim, num_head, ff_dim)
    encoder(g, "emb", "edge_emb")
