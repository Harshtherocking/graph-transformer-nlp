import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings

import os
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = os.getenv('PROJECT_ROOT')
os.chdir(project_root)
sys.path.append(project_root)

MAX_SEQ_LEN = int(os.getenv('MAX_SEQ_LEN'))
NUM_HEADS = int(os.getenv('NUM_HEADS'))

embed_path = os.path.join(os.getcwd(), "utils", "embed.pt") 

def load_embeds (path : str = embed_path ): 
    return torch.load(path,weights_only=False)



class DepedencyEmbedding(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.dep_emb = nn.Embedding(36+1, num_features, padding_idx= 36)
    
    def forward (self, dep_mask) -> torch.Tensor:
        dep_mask = dep_mask.clone()

        # replace -1 by last element
        dep_mask[dep_mask == -1] = 36
        embeddings = self.dep_emb(dep_mask)
        
        # mask out 
        embeddings[dep_mask == 36] = 0
        return embeddings



class SelfAttention(nn.Module):
    def __init__(self, hid_dim : int, rotary_emb : torch):
        super().__init__()
        self.hid_dim = hid_dim

    def forward(self, x : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        pass





class DependencyEncoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.dep_emb = DepedencyEmbedding(num_features)
        self.emb = load_embeds()
        self.rotary_emb = RotaryPositionalEmbeddings(dim=num_features//NUM_HEADS, max_seq_len= MAX_SEQ_LEN, base= 10000)
    
    def forward (self, in_ids : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        assert in_ids.shape[-1] == mask.shape[-2] == MAX_SEQ_LEN, f"size mismatch \ninput_ids : {in_ids.shape} \nmask : {mask.shape}"

        # tokens id to vector
        # batch_size x seq_len x num_features
        x = self.emb(in_ids)

        # dependency mask to mask vector
        # batch_sizer x seq_len x num_features
        mask_vec = self.dep_emb(mask)


        # padding the sequence 
        pass
        



if __name__ == "__main__":
    dep_mask = torch.tensor([[1, 2, 3], [4, 5, -1], [9, 5, -1]])
    in_ids = torch.tensor([1,3,34])

    encoder = DependencyEncoder(5)
    encoder(in_ids, dep_mask)

