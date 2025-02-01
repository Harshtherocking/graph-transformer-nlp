import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchtune

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



class DependencyEncoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.dep_emb = DepedencyEmbedding(num_features)
        self.emb = load_embeds()
        self.rotary_emb = torchtune.modules.RotaryPositionalEmbeddings(dim)
    
    def forward (self, in_ids : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        assert in_ids.shape[0] == mask.shape[0], f"size mismatch \ninput_ids : {in_ids.shape} \nmask : {mask.shape}"
        
        pass



if __name__ == "__main__":
    dep_mask = torch.tensor([[1, 2, 3], [4, 5, -1], [9, 5, -1]])
    in_ids = torch.tensor([1,3,34])

    encoder = DependencyEncoder(5)
    encoder(in_ids, dep_mask)

