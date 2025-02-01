import os
import torch
from utils.depen import DependencyParser
from utils.tokenizer import Tokenizer

class Preprocessor: 
    def __init__(self, tokenizer_path : str):
        self.tokenizer = Tokenizer(tokenizer_path)
        self.parser = DependencyParser()
    
    def __call__(self, text : str) -> tuple[torch.Tensor, torch.Tensor]:
        in_ids = self.tokenizer.tokenize(text)
        mask = self.parser.get_dep_mask(text)
        return in_ids, mask


