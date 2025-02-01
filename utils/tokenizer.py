import torch
import os
import stanza
import pickle

import sys
from dotenv import load_dotenv

load_dotenv()

project_root = os.getenv('PROJECT_ROOT')
os.chdir(project_root)
sys.path.append(project_root)

MAX_SEQ_LEN = int(os.getenv('MAX_SEQ_LEN'))

class Tokenizer:
    def __init__(self,tokenizer_path :str):
        self.tokenizer : dict | None = self.load_tokenizer(tokenizer_path)
        self.pipeline : stanza.Pipeline | None = self.load_pipeline()

    # tokenizer
    def load_tokenizer(self, tokenizer_path) -> dict | None :
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from file from {tokenizer_path}")
            with open(tokenizer_path, "rb") as f:
                return pickle.load(f)
        else:
            print(f"Tokenizer file not found in path {tokenizer_path}")
    
    # pipeline
    def load_pipeline(self) -> stanza.Pipeline | None: 
        print("Loading Stanza pipeline")
        try : 
            return stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', verbose=False)
        except:
            return None

    def tokenize(self, text : str) -> torch.Tensor:
        if len(text) > MAX_SEQ_LEN : 
            text = text[:MAX_SEQ_LEN]
        words = []
        # get raw tokens
        if self.pipeline:
            doc = self.pipeline(text.lower())
            # print(doc)
            for sentence in doc.sentences:
                for word in sentence.words:
                    words.append(word.text)
        else:
            print(f"Pipeline not loaded")
            exit()
        
        # get input ids
        input_ids : torch.Tensor | None = None
        if self.tokenizer:
            unk_id = self.tokenizer.get("<unk>") 
            input_ids = torch.tensor([self.tokenizer.get(word, unk_id) for word in words], dtype=torch.long)
        else:
            print(f"Tokenizer not loaded")
            exit()

        return input_ids 


if __name__ == "__main__":
    token_path = os.path.join(os.getcwd(),"utils","tokenizer.pkl")
    tokenizer = Tokenizer(tokenizer_path=token_path)
    text = "The quick brown fox jumps over the lazy dog."
    print(tokenizer.tokenize(text))