import os
import pickle
import torch
import torch.nn as nn

glove_save_path = os.path.join(os.getcwd(), "utils")

class GloveEmbedding():
    def __init__(self, glove_file_path : str, save_path : str = glove_save_path) -> None:
        tokenizer, Embed = self.load_glove_embeddings(glove_file_path)
        self.save_glove_embeddings(save_path, tokenizer, Embed)

    def load_glove_embeddings(self, file_path : str) -> tuple[dict, nn.Embedding]: 
        print(f"Loading Glove Embeddings from {file_path}")
        word_to_int = {}
        embeddings = []
        dim = 0 
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                values = line.split()
                word = values[0]
                word = word.encode('ascii', 'ignore').decode('ascii')
                # print(f"word : {word.encode('ascii', 'ignore').decode('ascii')}")
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                if idx == 0:
                    dim = vector.shape[0]
                if vector.shape[0] < dim:
                    vector = torch.concat([vector, torch.zeros(dim - vector.shape[0])])
                word_to_int[word] = idx
                embeddings.append(vector)
        
        embedding_matrix = torch.stack(embeddings)
        embedding_layer = nn.Embedding.from_pretrained(embedding_matrix)
        
        return word_to_int, embedding_layer

    def save_glove_embeddings(self, save_path : str, tokenizer : dict, embedding_layer : nn.Embedding) -> None:
        print(f"Saving Tokenizer")
        with open(os.path.join(save_path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"SavedTokenizer to {save_path + "tokenizer.pkl"}")

        print("Saving Embeddings")
        torch.save(embedding_layer, os.path.join(save_path, 'embed.pt'))
        print(f"Saved Embeddings to {save_path + "embed.pt"}")


if __name__ == "__main__" :
    glove_file_path = os.path.join(os.getcwd(),"glove.twitter.27B", 'glove.twitter.27B.25d.txt')
    glove = GloveEmbedding(glove_file_path)