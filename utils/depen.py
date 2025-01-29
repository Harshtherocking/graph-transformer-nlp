import torch 
import stanza

# Universal Dependencies
dependencies = {
    'acl': 0, 'advcl': 1, 'advmod': 2, 'amod': 3, 'appos': 4, 'aux': 5, 'case': 6, 'cc': 7, 'ccomp': 8,
    'clf': 9, 'compound': 10, 'conj': 11, 'cop': 12, 'csubj': 13, 'dep': 14, 'det': 15, 'discourse': 16,
    'dislocated': 17, 'expl': 18, 'fixed': 19, 'flat': 20, 'goeswith': 21, 'iobj': 22, 'list': 23, 'mark': 24,
    'nmod': 25, 'nsubj': 26, 'nummod': 27, 'obj': 28, 'obl': 29, 'orphan': 30, 'parataxis': 31, 'punct': 32,
    'reparandum': 33, 'root': 34, 'vocative': 35, 'xcomp': 36
}

# for easy reverse lookup
reverse_dependencies = {index: dep for dep, index in dependencies.items()}

def dep_to_idx(dep: str) -> int:
    assert dep in dependencies, f"Dependency {dep} not found in Universal Dependencies" 
    return dependencies.get(dep, -1) 

def idx_to_dep(idx: int) -> str | None:
    return reverse_dependencies.get(idx, None)


class DependencyParser:
    def __init__(self):
        try : 
            self.pipeline = stanza.Pipeline(lang='en', verbose= False, processors='tokenize,mwt,pos,lemma,depparse')
        except Exception as e:
            print(f"Error in Loading Parsing Stanza Pipeline")
            exit()
        
    def get_dep_mask(self, text):
        doc = self.pipeline(text)
        num_tokens = sum(len(sentence.words) for sentence in doc.sentences)
        dep_matrix = torch.full((num_tokens, num_tokens), -1, dtype=torch.long)
        
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.head > 0:  # head == 0 means the word is the root
                    head_idx = word.head - 1
                    tail_idx = word.id - 1
                    dep_idx = dep_to_idx(word.deprel.split(":")[0])
                    dep_matrix[head_idx, tail_idx] = dep_idx
        
        return dep_matrix


if __name__ == '__main__':
    
    text = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    text = " ".join(text)
    print(DependencyParser().get_dep_mask(text))