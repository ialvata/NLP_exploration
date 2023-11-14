"""
This Module uses PyTorch multiprocessing feature.
It is used to create some embeddings from the reviews.
It is GPU compatible.
Read the function docstring for more informatio.
"""

from sentence_transformers import SentenceTransformer
import torch.multiprocessing as torch_mp
import pandas as pd
from utils import save_to_binary, load_binary

from pathlib import Path

cwd_path = Path().absolute()
path = cwd_path.parent
data_path = path/'data'

# if you want to do embedding from the raw reviews, uncomment below, and
# comment loading of corpus_nouns code line.
# df = pd.read_csv (data_path/"preprocessed.csv")
# docs = df["Reviews"].tolist()

docs = load_binary(data_path/"corpus_nouns.pickle")

torch_mp.set_start_method('spawn',force=True)


def torch_parallelize(callable, data, workers= torch_mp.cpu_count()//4):
    """
    This function will parallelize the task of creating embeddings, using processes.
    The number of workers may need to be optimized according 
    to your RAM quantity, and GPU RAM.
    This function must be run under a __name__ =='__main__' clause, and you cannot 
    import python's standard multiprocessing lib.
    For more information: https://pytorch.org/docs/stable/notes/multiprocessing.html
    """
    with torch_mp.Pool(workers) as p:
        return p.map(callable, data)

if __name__ == '__main__':
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Starting creation of embeddings")
    embeddings = torch_parallelize(model.encode,docs)
    save_to_binary(data_path/"corpus_embeddings.pickle",embeddings)
    print("Embeddings have been created and saved to file")