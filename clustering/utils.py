"""
This module serves as an utilities module.
"""

import  pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Callable, Iterator   
from pathlib import Path
import subprocess

class FeatNamesMissingError(Exception):
     def __repr__(self) -> str:
          return "Features Names are missing! \
            You probably should run fit method of a topic extractor first."


def save_to_binary(file_path, data):
    with open(file_path,mode="wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_binary(file_path):
    with open(file_path, mode="rb") as file:
                # file will still close properly
                return pickle.load(file)

def parallelize(callable:Callable, 
                data:Iterator[str] | pd.Series, workers:int = mp.cpu_count() - 1):
    with mp.Pool(workers) as p:
        return p.map(callable, data)

def fetch_data(transformation:Callable, data:Iterator[str] | pd.Series, 
               file_path:Path | None = None, workers:int = mp.cpu_count() - 1):
    if file_path is None:
         return parallelize(transformation, data, workers=workers)
    if not file_path.exists():
        transformed_data = parallelize(transformation, data, workers=workers)
        save_to_binary(file_path,transformed_data)
        return transformed_data
    else:
        return load_binary(file_path)
    
def fetch_embeddings(file_path:Path)-> list[np.ndarray]:
     if not file_path.exists():
          subprocess.call(['python', 'embeddings.py'])
          return load_binary(file_path)
     else:
          return load_binary(file_path)