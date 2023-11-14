import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from utils import save_to_binary

nltk.download("punkt")

cluster = pd.read_csv("./Siemens/data/clusterized_dataframe.csv")

cluster_reviews_joined = " ".join(cluster["Reviews"])
# some reviews may contain important sentences.
sentences = cluster_reviews_joined.split(".")
# we could also have used the embeddings created by sentence transformers, instead of
# CountVectorizer.
vectorizer = CountVectorizer(stop_words="english")
sentence_vectors = vectorizer.fit_transform(cluster["Reviews"])
similarity_matrix = cosine_similarity(sentence_vectors, dense_output=False)
similarity_matrix = similarity_matrix.astype(dtype=np.float16)
# print("Creating Graph...") 
# graph = nx.from_numpy_array(similarity_matrix)
print("Creating Scores...")
scores = nx.pagerank_numpy(similarity_matrix,max_iter = 50)
print("Saving Scores...")
save_to_binary("./Siemens/data/page_rank_scores_all.pickle",scores)
print("Done!")


# num_sentences = 5
# top_sentence_indices = sorted(
#     range(len(scores)), 
#     key=lambda i: scores[i], 
#     reverse=True
# )[:num_sentences]

# summary = [sentences[i] for i in top_sentence_indices]
# print(summary)
