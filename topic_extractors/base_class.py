from  abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation
from typing import Iterable
from utils import FeatNamesMissingError
from math import ceil


class TopicExtractor(ABC):
    @abstractmethod
    def __init__(self):
        self.cleaned_docs:Iterable[str]
        self.vectorizer: TfidfVectorizer | CountVectorizer
        self.model:NMF | MiniBatchNMF | LatentDirichletAllocation
        self.parameters:dict[str, float|int|str]
        self.feature_names = []
    
    # @abstractmethod
    # def create_term_doc_matrix(self):...

    @abstractmethod
    def fit(self,matrix:np.ndarray):...

    def plot_top_words(self, n_top_words, title):
        """
        Warning:
        -----
        I only tested this function for n_components = 5, 10 and 15. We may get errors for 
        other possible values.
        """
        if self.feature_names == []:
            raise FeatNamesMissingError
        n_components = self.model.components_.shape[0]
        columns = min(n_components,5)
        rows = ceil(n_components//columns)
        fig, axes = plt.subplots(rows, columns, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(self.model.components_):
            # components_:  shape (n_components, n_features)
            # components_[i, j] can be viewed as pseudocount that 
            # represents the number of times word j was assigned to topic i. 
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=1)
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 20})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=20)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=30)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()


