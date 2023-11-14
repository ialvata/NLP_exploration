
from topic_extractors.base_class import TopicExtractor
from typing import Iterable,Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from dataclasses import dataclass, asdict

@dataclass
class LDAParameters:
    """
    Class that represents some the kwargs sent to LDA.
    Not strictly necessary to use, since LDA class accepts a dict.
    """
    n_components:int=10 # number of topics
    max_iter:int=5 # maximum number of passes over the training data
    learning_method="online", # if the data size is large, 
    # the online update will be much faster than the batch update.
    learning_offset=10.0,
    random_state=0,
    def dict(self)-> dict[str,Any]:
        return {k: v for k, v in asdict(self).items()}
    
class LDA(TopicExtractor):
    def __init__(
            self, 
            cleaned_docs:Iterable[str],
            vectorizer: TfidfVectorizer | CountVectorizer,
            parameters: dict[str, float|int|str]
        ):
        self.cleaned_docs = cleaned_docs
        self.vectorizer = vectorizer
        self.model = LatentDirichletAllocation(**parameters)
        self.parameters = parameters
        self.feature_names = []

    def fit(self):
        # Extracting term features for LDA
        doc_word_matrix = self.vectorizer.fit_transform(self.cleaned_docs)
        # each row is a vector representing a review.
        self.model.fit(doc_word_matrix)
        self.feature_names = self.vectorizer.get_feature_names_out()


