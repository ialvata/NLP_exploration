import spacy
from bertopic import BERTopic
import pandas as pd
from pathlib import Path

cwd_path = Path().absolute()
print("My current directory is : " + str(cwd_path))
path = cwd_path
data_path = path/'Siemens'/'data'
df = pd.read_csv (data_path/"preprocessed.csv")
docs = df["Reviews"].tolist()
# nlp = spacy.load('en_core_web_md', 
#                  exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

fig = topic_model.visualize_topics()
fig.show()