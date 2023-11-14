"""
This module has some utility functions to create features from text.
"""


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def lowercase(input):
  """
  Returns lowercase text
  """
  return input.lower()

def remove_punctuation(input):
  """
  Returns text without punctuation
  """
  return input.translate(str.maketrans('','', string.punctuation))

def remove_whitespaces(input):
  """
  Returns text without extra whitespaces
  """
  return " ".join(input.split())

def tokenize(input):
  """
  Returns tokenized version of text
  """
  return word_tokenize(input)

def remove_stop_words(input):
  """
  Returns text without stop words
  """
  input = word_tokenize(input)
  return [word for word in input if word not in stopwords.words('english')]

def lemmatize(input):
  """
  Lemmatizes input using NLTK's WordNetLemmatizer
  """
  lemmatizer=WordNetLemmatizer()
  input_str=word_tokenize(input)
  new_words = []
  for word in input_str:
    new_words.append(lemmatizer.lemmatize(word))
  return ' '.join(new_words)


def nlp_pipeline(sentence):
  """
  Function that calls all other functions together to perform 
  feature engineering from a given text.
  """
  return lemmatize(' '.join(remove_stop_words(remove_whitespaces(remove_punctuation(lowercase(sentence))))))


def extract_nouns(sentence:str)->str:
    """
    This function is used to extract the nouns from english sentences.
    """
    sentence = remove_punctuation(lowercase(sentence))
    tokens = word_tokenize(sentence, language="english")
    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    post_tag = pos_tag(tokens, lang="eng")
    return " ".join([
        word
        for word,tag in post_tag
        if tag == "NN"
      ])