In these notebooks, I've tried to predict Inflations Expectations from FED's speeches.
I run simulations for classification and regression experiments.

The LightGBM notebooks form a sequence of experiments. 
I started with a simple bag of words experiment, noticed things to improve, 
culminating in a lightGBM model with a word embedding Doc2Vec from the gensim package, plus hyperparameter optimization with Optuna.
