# /usr/bin/env python
"""
This module contains the vectorizers used to generate embeddings for text data.
Options include:
    - `sentence-transformer` 
    - `spacy`
    - `scikit-learn`
    - `saved-dictionary`
"""
import spacy
import scipy.sparse
import numpy as np
import sklearn.feature_extraction.text
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import save_pickle_data, load_pickle_data


def get_vectorizer_options():
    """Return a list of vectorizer options"""
    return ["scikit-learn", "sentence-transformers", "saved-dictionary", "spacy"]


def load_vectorizer(model, env, **kwargs):
    """
    Load a vectorizer using the `env` and `model` parameters.

    Args:
    """
    assert env in get_vectorizer_options(), f"{env} is not a valid vectorizer option"
    if env == "scikit-learn":
        return ScikitLearnVectorizer(model, **kwargs)
    elif env == "sentence-transformers":
        return SentenceTransformerVectorizer(model, **kwargs)
    elif env == "saved-dictionary":
        return DictionaryVectorizer(model, **kwargs)
    elif env == "spacy":
        return SpacyVectorizer(model, **kwargs)
    else:
        raise ValueError("{} is not a valid env".format(env))


class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    """
    Replace text with vectors using a saved pickle dictionary
    """

    def __init__(self, model="", **kwargs):
        self.model = model

    def load_embeddings(self):
        embedding_dictionary = load_pickle_data(self.model)
        return embedding_dictionary

    def fit(self, *fit_args, **fit_kwargs):
        """ """
        return self

    def transform(self, X, **transform_kwargs):
        """ """
        embeddings = self.load_embeddings()
        return scipy.sparse.csr_matrix(
            np.vstack([embeddings[x] for x in X]), dtype=np.float64
        )

    def get_params(self, **kwargs):
        return {"model": self.model, "env": "saved-dictionary"}

    def set_params(self, **param_args):
        self.model = param_args.get("model", self.model)


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):

    """Generate embeddings using the `sentence-transformer` package"""

    def __init__(self, model="average_word_embeddings_glove.6B.300d", **kwargs):
        self.model = model
        self.vectorizer = SentenceTransformer(self.model, **kwargs)

    def fit(self, *fit_args, **fit_kwargs):
        return self

    def transform(self, X, **transform_params):
        """
        Generates embeddings for sentences using the aforementioned initalized `SentenceBert` model
        Args:
             X                   : list of sentences to encode                  (List[str])
             convert_to_numpy    : whether to convert output to numpy array     (Bool)
             normalize_embeddings : whether to normalize embeddings              (Bool)
             device              : device to use for encoding                   (str)
        Returns:
            Encoded embeddings of sentences

        """
        return self.vectorizer.encode(X, **transform_params)

    def get_params(self, **kwargs):
        return {"model": self.model, "env": "sentence-transformers"}

    def set_params(self, **param_args):
        self.model = param_args.get("model", self.model)
        self.vectorizer = SentenceTransformer(self.model)


class SpacyVectorizer(BaseEstimator, TransformerMixin):
    """Generate vectors using the `spacy` package"""

    def __init__(self, model="en_core_web_sm", **kwargs):
        self.model = model
        self.vectorizer = spacy.load(self.model, **kwargs)

    def fit(self, *fit_args, **fit_kwargs):
        return self

    def transform(self, X, **transform_params):
        return [x.vector for x in self.vectorizer.pipe(X, **transform_params)]

    def get_params(self, **kwargs):
        return {"model": self.model, "env": "spacy"}

    def set_params(self, **param_args):
        self.model = param_args.get("model", self.model)
        self.vectorizer = spacy.load(self.model)


class ScikitLearnVectorizer(BaseEstimator, TransformerMixin):
    """Generate vectors using the `scikit-learn` text-feature extraction modules"""

    def __init__(self, model, **kwargs):
        self.model = model
        self.vectorizer = getattr(sklearn.feature_extraction.text, self.model)()
        self.vectorizer.set_params(
            **{k: v for k, v in kwargs.items() if k not in ["model", "env"]}
        )

    def fit(self, *fit_args, **fit_kwargs):
        """ """
        self.vectorizer.fit(*fit_args, **fit_kwargs)
        return self

    def transform(self, X, **kwargs):
        """ """
        vect = self.vectorizer.transform(X, **kwargs)
        return vect

    def get_params(self, **kwargs):
        params = self.vectorizer.get_params(**kwargs)
        params["model"] = self.model
        params["env"] = "scikit-learn"
        return params

    def set_params(self, **kwargs):
        self.model = kwargs.get("model", "CountVectorizer")
        self.vectorizer = getattr(sklearn.feature_extraction.text, self.model)()
        self.vectorizer.set_params(
            **{k: v for k, v in kwargs.items() if k not in ["model", "env"]}
        )


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    Base class for vectorizers
    """

    def __init__(self, model, env, **kwargs):
        self.model = model
        self.env = env
        self.vectorizer = load_vectorizer(model, env, **kwargs)

    def fit(self, *fit_args, **fit_kwargs):
        """ """
        self.vectorizer.fit(*fit_args, **fit_kwargs)
        return self

    def transform(self, X, **kwargs):
        """ """
        vect = self.vectorizer.transform(X, **kwargs)
        return vect

    def get_params(self, **kwargs):
        params = self.vectorizer.get_params(**kwargs)
        params["model"] = self.model
        params["env"] = self.env
        return params

    def set_params(self, **kwargs):
        self.model = kwargs.get("model", "CountVectorizer")
        self.env = kwargs.get("env", "scikit-learn")
        self.vectorizer = load_vectorizer(
            str(self.model),
            str(self.env),
            **{k: v for k, v in kwargs.items() if k not in ["model", "env"]},
        )
