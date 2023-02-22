# !/usr/bin/python3
"""Feature Scaling & Dimensionality Reduction Classes"""

import umap
import sklearn.preprocessing
import sklearn.decomposition
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Feature Scaling Methods
    """

    def __init__(self, name="Normalizer", **kwargs):
        """
        Feature Scaling Methods

        Args:
            name (str, optional): Name of the Scaler. Defaults to 'Normalizer'.
            **kwargs: Keyword arguments to pass to the Scaler.
        """
        self.name = name
        if isinstance(self.name, str):
            self.scaler = getattr(sklearn.preprocessing, self.name)()
            self.scaler.set_params(**kwargs)
        else:
            self.scaler = None

    def get_params(self, **kwargs):
        if self.scaler is not None:
            params = self.scaler.get_params(**kwargs)
            params["name"] = self.name
            return params
        else:
            return {"name": None}

    def set_params(self, **kwargs):
        self.name = kwargs.get("name", "Normalizer")
        if isinstance(self.name, str):
            self.scaler = getattr(sklearn.preprocessing, self.name)()
            self.scaler.set_params(
                **{k: v for k, v in kwargs.items() if k not in ["name"]}
            )
        else:
            self.scaler = None

    def fit(self, X, y=None, **kwargs):
        """
        Fit the scaler to the data.

        Args:
            X (np.array): Data to fit the scaler to.
            y (np.array, optional): Target data. Defaults to None.
            **kwargs: Keyword arguments to pass to the Scaler.
        Returns:
            self            
        """
        if self.scaler is not None:
            self.scaler.fit(X,y, **kwargs)
            return self
        else:
            return self

    def transform(self, X, **kwargs):
        """
        Transform the data.
        """
        if self.scaler is None:
            return X
        else:
            return self.scaler.transform(X, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the scaler to the data and transform it.
        """
        if self.scaler is not None:
            return self.scaler.fit_transform(X, **kwargs)
        else:
            return X


def apply_feature_scaling(X, name="Normalizer", **kwargs):
    """
    Callable function to apply a `stateless` scaler to a matrix. For use with FunctionTransformer.
    """
    return FeatureScaler(name=name, **kwargs).fit_transform(X)


class DimensionalityReduction(BaseEstimator, TransformerMixin):
    """
    Dimensionality Reduction Methods
    """

    def __init__(self, name="TruncatedSVD", **kwargs):
        """
        Dimensionality Reduction Methods
        Args:
            name (str, optional): Name of the Dimensionality Reduction Method. Defaults to 'TruncatedSVD'.
            **kwargs: Keyword arguments to pass to the Dimensionality Reduction Method.
        """
        self.name = name
        if self.name == "UMAP":
            self.method = umap.UMAP()
        else:
            self.method = getattr(sklearn.decomposition, self.name)()
        self.method.set_params(**kwargs)

    def get_params(self, **kwargs):
        return self.method.get_params(**kwargs)

    def set_params(self, **kwargs):
        self.method.set_params(**kwargs)

    def fit(self, *args, **kwargs):
        """
        Fit the Dimensionality Reduction Method to the data.
        """
        self.method.fit(*args, **kwargs)
        return self

    def transform(self, X, **kwargs):
        """
        Transform the data.
        """
        return self.method.transform(X, **kwargs)

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the Dimensionality Reduction Method to the data and transform it.
        """
        return self.method.fit_transform(X, **kwargs)


class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Convert sparse matrix to dense matrix
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray()
    
class SparseTransformer(BaseEstimator, TransformerMixin):
    """
    Convert dense matrix to sparse matrix
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.tocsr()