# !/usr/bin/python3
"""Feature Scaling & Dimensionality Reduction Classes"""

import umap
import sklearn.cluster
import sklearn.preprocessing
import sklearn.decomposition
from sklearn.base import BaseEstimator,TransformerMixin


class FeatureScaler(BaseEstimator,TransformerMixin):
    """
    """
    def __init__(self,name='StandardScaler',**kwargs):
        """
        """
        self.name = name
        if isinstance(self.name,str):
            self.scaler = getattr(sklearn.preprocessing,self.name)()
            self.scaler.set_params(**kwargs)
        else:
            self.scaler = None
        
    def get_params(self,**kwargs):
        if self.scaler is not None:
            params =  self.scaler.get_params(**kwargs)
            params['name'] = self.name
            return params
        else:
            return {'name':None}

    def set_params(self,**kwargs):
        self.name = kwargs.get("name",'StandardScaler')
        if isinstance(self.name,str):
            self.scaler = getattr(sklearn.preprocessing,self.name)()
            self.scaler.set_params(**{k:v for k,v in kwargs.items() if k not in ["name"]})
        else:
            self.scaler = None
    
    def fit(self,*args,**kwargs):
        """
        """
        if self.scaler is not None:
            self.scaler.fit(*args,**kwargs)
            return self
        else:
            return self

    def transform(self,X,**kwargs):
        """
        """
        if self.scaler is None:
            return X
        else:
            return self.scaler.transform(X,**kwargs)

    def fit_transform(self,X,y=None,**kwargs):
        """
        """

        if self.scaler is not None:
            return self.scaler.fit_transform(X,**kwargs)
        else:
            return X

def apply_feature_scaling(X,name='StandardScaler',**kwargs):
    """
    Callable function to apply a `stateless` scaler to a matrix.

        For use with FunctionTransformer.

    """
    return FeatureScaler(name=name,**kwargs).fit_transform(X)



class DimensionalityReduction(BaseEstimator,TransformerMixin):
    """
        Dimensionality Reduction Methods
    """
    def __init__(self,name='TruncatedSVD',**kwargs):
        """
        """
        self.name = name        
        if self.name=="UMAP":        
            self.method      = umap.UMAP()
        else:
            self.method      = getattr(sklearn.decomposition,self.name)()
        self.method.set_params(**kwargs)

    def get_params(self,**kwargs):
        return self.method.get_params(**kwargs)

    def set_params(self,**kwargs):
        self.method.set_params(**kwargs)
        

    def fit(self, *args, **kwargs):
        """
        """
        self.method.fit(*args, **kwargs)
        return self

    def transform(self,X,**kwargs):
        """
        """
        return self.method.transform(X,**kwargs)
    
    def fit_transform(self,X,y=None,**kwargs):
        """
        """
        return self.method.fit_transform(X,**kwargs)    