# !/usr/bin/python3
"""Feature Scaling & Dimensionality Reduction Classes"""

import umap
import sklearn.cluster
from sklearn.base import BaseEstimator,TransformerMixin
import sklearn.preprocessing
import sklearn.decomposition
import scipy.sparse


class FeatureScaler(BaseEstimator,TransformerMixin):
    """
    """
    def __init__(self,scaler_name='StandardScaler',**kwargs):
        """
        """
        self.scaler_name = scaler_name
        if isinstance(self.scaler_name,str):
            self.scaler = getattr(sklearn.preprocessing,self.scaler_name)()
            self.scaler.set_params(**kwargs)
        else:
            self.scaler = None
        
    def get_params(self,**kwargs):
        if self.scaler is not None:
            params =  self.scaler.get_params(**kwargs)
            params['scaler_name'] = self.scaler_name
            return params
        else:
            return {'scaler_name':None}

    def set_params(self,**kwargs):
        self.scaler_name = kwargs.get("scaler_name",'StandardScaler')
        if isinstance(self.scaler_name,str):
            self.scaler = getattr(sklearn.preprocessing,self.scaler_name)()
            self.scaler.set_params(**{k:v for k,v in kwargs.items() if k not in ["scaler_name"]})
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

def apply_feature_scaling(X,scaler_name='StandardScaler',**kwargs):
    """
    Callable function to apply a `stateless` scaler to a matrix.

        For use with FunctionTransformer.

    """
    return FeatureScaler(scaler_name=scaler_name,**kwargs).fit_transform(X)



class DimensionalityReduction(BaseEstimator,TransformerMixin):
    """
        Dimensionality Reduction Methods
    """
    def __init__(self,method_name='TruncatedSVD',**kwargs):
        """
        """
        self.method_name = method_name        
        if self.method_name=="UMAP":        
            self.method      = umap.UMAP()
        else:
            self.method      = getattr(sklearn.decomposition,self.method_name)()
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