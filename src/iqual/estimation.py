"""Post-Vectorization Estimation Module"""

import sklearn
import warnings
import numpy as np
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.isotonic
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.neural_network
import sklearn.semi_supervised
from sklearn.base import BaseEstimator, TransformerMixin
from .evaluation import get_scorer

module_options = {
    "naive_bayes": sklearn.naive_bayes.__all__,
    "linear_model": sklearn.linear_model.__all__,
    "semi_supervised": sklearn.semi_supervised.__all__,
    "svm": sklearn.svm.__all__,
    "isotonic": sklearn.isotonic.__all__,
    "tree": sklearn.tree.__all__,
    "ensemble": sklearn.ensemble.__all__,
    "neighbors": sklearn.neighbors.__all__,
    "neural_network": sklearn.neural_network.__all__,
    
}
module_dict_list = [
    {f: module for f in funcmethods} for module, funcmethods in module_options.items()
]
estimator_options = {k: v for element in module_dict_list for k, v in element.items()}


def get_probas_from_decisions(X):
    """
    Reverse decision function values to get \
        pseudo-probabilities between 0 & 1

    Args:
        X: decision function output

    Returns: pseudo-probabilities between 0 & 1
        dtype: float
    """
    return 1 / (1 + np.exp(-X))


def load_estimator(model, estimator_options=estimator_options):
    """
    Load an estimator from sklearn.
    """
    if model in estimator_options:
        return getattr(getattr(sklearn, estimator_options.get(model)), model)
    else:
        raise ValueError("{} is not a valid estimator model name".format(model))


class BinaryThresholder(BaseEstimator):
    """
    Custom Threshold Predictor to be used in the pipeline.
    """

    def __init__(
        self,
        use_threshold=True,
        threshold_range=(0.001, 0.999),
        steps=100,
        threshold=0.5,
        scoring_metric="f1",
    ):
        """
        Args:
            use_threshold (bool): Whether to use thresholding or not. Defaults to True.
            threshold_range (tuple): Range of thresholds to search over. Defaults to (0.001, 0.999).
            steps (int): Number of steps to search over. Defaults to 100.
            threshold (float): Threshold to use. Defaults to 0.5.
            scoring_metric (str): Scoring metric to use. Defaults to "f1".
        """
        self.use_threshold = use_threshold
        self.steps = steps
        self.threshold_range = threshold_range
        self.thresholds = np.linspace(
            self.threshold_range[0], self.threshold_range[1], self.steps
        )
        self.scoring_metric = scoring_metric
        self.scorer = get_scorer(self.scoring_metric)
        self.threshold = threshold
        self.params = {
            "use_threshold": self.use_threshold,
            "threshold_range": self.threshold_range,
            "steps": self.steps,
            "threshold": self.threshold,
            "scoring_metric": self.scoring_metric,
        }

    def get_params(self, **kwargs):
        return self.params.copy()

    def set_params(self, **kwargs):
        """ """
        self.use_threshold = kwargs.get("use_threshold", self.use_threshold)
        self.threshold_range = kwargs.get("threshold_range", self.threshold_range)
        self.steps = kwargs.get("steps", self.steps)
        self.thresholds = np.linspace(
            self.threshold_range[0], self.threshold_range[1], self.steps
        )
        self.scoring_metric = kwargs.get("scoring_metric", self.scoring_metric)
        self.scorer = get_scorer(self.scoring_metric)
        self.threshold = kwargs.get("threshold", self.threshold)
        self.params = {
            "use_threshold": self.use_threshold,
            "threshold_range": self.threshold_range,
            "steps": self.steps,
            "threshold": self.threshold,
            "scoring_metric": self.scoring_metric,
        }

    def fit(self, X=None, Y=None, *args, **kwargs):
        """
        Fit the estimator.
        Args:
            X : probas of the classifier
            Y : labels
        Returns:
            self
        """
        if self.use_threshold is True:
            t0, t1 = min(X[:, 1]), max(X[:, 1])
            self.threshold_range = (t0, t1)
            self.thresholds      = np.linspace(t0, t1, self.steps)
            y_pred_array         = [self.predict(X, thresh=t) for t in self.thresholds]
            scores               = [self.scorer._score_func(Y, y_preds) for y_preds in y_pred_array]
            self.threshold = self.thresholds[np.argmax(scores)]
        else:
            self.threshold = None
            self.thresholds = []
            self.steps      = 0
            self.threshold_range = None
        self.set_params(threshold=self.threshold, use_threshold=self.use_threshold)
        return self

    def predict_proba(self, X, **kwargs):
        """
        Predict probabilities of the positive class.
        """
        return X

    def predict_log_proba(self, X, **kwargs):
        """
        Predict the log probability of the positive class.
        """
        return np.log(X)

    def decision_function(self, X, **kwargs):
        """ """
        log_probs = self.predict_log_proba(X)
        return log_probs[:, 1] - log_probs[:, 0]

    def predict(self, X, thresh=None, validate=False, **pred_kwargs):
        """
        Predict the class labels for the provided data.
        """
        if self.use_threshold is False:
            return X
        else:
            if thresh is not None:
                t = thresh
            else:
                t = self.threshold
            if validate is False:
                return [1 if v >= t else 0 for v in X[:, 1]]
            else:
                proba_valid = self.validate_proba(X[:, 1])
                if proba_valid is True:
                    return [1 if v >= t else 0 for v in X[:, 1]]
                else:
                    return [None] * len(X[:, 1])

    def validate_proba(self, probas, **kwargs):
        """
        Check if the probabilities are valid
        """
        if len(set(probas)) > 1:
            return True
        else:
            return False


class Classifier(BaseEstimator, TransformerMixin):
    """
    Main classifier methods
    """

    def __init__(self, model, is_final=False, **kwargs) -> callable:
        """ """
        self.model = model
        self.is_final = is_final
        self.classifier = load_estimator(model=self.model)(**kwargs)

    def get_params(self, **kwargs) -> dict:
        params = self.classifier.get_params(**kwargs)
        params["model"] = self.model
        params["is_final"] = self.is_final
        return params

    def set_params(self, **kwargs):
        self.model    = kwargs.get("model", "LogisticRegression")
        self.is_final = kwargs.get("is_final", False)
        self.classifier = load_estimator(model=str(self.model))()
        self.classifier.set_params(
            **{k: v for k, v in kwargs.items() if k not in ["model", "is_final"]}
        )

    def fit(self, *args, **kwargs):
        """
        Fit the estimator.
        """
        self.classifier.fit(*args, **kwargs)
        return self

    def get_normalized_decisions(self, X, **kwargs):
        """
        Artificial probabilities - normalized decisions.
            There's no need to fit the estimator for training,
                since we will only use it on the decision-values to classify.
        """
        if hasattr(self.classifier, "decision_function"):
            decisions = self.classifier.decision_function(X, **kwargs)
        elif hasattr(self.classifier, "_decision_function"):
            decisions = self.classifier._decision_function(X, **kwargs)
        else:
            print("No decision function found.")
        positive_proba = get_probas_from_decisions(decisions)
        negative_proba = 1 - positive_proba
        return np.vstack((negative_proba, positive_proba)).T

    def get_coefficient(self, X, **kwargs):
        """
        """
        if hasattr(self.classifier, "coef_"):
            return self.classifier.coef_
        else:
            warnings.warn("No coefficient found.")
            return None

    def transform(self, X, **kwargs):
        """ 
        """
        if self.is_final is True:
            return self.predict(X, **kwargs)
        else:
            return self.predict_proba(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        """
        Predict the probability of each class for each sample.
        """
        # If `predict_proba` is available, use it
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X, **kwargs)
        else:
            # Otherwise, use `decision_function`
            if hasattr(self.classifier, "decision_function") or hasattr(
                self.classifier, "_decision_function"
            ):
                return self.get_normalized_decisions(X, **kwargs)
            else:
                warnings.warn("No probability function found. Returning decisions.")
                return self.predict(X, **kwargs)

    def predict(self, X, **kwargs):
        """
        Predict the class for each sample.
            The last layer of the pipeline will use this function.
            As long as the thresholdlayer is called, this function will not be used.
        """
        predictions = self.classifier.predict(X, **kwargs)
        return np.nan_to_num(predictions, nan=0)
