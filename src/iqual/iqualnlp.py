# /usr/bin/env python

""" NLP Modelling Classes """

import os
import joblib
from shutil import rmtree
from tempfile import mkdtemp
from sklearn import set_config
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from .vectorizers import Vectorizer
from .estimation import Classifier, BinaryThresholder
from .feature_transform import FeatureScaler, DimensionalityReduction, DenseTransformer, SparseTransformer
from .crossval import CrossValidator, count_hyperparameters


def column_selector(data, column_name, reshape=False):
    """
    Column selector function for feature concatenation
    Args:
        data: pandas.DataFrame
        column_name: string
    Returns:
        data: pandas.DataFrame
    """
    if reshape:
        return data.loc[:, column_name].values.reshape(-1, 1)
    else:
        return data.loc[:, column_name].values


def construct_text_pipelines(
    q_col, a_col, env="scikit-learn", model="TfidfVectorizer", **kwargs
):
    """
    Construct Text-Pipeline
    Args:
        column_names:
                list of column names
        kwargs:
                dictionary of keyword arguments for vectorization
        env:
                string, type of model to use
        name:
                string, name of model to use
    Returns:
        text_pipeline:
                sklearn.pipeline.Pipeline
    """
    if q_col is not None:
        q_pipe = Pipeline(
            [
                (
                    "selector",
                    FunctionTransformer(
                        column_selector,
                        kw_args=dict(column_name=q_col),
                        validate=False,
                    ),
                ),
                (
                    "vectorizer",
                    Vectorizer(model, env=env, **kwargs),
                ),
            ]
        )
    if a_col is not None:
        a_pipe = Pipeline(
            [
                (
                    "selector",
                    FunctionTransformer(
                        column_selector,
                        kw_args=dict(column_name=a_col),
                        validate=False,
                    ),
                ),
                (
                    "vectorizer",
                    Vectorizer(model, env=env, **kwargs),
                ),
            ]
        )
    if q_col is not None and a_col is not None:
        text_pipeline = FeatureUnion([("question", q_pipe), ("answer", a_pipe)])
    elif q_col is not None:
        text_pipeline = FeatureUnion([("question", q_pipe)])
    elif a_col is not None:
        text_pipeline = FeatureUnion([("answer", a_pipe)])
    else:
        text_pipeline = None
    return text_pipeline


class Model:
    """
    Main NLP Class for Model Training and Prediction.
    """

    def __init__(self, enable_caching=False, diagram=True):
        """
        Initialize Model
        Args:
            enable_caching (bool): Enable caching of the model
            diagram (bool): Create model diagram
        """
        if enable_caching is True:
            self.cache_dir = mkdtemp(prefix="iqual_")
        else:
            self.cache_dir = None
        self.diagram          = diagram
        self.text_vector_pipe = None
        self.transformation_steps = []
        self.estimation_steps = []
        self.post_estimation_steps = []

    def add_text_features(
        self, q_col, a_col, env="scikit-learn", model="TfidfVectorizer", **kwargs
    ):
        """
        Configure text features

        Args:
            q_col (str): Name of the question column
            a_col (str): Name of the answer column
            env (str): type of the vectorizer model to be used for text features
            name (str): name of the vectorizer model to be used for text features
            kwargs: keyword arguments for vectorization
        """

        self.text_vector_pipe = construct_text_pipelines(
            q_col, a_col, env=env, model=model, **kwargs
        )
        self.input_steps = [("Input", self.text_vector_pipe)]

    def add_densifier(self):
        """
        Add a densifier to the pipeline
        """
        self.transformation_steps.append(("Densifier", DenseTransformer()))
    
    def add_sparsifier(self):
        """
        Add a sparsifier to the pipeline
        """
        self.transformation_steps.append(("Sparsifier", SparseTransformer()))

    def add_feature_scaler(self, name="Normalizer", **kwargs):
        """
        Scale the features
        Args:
            name (str):
                Name of the feature scaling method
            **kwargs:
                Keyword arguments for the scaler method
        """
        self.transformation_steps.append(
            ("FeatureTransformation", FeatureScaler(name=name, **kwargs))
        )

    def add_dimensionality_reducer(self, name="TruncatedSVD", **kwargs):
        """
        Reduce the features
        Args:
            name (str): Name of the reducer method
            **kwargs: Keyword arguments for the reducer method
        """
        self.transformation_steps.append(
            ("FeatureTransformation", DimensionalityReduction(name=name, **kwargs))
        )

    def add_feature_transformer(
        self, name="Normalizer", transformation="DimensionalityReduction", **kwargs
    ):
        """ """
        if transformation == "DimensionalityReduction":
            self.add_dimensionality_reducer(name=name, **kwargs)
        elif transformation == "FeatureScaler":
            self.add_feature_scaler(name=name, **kwargs)
        else:
            raise ValueError(
                "Invalid transformation method, please choose from 'DimensionalityReduction' or 'FeatureScaler'"
            )

    def add_classifier(self, name="LogisticRegression", **kwargs):
        """
        Add a classifier to the pipeline
        Args:
            name (str): Name of the classifier method. Any classifier from sklearn can be used
            **kwargs: Keyword arguments for the classifier
        """
        self.estimation_steps.append(("Classifier", Classifier(name, **kwargs)))

    def add_threshold(self, scoring_metric='f1',**kwargs):
        """
        Add a custom threshold to the pipeline
        Args:
            **kwargs: Keyword arguments for the threshold

        """
        self.post_estimation_steps.append(("Threshold", BinaryThresholder(scoring_metric=scoring_metric,**kwargs)))

    def compile(self):
        """
        Compile the model
        """
        self.steps = [
            *self.input_steps,            # Input (Text Columns)
            *self.transformation_steps,   # Post Vectorization Steps, Densify/Sparsify, Feature Scaling, Dimensionality Reduction
            *self.estimation_steps,       # Classifier
            *self.post_estimation_steps,  # Post Estimation Steps, Custom Threshold
        ]

        #### Create the pipeline
        self.model = Pipeline(self.steps, memory=self.cache_dir)
        if len(self.post_estimation_steps)==0:
            self.model.set_params(Classifier__is_final=True)
        else:
            self.model.set_params(Classifier__is_final=False)

        if self.diagram is True:
            set_config(display="diagram")
            return self.model

    def cross_validate_fit(
        self,
        X,
        Y,
        cv_method="GridSearchCV",
        search_parameters={},
        scoring="f1",
        cv_splits=5,
        refit=False,
        **kwargs,
    ):
        """
        Cross validate the model and fit it
        Args:
            X (pd.DataFrame): Dataframe of features
            Y (pd.Series): Series of labels
            cv_method (str): Cross validation method
            search_parameters (dict): Search parameters for GridSearchCV
            scoring (str): Scoring method for GridSearchCV
            cv (int): Number of folds for GridSearchCV
            n_iter (int): Number of iterations for RandomizedSearchCV
            refit (bool): Whether to refit the model
            **kwargs: Keyword arguments for the model
        
        Returns:
            cv_scores (dict): Cross validation scores
        """
        print(
            f".......{count_hyperparameters(search_parameters)} hyperparameters configurations possible.....\r",
            end="",
        )

        self.cv = CrossValidator(
            model_pipe=self.model,
            cv_method=cv_method,
            search_parameters=search_parameters,
            scoring=scoring,
            cv=cv_splits,
            refit=refit,
            **kwargs,
        )
        self.cv.fit(X, Y)
        self.set_params(**self.cv.get_best_params())
        self.model.fit(X, Y)
        return self.cv.get_cv_scores()

    def get_params(self):
        """
        Get the parameters of the model
        """
        params = self.model.get_params()
        return params.copy()

    def set_params(self, **kwargs):
        """
        Set the parameters of the model

        """
        self.model.set_params(**kwargs)

    def fit(self, X, y=None, *args, **kwargs):
        """
        Fit the model
        """
        self.model.fit(X, y, *args, **kwargs)
        return self

    def predict_proba(self, X):
        """
        Predict the probabilities of the labels for the positive class
        """
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            if probas.ndim == 2:
                return probas[:, 1]
            elif probas.ndim == 1:
                return probas
            else:
                raise ValueError("The predicted probabilities have an unexpected shape")
        else:
            return self.model.predict(X)

    def predict(self, X):
        """
        Predict the labels  of the samples in X
        """
        return self.model.predict(X)

    def score(self, x, y_true, scoring_function=None):
        """
        Score the model

        Args:
            x (pandas.DataFrame): The data to score
            y_true (pandas.Series): The true labels
            scoring_function (function): The scoring function to use

        Returns:
            score (float): The score of the model
        """
        y_pred = self.model.predict(x)
        return scoring_function(y_true, y_pred)

    def load_model(self, model_path):
        """
        Load the model from a pickle file
        
        Args:
            model_path (str): Path to the model file

        """
        self.model = joblib.load(model_path)

    def save_model(self, model_path):
        """
        Save the model to disk as a pickle file
        
        Args:
            model_path (str): Path to the model file
        """
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def clear_cache(self):
        """
        Clear the cache directory
        """
        if self.cache_dir is not None:
            try:
                rmtree(self.cache_dir)
                print("Cache cleared")
            except FileNotFoundError:
                print("Cache already cleared")
        else:
            print("Cache directory not set")
