# /usr/bin/env python
"""
    This module contains the `Evalutation` class which contains
        common metrics for measuring the classification performance.
"""

import pandas as pd
import sklearn.metrics


def count_outcomes(y_true, y_pred, outcome_type="true_positives"):
    """
    Count the number of -
        true positives, false positives, false negatives, true negatives.
    Args:
        y_true: true labels
        y_pred: predicted labels
        outcome_type: "TP", "FP", "FN", or "TN"
    """
    outcomes = [
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        ]
    outcome_values = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return outcome_values[outcomes.index(outcome_type.lower())]


SKLEARN_SCORERS  = sklearn.metrics.SCORERS.copy()

    # IMPORTANT: These are not suitable to be used as a scorer to *fit* the model during cross-validation.
    # For the `make_scorer` functions below, the `greater_is_better is not utilized. [bool]
    # TODO: Check source code [of GridSearch, and make_scorer] to see if this makes a difference.

OUTCOMES         = ["true_negatives","false_positives","false_negatives","true_positives"]
MORE_SCORERS     = {"count_"+outcome: sklearn.metrics.make_scorer(count_outcomes,needs_proba=False,needs_threshold=False,outcome_type=outcome) for outcome in OUTCOMES}

CUSTOM_SCORERS   = {**SKLEARN_SCORERS, **MORE_SCORERS}

def get_scoring_dict(keys=['count_true_negatives',
                            'count_false_positives',
                            'count_false_negatives',
                            'count_true_positives',
                            'f1',
                            ]):
    """
    Returns a dictionary of scoring functions.
    """
    return {key: CUSTOM_SCORERS[key] for key in keys}


def calc_f1_score_from_outcomes(true_positives, false_positives, false_negatives, true_negatives=None):
    """
    Calculates the F1 score from a
        True-Positives, False-Postives, and False-Negatives.
    """
    return (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)



def calc_f1_score_from_labels(y_true, y_pred):
    """
    # TODO: Redundant? Check if this is used anywhere.
    Calculates the F1 score from true-labels and predicted-labels.
    """
    return sklearn.metrics.f1_score(y_true, y_pred)







class EvaluationScorer:
    """
    This function calculates the performance of a model.
    """

    metrics = [
        "accuracy_score",
        "adjusted_mutual_info_score",
        "adjusted_rand_score",
        "average_precision_score",
        "balanced_accuracy_score",
        "brier_score_loss",
        "calinski_harabasz_score",
        "cohen_kappa_score",
        "completeness_score",
        "consensus_score",
        "d2_tweedie_score",
        "davies_bouldin_score",
        "dcg_score",
        "explained_variance_score",
        "f1_score",
        "fbeta_score",
        "fowlkes_mallows_score",
        "homogeneity_score",
        "jaccard_score",
        "label_ranking_average_precision_score",
        "make_scorer",
        "mutual_info_score",
        "ndcg_score",
        "normalized_mutual_info_score",
        "precision_recall_fscore_support",
        "precision_score",
        "r2_score",
        "rand_score",
        "recall_score",
        "roc_auc_score",
        "silhouette_score",
        "top_k_accuracy_score",
        "v_measure_score",
    ]

    def __init__(
        self,
        metrics=[
            "f1_score",
            "recall_score",
            "precision_score",
            "accuracy_score",
        ],
    ):
        """
        Initializes the ScoreCalculator class.
        """
        self.metrics = metrics
        self.scorers = [
            getattr(sklearn.metrics, metric) for metric in self.metrics
        ]

    def calc_scores(self, y_true, y_pred, **kwargs):
        """
        Calculates the scores of a model.
        """
        score_dict = {
            self.metrics[s]: scorer(y_true, y_pred, **kwargs)
            for s, scorer in enumerate(self.scorers)
        }
        return pd.DataFrame([score_dict])
