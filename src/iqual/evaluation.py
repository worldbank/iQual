# /usr/bin/env python
"""This script contains common metrics for measuring the classification performance."""

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


def calc_f1_score_from_outcomes(
    true_positives, false_positives, false_negatives, true_negatives=None
):
    """
    Calculates the F1 score from a
        True-Positives, False-Postives, and False-Negatives.
    """
    return (2 * true_positives) / (
        2 * true_positives + false_positives + false_negatives
    )


def calc_f1_score_from_labels(y_true, y_pred):
    """
    Calculates the F1 score from true-labels and predicted-labels.
    """
    return sklearn.metrics.f1_score(y_true, y_pred)


def get_scoring_dict(
    keys=[
        "count_true_negatives",
        "count_false_positives",
        "count_false_negatives",
        "count_true_positives",
        "f1",
    ]
):
    """
    Returns a dictionary of scoring functions. Useful for passing to `sklearn.model_selection.cross_validate`.

    Any scoring function in `sklearn.metrics.SCORERS` can be used.

    Args:
        keys: list of scoring functions to return.

    """
    return {key: ALL_SCORERS[key] for key in keys}


def list_scoring_functions():
    """
    List all available scoring functions.
    """
    return list(ALL_SCORERS.keys())


SKLEARN_SCORERS = sklearn.metrics.SCORERS.copy()
MORE_SCORERS = {
    "count_"
    + outcome: sklearn.metrics.make_scorer(
        count_outcomes, needs_proba=False, needs_threshold=False, outcome_type=outcome
    )
    for outcome in [
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]
}
ALL_SCORERS = {**SKLEARN_SCORERS, **MORE_SCORERS}