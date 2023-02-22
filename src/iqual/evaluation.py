"""This script contains common metrics for measuring the classification performance."""

import sklearn.metrics

SKLEARN_SCORERS = sklearn.metrics.get_scorer_names()
CUSTOM_SCORERS  = ['count_' + c for c in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']]
ALL_SCORERS     = [*SKLEARN_SCORERS, *CUSTOM_SCORERS]

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

def list_scorers():
    """
    List all available scoring functions.
    """
    return ALL_SCORERS

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

    Returns:
        scoring_dict: a dictionary of scoring functions.
    """        
    scoring_dict = {}
    for key in keys:
        if key in SKLEARN_SCORERS:
            scoring_dict[key] = sklearn.metrics.get_scorer(key)
        elif key in CUSTOM_SCORERS:
            scoring_dict[key] = sklearn.metrics.make_scorer(count_outcomes, needs_proba=False, needs_threshold=False, outcome_type=key.split("count_")[-1])  
        else:
            raise ValueError(f"Invalid scorer: {key}")
    return scoring_dict

def get_scorer(scorer_name):
    """
    Returns a scorer function from a string.

    Args:
        scorer_name: name of the scorer to return. See `list_scorers()` for a list of valid scorers.

    Returns:
        scorer: a scorer function.
    """
    assert scorer_name in list_scorers(), f"{scorer_name} is not a valid scorer. See `list_scorers()` for a list of valid scorers."
    if scorer_name in SKLEARN_SCORERS:
        return sklearn.metrics.get_scorer(scorer_name)
    elif scorer_name in CUSTOM_SCORERS:
        return sklearn.metrics.make_scorer(count_outcomes, needs_proba=False, needs_threshold=False, outcome_type=scorer_name.split("count_")[-1])
    else:
        raise ValueError(f"Invalid scorer: {scorer_name}")
    

def calc_f1_score_from_outcomes(
    true_positives, false_positives, false_negatives, true_negatives=None):
    """
    Calculates the F1 score from a
        True-Positives, False-Postives, and False-Negatives.
    """
    return (2 * true_positives) / (
        2 * true_positives + false_positives + false_negatives
    )

def get_metric(metric_name='f1_score'):
    """
    Returns a metric function from a string.
    See sklearn.metrics for a list of valid metrics.
    """
    return getattr(sklearn.metrics, metric_name)
