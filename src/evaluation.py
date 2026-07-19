"""
Evaluation Module

Reusable evaluation functions for
classification and anomaly detection models.
"""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

def evaluate_classifier(y_true, y_pred):
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    metrics = {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

    }

    return metrics

def compute_confusion_matrix(
    y_true,
    y_pred
):
    """
    Compute confusion matrix.

    Returns
    -------
    ndarray
    """

    return confusion_matrix(
        y_true,
        y_pred
    )

def generate_classification_report(
    y_true,
    y_pred
):
    """
    Generate sklearn classification report.

    Returns
    -------
    dict
    """

    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

def compute_roc_data(y_true, y_scores):
    """
    Compute ROC curve data.

    Parameters
    ----------
    y_true : array-like

    y_scores : array-like
        Predicted probabilities.

    Returns
    -------
    dict
    """

    fpr, tpr, _ = roc_curve(
        y_true,
        y_scores
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    return {

        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc

    }

def compare_models(results):
    """
    Create comparison dataframe.

    Parameters
    ----------
    results : dict

    Returns
    -------
    pandas.DataFrame
    """

    comparison = []

    for model_name, metrics in results.items():

        comparison.append({

            "Model": model_name,

            "Accuracy": metrics["Accuracy"],

            "Precision": metrics["Precision"],

            "Recall": metrics["Recall"],

            "F1 Score": metrics["F1 Score"]

        })

    return pd.DataFrame(comparison)

def save_results(
    dataframe,
    filename
):
    """
    Save evaluation results.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    filename : str
    """

    dataframe.to_csv(
        filename,
        index=False
    )

    print(f"Results saved to {filename}")

def evaluate_all_models(
    y_true,
    predictions,
    probability_scores=None
):
    """
    Evaluate multiple classification models.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    predictions : dict
        Dictionary of model predictions.

    probability_scores : dict, optional
        Dictionary of prediction probabilities for ROC computation.

    Returns
    -------
    dict
        Evaluation results for all models.
    """

    results = {}

    for model_name, y_pred in predictions.items():

        metrics = evaluate_classifier(
            y_true,
            y_pred
        )

        cm = compute_confusion_matrix(
            y_true,
            y_pred
        )

        report = generate_classification_report(
            y_true,
            y_pred
        )

        results[model_name] = {

            "metrics": metrics,

            "confusion_matrix": cm,

            "classification_report": report

        }

        if (
            probability_scores is not None
            and model_name in probability_scores
        ):

            roc_data = compute_roc_data(
                y_true,
                probability_scores[model_name]
            )

            results[model_name]["roc_data"] = roc_data

    return results