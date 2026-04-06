"""
Module for the basic functionq for Linear Models.
"""

import numpy as np
import pandas as pd
from collections import Counter
from linear_models.src.metrics import *

#-----------------------#
# Basic Model Functions #
#-----------------------#
def pred(X, coeffs) -> np.ndarray:
    """
    Takes in a matrix of the input data and the coefficient vector

    Returns
    ----
    predictions : np.ndarray
        Linear combinaison of the inputs
    """
    predictions = np.matmul(X, coeffs)
    return predictions

def loss_function(y_true, y_pred, loss='mse'):
    """
    Computes loss function

    Parameter
    ----
    loss : str
        - 'mse'
        - 'log_loss'

    Returns
    ----
    loss_result : float
    """
    loss_dict = {'mse' : mse,
                 'log_loss' : log_loss}

    if loss in loss_dict:
        loss_result = loss_dict[loss](y_true, y_pred)
        return loss_result
    else:
        raise ValueError(f"Wrong Input, unknown loss function '{loss}'")

def ordinary_least_square(X, y) -> np.ndarray:
    """
    Function computing closed form solution equation for the linear regresion algorithm (ordinary least square).

    Returns
    ----
    ols : np.ndarray
    """
    # ols = np.linalg.inv(np.matmul(X.T, X)) @ np.matmul(X.T, y)
    # ols = np.linalg.inv(np.matmul(np.matmul(X.T, X)), np.matmul(X.T, y))
    # ols = np.linalg.solve(X.T @ X, X.T @ y)
    ols = np.linalg.inv(np.array(X).T @ np.array(X)) @ (np.array(X).T @ y)
    return ols

def sigmoid(y_pred) -> np.ndarray:
    """
    Compute sigmoïd function

    Returns
    ----
    sigmoid_ : np.ndarray
        Binary classification probabilities for each observations
    """
    sigmoid_ = np.where(y_pred >= 0,
                    1 / (1 + np.exp(-y_pred)),
                    np.exp(y_pred) / (1 + np.exp(y_pred)))

    # sigmoid_ = 1 / (1 + np.exp(-y_pred)) #TODO Choose

    # sigmoid_f = [1 / (1 + np.exp(-y_pred))
    #                         if x >= 0
    #                         else np.exp(y_pred) / (1 + np.exp(y_pred))
    #                         for x in y_pred]

    return sigmoid_

def gradient(X_train, y_train, coeffs, loss='mse') -> np.ndarray:
    """
    Computes the gradient according to a given loss function.

    Parameter
    ----
    loss : str
        - 'mse'
        - 'logloss'

    Returns
    ----
    gradient_ : np.ndarray
        - array of partial derivatives
    """
    n = len(X_train)

    if loss=='mse':
        gradient_ = (2/n) * np.dot(X_train.T, (pred(X_train, coeffs) - y_train))

    elif loss == 'log_loss':
        gradient_ = (1/n) * np.dot(X_train.T, sigmoid(pred(X_train, coeffs)) - y_train)

    else:
        raise ValueError(f"Unknown loss '{loss}', unable to compute the gradient")

    return gradient_

def evaluate_score(y_true, y_pred, metric='r2', task='regression'):
    """
    Function that computes the errors of a model based on a selected metric.

    Parameter
    ----
    metric : str
    - Regression metrics
        - 'mse'
        - 'rmse'
        - 'mae'
        - 'r2'
    - Classification metrics
        - 'accuracy'
        - 'recall'
        - 'precision'
        - 'f1'

    task : str
        - 'regression'
        - 'classification'

    Returns
    ----
    eval_score : float
    """
    regression_dict = {'mse' : mse,
                       'rmse' : rmse,
                       'mae' : mae,
                       'r2' : r_squared}

    classification_dict = {'accuracy' : accuracy,
                           'recall' : recall,
                           'precision' : precision,
                           'f1' : f1}

    if task == 'regression':
        if metric in regression_dict:
            eval_score = regression_dict[metric](y_true, y_pred)
            return round(eval_score, 4)
        else:
            raise ValueError(f"Wrong Input, unknown regression metric '{metric}'")

    elif task == 'classification':
        if metric in classification_dict:
            eval_score = classification_dict[metric](y_true, y_pred)
            return round(eval_score, 4)
        else:
            raise ValueError(f"Wrong Input, unknown classification metric '{metric}'")

    else:
        raise ValueError(f"Unknown task '{task}'")

def baseline_score(y_test, y_train,  metric='mse', task='regression'):
    """
    Function computing baseline score.

    For regression tasks, the baseline predictions will be the training dataset's mean.

    While for classification tasks, the predictions will be the most common category.

    Parameters
    ----
    task : str
        - 'regression'
        - 'classification'

    metric : str
    - Regression metrics
        - 'mse'
        - 'rmse'
        - 'mae'
        - 'r2'
    - Classification metrics
        - 'accuracy'
        - 'recall'
        - 'precision'
        - 'f1'

    Returns
    ----
    baseline_score : float
    """
    if task=='regression':
        baseline_pred = np.mean(y_train)
    elif task=='classification':
        baseline_pred = np.full(shape=len(y_test),
                                fill_value=Counter(y_train).most_common(1)[0][0])
    else:
        raise ValueError(f"Unknown task '{task}'")

    baseline_score = evaluate_score(y_true=y_test, y_pred=baseline_pred, metric=metric, task=task)
    return float(baseline_score)
