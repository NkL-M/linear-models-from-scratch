"""
Module for metrics and loss function for Linear Models built from scratch without ML librairies like Scikit-Learn.
"""
import numpy as np

#-----------------------------#
# Regression Metrics and Loss #
#-----------------------------#

def ssr(y_true, y_pred) -> float:
    """
    Function for computing Sum of Squared Residuals.

    Returns
    ----
    ssr_score : float
    """
    ssr_score = np.sum((y_true - y_pred)**2)
    return float(ssr_score)

def r_squared(y_true, y_pred) -> float:
    """
    Function for computing R-squared.

    Returns
    ----
    r2_score : float
    """
    ss_residuals = ssr(y_true, y_pred)
    ss_mean = ssr(y_true, np.mean(y_true))
    r2_score = 1 - (ss_residuals)/(ss_mean)
    return float(r2_score)

def mse(y_true, y_pred) -> float:
    """
    Function for computing Mean Squared Error.

    Returns
    ----
    mse_score : float
    """
    mse_score = np.mean((y_true - y_pred)**2)
    return float(mse_score)

def rmse(y_true, y_pred) -> float:
    """
    Function for computing Root Mean Squared Error.

    Returns
    ----
    rmse_score : float
    """
    rmse_score = np.sqrt(np.mean((y_true - y_pred)**2))
    return float(rmse_score)

def mae(y_true, y_pred) -> float:
    """
    Function for computing Mean Absolute Error.

    Returns
    ----
    mae_score : float
    """
    mae_score = np.mean(np.abs(y_true - y_pred))
    return float(mae_score)

#---------------------------------#
# Classification Metrics and Loss #
#---------------------------------#
def log_loss(y_true, y_pred) -> float:
    """
    Function for computing log loss aka Binary cross-entropy loss (BCE).

    Returns
    ----
    log_loss : float
    """
    logloss = (-1/len(y_true)) * np.sum((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))
    return logloss

def accuracy(y_true, y_pred) -> float:
    """
    Function for computing accuracy score.

    Returns
    ----
    accuracy_score : float
    """
    tp_tn = np.sum([1 for true, pred in zip(y_true, y_pred) if true==pred])
    accuracy_score = tp_tn / len(y_true)
    return float(accuracy_score)

def recall(y_true, y_pred) -> float:
    """
    Function for computing recall score.

    Returns
    ----
    recall_score : float
    """
    tp = np.sum([1 for true, pred in zip(y_true, y_pred) if true==1 and pred==1])
    fn = np.sum([1 for true, pred in zip(y_true, y_pred) if true==1 and pred==0])
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(recall_score)

def precision(y_true, y_pred) -> float:
    """
    Function for computing precision score.

    Returns
    ----
    precision_score : float
    """
    tp = np.sum([1 for true, pred in zip(y_true, y_pred) if true==1 and pred==1])
    fp = np.sum([1 for true, pred in zip(y_true, y_pred) if true==0 and pred==1])
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return float(precision_score)

def f1(y_true, y_pred) -> float:
    """
    Function for computing F1 score.

    Returns
    ----
    f1_score : float
    """
    recall_ = recall(y_true, y_pred)
    precision_ = precision(y_true, y_pred)
    f1_score = 2 * ((recall_ * precision_) / (recall_ + precision_)) if (recall_ + precision_) > 0 else 0.0
    return float(f1_score)
