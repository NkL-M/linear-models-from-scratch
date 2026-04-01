"""
Module for Linear Regression and Logistic Regression built from scratch without ML librairies like Scikit-Learn.
"""

import numpy as np
import pandas as pd
from linear_models.src.metrics import *
from linear_models.utils import *

#-----------------------#
# Basic Model Functions #
#-----------------------#
def pred(X, coeffs) -> np.ndarray:
    """
    Takes in a matrix of the input data and the coefficient vector

    Returns
    ----
    predictions : np.ndarray
    """
    predictions = np.matmul(X, coeffs)
    return predictions

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
    obs_train = len(X_train)

    if loss=='mse':
        # gradient_ = (2/obs_train) * (X_train.T @ ((X_train @ coeffs) - np.resize(y_train, new_shape=(len(y_train),1))))
        gradient_ = (2/obs_train) * (X_train.T @ ((X_train @ coeffs) - y_train))

    elif loss == 'log_loss':
        gradient_ = 0 # TODO add grandient BCE

    else:
        raise ValueError(f"Unknown loss '{loss}', unable to compute the gradient")

    return gradient_

def loss_function(y_true, y_pred, loss='mse'):
    """
    Computes loss function

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

    Arg
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
    baseline_pred = np.mean(y_train)
    baseline_score = evaluate_score(y_true=y_test, y_pred=baseline_pred, metric=metric, task=task)
    return float(baseline_score)

def ordinary_least_square(X, y) -> np.ndarray:
    """
    Function computing closed form solution equation for the linear regresion algorithm (ordinary least square).

    Returns
    ----
    ols : np.ndarray
    """
    assert(y.shape==(y.shape[0],))
    ols = np.linalg.inv(np.matmul(X.T, X)) @ np.matmul(X.T, y)
    return ols

#---------------------------#
# Linear Regression Classes #
#---------------------------#

class LinearRegOLS():
    """
    Linear Regression Class with closed form solution, Ordinary Least Squares (OLS)
    """

    def train(self, X_train, y_train, X_val=None, y_val=None, loss='mse') -> dict:
        """
        Train linear model on the training dataset and evaluate loss bith on validation set.

        Arg
        ----
        metric : str
            - 'mse'
            - 'rmse'
            - 'mae'
            - 'r2'

        Returns
        ----
        ols_results : dict
            - 'loss_train' loss on the training dataset
            - 'loss_val' loss on the validation dataset
        """
        self.y_train = y_train
        ols_results = {}
        coeffs_ols = ordinary_least_square(X_train, y_train)
        self.coeffs_ = coeffs_ols

        # -------- Compute Loss / Val -------- #
        train_loss = evaluate_score(y_train, pred(X_train, self.coeffs_), metric=loss, task='regression')

        if X_val.all()!=None and y_val.all()!=None:
            val_loss = evaluate_score(y_train, pred(X_train, self.coeffs_), metric=loss, task='regression')
            ols_results['train_loss'], ols_results['val_loss'] = train_loss, val_loss
        else:
            ols_results['train_loss'] = train_loss

        return ols_results

    def evaluate(self, X_test, y_test, metric='r2') -> float:
        """
        Evaluates model's performance according to selected metric and compare the result to a baseline.

        Arg
        ----
        metric : str
            - 'mse'
            - 'rmse'
            - 'mae'
            - 'r2'

        Returns
        ----
        test_score : float
            - Score on the test dataset
        """
        # ------------ Baseline ------------ #
        y_train = self.y_train
        self.baseline = baseline_score(y_train, y_test, metric=metric, task='regression')

        # -------- Model Evaluation -------- #
        y_pred = pred(X_test, self.coeffs_)
        test_score = evaluate_score(y_true=y_test, y_pred=y_pred, metric=metric, task='regression')

        if metric=='r2':
            if self.baseline < test_score:
                print(f"✅ Model performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")
            else:
                print(f"❌ Model did not performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")

        else:
            if self.baseline > test_score:
                print(f"✅ Model performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")
            else:
                print(f"❌ Model did not performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")

        return test_score

    def predict(self, X_new) -> np.ndarray:
        """
        Function that predicts target values with the weights computed during the training phase.

        Returns
        ----
        predictions : np.ndarray
            - array of prediction on new observation(s)
        """
        predictions = pred(X_new, self.coeffs_)
        return predictions

class LinearRegSGD():
    """
    Linear Regression Class with Gradient Descent
    """

    def __init__(self, loss='mse'):
        self.loss = loss

    def train(self,
              X_train,
              y_train,
              X_val,
              y_val,
              epochs=10,
              learning_rate=0.01,
              #  batch_size=None, # TODO Implement batch_size
              early_stopping=None,
              verbose=True) -> dict:
        """
        Train the model on the training dataset and evaluates on validation set.

        Args
        ----
        epoch : int
            - Number of times the model sees the full training dataset.

        learning_rate : float
            - Size of parameters updates.

        batch_size : int
            - Number of observations needed to updated model's parameters.

        early_stopping : int
            - Indicates to stop training after the number of times the val loss hasn't increases in a row.

        verbose : bool
            - If True, output train and val loss at each epoch

        Returns
        ----
        history : dict
            - 'loss_train_history' list of loss on the training dataset at each epoch
            - 'loss_val_history' list of loss on the validation dataset at each epoch
            - 'coeffs_history' list of weights (np.ndarray) calculated at each epoch
        """
        nb_features = X_train.shape[1] # Number of features
        coeffs = np.zeros(shape=(nb_features,)) # Weights initialization

        loss_train_history = []
        loss_val_history = []
        coeffs_history = []
        early_stopping_count = 0

        print(f"Loss function: {self.loss}")

        # if isinstance(y_train, pd.Series):
        #     y_train = y_train.to_numpy()

        for epoch in range(epochs):
            # ----------- Compute Train / Val Loss ----------- #
            # indices = np.random.permutation(n_train)
            # X_train = X_train[indices]
            # y_train = y_train[indices]

            # ----------- Compute Train / Val Loss ----------- #
            loss_train = loss_function(y_true=y_train, y_pred=pred(X_train, coeffs), loss=self.loss)
            loss_val = loss_function(y_true=y_val, y_pred=pred(X_val, coeffs), loss=self.loss)

            # --------------- Loss Histories ----------------- #
            loss_train_history.append(loss_train)
            loss_val_history.append(loss_val)

            if verbose==True:
                print(f"Epoch{epoch+1}: Train loss = {loss_train}, Val loss = {loss_val}")

            # -------------- Gradient Computation ------------- #
            gradients = gradient(X_train, y_train, coeffs, loss=self.loss)
            coeffs_history.append(coeffs)
            coeffs = coeffs - (learning_rate * gradients)

            # ------------------ Early Stopping -------------- #
            if early_stopping != None:
                if epoch > 0:
                    if loss_val > loss_val_history[-2]:
                        early_stopping_count += 1
                        if early_stopping_count >= early_stopping:
                            coeffs_history = coeffs_history[:len(coeffs_history) - early_stopping] # Restore Best Weights
                            break
                    else:
                        early_stopping_count = 0

        if early_stopping==None:
            print(f"✅ Linear regresion model successfully trained on {epochs} epochs")
        else:
            if early_stopping_count >= early_stopping:
                print(f"✅ Linear regresion model successfully trained, early-stopping stopped training phase at the epoch {epoch+1}")
            else:
                print(f"✅ Linear regresion model successfully trained on {epoch+1} epochs")

        # ----------------- History Dict -------------------#
        history = {}
        history['loss_train_history'] = loss_train_history
        history['loss_val_history'] = loss_val_history
        history['coeffs_history'] = coeffs_history

        self.history = history
        self.best_coeffs_ = history['coeffs_history'][-1]
        self.y_train = y_train

        return history

    def evaluate(self, X_test, y_test, metric='mse') -> float :
        """
        Evaluates model's performance according to selected metric and compare the result to a baseline.

        Arg
        ----
        metric : str
            - 'mse'
            - 'rmse'
            - 'mae'
            - 'r2'

        Returns
        ----
        test_score : float
            - Score on the test dataset
        """
        # ------ Baseline Computation ------ #
        y_train = self.y_train
        self.baseline = baseline_score(y_train, y_test, metric=metric, task='regression')

        # -------- Model Evaluation -------- #
        y_pred = pred(X_test, self.best_coeffs_)
        test_score = evaluate_score(y_true=y_test, y_pred=y_pred, metric=metric, task='regression')

        if metric=='r2':
            if self.baseline < test_score:
                print(f"✅ Model performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")
            else:
                print(f"❌ Model did not performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")

        else:
            if self.baseline > test_score:
                print(f"✅ Model performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")
            else:
                print(f"❌ Model did not performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {self.baseline})")

        return test_score

    def plot_learning_curves(self):
        """
        Plot learning curves which shows the scores of training and validation datasets for each epoch.
        """
        return plot_loss(self.history, self.baseline)

    def predict(self, X_new) -> np.ndarray:
        """
        Function that predicts target values with the weights computed during the training phase.

        Returns
        ----
        predictions : np.ndarray
            - array of prediction on new observation(s)
        """
        predictions = pred(X_new, self.best_coeffs_)
        return predictions

#-----------------------------#
# Logistic Regression Classes #
#-----------------------------#

def sigmoid(y_pred):
    """
    Compute sigmoïd function
    """
    sigmoid_ = 1 / (1 + np.exp(-1 * y_pred))
    return sigmoid_

class LogisticRegOLS():
    pass

class LogisticRegSGD():

    def __init__(self, loss='log_loss'):
        self.log_loss = loss

    def train(self):
        pass

    def evaluate(self):
        metric_train_history = []
        metric_val_history = []

        # for epoch in range(epochs):
        #     # ----------- Compute Train / Val Loss ----------- #
        #     loss_train = loss_function(y_true=y_train, y_pred=pred(X_train, coeffs), loss=self.loss)
        #     loss_val = loss_function(y_true=y_val, y_pred=pred(X_val, coeffs), loss=self.loss)

        #     # --------------- Loss Histories ----------------- #
        #     loss_train_history.append(loss_train)
        #     loss_val_history.append(loss_val)

        #     # ----------- Compute Train / Val Metric ----------- #
        #     metric_train_history = evaluate_score(y_true=y_train, y_pred=pred(X_train, coeffs), metric='r2', task='classification')
        #     metric_train_history = loss_function(y_true=y_val, y_pred=pred(X_val, coeffs), loss=self.loss)

        #     # --------------- Metric Histories ----------------- #
        #     loss_train_history.append(loss_train)
        #     loss_val_history.append(loss_val)
        pass

    def plot_learning_curves(self, plot_metric=True):
        """
        Plot learning curves which shows the scores of training and validation datasets for each epoch.

        Arg
        ---
        plot_metric : bool
            - Allows to choose whether to returns both loss and metric curves plots or only the loss curves.
        """
        # if plot_metric==True:
        #     print("Model's training and validation losses and metric plotted:")
        #     return plot_loss_metric(self.history, self.baseline)
        # else:
        #     print("Model's training and validation losses plotted:")
        #     return plot_loss(self.history, self.baseline)
        pass

    def predict(self):
        pass
