"""
Module for Linear Regression and Logistic Regression built from scratch without ML librairies like Scikit-Learn.
"""

import numpy as np
import pandas as pd
from collections import Counter
from linear_models.src.metrics import *
from linear_models.utils import *
from linear_models.src.basic import *

#---------------------------#
# Linear Regression Classes #
#---------------------------#

class LinearRegOLS():
    """
    Linear Regression Class with closed form solution, Ordinary Least Squares (OLS).
    """

    def train(self, X_train, y_train, X_val=None, y_val=None, loss='mse') -> dict:
        """
        Train linear model on the training dataset and evaluate loss bith on validation set.

        Parameter
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
        assert (X_train == 1).all(axis=0).any(), "No intercept column within the training dataset"

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

        Parameter
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

class LinearRegressionGD():
    """
    Linear Regression Class with an iterative solution (i.e. Gradient Descent).
    """

    def __init__(self, loss='mse'):
        self.loss = loss

    def train(self,
              X_train,
              y_train,
              X_val,
              y_val,
              epochs=100,
              learning_rate=0.01,
              batch_size=None,
              early_stopping=None,
              verbose=True) -> dict:
        """
        Train the model on the training dataset and evaluates on validation set.

        Parameters
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
            - If True, output train and val loss and metric at each epoch.

        Returns
        ----
        history : dict
            - 'loss_train_history' list of loss on the training dataset at each epoch.
            - 'loss_val_history' list of loss on the validation dataset at each epoch.
            - 'metric_train_history' list of metric on the training dataset at each epoch.
            - 'metric_val_history' list of metric on the validation dataset at each epoch.
            - 'coeffs_history' list of weights calculated at each epoch.
        """
        assert (X_train == 1).all(axis=0).any(), "No intercept column within the training dataset" # Check wether there is a intercept column full of ones

        coeffs = np.zeros(X_train.shape[1]) # Weights initialization, with shape number of features
        n = X_train.shape[0]

        loss_train_history = []
        loss_val_history = []
        coeffs_history = []
        early_stopping_count = 0

        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()

        for epoch in range(epochs):
            # ----------- Compute Train / Val Loss ----------- #
            loss_train = loss_function(y_true=y_train, y_pred=pred(X_train, coeffs), loss=self.loss)
            loss_val = loss_function(y_true=y_val, y_pred=pred(X_val, coeffs), loss=self.loss)

            # --------------- Loss Histories ----------------- #
            loss_train_history.append(loss_train)
            loss_val_history.append(loss_val)

            if verbose==True:
                print(f"Epoch {epoch+1}: Train loss = {loss_train}, Val loss = {loss_val}")

            if batch_size==None or batch_size==0:
                # -------------- Full Batch Gradient Compute ------------- #
                gradients = gradient(X_train, y_train, coeffs, loss=self.loss)
                coeffs = coeffs - (learning_rate * gradients)
                coeffs_history.append(coeffs)

            else:
                # ---------- Shuffling Training Datasets ---------- #
                indices = np.random.permutation(n)
                X_train = X_train[indices]
                y_train = y_train[indices]

                for batch in range(0, n, batch_size):
                    X_mini = X_train[batch:batch+batch_size]
                    y_mini = y_train[batch:batch+batch_size]

                    # -------------- Mini Batch Gradient Compute ------------- #
                    gradients = gradient(X_mini, y_mini, coeffs, loss=self.loss)

                    # -------------- Mini Batch Update coeffs ------------- #
                    coeffs = coeffs - (learning_rate * gradients)

                coeffs_history.append(coeffs)

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

        # ---------- Prints ---------- #
        print(f"-------------- {int(n / batch_size) + (n % batch_size > 0)} Batchs computed per epoch --------------")
        if early_stopping == None:
            if batch_size != None or batch_size != 0:
                print(f"✅ Linear regresion model successfully trained. Epochs: {epochs}, Batch size: {batch_size}")
            else:
                print(f"✅ Linear regresion model successfully trained. Epochs: {epochs}")
        else:
            if early_stopping_count >= early_stopping:
                if batch_size != None or batch_size != 0:
                    print(f"✅ Linear regresion model successfully trained. Epochs: {epoch+1} (Early-stopping), Batch size: {batch_size}")
                else:
                    print(f"✅ Linear regresion model successfully trained. Epochs: {epoch+1} (Early-stopping)")
            else:
                if batch_size != None or batch_size != 0:
                    print(f"✅ Linear regresion model successfully trained. Epochs: {epoch+1} (No Early-stopping), Batch size: {batch_size}")
                else:
                    print(f"✅ Linear regresion model successfully trained. Epochs: {epoch+1} (No Early-stopping)")

        # ----------------- History Dict -------------------#
        history = {}
        history['loss_train_history'] = loss_train_history
        history['loss_val_history'] = loss_val_history
        history['coeffs_history'] = coeffs_history

        self.history = history
        self.early_stopping = early_stopping
        self.early_stopping_count = early_stopping_count
        self.best_coeffs_ = history['coeffs_history'][-1]
        self.y_train = y_train

        return history

    def evaluate(self, X_test, y_test, metric='mse') -> float :
        """
        Evaluates model's performance according to selected metric and compare the result to a baseline.

        Parameter
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
        self.baseline = baseline_score(self.y_train, y_test, metric=metric, task='regression')

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
        return plot_loss(self.history, self.baseline, early_stopping=self.early_stopping, es_count=self.early_stopping_count)

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

class LogisticReg():
    """
    Logistic Regression Class with closed form solution. # TODO specify binary or multiclass predictions
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def train(self, X_train, y_train, X_val=None, y_val=None, metric='accuracy') -> dict:
        """
        Train logistic regression model on the training dataset and evaluate loss bith on validation set.

        Parameter
        ----
        metric : str
            - 'accuracy'
            - 'recall'
            - 'precision'
            - 'f1'

        Returns
        ----
        logit_results : dict
            - 'loss_train' loss on the training dataset
            - 'loss_val' loss on the validation dataset
        """
        assert (X_train == 1).all(axis=0).any(), "No intercept column within the training dataset"

        self.y_train = y_train
        logit_results = {}
        self.coeffs_ = ordinary_least_square(np.array(X_train), np.array(y_train))
        self.metric = metric


        # -------- Compute Train Scores -------- #
        probas_train = sigmoid(pred(X_train, self.coeffs_))
        class_preds = pd.Series([1 if proba > self.threshold else 0 for proba in probas_train])
        train_score = evaluate_score(y_train, class_preds, metric=metric, task='classification')

        # --------- Compute Val Scores --------- #
        # if X_val.all()!=None and y_val.all()!=None:
        probas_val = sigmoid(pred(X_val, self.coeffs_))
        class_preds_val = pd.Series([1 if proba > self.threshold else 0 for proba in probas_val])
        val_score = evaluate_score(y_val, class_preds_val, metric=metric, task='classification')
        # else:
        #     logit_results['train_score'] = train_score

        logit_results['train_score'], logit_results['val_score'] = train_score, val_score

        return logit_results

    def evaluate(self, X_test, y_test, metric=None) -> float:
        """
        Evaluates model's performance according to selected metric and compare the result to a baseline.

        Parameter
        ----
        metric : str
            -  None -> will use the same metrics defined in train()
            - 'accuracy'
            - 'recall'
            - 'precision'
            - 'f1'

        Returns
        ----
        test_score : float
            - Score on the test dataset
        """
        if metric!=None:
            current_metric=metric
        else:
            current_metric=self.metric

        # ------ Baseline Computation ------ #
        self.baseline = baseline_score(self.y_train, y_test, metric=current_metric, task='classification')

        # -------- Model Evaluation -------- #
        probas_test = sigmoid(pred(X_test, self.coeffs_))
        class_preds_test = pd.Series([1 if proba > self.threshold else 0 for proba in probas_test])
        test_score = evaluate_score(y_true=y_test, y_pred=class_preds_test, metric=current_metric, task='classification')

        if self.baseline < test_score:
            print(f"✅ Model performed better ({current_metric}_score = {test_score}) than the baseline score ({current_metric}_score = {self.baseline})")
        else:
            print(f"❌ Model did not performed better ({current_metric}_score = {test_score}) than the baseline score ({current_metric}_score = {self.baseline})")

        return test_score

    def predict(self, X_new):
        """
        Function that predicts target class with the weights computed during the training phase.

        Returns
        ----
        predictions : np.ndarray
            - array of class prediction on new observation(s)
        """
        probabilities = sigmoid(pred(X_new, self.coeffs_))
        class_predictions = pd.Series([1 if proba > self.threshold else 0 for proba in probabilities])
        return class_predictions

class LogisticRegressionGD():
    """
    Logistic Regression Algorithm with Gradient Descent. TODO add description
    """

    def __init__(self, threshold=0.5, loss='log_loss'):
        self.threshold = threshold
        self.loss = loss

    def train(self,
              X_train,
              y_train,
              X_val=None,
              y_val=None,
              learning_rate=0.1,
              epochs=10,
              metric='accuracy',
              # batch_size=None,  # TODO Add Mini Batch Gradient Descent
              early_stopping=None,
              verbose=True
              ):
        """
        Train the model on the training dataset and evaluates on validation set.

        Parameters
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
            - If True, output train and val loss and metric at each epoch.

        Returns
        ----
        history : dict
            - 'loss_train_history' list of loss on the training dataset at each epoch.
            - 'loss_val_history' list of loss on the validation dataset at each epoch.
            - 'metric_train_history' list of metric on the training dataset at each epoch.
            - 'metric_val_history' list of metric on the validation dataset at each epoch.
            - 'coeffs_history' list of weights calculated at each epoch.
        """
        assert (X_train == 1).all(axis=0).any(), "No intercept column within the training dataset"
        coeffs = np.zeros(X_train.shape[1]) # Weights initialization, with shape number of features

        loss_train_history = []
        loss_val_history = []
        metric_train_history = []
        metric_val_history = []
        coeffs_history = []
        early_stopping_count = 0

        for epoch in range(epochs):
            # ------------- Predictions ------------ #
            probas_train = sigmoid(pred(X_train, coeffs))
            class_preds = pd.Series([1 if proba > self.threshold else 0 for proba in probas_train])
            probas_val = sigmoid(pred(X_val, coeffs))
            class_preds_val = pd.Series([1 if proba > self.threshold else 0 for proba in probas_val])

            assert(len(X_train)==len(class_preds)), 'Unequal Prediction Size'

            # ------------ Compute Loss ------------ #
            loss_train = loss_function(y_true=y_train, y_pred=probas_train, loss=self.loss)
            loss_val = loss_function(y_true=y_val, y_pred=probas_val, loss=self.loss)

            # ------------ Compute Metric ---------- #
            metric_train = evaluate_score(y_true=y_train, y_pred=class_preds, metric=metric, task='classification')
            metric_val = evaluate_score(y_true=y_val, y_pred=class_preds_val, metric=metric, task='classification')

            # ------------ Loss Histories ---------- #
            loss_train_history.append(loss_train)
            loss_val_history.append(loss_val)

            # ---------- Metric Histories ---------- #
            metric_train_history.append(metric_train)
            metric_val_history.append(metric_val)

            if verbose==True:
                print(f"Epoch{epoch+1}: Train - [loss = {loss_train}, {metric} = {metric_train}], Val - [loss = {loss_val}, {metric} = {metric_val}]")

            # ---------- Gradient Compute ---------- #
            gradient_ = gradient(X_train, y_train, coeffs, loss=self.loss)
            coeffs = coeffs - (learning_rate * gradient_)
            coeffs_history.append(coeffs)

            # print(f"  coeffs range: [{coeffs.min():.3f}, {coeffs.max():.3f}]")
            # print(f"  gradient range: [{gradient_.min():.3f}, {gradient_.max():.3f}]")
            # print(f"  dot product range: [{np.dot(X_train, coeffs).min():.3f}, {np.dot(X_train, coeffs).max():.3f}]")

            # ---------- Early Stopping ---------- #
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
            print(f"✅ Logistic regresion model successfully trained on {epochs} epochs")
        else:
            if early_stopping_count >= early_stopping:
                print(f"✅ Logistic regresion model successfully trained, early-stopping stopped training phase at the epoch {epoch+1}")
            else:
                print(f"✅ Logistic regresion model successfully trained on {epoch+1} epochs")

        history = {}
        history['loss_train_history'] = loss_train_history
        history['loss_val_history'] = loss_val_history
        history['metric_train_history'] = metric_train_history
        history['metric_val_history'] = metric_val_history
        history['coeffs_history'] = coeffs_history

        self.y_train = y_train
        self.metric = metric
        self.coeffs_ = coeffs_history[-1]
        self.early_stopping = early_stopping
        self.es_count = early_stopping_count
        self.history = history
        return history

    def evaluate(self, X_test, y_test, metric=None):
        """
        Evaluates model's performance according to selected metric and compare the result to a baseline.

        Parameter
        ----
        metric : str
            -  None -> will use the same metrics defined in train()
            - 'accuracy'
            - 'recall'
            - 'precision'
            - 'f1'

        Returns
        ----
        test_score : float
            - Score on the test dataset
        """
        if metric!=None:
            current_metric=metric
        else:
            current_metric=self.metric

        # ------ Baseline Computation ------ #
        self.baseline = baseline_score(self.y_train, y_test, metric=current_metric, task='classification')

        # -------- Model Evaluation -------- #
        probas_test = sigmoid(pred(X_test, self.coeffs_))
        class_preds_test = pd.Series([1 if proba > self.threshold else 0 for proba in probas_test])
        test_score = evaluate_score(y_true=y_test, y_pred=class_preds_test, metric=current_metric, task='classification')

        if self.baseline < test_score:
            print(f"✅ Model performed better ({current_metric}_score = {test_score}) than the baseline score ({current_metric}_score = {self.baseline})")
        else:
            print(f"❌ Model did not performed better ({current_metric}_score = {test_score}) than the baseline score ({current_metric}_score = {self.baseline})")

        return test_score

    def plot_learning_curves(self, plot_metric=True):
        """
        Plot learning curves which shows the scores of training and validation datasets for each epoch.

        Parameter
        ---
        plot_metric : bool
            - Allows to choose whether to returns both loss and metric curves plots or only the loss curves.
        """
        if plot_metric==True:
            return plot_loss_metric(self.history, self.baseline, metric_name=self.metric, early_stopping=self.early_stopping, es_count=self.es_count)
        else:
            return plot_loss(self.history, self.baseline, early_stopping=self.early_stopping)

    def predict(self, X_new) -> np.ndarray:
        """
        Function that predicts target class with the weights computed during the training phase.

        Returns
        ----
        predictions : np.ndarray
            - array of class prediction on new observation(s)
        """
        probabilities = sigmoid(pred(X_new, self.coeffs_))
        class_predictions = pd.Series([1 if proba > self.threshold else 0 for proba in probabilities])
        return class_predictions
