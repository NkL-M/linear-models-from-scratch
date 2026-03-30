import numpy as np
from linear_models.src.metrics import *
from linear_models.utils import *

"""
Module for Linear Models built from scratch with libraires like Scikit-Learn
"""

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
    loss_dict = {'mse' : mse(y_true, y_pred),
                 'log_loss' : log_loss(y_true, y_pred)}

    if loss in loss_dict.keys():
        loss_result = loss_dict[loss]
        return loss_result

    else:
        raise ValueError(f"Wrong Input, unknown loss function '{loss}'")

def evaluate_score(y_true, y_pred, metric='r2', task='regression'):
    """
    Function that computes the errors of a model based on a selected metric.

    Parameter
    ----
    metric : str

    Regression metrics
        - 'mse'
        - 'rmse'
        - 'mae'
        - 'r2'
    Classification metrics
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
    regression_dict = {'mse' : mse(y_true, y_pred),
                       'rmse' : rmse(y_true, y_pred),
                       'mae' : mae(y_true, y_pred),
                       'r2' : r_squared(y_true, y_pred)}

    # TODO Finish regression metrics
    clasification_dict = {'accuracy' : accuracy(y_true, y_pred)}#,
    #                       'recall' : recall(y_true, y_pred),
    #                       'precision' : precision(y_true, y_pred),
    #                       'f1' : f1(y_true, y_pred)}

    if task == 'regression':
        if metric in regression_dict.keys():
            eval_score = regression_dict[metric]
            return round(eval_score, 4)
        else:
            raise ValueError(f"Wrong Input, unknown regression metric '{metric}'")

    elif task == 'classification':
        if metric in clasification_dict.keys():
            eval_score = clasification_dict[metric]
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
        - 'mse'
        - 'rmse'
        - 'mae'
        - 'r2'

    Returns
    ----
    baseline_score : float
    """
    #TODO Adapt function for logistic regression
    # if task=='regression':
    #     baseline_pred = np.mean(y_train)
    #     baseline_score = evaluate_score(y_true=y_test, y_pred=baseline_pred, metric=metric, task=task)
    baseline_pred = np.mean(y_train)
    baseline_score = evaluate_score(y_true=y_test, y_pred=baseline_pred, metric=metric, task=task)
    return float(baseline_score)


#-------------------------#
# Linear Regression Class #
#-------------------------#
class LinearRegOLS():
    """
    Linear Regression Class with closed solition, Ordinary Least Squares (OLS)
    """

    def __init__(self):
        pass

    def train(self, X_train, y_train):
        # formula = np.linalg.inv(np.matmul(X_train.T, X_train)) * np.matmul(X_train.T, y_train)
        # return formula
        pass

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
              verbose=True) -> tuple:
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
        # loss = self.loss

        loss_train_history = []
        loss_val_history = []
        coeffs_history = []
        early_stopping_count = 0

        print(f"Loss function: {self.loss}")

        for epoch in range(epochs):
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
            # coeffs -= learning_rate * gradients # TODO This one render history coef impossible
            coeffs = coeffs - (learning_rate * gradients)

            # ------------------ Early Stopping -------------- #
            if early_stopping != None:
                if epoch > 0:
                    if loss_val > loss_val_history[-2]:
                        early_stopping_count += 1
                        if early_stopping_count >= early_stopping:
                            # ------ Restore Best Weight ------ #
                            coeffs_history = coeffs_history[ : len(coeffs_history) - early_stopping] # TODO verify the Restore Best Weights
                            # print(f"Test with removal : {coeffs_history_n[-1]}")
                            # print(f"Test without removal : {coeffs_history[-1]}")
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

        # -------------- Variables to Pass ---------------#
        self.best_coeffs = history['coeffs_history'][-1]
        self.y_train = y_train
        self.history = history

        return history

    def evaluate(self, X_test, y_test, metric='mse') -> float :
        """
        Evaluates model's performance according to selected metric and compare the result to a baseline.

        Arg
        ----
        metric : str

        Returns
        ----
        test_score : float
            - Score on the test dataset
        """
        # ------ Baseline Computation ------ #
        y_train = self.y_train
        score_baseline = baseline_score(y_train, y_test, metric=metric, baseline_task='regression')

        # -------- Model Evaluation -------- #
        y_pred = pred(X_test, self.best_coeffs)
        test_score = evaluate_score(y_true=y_test, y_pred=y_pred, metric=metric, task='regression')


        if metric=='r2':
            if score_baseline < test_score:
                print(f"✅ Model performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {score_baseline})")
            else:
                print(f"❌ Model did not performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {score_baseline})")

        else:
            if score_baseline > test_score:
                print(f"✅ Model performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {score_baseline})")
            else:
                print(f"❌ Model did not performed better ({metric}_score = {test_score}) than the baseline score ({metric}_score = {score_baseline})")

        return test_score

    def plot_learning_curves(self, plot_metric=True):
        """
        Plot learning curves which shows the scores of training and validation datasets for each epoch.

        Arg
        ---
        plot_metric : bool
            - Allows to choose whether to returns both loss and metric curves plots or only the loss curves.
        """
        if plot_metric==True:
            print("Model's training and validation losses and metric plotted:")
            return plot_loss_metric(self.history)
        else:
            print("Model's training and validation losses plotted:")
            return plot_loss(self.history)

    def predict(self, X_new) -> np.ndarray:
        """
        Function that predicts target values with the weights computed.

        Returns
        ----
        predictions : np.ndarray
            - array of prediction on new observation(s)
        """
        predictions = pred(X_new, self.best_coeffs)
        return predictions

# TODO Add a coeffs_, params_, epochs_, etc...

    def kfold_cross_validation(X, y, fold=5):
        pass

#---------------------------#
# Logistic Regression Class #
#---------------------------#

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
        pass

    def plot_learning_curves(self):
        pass

    def predict(self):
        pass
