import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


def data_loader(task='regression') -> tuple:
    """
    Function that load simple datasets for this project.
        'regression': Import the diabetes dataset from sklearn
        'classification': Import the...

    Parameter
    ----
    task : str
        - 'regression'
            Import the diabetes dataset from sklearn's datasets module
        - 'classification'
            Import the breast cancer dataset from sklearn's datasets module

    Returns
    ----
    X, y : tuple
    Tuple of feature dataset X (Pandas.DataFrame) and a target vector y (Pandas.Series)
    """
    if task=="regression":
        X, y = datasets.load_diabetes(return_X_y = True, as_frame = True)
        print(f"The feature matrix has {X.shape[0]} cols and {X.shape[1]} rows")
        print(f"The target vector has {y.shape[0]} values")
        return X, y

    elif task=="classification":
        X, y = datasets.load_breast_cancer(return_X_y=True, as_frame = True)
        print(f"The feature matrix has {X.shape[0]} cols and {X.shape[1]} rows")
        print(f"The target vector has {y.shape[0]} values")
        return X, y

    else:
        raise ValueError(f"Dataset for the {task} task not found")


def features_dataset(X) -> np.ndarray:
    """
    Function that create a feature dataset ready to be inputed in an ML Algorithm.

    Returns
    ----
    X_mat : np.ndarray
    Feature matrix
    """
    X_mat = np.hstack((np.ones((X.shape[0], 1)), X))
    assert X_mat.shape[1] == X.shape[1] + 1, 'Error in feature matrix shape'
    return X_mat


def split_data(X_mat, y, test_size=0.2) -> tuple:
    """
    Function that split the data into a train, val and test datasets

    Parameter
    ----
    test_size : float

    Returns
    ----
    X_train, X_val, X_test, y_train, y_val, y_test : tuple
    """
    X_train_tmp, X_test, y_train_tmp, y_test = train_test_split(X_mat,
                                                                y,
                                                                test_size=test_size,
                                                                random_state=42,
                                                                shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_tmp,
                                                      y_train_tmp,
                                                      test_size=0.12,
                                                      random_state=42,
                                                      shuffle=True)

    print(f"X_train: {X_train.shape} - y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape} - y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape} - y_test: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data(X, y, test_size=0.2) -> tuple:
    """
    Function that get the data ready for ML algorithms and split the data into
    a train, val and test datasets.

    Arg
    ----
    test_size : float

    Returns
    ----
    (X_train, X_val, X_test, y_train, y_val, y_test) : tuple
    """
    X_mat = features_dataset(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_mat, y, test_size=test_size)

    print("Train - Val - Test split ratio done ✅")
    print(f"Initial dataset size: {X_mat.shape[0]} rows")
    print(f"Test set is {round(test_size * 100)}% of the full dataset")
    print(f"Val set is {round(X_val.shape[0] / (X_mat.shape[0] / 100))}% of the full dataset")
    print(f"Train set is {round(X_train.shape[0] / (X_mat.shape[0] / 100))}% of the full dataset")

    return X_train, X_val, X_test, y_train, y_val, y_test
