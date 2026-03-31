"""
Module for plots
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history, baseline):
    """
    Function to plot the loss function history

    Returns
    ----
    Plot
    """
    plt.figure(figsize=(15,7))
    plt.plot(history['loss_train_history'])
    plt.plot(history['loss_val_history'])
    plt.axhline(y=baseline, color='red', linestyle='--', linewidth=0.8, label='aze')

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Training loss', 'Validation loss', 'Baseline'], loc='best')

    plt.grid(axis="x",linewidth=0.5)
    plt.grid(axis="y",linewidth=0.5)

def plot_loss_metric(history, baseline):
    """
    Function to plot the loss and metric histories

    Returns
    ----
    Subplots
    """
    fig, ax = plt.subplots(1,2, figsize=(15,5))

    # --- LOSS --- #
    ax[0].plot(history['loss_train_history'])
    ax[0].plot(history['loss_val_history'])

    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')

    ax[0].hlines(y=baseline, xmin=-20, xmax=len(history['loss_train_history']), colors='red', linestyles='--', linewidth=0.8)

    ax[0].legend(['Training', 'Validation', 'Basline'], loc='best')

    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- METRIC --- # TODO Test with metrics in Logistic Regression
    # ax[1].plot(history['metric_train_history'])
    # ax[1].plot(history['metric_val_history'])

    ax[1].set_title('Model metric')
    ax[1].set_ylabel('Metric')
    ax[1].set_xlabel('Epoch')

    ax[1].legend(['Training', 'Validation', 'Basline'], loc='best')

    ax[1].hlines(y=baseline, xmin=-20, xmax=len(history['loss_train_history']), colors='red', linestyles='--', linewidth=0.8)

    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)
