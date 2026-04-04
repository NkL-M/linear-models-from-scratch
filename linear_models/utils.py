"""
Module for plots
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history, baseline, early_stopping=None, es_count=None):
    """
    Function to plot the loss function history.
    Plot the early stopping as a vertical green line and the baseline score as a horizontal red line.

    Returns
    ----
    Matplotlib Plot
    """
    plt.figure(figsize=(15,7))
    plt.plot(history['loss_train_history'])
    plt.plot(history['loss_val_history'])
    plt.axhline(y=baseline, color='red', linestyle='--', linewidth=0.8)

    if early_stopping != None and es_count >= early_stopping:
        plt.axvline(x=len(history['loss_train_history'])-early_stopping, color='green', linestyle='--', linewidth=0.8)

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Training loss', 'Validation loss', 'Baseline', 'Early stopping'], loc='best')

    plt.grid(axis="x",linewidth=0.5)
    plt.grid(axis="y",linewidth=0.5)

def plot_loss_metric(history, baseline, metric_name=None, early_stopping=None, es_count=None):
    """
    Function to plot the loss and metric histories.
    Plot the early stopping as a vertical green line and the baseline score as a horizontal red line.

    Returns
    ----
    Matplotlib Subplots
    """
    fig, ax = plt.subplots(1,2, figsize=(15,5))

    # --- LOSS --- #
    ax[0].plot(history['loss_train_history'])
    ax[0].plot(history['loss_val_history'])

    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')

    if early_stopping != None and es_count >= early_stopping:
        ax[0].axvline(x=len(history['loss_train_history']) - early_stopping, color='green', linestyle='--', linewidth=0.8)

    ax[0].legend(['Training', 'Validation', 'Early stopping'], loc='best')

    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- METRIC --- #
    ax[1].plot(history['metric_train_history'])
    ax[1].plot(history['metric_val_history'])

    ax[1].set_title(f'Model metric ({metric_name})')
    ax[1].set_ylabel(f'{metric_name}')
    ax[1].set_xlabel('Epoch')

    ax[1].axhline(y=baseline, color='red', linestyle='--', linewidth=0.8)

    ax[1].legend(['Training', 'Validation', 'Baseline', 'Early stopping'], loc='best')

    if early_stopping != None and es_count >= early_stopping:
        ax[1].axvline(x=len(history['metric_train_history']) - early_stopping, color='green', linestyle='--', linewidth=0.8)

    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)
