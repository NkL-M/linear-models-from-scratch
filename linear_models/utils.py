import matplotlib.pyplot as plt
import numpy as np
#TODO - Create Seaborn or Plotly.express version of these functions
# import seaborn as sns
# import plotly.express as px


def old_plot(loss_train_history, loss_val_history):
    # Plot train and test histories
    plt.plot(loss_train_history, label = 'training loss')
    plt.plot(loss_val_history, label = 'validation loss')

    # Set title and labels
    plt.title('Learning Curves Train/Val Loss - Hand-Made Model')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')

    # Change limits
    plt.ylim(ymin = 00, ymax = 15_000)

    # Generate legend
    plt.legend()

def plot_loss(history):
    """
    Function to plot the loss function history

    Returns
    ----
    Plot
    """
    # --- LOSS --- #
    plt.figure(figsize=(7,4))
    plt.plot(history['loss_train_history'])
    plt.plot(history['loss_val_history'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # inf = np.min(history['loss_train_history'])#-1000
    # sup = np.max(history['loss_train_history']) + (np.max(history['loss_train_history']) * 1.1)
    # plt.ylim((inf, sup))

    plt.legend(['Training loss', 'Validation loss'], loc='best')

    plt.grid(axis="x",linewidth=0.5)
    plt.grid(axis="y",linewidth=0.5)

def plot_loss_metric(history):
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

    # ax[0].set_ylim((0,3))

    ax[0].legend(['Train', 'Val'], loc='best')

    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # # --- METRIC --- #
    # ax[1].plot(history['metric_train_history'])
    # ax[1].plot(history['metric_val_history'])

    ax[1].set_title('Model metric')
    ax[1].set_ylabel('Metric')
    ax[1].set_xlabel('Epoch')

    ax[1].legend(['Train', 'Val'], loc='best')

    # ax[1].ylim((0,1))

    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)
