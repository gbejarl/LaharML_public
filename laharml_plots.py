import obspy.signal.filter
import os
import sys
import obspy
import sklearn
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy import signal
from obspy import UTCDateTime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib as mpl

import seaborn as sns

# %% plot_sampled


def plot_sampled(dataframe,
                 variable,
                 hue):

    dataframe_copy = dataframe.copy()
    dataframe_copy['Times'] = pd.to_datetime(dataframe_copy['Times'], unit='s')

    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=200)
    sns.scatterplot(data=dataframe_copy,
                    x='Times',
                    y=variable,
                    hue=hue,
                    ax=ax,
                    edgecolor='none')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.show()

# %% plot_interval


def plot_interval(trace,
                  starttime=None,
                  endtime=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=200)
    sns.lineplot(x=pd.to_datetime(trace.times('timestamp'), unit='s'),
                 y=trace.data,
                 ax=ax,
                 color='black')
    if starttime and endtime:
        ax.axvspan(starttime, endtime, color='red', alpha=0.15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(formatter)
    # Get the x-axis tick labels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    # Modify the first label
    labels[0] = pd.to_datetime(trace.times('timestamp')[
                               0], unit='s').strftime('%H:%M\n%Y-%m-%d')
    # Set the modified labels back to the x-axis
    ax.set_xticklabels(labels)
    ax.set_xlim(pd.to_datetime(trace.times('timestamp')[0], unit='s'),
                pd.to_datetime(trace.times('timestamp')[-1], unit='s'))
    ax.set_ylabel('m/s')
    ax.set_ylim([-5e-4, 5e-4])
    plt.show()
