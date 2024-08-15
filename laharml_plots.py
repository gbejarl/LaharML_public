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

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=200)
    sns.scatterplot(data=dataframe_copy,
                    x='Times',
                    y=variable,
                    hue=hue,
                    ax=ax,
                    edgecolor='none')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.show()

#
