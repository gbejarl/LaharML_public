##############################
# %% 1 Import packages
# 1 Import packages
##############################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy import signal

from laharml_modules import (extract_from_local_directory,
                             train_test_knn,
                             predict_knn,
                             clean_detections,
                             retrieve_dates,
                             plot_detections)

##############################
# %% 2 Initial parameters
# 2 Initial parameters
##############################


def laharml_traintest():
