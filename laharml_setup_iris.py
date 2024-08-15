##############################
# %% 1 Import packages
# 1 Import packages
##############################

import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy import signal

from laharml_modules import (preprocess_stream,
                             extract_samples,
                             train_test_knn)

##############################
# %% 2 Initial parameters
# 2 Initial parameters
##############################

# Set FDSN client parameters
client_id = 'IRIS'
username = 'gbejarlo@mtu.edu'
token = 'GiHPgTRUXbvde2l0'

# Set model name
model_name = 'FuegoExample'

# Set station parameters
network = '6Q'
station = 'FEC1'
location = ''
channel = 'HHZ'

# Load files with lahar events and noise events
list_of_lahars = '/Users/gustavo/Developer/LaharML_public/events_lahar.csv'
list_of_noise = '/Users/gustavo/Developer/LaharML_public/events_noise.csv'

# Set model parameters
features = [4, 6, 34, 39, 44]
window_length = 10  # [required] in minutes
window_overlap = 0.75  # [required] in fraction of window length

# Set other classifier parameters
remove_response = True  # Remove instrument response
decimation_factor = 4  # Decimate seismic data by this factor
minimum_frequency = None  # High-pass filter above this frequency (in Hz)
maximum_frequency = None  # Low-pass filter above this frequency (in Hz)
minimum_lahar_duration = 30  # Filter out smaller events (in minutes)

# Other outputs
save_parameters = True  # Save sampled parameters
save_model = True  # Save trained model and scaler

##############################
# %% 3 Script: Train and test classifier
# 3 Script: Train and test classifier
##############################

# Load training lahar and noise events
list_lahar = pd.read_csv(list_of_lahars, header=None)
list_noise = pd.read_csv(list_of_noise, header=None)

# Set up FDSN client
client = Client(client_id, user=username, password=token)

training_lahar = []
training_noise = []
total_lahar_time = 0

# Get instrument response
try:
    inventory = client.get_stations(network=network,
                                    station=station,
                                    location=location,
                                    channel=channel,
                                    level='response')
    sensitivity = inventory[0][0][0].response.instrument_sensitivity.value
except:
    sensitivity = None

# Request waveforms for lahar events
for i in list_lahar.index:
    request_t1 = UTCDateTime(list_lahar.loc[i][0])
    request_t2 = UTCDateTime(list_lahar.loc[i][1])
    stream = client.get_waveforms(network,
                                  station,
                                  location,
                                  channel,
                                  request_t1,
                                  request_t2)
    total_lahar_time += stream[-1].stats.endtime - stream[0].stats.starttime
    training_lahar.append(stream)
    print(
        f'Training lahar event {i+1} of {len(list_lahar)} requested.', flush=True)
print(f'---', flush=True)

duration_per_noise = total_lahar_time / len(list_noise)

# Request waveforms for noise events
for i in list_noise.index:
    request_t1 = UTCDateTime(list_noise.loc[i][1])-(int(duration_per_noise/2))
    request_t2 = UTCDateTime(list_noise.loc[i][1])+(int(duration_per_noise/2))
    stream = client.get_waveforms(network,
                                  station,
                                  location,
                                  channel,
                                  request_t1,
                                  request_t2)
    training_noise.append(stream)
    print(
        f'Training noise signal {i+1} of {len(list_noise)} requested.', flush=True)
print(f'---', flush=True)

# Preprocess training data (lahars)
for i in range(len(training_lahar)):
    training_lahar[i] = preprocess_stream(training_lahar[i],
                                          starttime=None,
                                          endtime=None,
                                          taper=False,
                                          decimate=decimation_factor,
                                          freqmin=minimum_frequency,
                                          freqmax=maximum_frequency,
                                          sensitivity=sensitivity)
print(f'Preprocessed routine in lahar training data done.', flush=True)
print(f'---', flush=True)

# Preprocess training data (noise)
for i in range(len(training_noise)):
    training_noise[i] = preprocess_stream(training_noise[i],
                                          starttime=None,
                                          endtime=None,
                                          taper=False,
                                          decimate=decimation_factor,
                                          freqmin=minimum_frequency,
                                          freqmax=maximum_frequency,
                                          sensitivity=sensitivity)
print(f'Preprocessed routine in noise training data done.', flush=True)
print(f'---', flush=True)

training = pd.DataFrame()

# Extract samples from training data (lahars
for i in range(len(training_lahar)):
    samples = extract_samples(training_lahar[i],
                              features=features,
                              window_length=window_length,
                              window_overlap=window_overlap)
    samples['Classification'] = 1
    training = pd.concat([training, samples], ignore_index=True, sort=False)
    print(
        f'Samples extracted for lahar {i+1} of {len(training_lahar)}.', flush=True)
print(f'---', flush=True)

# Extract samples from training data (noise)
for i in range(len(training_noise)):
    samples = extract_samples(training_noise[i],
                              features=features,
                              window_length=window_length,
                              window_overlap=window_overlap)
    samples['Classification'] = 0
    training = pd.concat([training, samples], ignore_index=True, sort=False)
    print(
        f'Samples extracted for noise {i+1} of {len(training_noise)}.', flush=True)
print(f'---', flush=True)

# Train/test classifier

model, scaler, classification_report, confusion_matrix, neighbors = \
    train_test_knn(training,
                   scale=True,
                   get_n=True,
                   plot_n=True)
print(f'Training-testing complete.', flush=True)
print(f'---', flush=True)

model_filename = f'{model_name}_model.joblib'
scaler_filename = f'{model_name}_scaler.joblib'
joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)

input_parameters = {
    'client': client_id,
    'network': network,
    'station': station,
    'location': location,
    'channel': channel,
    'features': features,
    'window_length': window_length,
    'window_overlap': window_overlap,
    'remove_response': remove_response,
    'decimation_factor': decimation_factor,
    'minimum_frequency': minimum_frequency,
    'maximum_frequency': maximum_frequency,
    'minimum_lahar_duration': minimum_lahar_duration,
    'model_name': model_name,
    'model_filename': model_filename,
    'scaler_filename': scaler_filename
}

# Save the dictionary to a JSON file
with open(f'{model_name}.json', 'w') as json_file:
    json.dump(input_parameters, json_file, indent=4)
# %%
