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
                             predict_knn,
                             clean_detections,
                             retrieve_dates)

##############################
# %% 2 Initial parameters
# 2 Initial parameters
##############################

# Open model
model_name = 'FuegoExample'

# Set FDSN client parameters
username = 'gbejarlo@mtu.edu'
token = 'GiHPgTRUXbvde2l0'

# Set time interval
start_time = '2022-06-06T00:00:00'
end_time = '2022-06-08T00:00:00'

# Set ouput parameters
show_prediction = True  # Plot raw classifier results
show_detections = True  # Plot post-processed classifier results
save_log = True  # Save all events prior to post-processing/verification

##############################
# %% 3 Script
# 3 Script
##############################

x1 = np.array([])  # Start time
x2 = np.array([])  # End time
x3 = np.array([])  # Average power 0-5Hz
x4 = np.array([])  # Average power 5-10Hz2
x5 = np.array([])  # Ratio of x3/x4

# Load model
with open(f'{model_name}.json', 'r') as json_file:
    input_parameters = json.load(json_file)
client_id = input_parameters['client']
network = input_parameters['network']
station = input_parameters['station']
location = input_parameters['location']
channel = input_parameters['channel']
features = input_parameters['features']
window_length = input_parameters['window_length']
window_overlap = input_parameters['window_overlap']
remove_response = input_parameters['remove_response']
decimation_factor = input_parameters['decimation_factor']
minimum_frequency = input_parameters['minimum_frequency']
maximum_frequency = input_parameters['maximum_frequency']
minimum_lahar_duration = input_parameters['minimum_lahar_duration']
model_filename = input_parameters['model_filename']
scaler_filename = input_parameters['scaler_filename']

# Set FDSN client
client = Client(client_id, user=username, password=token)

# Load model and scaler if any

model = joblib.load(model_filename)
try:
    scaler = joblib.load(scaler_filename)
except:
    scaler = None

raw_predictions = pd.DataFrame()
dt1 = UTCDateTime(start_time)
dt2 = UTCDateTime(end_time)

starttime = dt1

try:
    inventory = client.get_stations(network=network,
                                    station=station,
                                    location=location,
                                    channel=channel,
                                    level='response')
    sensitivity = inventory[0][0][0].response.instrument_sensitivity.value
except:
    sensitivity = None

while starttime < dt2:
    endtime = starttime + (3600*24)
    print('Starting '+starttime.strftime('%Y-%m-%dT%H:%M:%S'), flush=True)

    stream = client.get_waveforms(network,
                                  station,
                                  location,
                                  channel,
                                  starttime,
                                  endtime)
    print(f'Waveform requested.', flush=True)
    stream = preprocess_stream(stream,
                               decimate=decimation_factor,
                               freqmin=minimum_frequency,
                               freqmax=maximum_frequency,
                               sensitivity=sensitivity)
    print(f'Preprocessed routine in lahar training data done.', flush=True)

    try:
        samples = extract_samples(stream,
                                  features,
                                  window_length,
                                  window_overlap)
        print(f'Samples extracted.', flush=True)
    except:
        print(f'No data for this time period.', flush=True)
        print('---', flush=True)
        continue

    try:
        classified_data_frame = predict_knn(
            samples, model, scaler=scaler)
        print(f'Predictions generated.', flush=True)
        cleaned_data_frame = clean_detections(classified_data_frame)
        print(f'Results cleaned.', flush=True)
        lah_0, lah_1, lah_0l, lah_1l = retrieve_dates(cleaned_data_frame)
        print(f'Dates retrieved.', flush=True)
        lah_count = len(lah_0)
        x1 = np.append(x1, lah_0)
        x2 = np.append(x2, lah_1)
    except:
        lah_count = 0
        tot_count = None
        starttime = starttime+(3600*12)
        print('No detections found for this time period.')
        print('---', flush=True)
        continue

    # Count number of detections in iteration
    if np.any(x1):
        tot_count = len(x1)
    else:
        tot_count = 0

    if lah_count:
        for i in range(-len(lah_0), 0):
            sts = stream.slice(x1[i], x2[i])
            st_data = sts[0].data
            ffx, ppx = signal.welch(st_data, fs=sts[0].stats.sampling_rate)
            avg_lo = np.mean(ppx[np.where(ffx < 5)])
            # avg_hi = np.mean(ppx[np.where(ffx < 15)])
            avg_hi = np.mean(ppx[np.where((ffx >= 5) & (ffx <= 10))])
            x3 = np.append(x3, avg_lo)
            x4 = np.append(x4, avg_hi)
            x5 = np.append(x5, avg_lo/avg_hi)
        xts = np.stack(([i.strftime('%Y-%m-%dT%H:%M:%S') for i in x1],
                        [i.strftime('%Y-%m-%dT%H:%M:%S') for i in x2],
                        x3,
                        x4,
                        x5), axis=-1)

    # Update start time for next iteration and print results
    starttime = starttime+(3600*12)
    print(f'Detections found in iteration = {lah_count}', flush=True)
    print(f'Total detections found = {tot_count}', flush=True)
    print('---', flush=True)

# Automated post processing
r1a = []  # Start time, step 1
r1b = []  # End time, step 1
r2a = []  # Start time, step 2
r2b = []  # End time, step 2
r3a = []  # Start time, step 3
r3b = []  # End time, step 3

if tot_count:

    # Step 1: Remove detections that are likely noise (use frequency ratios)
    for i in range(len(xts)):
        if (float(xts[i][2])/float(xts[i][3])) <= 0.75:
            r1a.append(xts[i][0])
            r1b.append(xts[i][1])

    # Step 2: Remove detection of less than minimum_lahar_duration minutes
    for i in range(len(r1a)):
        if (UTCDateTime(r1b[i])-UTCDateTime(r1a[i])) >= \
                (60*minimum_lahar_duration):
            r2a.append(r1a[i])
            r2b.append(r1b[i])
    r2a = np.array(r2a)
    r2b = np.array(r2b)

    # Step 3: Remove overlapping detections
    for i in range(len(r2a)):
        skipped = [*range(len(r2a))]
        skipped.remove(i)
        overlap = 0
        for j in skipped:
            a1 = UTCDateTime(r2a[i])
            a2 = UTCDateTime(r2b[i])
            if (a1 <= UTCDateTime(r2b[j])) and (a2 >= UTCDateTime(r2a[j])):
                overlap += 1
                if ((UTCDateTime(r2b[j])-UTCDateTime(r2a[j])) <= (a2-a1)):
                    r3a.append(a1)
                    r3b.append(a2)
        if overlap == 0:
            r3a.append(a1)
            r3b.append(a2)
    x1 = np.unique(np.array([i.strftime('%Y-%m-%dT%H:%M:%S') for i in r3a]))
    x2 = np.unique(np.array([i.strftime('%Y-%m-%dT%H:%M:%S') for i in r3b]))

else:

    # Final list of detections (empty, no detections)
    x1 = np.array([])
    x2 = np.array([])

# Save results
if len(x1) == 0:
    print('No detections found.', flush=True)
    print('---', flush=True)
else:
    xts = np.stack((x1, x2), axis=-1)
    out_dts = 'detection_'+station+'_' + f"{window_length:02d}" + '_' +\
        UTCDateTime(x1[0]).strftime('%Y%m%d')+'_' + \
        UTCDateTime(x2[-1]).strftime('%Y%m%d')+'.txt'
    np.savetxt(f'{model_name}_events.csv', xts, delimiter=",", fmt='%s')
    print(f'Detections saved.', flush=True)

# %%
