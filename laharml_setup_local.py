##############################
# %% 1 Import packages
# 1 Import packages
##############################

from obspy import UTCDateTime
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

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

# Set path to training files. If existing_data_frame is True, it opens a
# file in the current directory with training data sampled and ready for
# use in the classifier. The name of this file is defined by the station
# parameters below. If existing_data_frame is False, it opens two files with
# datetimes of events and noise data and gets them ready for training the
# classifier. An existing_data_frame is used to keep training consistency, so
# keep existing_data_frame as False in the first run.
existing_training_file = False
event_file_path = 'training_v2.txt'  # File with datetimes of lahar events
noise_file_path = 'noise.txt'  # File with datetimes of noise events
save_training_file = True

# Set date range
start_date = '2022-01-01'
end_date = '2023-01-01'

# Set data directory path. The script reads .mseed files with daily seismic
# record named under the convention 'AA.BBBB.CC.DDD.YYYY.JJJ.mseed' and
# inside the subdirectory 'BBBB' where AA is the two-character network code,
# BBBB is the four-character station code, CC is the two-character location
# code, DDD is the three-character channel code, YYYY is the four-digit year,
# JJJ is the three-digit zero-padded julian day (day of the year).
directory = '/Volumes/Tungurahua/FUEGO/SEISMIC'

# Set station parameters
network = 'GI'
station = 'FG12'
location = '00'
channel = 'BHZ'
sensitivity = 4.81e8  # Station response

# Set features for training. Check the README file for a list of codes.
features = [4, 6, 34, 39, 44]
window_length = 10  # (in minutes)
window_overlap = 0.75  # Fraction of percentage

# Set output file path. If output_folder_path is None, results will be saved
# in current directory. If save_log is True, all detections prior to
# postprocessing are saved. If save_detections is True, all detections after
# postprocessing are saved. If save_detections is False, the script performs
# feature extraction for training files, trains the classifier, and outputs
# performance metrics but does not generate predictions or detections.
output_folder_path = None
save_log = True
save_detections = True

# Other classifier parameters
decimation_factor = None  # Decimate seismic data by this factor
minimum_frequency = 5  # High-pass filter above this frequency (in Hz)
maximum_frequency = None  # Low-pass filter above this frequency (in Hz)
minimum_lahar_duration = 30  # Minimum duration for post-processing

# Plot parameters
vmin = -150  # in dB
vmax = 150  # in dB
vertical_limit = 5e-4  # (in counts if sensitivity = None)

##############################
# %% 3 Script
# 3 Script
##############################

# Codes for file naming
datacode = f'{network}.{station}.{location}.{channel}'

# Start and end dates to UTCDateTime
dt1 = UTCDateTime(start_date)
dt2 = UTCDateTime(end_date)

# File name for existing training dataframe file (or for exporting)
training_file_name = ('LaharMLTrainTest_'
                      + datacode
                      + f'_{window_length:02d}'
                      + f'.{window_overlap*100:03d}'
                      + f'.{decimation_factor:02d}'
                      + '.csv')

# If training file exist, then load it
if existing_training_file:
    training = pd.read_csv(training_file_name, index_col=0)
else:
    s_fm = np.dtype([('station',
                      (str, 4)),
                     ('datestart', (str, 19)),
                     ('dateend', (str, 19))])
    s_fl = np.loadtxt(event_file_path, dtype=s_fm,
                      delimiter='\t', unpack=True, usecols=(0, 1, 2))

    if station in s_fl[0]:
        ids = np.where(s_fl[0] == station)[0]
        dtm1 = [UTCDateTime(i) for i in s_fl[1][ids]]
        dtm2 = [UTCDateTime(i) for i in s_fl[2][ids]]
    else:
        print('Station not found in training date file, try again.')

    # Initiliaze data frame

    training = pd.DataFrame()
    signal_time = 0

    # Add clasification column (101 = lahar signal)

    features.append(101)

    # Extract samples from record in local directory

    for i in range(len(dtm1)):
        starttime = UTCDateTime(dtm1[i])
        endtime = UTCDateTime(dtm2[i])
        dl = extract_from_local_directory(directory,
                                          network,
                                          station,
                                          location,
                                          channel,
                                          starttime,
                                          endtime,
                                          features,
                                          extended,
                                          min_freq=minimum_frequency,
                                          decimate=decimation_factor,
                                          window=window_length,
                                          overlap=window_overlap,
                                          plot=True,
                                          vmin=vmin,
                                          vmax=vmax,
                                          count_lim=count_lim)
        signal_time += endtime-starttime
        training = pd.concat([training, dl], ignore_index=True, sort=False)
    print('Total signal time: '+str(signal_time/3600)+' hours')

    # Read dates from noise event file

    s_fm = np.dtype([('station', (str, 4)), ('datestart',
                                             (str, 19)), ('type', (str, 2))])
    s_fl = np.loadtxt(noise_file_path, dtype=s_fm,
                      delimiter='\t', unpack=True, usecols=(0, 1, 2))

    if station in s_fl[0]:
        ids = np.where(s_fl[0] == station)[0]
        dtm1 = [UTCDateTime(i) for i in s_fl[1][ids]]
    else:
        print('Station not found in training date file, try again.')
        exit()

    features[-1] = 100
    duration_noise_event = signal_time/len(dtm1)

    # Extract samples from record in local directory

    for i in range(len(dtm1)):
        starttime = UTCDateTime(dtm1[i])-(duration_noise_event/2)
        endtime = UTCDateTime(dtm1[i])+(duration_noise_event/2)
        dl = extract_from_local_directory(directory,
                                          network,
                                          station,
                                          location,
                                          channel,
                                          starttime,
                                          endtime,
                                          features,
                                          extended,
                                          min_freq=minimum_frequency,
                                          decimate=decimation_factor,
                                          window=window_length,
                                          overlap=window_overlap,
                                          plot=True,
                                          vmin=vmin,
                                          vmax=vmax,
                                          count_lim=count_lim)
        training = pd.concat([training, dl], ignore_index=True, sort=False)

    # Remove classification column from feature indicator list

    del features[-1]

else:

    # Read existing data frame file

    training = pd.read_csv(data_frame_path, index_col=0)
    columns = list(training.columns[features])
    columns.append('Times')
    columns.append('Classification')
    training = training[columns]

# Train and test model

model, scaler, classification_report, confusion_matrix, neighbors = train_test_knn(training,
                                                                                   scale=True,
                                                                                   get_n=True,
                                                                                   plot_n=True)

print(classification_report, confusion_matrix)
print("Neighbors: "+str(neighbors))

# Generate empty arrays for detections

x1 = np.array([])  # Start time
x2 = np.array([])  # End time
x3 = np.array([])  # Average power 0-5Hz
x4 = np.array([])  # Average power 5-10Hz2
x5 = np.array([])  # Ratio of x3/x4

# Loop through dates

starttime = dt1

while starttime < dt2:

    endtime = starttime + (3600*24)

    print('\rStarting '+starttime.strftime('%Y-%m-%dT%H:%M:%S') +
          '----\n', end="", flush=True)

    # Extract samples from record in local directory

    try:
        unclassified_data_frame, st = extract_from_local_directory(
            directory,
            network,
            station,
            location,
            channel,
            starttime,
            endtime,
            features,
            min_freq=minimum_frequency,
            decimate=decimation_factor,
            window=window_length,
            overlap=window_overlap,
            keep=True)
    except:
        lah_count = 0
        tot_count = len(x1)
        starttime = starttime+(3600*12)
        print('No data found for this time period.')
        continue

    # Classify samples

    try:
        classified_data_frame = predict_knn(
            unclassified_data_frame, model, scaler=scaler)
        cleaned_data_frame = clean_detections(classified_data_frame)
        lah_0, lah_1, lah_0l, lah_1l = retrieve_dates(cleaned_data_frame)
        lah_count = len(lah_0)
        x1 = np.append(x1, lah_0)
        x2 = np.append(x2, lah_1)
        if show_predictions:
            sns.scatterplot(classified_data_frame,
                            x='Times',
                            y=unclassified_data_frame.columns[0],
                            hue='Prediction')
            plt.show()
    except:
        lah_count = 0
        tot_count = len(x1)
        starttime = starttime+(3600*12)
        print('No detections found for this time period.')
        continue

    # Plot detections and calculate periodogram for each detection

    if lah_count:
        for i in range(-len(lah_0), 0):
            sts = st.slice(x1[i], x2[i])
            st_data = sts.data
            ffx, ppx = signal.welch(st_data, fs=sts.stats.sampling_rate)
            avg_lo = np.mean(ppx[np.where(ffx < 5)])
            # avg_hi = np.mean(ppx[np.where(ffx < 15)])
            avg_hi = np.mean(ppx[np.where((ffx >= 5) & (ffx <= 10))])
            x3 = np.append(x3, avg_lo)
            x4 = np.append(x4, avg_hi)
            x5 = np.append(x5, avg_lo/avg_hi)
            if show_detections:
                plot_detections(
                    cleaned_data_frame,
                    st, show=True,
                    save=False,
                    target='Detection',
                    vmin=vmin,
                    vmax=vmax,
                    count_lim=count_lim)
                plt.plot(ffx, 10*np.log10(ppx))
                plt.title(
                    '0-5Hz: '+str(round(avg_lo, 2)) +
                    '; 5-10Hz: '+str(round(avg_hi, 2))
                )
                plt.show()

        xts = np.stack(([i.strftime('%Y-%m-%dT%H:%M:%S') for i in x1],
                        [i.strftime('%Y-%m-%dT%H:%M:%S') for i in x2],
                        x3,
                        x4,
                        x5), axis=-1)

        # Save log file

        if save_log:
            out_log = 'log_'+station+'_' + f"{window_length:02d}" + '_' + \
                dt1.strftime('%Y%m%d')+'_'+dt2.strftime('%Y%m%d')+'.txt'
            np.savetxt(out_log, xts, delimiter=",", fmt='%s')

    # Count number of detections in iteration

    if np.any(x1):
        tot_count = len(x1)
    else:
        tot_count = 0

    # Update start time for next iteration and print results

    starttime = starttime+(3600*12)

    print('''\r\n
        Finished detections for the following dates:
        {date_1} to {date_2}.
        Detections found = {number_1}
        Number of detections saved = {number_2}
        '''.format(date_1=starttime.strftime('%Y-%m-%dT%H:%M:%S'),
                   date_2=endtime.strftime('%Y-%m-%dT%H:%M:%S'),
                   number_1=lah_count,
                   number_2=tot_count))

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

# Save detections file

if save_detections and (len(x1) > 0):

    xts = np.stack((x1, x2), axis=-1)
    out_dts = 'detection_'+station+'_' + f"{window_length:02d}" + '_' +\
        UTCDateTime(x1[0]).strftime('%Y%m%d')+'_' + \
        UTCDateTime(x2[-1]).strftime('%Y%m%d')+'.txt'
    np.savetxt(out_dts, xts, delimiter=",", fmt='%s')

elif (len(x1) > 0):

    print('Detections found but not saved.')

elif len(x1) == 0:

    print('No detections found.')

# %%
