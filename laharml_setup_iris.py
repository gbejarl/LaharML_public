##############################
# %% 1 Import packages
# 1 Import packages
##############################

import json
import joblib
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

from laharml_modules import (preprocess_stream,
                             extract_samples,
                             train_test_knn)

##############################
# %% 2 Script
# 2 Script
##############################


def setup_iris(client_id,
               username,
               token,
               model_name,
               network,
               station,
               location,
               channel,
               list_of_lahars,
               list_of_noise,
               features,
               window_length,
               window_overlap,
               remove_response,
               decimation_factor,
               minimum_frequency,
               maximum_frequency,
               minimum_lahar_duration,
               save_parameters=True,
               save_model=True):

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
        total_lahar_time += stream[-1].stats.endtime - \
            stream[0].stats.starttime
        training_lahar.append(stream)
        print(
            f'Training lahar event {i+1} of {len(list_lahar)} requested.', flush=True)
    print(f'---', flush=True)

    duration_per_noise = total_lahar_time / len(list_noise)

    # Request waveforms for noise events
    for i in list_noise.index:
        request_t1 = UTCDateTime(
            list_noise.loc[i][1])-(int(duration_per_noise/2))
        request_t2 = UTCDateTime(
            list_noise.loc[i][1])+(int(duration_per_noise/2))
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
        training = pd.concat([training, samples],
                             ignore_index=True, sort=False)
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
        training = pd.concat([training, samples],
                             ignore_index=True, sort=False)
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
    print(f'Model saved.', flush=True)
    print(f'---', flush=True)

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
