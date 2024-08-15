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

# %% preprocess stream


def preprocess_stream(stream,
                      starttime=None,
                      endtime=None,
                      decimate=None,
                      detrend=True,
                      detrend_type='linear',
                      taper=True,
                      taper_fraction=0.05,
                      freqmin=None,
                      freqmax=None,
                      merge=True,
                      merge_fill_value='interpolate',
                      merge_method=0,
                      merge_interpolation_samples=0,
                      sensitivity=None):
    """
    Preprocesses a stream of seismic data by applying various filters and corrections.

    Args:
        stream (obspy.core.stream.Stream): The stream of seismic data to preprocess.
        starttime (obspy.UTCDateTime, optional): The start time to trim the stream to. Defaults to None.
        endtime (obspy.UTCDateTime, optional): The end time to trim the stream to. Defaults to None.
        decimate (int, optional): The factor to decimate the stream by. Defaults to None.
        detrend (bool, optional): Whether or not to detrend the stream. Defaults to True.
        detrend_type (str, optional): The type of detrending to apply. Defaults to 'linear'.
        taper (bool, optional): Whether or not to taper the stream. Defaults to True.
        taper_fraction (float, optional): The fraction of the stream to taper. Defaults to 0.05.
        freqmin (float, optional): The minimum frequency to filter the stream to. Defaults to None.
        freqmax (float, optional): The maximum frequency to filter the stream to. Defaults to None.
        merge (bool, optional): Whether or not to merge the stream. Defaults to True.
        merge_fill_value (str, optional): The fill value to use when merging the stream. Defaults to 'interpolate'.
        merge_method (int, optional): The method to use when merging the stream. Defaults to 0.
        merge_interpolation_samples (int, optional): The number of samples to interpolate when merging the stream. Defaults to 0.
        calibration (float, optional): The calibration factor to apply to the stream. Defaults to None.

    Returns:
        obspy.core.stream.Stream: The preprocessed stream of seismic data.
    """

    if merge:
        stream.merge(fill_value=merge_fill_value,
                     method=merge_method,
                     interpolation_samples=merge_interpolation_samples)

    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    if decimate:
        stream.decimate(factor=decimate)

    if detrend:
        stream.detrend(type=detrend_type)

    if taper:
        stream.taper(max_percentage=taper_fraction)

    if freqmin:
        stream.filter('highpass', freq=freqmin)

    if freqmax:
        stream.filter('lowpass', freq=freqmax)

    if sensitivity:
        for traces in stream:
            traces.data /= sensitivity

    return stream

# %% extract_samples


def extract_samples(stream,
                    features,
                    window_length,
                    window_overlap):

    feature_labels = ['00_Envelope_Unfiltered',
                      '01_Envelope_5Hz',
                      '02_Envelope_5_10Hz',
                      '03_Envelope_5_20Hz',
                      '04_Envelope_10Hz',
                      '05_Freq_Max_Unfiltered',
                      '06_Freq_25th',
                      '07_Freq_50th',
                      '08_Freq_75th',
                      '09_Kurtosis_Signal_Unfiltered',
                      '10_Kurtosis_Signal_5Hz',
                      '11_Kurtosis_Signal_5_10Hz',
                      '12_Kurtosis_Signal_5_20Hz',
                      '13_Kurtosis_Signal_10Hz',
                      '14_Kurtosis_Envelope_Unfiltered',
                      '15_Kurtosis_Envelope_5Hz',
                      '16_Kurtosis_Envelope_5_10Hz',
                      '17_Kurtosis_Envelope_5_20Hz',
                      '18_Kurtosis_Envelope_10Hz',
                      '19_Kurtosis_Frequency_Unfiltered',
                      '20_Kurtosis_Frequency_5Hz',
                      '21_Kurtosis_Frequency_5_10Hz',
                      '22_Kurtosis_Frequency_5_20Hz',
                      '23_Kurtosis_Frequency_10Hz',
                      '24_Skewness_Signal_Unfiltered',
                      '25_Skewness_Signal_5Hz',
                      '26_Skewness_Signal_5_10Hz',
                      '27_Skewness_Signal_5_20Hz',
                      '28_Skewness_Signal_10Hz',
                      '29_Skewness_Env_Unfiltered',
                      '30_Skewness_Env_5Hz',
                      '31_Skewness_Env_5_10Hz',
                      '32_Skewness_Env_5_20Hz',
                      '33_Skewness_Env_10Hz',
                      '34_Skewness_Frequency_Unfiltered',
                      '35_Skewness_Frequency_5Hz',
                      '36_Skewness_Frequency_5_10Hz',
                      '37_Skewness_Frequency_5_20Hz',
                      '38_Skewness_Frequency_10Hz',
                      '39_Spectral_Entropy_Unfiltered',
                      '40_Spectral_Entropy_5Hz',
                      '41_Spectral_Entropy_5_10Hz',
                      '42_Spectral_Entropy_5_20Hz',
                      '43_Spectral_Entropy_10Hz',
                      '44_Ratio_Unfiltered_5Hz_10Hz']

    tr = stream[0]
    tr_x = tr.data
    tr_t = tr.times(type='timestamp')

    bool_unfiltered = True in (
        i in [0, 5, 9, 14, 19, 24, 29, 34, 39, 44] for i in features)
    bool_5Hz = True in (i in [1, 10, 15, 20, 25, 30, 35, 40] for i in features)
    bool_5_10Hz = True in (
        i in [2, 11, 16, 21, 26, 31, 36, 41] for i in features)
    bool_5_20Hz = True in (
        i in [3, 12, 17, 22, 27, 32, 37, 42] for i in features)
    bool_10Hz = True in (i in [4, 13, 18, 23, 28, 33, 38, 43]
                         for i in features)
    bool_freq_unfiltered = True in (i in [5, 6, 7, 8, 44] for i in features)
    bool_freq_5Hz = True in (i in [20, 25, 30, 35] for i in features)
    bool_freq_5_10Hz = True in (i in [21, 26, 31, 36] for i in features)
    bool_freq_5_20Hz = True in (i in [22, 27, 32, 37] for i in features)
    bool_freq_10Hz = True in (i in [23, 28, 33, 38] for i in features)

    if bool_unfiltered:
        tr_unfiltered = tr.copy()

    if bool_5Hz:
        tr_5Hz = tr.copy()
        tr_5Hz.filter('highpass', freq=5)

    if bool_5_10Hz:
        tr_5_10Hz = tr.copy()
        tr_5_10Hz.filter('bandpass', freqmin=5, freqmax=10)

    if bool_5_20Hz:
        tr_5_20Hz = tr.copy()
        tr_5_20Hz.filter('bandpass', freqmin=5, freqmax=20)

    if bool_10Hz:
        tr_10Hz = tr.copy()
        tr_10Hz.filter('highpass', freq=10)

    ww = (1/(tr.stats.delta))*60*window_overlap
    ll = ww*(1-window_overlap)

    if (100 in features) or (101 in features):
        featfeat = features[:-1].copy()
    else:
        featfeat = features

    selected_features = [feature_labels[i] for i in featfeat]
    feature_dict = {i: [] for i in selected_features}

    timestamp = np.array([])

    for i in np.arange(0, (len(tr_x))-ww, ll):

        timestamp = np.append(timestamp, tr_t[int(i+(ww/2))])

        if bool_freq_unfiltered:
            segleng = 20*(1/tr_unfiltered.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_unfiltered, pp_unfiltered = signal.welch(tr_unfiltered.data[int(i):int(i+ww)],
                                                        fs=(1/tr_unfiltered.stats.delta),
                                                        window='hann',
                                                        nperseg=nperseg,
                                                        noverlap=nperseg/2,
                                                        nfft=nfft)
            csd_unfiltered = np.cumsum(pp_unfiltered)
            csd_unfiltered = csd_unfiltered-np.min(csd_unfiltered[1:])
            csd_unfiltered = csd_unfiltered/csd_unfiltered.max()

        if bool_freq_5Hz:
            segleng = 20*(1/tr_5Hz.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_5Hz, pp_5Hz = signal.welch(tr_5Hz.data[int(i):int(i+ww)],
                                          fs=(1/tr_5Hz.trats.delta),
                                          window='hann',
                                          nperseg=nperseg,
                                          noverlap=nperseg/2,
                                          nfft=nfft)
            csd_5Hz = np.cumsum(pp_5Hz)
            csd_5Hz = csd_5Hz-np.min(csd_5Hz[1:])
            csd_5Hz = csd_5Hz/csd_5Hz.max()

        if bool_freq_5_10Hz:
            segleng = 20*(1/tr_5_10Hz.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_5_10Hz, pp_5_10Hz = signal.welch(tr_5_10Hz.data[int(i):int(i+ww)],
                                                fs=(1/tr_5_10Hz.stats.delta),
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=nperseg/2,
                                                nfft=nfft)
            csd_5_10Hz = np.cumsum(pp_5_10Hz)
            csd_5_10Hz = csd_5_10Hz-np.min(csd_5_10Hz[1:])
            csd_5_10Hz = csd_5_10Hz/csd_5_10Hz.max()

        if bool_freq_5_20Hz:
            segleng = 20*(1/tr_5_20Hz.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_5_20Hz, pp_5_20Hz = signal.welch(tr_5_20Hz.data[int(i):int(i+ww)],
                                                fs=(1/tr_5_20Hz.trats.delta),
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=nperseg/2,
                                                nfft=nfft)
            csd_5_20Hz = np.cumsum(pp_5_20Hz)
            csd_5_20Hz = csd_5_20Hz-np.min(csd_5_20Hz[1:])
            csd_5_20Hz = csd_5_20Hz/csd_5_20Hz.max()

        if bool_freq_10Hz:
            segleng = 20*(1/tr_10Hz.trats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_10Hz, pp_10Hz = signal.welch(tr_10Hz.data[int(i):int(i+ww)],
                                            fs=(1/tr_10Hz.stats.delta),
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=nperseg/2,
                                            nfft=nfft)
            csd_10Hz = np.cumsum(pp_10Hz)
            csd_10Hz = csd_10Hz-np.min(csd_10Hz[1:])
            csd_10Hz = csd_10Hz/csd_10Hz.max()

        if 0 in features:
            feature_dict['00_Envelope_Unfiltered'].append(np.mean(
                obspy.signal.filter.envelope(tr_unfiltered.data[int(i):int(i+ww)])))

        if 1 in features:
            feature_dict['01_Envelope_5Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_5Hz.data[int(i):int(i+ww)])))

        if 2 in features:
            feature_dict['02_Envelope_5_10Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_5_10Hz.data[int(i):int(i+ww)])))

        if 3 in features:
            feature_dict['03_Envelope_5_20Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_5_20Hz.data[int(i):int(i+ww)])))

        if 4 in features:
            feature_dict['04_Envelope_10Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_10Hz.data[int(i):int(i+ww)])))

        if 5 in features:
            feature_dict['05_Freq_Max_Unfiltered'].append(
                ff_unfiltered[np.argmax(pp_unfiltered)])

        if 6 in features:
            feature_dict['06_Freq_25th'].append(
                ff_unfiltered[np.where(csd_unfiltered <= 0.25)[0][-1]])

        if 7 in features:
            feature_dict['07_Freq_50th'].append(
                ff_unfiltered[np.where(csd_unfiltered <= 0.50)[0][-1]])

        if 8 in features:
            feature_dict['08_Freq_75th'].append(
                ff_unfiltered[np.where(csd_unfiltered <= 0.75)[0][-1]])

        if 9 in features:
            feature_dict['09_Kurtosis_Signal_Unfiltered'].append(
                stats.kurtosis(tr_unfiltered.data[int(i):int(i+ww)]))

        if 10 in features:
            feature_dict['10_Kurtosis_Signal_5Hz'].append(
                stats.kurtosis(tr_5Hz.data[int(i):int(i+ww)]))

        if 11 in features:
            feature_dict['11_Kurtosis_Signal_5_10Hz'].append(
                stats.kurtosis(tr_5_10Hz.data[int(i):int(i+ww)]))

        if 12 in features:
            feature_dict['12_Kurtosis_Signal_5_20Hz'].append(
                stats.kurtosis(tr_5_20Hz.data[int(i):int(i+ww)]))

        if 13 in features:
            feature_dict['13_Kurtosis_Signal_10Hz'].append(
                stats.kurtosis(tr_10Hz.data[int(i):int(i+ww)]))

        if 14 in features:
            feature_dict['14_Kurtosis_Envelope_Unfiltered'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_unfiltered.data[int(i):int(i+ww)])))

        if 15 in features:
            feature_dict['15_Kurtosis_Envelope_5Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_5Hz.data[int(i):int(i+ww)])))

        if 16 in features:
            feature_dict['16_Kurtosis_Envelope_5_10Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_5_10Hz.data[int(i):int(i+ww)])))

        if 17 in features:
            feature_dict['17_Kurtosis_Envelope_5_20Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_5_20Hz.data[int(i):int(i+ww)])))

        if 18 in features:
            feature_dict['18_Kurtosis_Envelope_10Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_10Hz.data[int(i):int(i+ww)])))

        if 19 in features:
            feature_dict['19_Kurtosis_Frequency_Unfiltered'].append(
                stats.kurtosis(pp_unfiltered))

        if 20 in features:
            feature_dict['20_Kurtosis_Frequency_5Hz'].append(
                stats.kurtosis(pp_5Hz))

        if 21 in features:
            feature_dict['21_Kurtosis_Frequency_5_10Hz'].append(
                stats.kurtosis(pp_5_10Hz))

        if 22 in features:
            feature_dict['22_Kurtosis_Frequency_5_20Hz'].append(
                stats.kurtosis(pp_5_20Hz))

        if 23 in features:
            feature_dict['23_Kurtosis_Frequency_10Hz'].append(
                stats.kurtosis(pp_10Hz))

        if 24 in features:
            feature_dict['24_Skewness_Signal_Unfiltered'].append(
                stats.skew(tr_unfiltered.data[int(i):int(i+ww)]))

        if 25 in features:
            feature_dict['25_Skewness_Signal_5Hz'].append(
                stats.skew(tr_5Hz.data[int(i):int(i+ww)]))

        if 26 in features:
            feature_dict['26_Skewness_Signal_5_10Hz'].append(
                stats.skew(tr_5_10Hz.data[int(i):int(i+ww)]))

        if 27 in features:
            feature_dict['27_Skewness_Signal_5_20Hz'].append(
                stats.skew(tr_5_20Hz.data[int(i):int(i+ww)]))

        if 28 in features:
            feature_dict['28_Skewness_Signal_10Hz'].append(
                stats.skew(tr_10Hz.data[int(i):int(i+ww)]))

        if 29 in features:
            feature_dict['29_Skewness_Env_Unfiltered'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_unfiltered.data[int(i):int(i+ww)])))

        if 30 in features:
            feature_dict['30_Skewness_Env_5Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_5Hz.data[int(i):int(i+ww)])))

        if 31 in features:
            feature_dict['31_Skewness_Env_5_10Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_5_10Hz.data[int(i):int(i+ww)])))

        if 32 in features:
            feature_dict['32_Skewness_Env_5_20Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_5_20Hz.data[int(i):int(i+ww)])))

        if 33 in features:
            feature_dict['33_Skewness_Env_10Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_10Hz.data[int(i):int(i+ww)])))

        if 34 in features:
            feature_dict['34_Skewness_Frequency_Unfiltered'].append(
                stats.skew(pp_unfiltered))

        if 35 in features:
            feature_dict['35_Skewness_Frequency_5Hz'].append(
                stats.skew(pp_5Hz))

        if 36 in features:
            feature_dict['36_Skewness_Frequency_5_10Hz'].append(
                stats.skew(pp_5_10Hz))

        if 37 in features:
            feature_dict['37_Skewness_Frequency_5_20Hz'].append(
                stats.skew(pp_5_20Hz))

        if 38 in features:
            feature_dict['38_Skewness_Frequency_10Hz'].append(
                stats.skew(pp_10Hz))

        if 39 in features:
            normalized_psd = pp_unfiltered/np.sum(pp_unfiltered)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['39_Spectral_Entropy_Unfiltered'].append(
                spectral_entropy)

        if 40 in features:
            normalized_psd = pp_5Hz/np.sum(pp_5Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['40_Spectral_Entropy_5Hz'].append(spectral_entropy)

        if 41 in features:
            normalized_psd = pp_5_10Hz/np.sum(pp_5_10Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['41_Spectral_Entropy_5_10Hz'].append(spectral_entropy)

        if 42 in features:
            normalized_psd = pp_5_20Hz/np.sum(pp_5_20Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['42_Spectral_Entropy_5_20Hz'].append(spectral_entropy)

        if 43 in features:
            normalized_psd = pp_10Hz/np.sum(pp_10Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['43_Spectral_Entropy_10Hz'].append(spectral_entropy)

        if 44 in features:
            feature_dict['44_Ratio_Unfiltered_5Hz_10Hz'].append(
                np.log10(np.mean(pp_unfiltered[np.where(ff_unfiltered < 5)]) /
                         np.mean(pp_unfiltered[np.where((ff_unfiltered >= 5) & (ff_unfiltered <= 10))])))

    feature_dict.update(
        {'Times': timestamp})

    if 100 in features:
        feature_dict.update(
            {'Classification': np.zeros(
                len(feature_dict[list(feature_dict.keys())[0]]))})

    if 101 in features:
        feature_dict.update(
            {'Classification': np.zeros(
                len(feature_dict[list(feature_dict.keys())[0]]))+1})

    df = pd.DataFrame.from_dict(feature_dict)

    return df


# %% train_test_knn

def train_test_knn(data_frame, min_n=5, scale=True, neighbors=None, get_n=True, plot_n=True):

    X_train, X_test, y_train, y_test = train_test_split(data_frame.drop(
        ['Times', 'Classification'], axis=1), data_frame['Classification'], test_size=0.50)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train, columns=data_frame.columns[:-2])
        X_test = pd.DataFrame(X_test, columns=data_frame.columns[:-2])

        if get_n:
            train = X_train.copy()
            train['Classification'] = y_train.values

            k_range = range(min_n, int(len(train)*(4/5)))
            k_scores = []
            kf = KFold(n_splits=5)

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, data_frame.drop(
                    ['Times', 'Classification'], axis=1), data_frame['Classification'], cv=kf)
                k_scores.append(1-scores.mean())

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]

            if plot_n:
                fig = plt.figure(figsize=(5, 3))
                plt.plot(k_range, k_scores)
                plt.xlabel('Value of K for KNN')
                plt.ylabel('Cross-Validated Error')
                plt.title('KNN Cross-Validation')
                plt.show()

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]
            neighbors = n

        model = KNeighborsClassifier(n_neighbors=int(neighbors))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        report = classification_report(y_test, pred)
        conmat = confusion_matrix(y_test, pred)

        return model, scaler, report, conmat, neighbors

    else:
        if get_n:
            train = pd.concat([X_train, y_train], axis=1)

            k_range = range(min_n, int(len(train)*(4/5)))
            k_scores = []
            kf = KFold(n_splits=5)

            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, data_frame.drop(
                    ['Times', 'Classification'], axis=1), data_frame['Classification'], cv=kf)
                k_scores.append(1-scores.mean())

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]

            if plot_n:
                fig = plt.figure(figsize=(5, 3))
                plt.plot(k_range, k_scores)
                plt.xlabel('Value of K for KNN')
                plt.ylabel('Cross-Validated Error')
                plt.title('KNN Cross-Validation')
                plt.show()

            minpos = k_scores.index(min(k_scores))
            n = k_range[minpos]
            neighbors = n

        model = KNeighborsClassifier(n_neighbors=int(neighbors))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        report = classification_report(y_test, pred)
        conmat = confusion_matrix(y_test, pred)

        return model, report, conmat, neighbors

# %% predict_knn


def predict_knn(data_frame, model, scaler=None):

    if 'Classification' in data_frame:
        drop = ['Times', 'Classification']
    else:
        drop = ['Times']

    dropped = data_frame[drop]
    data_frame = data_frame.drop(drop, axis=1)

    if isinstance(scaler, sklearn.preprocessing._data.StandardScaler):
        X_input = pd.DataFrame(scaler.transform(
            data_frame), columns=data_frame.columns)
        prd = model.predict(X_input)
    else:
        prd = model.predict(data_frame)

    classified_data_frame = data_frame.join(dropped)
    classified_data_frame['Prediction'] = prd

    return classified_data_frame

# %% clean_detections


def clean_detections(data_frame, min_gap=15, min_sig=15, min_duration=30):

    # Indices of all non-predictions
    zero_index = np.array(
        data_frame.index[data_frame['Prediction'] == 0].tolist())
    ones_index = np.array(
        data_frame.index[data_frame['Prediction'] == 1].tolist())

    if np.any(zero_index):

        # Time between samples
        tst = data_frame['Times'].iloc[1]-data_frame['Times'].iloc[0]

        # Shortened indices of non-predictions for comparison
        pred1 = zero_index[:-1]
        pred2 = zero_index[1:]

        # Time between non-predictions in seconds
        pred3 = pred2 - pred1
        pred3 = pred3*tst

        # Find continuous predictions that are longer than min_duration
        cont = np.where(pred3 > min_duration*60)[0]

    # Significant detections

    # If predictions start at index 0, include the first prediction
    if not (0 in zero_index):
        pair_0 = [0, zero_index[0]-1]

    # If predictions end at the last index, include the last prediction
        if not (len(data_frame['Prediction'])-1 in zero_index):
            sig_pairs = [[zero_index[i]+1, zero_index[i+1]-1]
                         for i in cont[:-1]]
            pair_1 = [zero_index[-1]+1, len(data_frame['Prediction'])-1]
            sig_pairs.insert(0, pair_0)
            sig_pairs.append(pair_1)
        else:
            sig_pairs = [[zero_index[i]+1, zero_index[i+1]-1] for i in cont]
            sig_pairs.insert(0, pair_0)

    else:
        if not (len(data_frame['Prediction'])-1 in zero_index):
            sig_pairs = [[zero_index[i]+1, zero_index[i+1]-1]
                         for i in cont[:-1]]
            pair_1 = [zero_index[-1]+1, len(data_frame['Prediction'])-1]
            sig_pairs.append(pair_1)
        else:
            sig_pairs = [[zero_index[i]+1, zero_index[i+1]-1] for i in cont]

    # If two detections are closer than min_sig, merge them
    for i in range(len(sig_pairs)-1):
        if ((sig_pairs[i+1][0]-sig_pairs[i][1])*tst) < min_sig*60:
            sig_pairs[i+1][0] = sig_pairs[i][0]
            sig_pairs[i][1] = sig_pairs[i+1][1]

    # Remove duplicates
    sig_pairs_x = []
    for i in sig_pairs:
        if i not in sig_pairs_x:
            sig_pairs_x.append(i)

    # Merge detections that are close to significant detections

    final_pairs = []

    for i in sig_pairs_x:
        bw_i = i[0]-1
        fw_i = i[1]+1
        bw_d = np.where(ones_index < bw_i)[0]
        fw_d = np.where(ones_index > fw_i)[0]

        if np.any(bw_d):
            while ((bw_i-ones_index[bw_d[-1]])*tst) < min_gap*60:
                bw_i = ones_index[bw_d[-1]]
                bw_d = np.where(ones_index < bw_i)[0]
                if not (np.any(bw_d)):
                    break

        if np.any(fw_d):
            while ((ones_index[fw_d[0]]-fw_i)*tst) < min_gap*60:
                fw_i = ones_index[fw_d[0]]
                fw_d = np.where(ones_index > fw_i)[0]
                if not (np.any(fw_d)):
                    break

        final_pairs.append([bw_i, fw_i])

    # Replace values

    detections = np.zeros(len(data_frame['Prediction']))

    for i in final_pairs:
        detections[i[0]:i[1]] = 1

    data_frame['Detection'] = detections

    return data_frame

# %% retrieve_dates


def retrieve_dates(data_frame, target='Detection'):

    det = np.array(data_frame.index[data_frame[target] == 1].tolist())

    if np.any(det):

        det1 = det[:-1]
        det2 = det[1:]
        det3 = det2-det1

        det_i1 = (np.where(abs(det3) > 1)[0])
        det_i2 = [det[i] for i in det_i1]
        det_i3 = [det[i+1] for i in det_i1]

        if np.any(det_i2):
            det_0 = det[0]
            det_1 = det_i2
            det_0 = np.append(det_0, det_i3)
            det_1 = np.append(det_1, det[-1])
            starttimes = [UTCDateTime(data_frame['Times'].iloc[i])
                          for i in det_0]
            endtimes = [UTCDateTime(data_frame['Times'].iloc[i])
                        for i in det_1]
        else:
            det_0 = det[0]
            det_1 = det[-1]
            starttimes = [UTCDateTime(data_frame['Times'].iloc[det_0])]
            endtimes = [UTCDateTime(data_frame['Times'].iloc[det_1])]

        if det[0] == 0:
            open_start = True
        else:
            open_start = False

        if det[-1] == len(data_frame[target]):
            open_end = True
        else:
            open_end = False

    else:
        starttimes = []
        endtimes = []
        open_start = False
        open_end = False

    return starttimes, endtimes, open_start, open_end
