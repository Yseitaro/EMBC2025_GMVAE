import numpy as np
from scipy import signal
import random
import os
import configparser

def rectification(data):
    """full-wave rectification
    """
    return np.abs(data)

def smoothing(data, order, sampling_freq, cutoff_freq):
    """Smoothing using butterworth lowpass filter
    """
    Wn = cutoff_freq / (sampling_freq/2)
    b, a = signal.butter(order, Wn, 'low')

    return signal.filtfilt(b, a, data, axis=0)

def remove_transient_state(filt_data, reject_rate):
    """Remove transient state from filtered signals
    """
    n_samples, _ = filt_data.shape
    start_point = n_samples*reject_rate

    return filt_data[int(start_point):]

def get_shuffled_data(data, random_state):
    """Shuffle data
    """
    n_samples = data.shape[0]

    ind = random_state.choice(np.arange(0, n_samples, 1), 
                            size=n_samples, replace=False)

    return data[ind]


def compute_RMS_features(X, window_size):
    """移動窓でRMSを計算
    """

    return RMS(rolling_window(X.T, window_size), -1).T


def compute_MAV_features(X, window_size):
    """移動窓でMAVを計算
    """

    return MAV(rolling_window(X.T, window_size), -1).T


def RMS(X, axis=0):
    """RMSを計算
    """
    rms = np.sqrt(np.sum(np.power(X,2), axis=axis)) / len(X)

    return rms


def MAV(X, axis=0):
    """MAVを計算
    """
    rms = np.sum(np.abs(X), axis=axis) / len(X)

    return rms


def rolling_window(a, window):
    """移動窓の設定
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)