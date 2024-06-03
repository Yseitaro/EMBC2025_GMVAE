import numpy as np
from scipy import signal
import numbers
import errno
import os
import configparser
import sys
import pandas as pd
from .process_emg import rectification, smoothing
from .process_emg import remove_transient_state, get_shuffled_data
from .process_emg import compute_MAV_features, compute_RMS_features
from sklearn.model_selection import train_test_split

def config_parser(path):
    """Read `config.ini` to get setting
    """
    config_ini = configparser.ConfigParser()

    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    config_ini.read(path + '/config.ini', encoding='utf-8')

    read_setting = config_ini['SETTING']

    n_sub = read_setting.getint('n_sub')
    n_trial = read_setting.getint('n_trial')
    n_channel = read_setting.getint('n_channel')
    n_class = read_setting.getint('n_class')
    sampling_freq = read_setting.getint('sampling_freq')

    return n_sub, n_trial, n_channel, n_class, sampling_freq


def _make_write_directory(root, category, sub):
    """Make directory for reshaped files
    """
    path = '{0}/{1}/sub{2}'.format(root, category, sub)
    os.makedirs(path, exist_ok=True)

def make_write_file_path(root, category, sub, clas, trial):
    """Generate a file path for reshaped files
    """
    _make_write_directory(root, category, sub)

    path = '{0}/{1}/sub{2}/class{3}trial{4}.dat'.\
            format(root, category, sub, clas, trial)
    return path

def get_delimiter(extension):
    """Get deliminater depending on extension of file.
    """
    if extension == 'txt':
        return None
    elif extension == 'csv':
        return ','
    elif extension == 'dat':
        return '\t'
    else:
        return None


def make_read_path(root, category, sub, clas, trial):
    """Generate a file path for raw data 
    """
    path = '{0}/{1}/sub{2}/class{3}trial{4}.dat'.\
            format(root, category, sub, clas, trial)

    return path

def process_data(data_path, order=2, sampling_freq=2000, 
                 cutoff_freq=1.0, reject_rate = 0.1):
    """Rectification, smoothing, and extraction for raw data
    """
    n_sub, n_trial, _, n_class, sampling_freq = config_parser(data_path)

    print('Processing EMG signals...')
    for s in range(n_sub):
        for t in range(n_trial):
            print('\r                                       ', end='', flush=True)
            for c in range(n_class):
                print('\r > subject: {0},  trial: {1}, class: {2}'.format(s+1, t+1, c+1), end='', flush=True)

                fname = make_read_path(data_path, 'raw', s+1, c+1, t+1)
                read_data = np.loadtxt(fname, delimiter='\t')

                # Rectified and smoothed data
                filtered_data = rectification(read_data)
                filtered_data = smoothing(filtered_data, order,         
                                                  sampling_freq, cutoff_freq)
                category = 'filtered/{:.1f}Hz'.format(cutoff_freq)
                fname = make_write_file_path(data_path, category, s+1, c+1, t+1)
                np.savetxt(fname=fname, X=filtered_data, fmt='%.7e', delimiter='\t')

                # extracted data
                extracted_data = remove_transient_state(filtered_data, reject_rate)
                category = 'extracted/{:.1f}Hz'.format(cutoff_freq)
                fname = make_write_file_path(data_path, category, s+1, c+1, t+1)
                np.savetxt(fname=fname, X=extracted_data, fmt='%.7e', delimiter='\t')
    print('\nDone!\n')


def concatenate_data(data_path, sub, cutoff, trial_set, extraction_rate=0, down_sampling_rate=1):
    """
    Concatenate data with class labels
    """
    _, _, n_channel, n_class, _ = config_parser(data_path)

    # Generate empty array
    stacked_data = np.empty((0, n_channel))
    class_label = np.empty(0)

    n_trial = len(trial_set)    

    # Data concatenation
    for c in range(n_class):
        stacked_data_trial = np.empty((0, n_channel))
        for t in range(n_trial):
            category = 'extracted/{:.1f}Hz'.format(cutoff)
            fname = make_read_path(data_path, category, sub+1, c+1, trial_set[t] + 1)
            
            read_data = pd.read_table(fname, header=None).values
            read_data = read_data[int(len(read_data)*extraction_rate):]
            read_data = read_data[::down_sampling_rate]

            # Stack data along with trial direction
            stacked_data_trial = np.vstack([stacked_data_trial, read_data])
        
        
        stacked_data = np.vstack([stacked_data, stacked_data_trial])
        class_label = np.hstack([class_label, np.full(len(stacked_data_trial), c, dtype='int8')])

    return stacked_data.astype(np.float64), class_label.astype(np.int8)


def concatenate_data_time(data_path, sub, cutoff, trial_set, extraction_rate=0, time_width=40, down_sampling_rate=1):
    """
    Concatenate data with class labels
    """
    _, _, n_channel, n_class, _ = config_parser(data_path)

    # Generate empty array
    stacked_data = np.empty((0, time_width, n_channel))
    class_label = np.empty(0)

    n_trial = len(trial_set)    

    # Data concatenation
    for c in range(n_class):
        stacked_data_trial = np.empty((0, time_width, n_channel))
        for t in range(n_trial):
            category = 'extracted/{:.1f}Hz'.format(cutoff)
            fname = make_read_path(data_path, category, sub+1, c+1, trial_set[t] + 1)
            
            read_data = pd.read_table(fname, header=None).values
            read_data = read_data[int(len(read_data)*extraction_rate):]
            read_data = read_data[::down_sampling_rate].reshape(-1, time_width, n_channel)

            # Stack data along with trial direction
            stacked_data_trial = np.vstack([stacked_data_trial, read_data])

        stacked_data = np.vstack([stacked_data, stacked_data_trial])
        class_label = np.hstack([class_label, np.full(len(stacked_data_trial), c, dtype='int8')])

    return stacked_data.astype(np.float64), class_label.astype(np.int8)

                
def _extract_data(X, n_extraction, random_state):
    """データから一部をランダムに取得
    """
    return get_shuffled_data(X, random_state)[:int(n_extraction)]


def check_random_state(seed):
    """ランダムシードの設定
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    

def normalize_train_data(X, type='norm', dim='whole'):
    """訓練データを正規化（標準化）する処理
    """
    stats = {}

    if dim == 'whole':
        stats['max'] = np.max(X)
        stats['mean'] = np.mean(X)
        stats['std'] = np.std(X)
    elif dim == 'each':
        stats['max'] = np.max(X, axis=0)
        stats['mean'] = np.mean(X, axis=0)
        stats['std'] = np.std(X, axis=0)

    if type == 'norm':
        X = X / stats['max']
    elif type == 'standard':
        X = (X - stats['mean']) / stats['std']
    
    return X, stats

def normalize_test_data(X, stats, type='norm'):
    """テストデータを正規化（標準化）する処理
    """

    if type == 'norm':
        X = X / stats['max']
    elif type == 'standard':
        X = (X - stats['mean']) / stats['std']
    
    return X