# python3 process_data.py

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile
from sklearn.decomposition import FastICA
import scipy.fft
import scipy.signal

NUM_SECS_DISPLAY = 10
BUFFER_SECS = 5
SAMPLE_RATE = 20
RENAME = {'gravity_x':'x', 'gravity_y':'y', 'gravity_z':'z', 'seconds_elapsed_rounded':'seconds_elapsed'}
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
def fourier(data, col):

    N = data.shape[0]

    mean = np.mean(data[col])
    minus_mean = data[col] - mean

    frequency = np.linspace (0.0, SAMPLE_RATE / 2, int (N/2))
    freq_data = scipy.fft.fft(np.array(minus_mean))
    y = 2/N * np.abs (freq_data [0:np.int (N/2)])

    return (frequency, y, minus_mean)

def filter(data, col, fft, frequency):
    max_idx = np.argmax(fft)
    highest_amplitude_freq = frequency[max_idx]
    low = highest_amplitude_freq - 0.1
    high = highest_amplitude_freq + 0.1
    if low <= 0.1:
        low = 0.1
    sos = scipy.signal.butter(10, [0.1, 4], 'bandpass', fs=SAMPLE_RATE, output='sos', analog=False)
    return np.abs(scipy.signal.sosfilt(sos, data[col]))

def filter2(data, col, fft, frequency):
    N = data.shape[0]

    mean = np.mean(data[col])
    minus_mean = data[col] - mean

    freq_data = scipy.fft.fft(np.array(minus_mean))
    y = 2/N * np.abs (freq_data [0:np.int (N/2)])

    max_idx = np.argmax(freq_data)
    freq_data[freq_data < max_idx - 0.1] = 0
    inverse = scipy.fft.ifft(freq_data)
    return np.abs(inverse)

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
files = [f for f in listdir('grouped/') if isfile(join('grouped', f))]
print(files)
for file in files:
    print(file)
    data = pd.read_csv('grouped/' + file)
    data = data[['gravity_x', 'gravity_y', 'gravity_z', 'seconds_elapsed_rounded']].rename(columns=RENAME).dropna()
    data = data[data['seconds_elapsed'] > 10]
    data = data[data['seconds_elapsed'] <= 15]
    # mean = np.mean(data['y'])
    # data['y_sub'] = data['y'] - mean
    # print(data)
    # SAMPLE_RATE = 20
    # N = data.shape[0]
    # frequency = np.linspace (0.0, SAMPLE_RATE / 2, int (N/2))
    # freq_data = scipy.fft.fft(np.array(data['y_sub']))
    # y = 2/N * np.abs (freq_data [0:np.int (N/2)])
    # inverse = scipy.fft.ifft(freq_data)
    # print(data.shape[0])
    # print(inverse)
    frequency_x, fft_x, minus_mean_x = fourier(data, 'x')
    frequency_y, fft_y, minus_mean_y = fourier(data, 'y')
    frequency_z, fft_z, minus_mean_z = fourier(data, 'z')
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    print(fft_x)
    axs[0, 0].set_title('fft')
    axs[0, 0].plot(frequency_x, np.abs(fft_x), 'r-')
    axs[0, 0].plot(frequency_y, np.abs(fft_y), 'g-')
    axs[0, 0].plot(frequency_z, np.abs(fft_z), 'b-')
    axs[0, 1].set_title('data - mean')
    axs[0, 1].plot(data['seconds_elapsed'], minus_mean_x, 'r-')
    axs[0, 1].plot(data['seconds_elapsed'], minus_mean_y, 'g-')
    axs[0, 1].plot(data['seconds_elapsed'], minus_mean_z, 'b-')
    axs[1, 0].set_title('filtered')
    axs[1, 0].plot(data['seconds_elapsed'], filter(data, 'x', fft_x, frequency_x), 'r-')
    axs[1, 0].plot(data['seconds_elapsed'], filter(data, 'y', fft_y, frequency_y), 'g-')
    axs[1, 0].plot(data['seconds_elapsed'], filter(data, 'z', fft_z, frequency_z), 'b-')
    # axs[1, 0].set_title('inverse fft')
    # axs[1, 0].plot(data['seconds_elapsed'], inverse)
    plt.savefig('fft/' + file.split('.')[0] + '_fft.png')
    # break
    

    



