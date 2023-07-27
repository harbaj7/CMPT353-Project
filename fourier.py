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

NUM_SECS_DISPLAY = 10
BUFFER_SECS = 5
RENAME = {'gravity_x':'x', 'gravity_y':'y', 'gravity_z':'z', 'seconds_elapsed_rounded':'seconds_elapsed'}

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
files = [f for f in listdir('grouped/') if isfile(join('grouped', f))]
print(files)
for file in files:
    data = pd.read_csv('grouped/' + file)
    data = data[['gravity_x', 'gravity_y', 'gravity_z', 'seconds_elapsed_rounded']].rename(columns=RENAME).dropna()
    data = data[data['seconds_elapsed'] > 10]
    data = data[data['seconds_elapsed'] <= 14]
    mean = np.mean(data['y'])
    data['y_sub'] = data['y'] - mean
    print(data)
    SAMPLE_RATE = 20
    N = data.shape[0]
    frequency = np.linspace (0.0, SAMPLE_RATE / 2, int (N/2))
    freq_data = scipy.fft.fft(np.array(data['y_sub']))
    y = 2/N * np.abs (freq_data [0:np.int (N/2)])
    inverse = scipy.fft.ifft(freq_data)
    print(data.shape[0])
    print(inverse)
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs[0, 0].set_title('fft')
    axs[0, 0].plot(frequency, y)
    axs[0, 1].set_title('y')
    axs[0, 1].plot(data['seconds_elapsed'], data['y_sub'])
    axs[1, 0].set_title('inverse fft')
    axs[1, 0].plot(data['seconds_elapsed'], inverse)
    plt.savefig('fft/' + file.split('.')[0] + '_fft.png')
    # break
    

    



