# does a fft on the grouped gravity vector data
import re
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile
from sklearn.decomposition import FastICA
import scipy.fft

SAMPLE_RATE = 20
NUM_SECS_DISPLAY = 10
BUFFER_SECS = 5
RENAME = {'gravity_x':'x', 'gravity_y':'y', 'gravity_z':'z', 'seconds_elapsed_rounded':'seconds_elapsed'}


########################################################################- Harbaj code update
data_list = []
def dominant_frequency(frequencies, magnitudes):
    # Find the index of the peak
    peak_index = np.argmax(magnitudes)
    # Return the corresponding frequency
    return frequencies[peak_index]


def fourier(data, col):
    N = data.shape[0]

    # prevents an unhelpful spike at 0
    mean = np.mean(data[col])
    minus_mean = data[col] - mean

    # the fft, copied from some site
    # N is the number of samples, SAMPLE_RATE is in Hertz
    # the x axis
    frequency = np.linspace (0.0, SAMPLE_RATE / 2, int (N/2)) 
    # the y axis
    freq_data = scipy.fft.fft(np.array(minus_mean)) 
    # I'm pretty sure this is to deal with negative and complex values
    # also the N/2 thing is because there's some principle that prevents you from analysing frequencies greater than half your sample rate
    y = 2/N * np.abs (freq_data [0:int (N/2)]) 

    return (frequency, y, minus_mean)

files = [f for f in listdir('grouped/') if isfile(join('grouped', f))]
print(files)
for file in files:
    data = pd.read_csv('grouped/' + file)
    data = data[['gravity_x', 'gravity_y', 'gravity_z', 'seconds_elapsed_rounded']].rename(columns=RENAME).dropna()
    data = data[data['seconds_elapsed'] > 10]
    data = data[data['seconds_elapsed'] <= 14]

    # calculate fft
    frequency_x, fft_x, minus_mean_x = fourier(data, 'x')
    frequency_y, fft_y, minus_mean_y = fourier(data, 'y')
    frequency_z, fft_z, minus_mean_z = fourier(data, 'z')

    ################################ - Harbaj code update - 
    # Find the dominant frequencies
    dominant_frequency_x = dominant_frequency(frequency_x, fft_x)
    dominant_frequency_y = dominant_frequency(frequency_y, fft_y)
    dominant_frequency_z = dominant_frequency(frequency_z, fft_z)
    
    match = re.search(r'jog|sprint|stand|walk|run', file)
    if match:
        activity = match.group()
        if activity in ['jog', 'sprint', 'run']:
            activity = 'running'
        elif activity in ['walk']:
            activity = 'walking'
    else:
        activity = 'unknown'

    data_list.append({
        'filename': file,
        'dominant_frequency_x': dominant_frequency_x,
        'dominant_frequency_y': dominant_frequency_y,
        'dominant_frequency_z': dominant_frequency_z,
        'activity': activity
    })
    
    ###############################
    
    # plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs[0, 0].set_title('fft')
    axs[0, 0].plot(frequency_x, fft_x, 'r-')
    axs[0, 0].plot(frequency_y, fft_y, 'g-')
    axs[0, 0].plot(frequency_z, fft_z, 'b-')
    axs[0, 1].set_title('data - mean')
    axs[0, 1].plot(data['seconds_elapsed'], minus_mean_x, 'r-')
    axs[0, 1].plot(data['seconds_elapsed'], minus_mean_y, 'g-')
    axs[0, 1].plot(data['seconds_elapsed'], minus_mean_z, 'b-')

    plt.savefig('fft/' + file.split('.')[0] + '_fft.png')
    plt.close()
    # break
    
################################################################ - Harbaj code update Analysis
df = pd.DataFrame(data_list)

# Check if the directory exists
if not os.path.exists('analysis'):
    # If not, create it
    os.makedirs('analysis')

# Then, you can save the DataFrame
df.to_csv('analysis/activity_analysis.csv', index=False)





    



