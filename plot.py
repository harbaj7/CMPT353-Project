# python3 plot.py 

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile

NUM_SECS_DISPLAY = 10
BUFFER_SECS = 5

# file = sys.argv[1]

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
files = [f for f in listdir('data/') if isfile(join('data', f))]

for file in files:
    data_accl = pd.read_csv(ZipFile('data/' + file).open('Accelerometer.csv'))
    data_accl = data_accl[data_accl['seconds_elapsed'] > 7]
    

    end_time = data_accl.iloc[data_accl.shape[0] - 1]['seconds_elapsed']
    data_accl = data_accl[data_accl['seconds_elapsed'] < 12]
    data_accl['mag'] = np.sqrt(data_accl['x'] ** 2 + data_accl['y'] ** 2 + data_accl['z'] ** 2)

    plt.figure(figsize=(16, 16))
    plt.title('acceleration mag')
    plt.plot(data_accl['seconds_elapsed'], data_accl['mag'])
    # plt.plot(data['time'], data['x'], 'r-', alpha=0.5)
    # plt.plot(data['time'], data['y'], 'g-', alpha=0.5)
    # plt.plot(data['time'], data['z'], 'b-', alpha=0.5)
    plt.savefig('figures/' + file.split('.')[0] + '_accl.png')

    data_orient = pd.read_csv(ZipFile('data/' + file).open('Orientation.csv'))
    data_orient = data_orient[data_orient['seconds_elapsed'] > 7]
    data_orient = data_orient[data_orient['seconds_elapsed'] < 12]

    plt.figure(figsize=(16, 16))
    plt.title('orientation')
    plt.plot(data_orient['seconds_elapsed'], data_orient['pitch'], 'r-')
    plt.plot(data_orient['seconds_elapsed'], data_orient['roll'], 'b-')
    plt.plot(data_orient['seconds_elapsed'], data_orient['yaw'], 'g-')
    # plt.plot(data['time'], data['x'], 'r-', alpha=0.5)
    # plt.plot(data['time'], data['y'], 'g-', alpha=0.5)
    # plt.plot(data['time'], data['z'], 'b-', alpha=0.5)
    plt.savefig('figures/' + file.split('.')[0] + '_orient.png')



