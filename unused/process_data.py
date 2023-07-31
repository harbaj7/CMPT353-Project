# python3 process_data.py

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile
from sklearn.decomposition import FastICA

NUM_SECS_DISPLAY = 10
BUFFER_SECS = 5
RENAME = {'gravity_x':'x', 'gravity_y':'y', 'gravity_z':'z', 'seconds_elapsed_rounded':'seconds_elapsed'}

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
files = [f for f in listdir('grouped/') if isfile(join('data', f))]

for file in files:
    data_accl = pd.read_csv(file)
    data_accl = data_accl[['gravity_x', 'gravity_y', 'gravity_z', 'seconds_elapsed_rounded']].rename(columns=RENAME)

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

    # smallestAngle(data_orient)

    # data_np = data_orient[['pitch', 'roll', 'yaw']].to_numpy()


    # print(data_orient)
    plt.figure(figsize=(16, 16))
    plt.title('orientation')
    plt.plot(data_orient['seconds_elapsed'], data_orient['pitch'], 'r-')
    plt.plot(data_orient['seconds_elapsed'], data_orient['roll'], 'b-')
    plt.plot(data_orient['seconds_elapsed'], data_orient['yaw'], 'g-')
    # plt.plot(data_orient['seconds_elapsed'], PCA(n_components=1).fit_transform(data_np), 'r-')
    # plt.plot(data['time'], data['x'], 'r-', alpha=0.5)
    # plt.plot(data['time'], data['y'], 'g-', alpha=0.5)
    # plt.plot(data['time'], data['z'], 'b-', alpha=0.5)
    plt.savefig('figures/' + file.split('.')[0] + '_orient.png')

    ica = FastICA(n_components=4).fit_transform(data_orient[['seconds_elapsed', 'pitch']])

    plt.figure(figsize=(16, 16))
    plt.title('ica orient')
    plt.plot(data_orient['seconds_elapsed'], ica[:,0] + ica[:,1], 'r-')
    plt.plot(data_orient['seconds_elapsed'], ica[:,0], 'g-')
    plt.plot(data_orient['seconds_elapsed'], ica[:,1], 'b-')
    plt.savefig('figures/' + file.split('.')[0] + '_ica.png')

    data_grav = pd.read_csv(ZipFile('data/' + file).open('Gravity.csv'))
    data_grav = data_grav[data_grav['seconds_elapsed'] > 7]
    data_grav = data_grav[data_grav['seconds_elapsed'] < 12]
    vectors = data_grav[['x', 'y', 'z']].to_numpy()
    length = np.linalg.norm(vectors, axis=1)
    data_grav['length'] = length

    dot = np.dot(vectors, np.array([0, 0, 1]))
    angle = np.arccos(dot / length)
    data_grav['angle'] = angle
    # data_grav = data_grav.sort_values('angle')
    print(data_grav)
    
    plt.figure(figsize=(16, 16))
    plt.title('grav angle')
    plt.plot(data_grav['seconds_elapsed'], data_grav['angle'], 'r-')
    plt.savefig('figures/' + file.split('.')[0] + '_grav_angle.png')

    plt.figure(figsize=(16, 16))
    plt.title('grav')
    plt.plot(data_grav['seconds_elapsed'], data_grav['x'], 'r-')
    plt.plot(data_grav['seconds_elapsed'], data_grav['y'], 'g-')
    plt.plot(data_grav['seconds_elapsed'], data_grav['z'], 'b-')
    plt.plot(data_grav['seconds_elapsed'], data_grav['angle'], 'y-')
    plt.savefig('figures/' + file.split('.')[0] + '_grav.png')
    # to get down, try calculating all angles with straight down, sorting by that, nad averaging all the vectors
    # break

    



