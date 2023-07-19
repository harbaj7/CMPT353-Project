# python3 plot.py 

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

# file = sys.argv[1]

def smallestAngle(data_orient):
    columns = ['roll', 'pitch', 'yaw']
    created = [ ]
    for column in columns:
        data_orient[column + '_cos'] = np.cos(data_orient[column])
        data_orient[column + '_sin'] = np.sin(data_orient[column])
        created.append(column + '_cos', column + '_sin')
    print(data_orient)

    data_orient['vec_x'] = -data_orient['roll_sin'] * data_orient['yaw_cos'] - data_orient['roll_cos'] * data_orient['pitch_sin'] * data_orient['yaw_sin']
    data_orient['vec_y'] = data_orient['roll_sin'] * data_orient['yaw_sin'] - data_orient['roll_cos'] * data_orient['pitch_sin'] * data_orient['yaw_sin']
    data_orient['vec_z'] = data_orient['roll_cos'] * data_orient['pitch_cos']


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
    data_grav = data_grav[data_grav['seconds_elapsed'] < 15]
    vectors = data_grav[['x', 'y', 'z']].to_numpy()
    dot = np.dot(vectors, np.array([1, 0, 0]))
    length = np.linalg.norm(vectors, axis=1)
    data_grav['length'] = length
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

    



