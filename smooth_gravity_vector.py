# the interpolating I said I would work on

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile
from sklearn.decomposition import FastICA
from scipy.interpolate import CubicHermiteSpline
import statsmodels.api as sm

files = [f for f in listdir('data/') if isfile(join('data', f))]
csv_names = [
    'Accelerometer.csv', 'AccelerometerUncalibrated.csv', 'Gravity.csv', 
    'Gyroscope.csv', 'GyroscopeUncalibrated.csv', 'Location.csv', 
    'Magnetometer.csv', 'MagnetometerUncalibrated.csv', 'Orientation.csv', 
    'TotalAcceleration.csv'
]
names = []

BASE = 0.05

for file in files:
    print(file)

    zip = ZipFile('data/' + file)
    gyro_data = pd.read_csv(zip.open('Gyroscope.csv'))
    orientation_data = pd.read_csv(zip.open('Orientation.csv'))

    gyro_data['seconds_elapsed_shift'] = gyro_data['seconds_elapsed'].shift(-1)
    columns = ['x', 'y', 'z']
    for col in columns:
        gyro_data[col + '_shift'] = gyro_data[col].shift(-1)
        gyro_data[col + '_rise'] = (gyro_data[col + '_shift'] - gyro_data[col]) / (gyro_data['seconds_elapsed_shift'] - gyro_data['seconds_elapsed'])
        gyro_data['finite'] = np.isfinite(gyro_data[col + '_rise'])
        gyro_data = gyro_data[gyro_data['finite'] == True]
    gyro_data.dropna()
    gyro_data.drop(['x_shift', 'y_shift', 'z_shift'], axis=1)

    # def test_spline(row):
    #     val = spline.__call__(row['seconds_elapsed'])
    #     return val

    gyro_data = gyro_data[gyro_data['seconds_elapsed'] < 7]
    gyro_data = gyro_data[gyro_data['seconds_elapsed'] > 5]
    orientation_data = orientation_data[orientation_data['seconds_elapsed'] < 7]
    orientation_data = orientation_data[orientation_data['seconds_elapsed'] > 5]

    spline_gyro_x = CubicHermiteSpline(gyro_data['seconds_elapsed'], gyro_data['x'], gyro_data['x_rise'])
    spline_gyro_y = CubicHermiteSpline(gyro_data['seconds_elapsed'], gyro_data['y'], gyro_data['y_rise'])
    spline_gyro_z = CubicHermiteSpline(gyro_data['seconds_elapsed'], gyro_data['z'], gyro_data['z_rise'])

    r = np.arange(5, 7, 0.0001)

    # yaw -> z, pitch -> x, roll -> y

    # plt.figure()
    # plt.plot(gyro_data['seconds_elapsed'], gyro_data['x'], 'r-')
    # plt.plot(gyro_data['seconds_elapsed'], gyro_data['y'], 'g-')
    # plt.plot(gyro_data['seconds_elapsed'], gyro_data['z'], 'b-')
    # plt.plot(orientation_data['seconds_elapsed'], orientation_data['pitch'], 'r.')
    # plt.plot(orientation_data['seconds_elapsed'], orientation_data['roll'], 'g.')
    # plt.plot(orientation_data['seconds_elapsed'], orientation_data['yaw'], 'b.')
    # plt.show()
    
    # plt.figure()
    # plt.plot(gyro_data['seconds_elapsed'], gyro_data['x'], 'r-')
    # plt.plot(r, spline_gyro.__call__(r), 'b.')

    spline_orientation_pitch = CubicHermiteSpline(orientation_data['seconds_elapsed'], orientation_data['pitch'], spline_gyro_x.__call__(orientation_data['seconds_elapsed']))
    spline_orientation_yaw = CubicHermiteSpline(orientation_data['seconds_elapsed'], orientation_data['yaw'], spline_gyro_y.__call__(orientation_data['seconds_elapsed']))
    spline_orientation_roll = CubicHermiteSpline(orientation_data['seconds_elapsed'], orientation_data['roll'], spline_gyro_z.__call__(orientation_data['seconds_elapsed']))



    plt.figure()
    plt.plot(orientation_data['seconds_elapsed'], orientation_data['pitch'], 'r.')
    plt.plot(r, spline_orientation_pitch.__call__(r), 'r-', alpha=0.25)
    plt.plot(orientation_data['seconds_elapsed'], orientation_data['yaw'], 'g.')
    plt.plot(r, spline_orientation_roll.__call__(r), 'g-', alpha=0.25)
    plt.plot(orientation_data['seconds_elapsed'], orientation_data['roll'], 'b.')
    plt.plot(r, spline_orientation_yaw.__call__(r), 'b-', alpha=0.25)
    # plt.plot(gyro_data['seconds_elapsed'], gyro_data['x'] / 40, 'g-')
    plt.show()

    # alpha = np.arctan(gyro_data['x'] / gyro_data['y'])
    # beta = np.arcsin(gyro_data['z'] / 9.81)
    # plt.plot(gyro_data['seconds_elapsed'], alpha, 'r-')
    # plt.plot(gyro_data['seconds_elapsed'], beta, 'b-')
    # plt.show()

    # # data.to_csv('test.csv')
    # break