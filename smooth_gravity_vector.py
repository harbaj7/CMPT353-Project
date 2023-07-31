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
    zip = ZipFile('data/' + file)
    data = pd.read_csv(zip.open('Gravity.csv'))

    data['seconds_elapsed_shift'] = data['seconds_elapsed'].shift(-1)
    columns = ['x', 'y', 'z']
    for col in columns:
        data[col + '_shift'] = data[col].shift(-1)
        data[col + '_rise'] = (data[col + '_shift'] - data[col]) / (data['seconds_elapsed_shift'] - data['seconds_elapsed'])
        data['finite'] = np.isfinite(data[col + '_rise'])
        data = data[data['finite'] == True]
    data.dropna()
    data.drop(['x_shift', 'y_shift', 'z_shift'], axis=1)

    spline = CubicHermiteSpline(data['seconds_elapsed'], data['x'], data['x_rise'])

    def test_spline(row):
        val = spline.__call__(row['seconds_elapsed'])
        return val
    
    # data['x_spline'] = data.apply(test_spline, axis=1)

    data = data[data['seconds_elapsed'] < 10]
    data = data[data['seconds_elapsed'] > 5]

    r = np.arange(5, 10, 0.001)

    plt.figure()
    plt.plot(data['seconds_elapsed'], data['x'], 'r.')
    plt.plot(r, spline.__call__(r), 'b.')
    plt.show()

    # data.to_csv('test.csv')
    print(data)
    break