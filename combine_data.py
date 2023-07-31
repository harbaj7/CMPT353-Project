# groups data into BASE second intervals for the fft

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile
from sklearn.decomposition import FastICA

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
    all_data = []
    for csv_name in csv_names:
        data = pd.read_csv(zip.open(csv_name))
        prefix = csv_name.split('.')[0].lower()

        new_column_names = {}
        columns = data.columns.values.tolist()
        for column in columns:
            new_column_names.update({column:prefix + '_' + column})

        data = data.rename(columns=new_column_names)
        data['seconds_elapsed_rounded'] = np.rint(data[new_column_names['seconds_elapsed']] / BASE)
        data = data.groupby('seconds_elapsed_rounded', axis=0).mean()
        all_data.append(data)

    # print(all_data)

    single_frame = all_data[0]
    for i in range(1, len(all_data)):
        data = all_data[i]
        single_frame = single_frame.join(data, how='outer')

    single_frame = single_frame.reset_index().sort_values(by='seconds_elapsed_rounded')
    single_frame['seconds_elapsed_rounded'] = single_frame['seconds_elapsed_rounded'] * BASE
    single_frame.to_csv('grouped/' + file + '_grouped.csv', index=False)