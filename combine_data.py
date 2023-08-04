from os import listdir, makedirs
from os.path import isfile, join, exists
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
    'Magnetometer.csv', 'MagnetometerUncalibrated.csv', 'Orientation.csv'
]
names = []

BASE = 0.05
name_count = {}

# Create the 'grouped' directory if it doesn't exist
if not exists('grouped'):
    makedirs('grouped')

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

        # Remove rows with negative time
        data = data[data[new_column_names['seconds_elapsed']] >= 0]

        data['seconds_elapsed_rounded'] = np.rint(data[new_column_names['seconds_elapsed']] / BASE)
        data = data.groupby('seconds_elapsed_rounded', axis=0).mean()
        all_data.append(data)

    single_frame = all_data[0]
    for i in range(1, len(all_data)):
        data = all_data[i]
        single_frame = single_frame.join(data, how='outer')

    single_frame = single_frame.reset_index().sort_values(by='seconds_elapsed_rounded')
    single_frame['seconds_elapsed_rounded'] = single_frame['seconds_elapsed_rounded'] * BASE

    # Clean the filename
    base_name = file.split('-')[0]
    if base_name in name_count:
        name_count[base_name] += 1
        clean_name = f'{base_name}_{name_count[base_name]}'
    else:
        name_count[base_name] = 0
        clean_name = base_name

    single_frame.to_csv('grouped/' + clean_name + '_grouped.csv', index=False)

# After all data is grouped and saved, start splitting by 15 second intervals
grouped_files = [f for f in listdir('grouped/') if isfile(join('grouped', f))]

for file in grouped_files:
    data = pd.read_csv('grouped/' + file)
    max_time = data['seconds_elapsed_rounded'].max()
    min_time = data['seconds_elapsed_rounded'].min()

    chunk_number = 0
    start_time = 0
    while start_time < max_time:
        end_time = start_time + 15
        chunk = data[(data['seconds_elapsed_rounded'] >= start_time) & (data['seconds_elapsed_rounded'] < end_time)]
        if chunk['seconds_elapsed_rounded'].max() - chunk['seconds_elapsed_rounded'].min() >= 5:
            # Reset time to start from 0
            chunk = chunk.copy()
            chunk['seconds_elapsed_rounded'] = chunk['seconds_elapsed_rounded'] - chunk['seconds_elapsed_rounded'].min()

            # Save the chunk with a new name
            chunk_number += 1
            chunk.to_csv('grouped/' + file.split('.')[0] + '_' + str(chunk_number) + '.csv', index=False)

        start_time += 15