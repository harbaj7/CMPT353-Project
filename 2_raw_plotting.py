import numpy as np
import pandas as pd
import datetime

import os
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq

########################################################################
#   PLOTTING THE RAW DATA FOR
#   'Accelerometer.csv', 'Gyroscope.csv', 'Magnetometer.csv',
#   'Gravity.csv', 'Orientation.csv', 'Location.csv'
#   'TotalAcceleration.csv' for each activity
########################################################################
def plot_time_series(csv_filepath, save_dir):
    # Load the CSV file
    df = pd.read_csv(csv_filepath)

    # Create a plot for each column other than 'time'
    for column in df.columns:
        if column != 'time':
            plt.figure(figsize=(10, 6))
            plt.plot(df['time'], df[column])
            plt.title(f'Time Series Data - {column}')
            plt.xlabel('Time (seconds)')
            plt.ylabel(column)

            # Save the plot
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{os.path.splitext(csv_filename)[0]}_{column}.png'))
            plt.close()

# Directory containing the combined data
combined_dir = 'extracted'

# Process each CSV file in each activity directory
for activity_name in os.listdir(combined_dir):
    activity_dir = os.path.join(combined_dir, activity_name)
    for csv_filename in os.listdir(activity_dir):
        if csv_filename in ['Accelerometer.csv', 'Gyroscope.csv', 'Magnetometer.csv', 'Gravity.csv', 'Orientation.csv', 'Location.csv', 'TotalAcceleration.csv']:
            csv_filepath = os.path.join(activity_dir, csv_filename)
            save_dir = os.path.join('plots', activity_name)
            plot_time_series(csv_filepath, save_dir)

 

