import os
import sys
import pandas as pd
import numpy as np
from zipfile import ZipFile
from pathlib import Path
import datetime

import scipy.signal
import shutil
# python3 raw_combine.py data
################################################################
#   This file combines all the data from raw zip file and combines them togehter
#   into a single csv file after changing time to seconds (with 1 decimal point),
#   drops the first and last 3 second data (time adjustment for putting phone into pocket) 
#   and removes data that does not contain time and gets it ready for testing.
################################################################
def count_steps(data, height):
    # Use scipy's find_peaks function to find peaks in the data
    peaks, _ = scipy.signal.find_peaks(data, height=height)

    # The number of peaks is the number of steps
    num_steps = len(peaks)

    # Calculate steps per second
    steps_per_second = num_steps / data.shape[0]

    return steps_per_second

# Function to convert time to seconds with one decimal point
def convert_time_to_seconds(time_ns):
    return round(time_ns / 1e9, 1)  # Convert nanoseconds to seconds

# Function to process CSV files
def process_csv_file(file_path):
    # Get the filename without extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert 'time' column to seconds and reset starting time
    df['time'] = df['time'].apply(convert_time_to_seconds)
    df['time'] -= df['time'].min()

    # Drop 'seconds_elapsed' column
    df.drop(columns=['seconds_elapsed'], inplace=True)

    # Create a datetime index for resampling
    start_time = datetime.datetime(2023, 1, 1)  # Arbitrarily chosen start date
    df.index = pd.to_datetime(start_time + pd.to_timedelta(df['time'], unit='s'))

    # Resample data
    df_resampled = df.resample('100ms').mean().interpolate()

    # Convert datetime index back to elapsed time in seconds
    df_resampled['time'] = (df_resampled.index - start_time) / np.timedelta64(1, 's')

    # Remove the first and last 3 seconds
    df_resampled = df_resampled[(df_resampled['time'] > 3) & (df_resampled['time'] < df_resampled['time'].max() - 3)]

    # Remove rows without time data
    df_resampled = df_resampled[df_resampled['time'].notnull()]

    # Rename columns
    for col in df_resampled.columns:
        if col != 'time':
            df_resampled = df_resampled.rename(columns={col: f'{base_name}_{col}'})

    # Reset index
    df_resampled.reset_index(drop=True, inplace=True)

    return df_resampled

# Iterate over all zip files in the 'data' directory
data_dir = sys.argv[1]

# Keep track of the names we have already encountered
name_count = {}

for zip_file in Path(data_dir).rglob('*.zip'):
    with ZipFile(zip_file, 'r') as zip_ref:
        # Extract files to a temporary directory
        temp_dir = os.path.join('temp', zip_file.stem)
        os.makedirs(temp_dir, exist_ok=True)
        zip_ref.extractall(temp_dir)

        # Process CSV files
        dfs = []
        for csv_file in Path(temp_dir).rglob('*.csv'):
            if csv_file.stem not in ['Metadata', 'Annotation']:
                df = process_csv_file(csv_file)
                dfs.append(df)

        # Combine all DataFrames
        df_combined = pd.concat(dfs, axis=1)
        df_combined = df_combined.loc[:,~df_combined.columns.duplicated()]  # remove duplicated 'time' columns

        # Save the combined DataFrame to the output directory
        output_dir = 'raw_combine'
        os.makedirs(output_dir, exist_ok=True)
        filename = zip_file.stem.split('-', 1)[0]

        df_combined.to_csv(os.path.join(output_dir, f'{filename}.csv'), index=False)

        # Remove the 'temp' directory after processing all zip files
        if os.path.exists('temp'):
            shutil.rmtree('temp')
            

