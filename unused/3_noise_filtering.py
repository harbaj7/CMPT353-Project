import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# Specify the directory where your CSV files are located
csv_directory = 'extracted/'

# Specify the directories to save your plots and filtered data
plot_directory = 'plots_fft/'
filtered_directory = 'filtered/'

# Specify the CSV files to process
csv_files = ['Accelerometer.csv', 'Gyroscope.csv', 'Magnetometer.csv', 'Gravity.csv', 'Orientation.csv', 'Location.csv', 'TotalAcceleration.csv']

# Define the cutoff frequency for the FFT
cutoff = 69

# Loop over each subdirectory in the CSV directory
for subdir in os.listdir(csv_directory):
    subdir_path = os.path.join(csv_directory, subdir)

    # Ensure that the subdirectory is indeed a directory
    if os.path.isdir(subdir_path):

        # Create a subdirectory in each plot directory for each activity
        plot_subdir = os.path.join(plot_directory, subdir)
        filtered_subdir = os.path.join(filtered_directory, subdir)
        os.makedirs(plot_subdir, exist_ok=True)
        os.makedirs(filtered_subdir, exist_ok=True)

        # Loop over each CSV file
        for csv_file in csv_files:
            csv_file_path = os.path.join(subdir_path, csv_file)

            # Ensure that the CSV file exists
            if os.path.exists(csv_file_path):
                # Load the CSV file
                df = pd.read_csv(csv_file_path)

                # Create a new DataFrame for the filtered data
                df_filtered = pd.DataFrame()
                df_filtered['time'] = df['time']

                # Loop over each column in the DataFrame (excluding the 'time' column)
                for column in df.columns.drop('time'):
                    # Compute the FFT of the data
                    data_fft = np.fft.fft(df[column])

                    # Apply a low-pass filter by zeroing out components beyond the cutoff
                    data_fft[cutoff:] = 0

                    # Compute the inverse FFT to get the filtered data
                    filtered_data = np.fft.ifft(data_fft)

                    # Add the filtered data to the DataFrame
                    df_filtered[column] = filtered_data.real

                    # Plot the original data and the filtered data
                    plt.figure(figsize=(10, 6))
                    plt.plot(df['time'], df[column], label='Original')
                    plt.plot(df_filtered['time'], df_filtered[column], label='Filtered')

                    # Label the axes
                    plt.xlabel('Time (s)')
                    plt.ylabel(f'{column}')

                    plt.legend()
                    plt.title(f'{csv_file} - {column}')

                    # Save the plot in the activity subdirectory
                    plt.savefig(os.path.join(plot_subdir, f'{os.path.splitext(csv_file)[0]}_{column}.png'))
                    plt.close()

                # Save the filtered data as a CSV file
                df_filtered.to_csv(os.path.join(filtered_subdir, csv_file), index=False)

