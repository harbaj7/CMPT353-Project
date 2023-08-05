import os
import pandas as pd
import numpy as np

# Specify the directory where your CSV files are located
csv_directory = 'filtered/'

# Specify the directory to save your merged CSV files
merged_directory = 'filtered_combined/'

# Ensure that the merged directory exists
os.makedirs(merged_directory, exist_ok=True)

# Loop over each subdirectory in the CSV directory
for subdir in os.listdir(csv_directory):
    subdir_path = os.path.join(csv_directory, subdir)

    # Ensure that the subdirectory is indeed a directory
    if os.path.isdir(subdir_path):
        # Create a list to store the DataFrames
        dfs = []

        # Loop over each CSV file in the subdirectory
        for csv_file in os.listdir(subdir_path):
            csv_file_path = os.path.join(subdir_path, csv_file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)

            # Remove rows with NaN time
            df = df[pd.notnull(df['time'])]

            # Rename the columns (except for the 'time' column)
            df.rename(columns={col: f'{os.path.splitext(csv_file)[0]}_{col}' for col in df.columns if col != 'time'}, inplace=True)

            # Add the DataFrame to the list
            dfs.append(df)

        # Merge the DataFrames on the 'time' column
        df_merged = pd.concat(dfs, axis=1)

        # Remove duplicate 'time' columns
        df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
        # Remove rows with NaN or negative time
        df_merged = df_merged[pd.notnull(df_merged['time'])]
        df_merged = df_merged[df_merged['time'] >= 0]
        # Save the merged DataFrame to a CSV file
        df_merged.to_csv(os.path.join(merged_directory, f'{subdir}.csv'), index=False)

