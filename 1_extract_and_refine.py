import os
import sys
import zipfile
import pandas as pd

# Iterate over all zip files in the 'data' directory
data_dir = sys.argv[1]

# Keep track of the names we have already encountered
name_count = {}

# Process each zip file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.zip'):
        # Extract the zip file
        zip_filepath = os.path.join(data_dir, filename)
        base_name = filename.split('-')[0]  # Get the base name from the filename

        # If the base name has already been encountered, add a suffix to it
        if base_name in name_count:
            name_count[base_name] += 1
            activity_name = f'{base_name}_{name_count[base_name]}'
        else:
            name_count[base_name] = 0
            activity_name = base_name

        extracted_dir = os.path.join('extracted', activity_name)
        os.makedirs(extracted_dir, exist_ok=True)  # Create the directory if it doesn't exist

        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        
        # Process each CSV file in the extracted directory
        for csv_filename in os.listdir(extracted_dir):
            csv_filepath = os.path.join(extracted_dir, csv_filename)
            if csv_filename in ['Annotation.csv', 'Metadata.csv']:
                # Delete the unwanted CSV files
                os.remove(csv_filepath)
            elif csv_filename.endswith('.csv'):
                df = pd.read_csv(csv_filepath)
                
                # Drop 'time' column and rename 'seconds_elapsed' to 'time'
                df.drop(columns=['time'], inplace=True)
                df.rename(columns={'seconds_elapsed': 'time'}, inplace=True)
                
                # Save the modified CSV file
                df.to_csv(csv_filepath, index=False)
                
