# Phone Sensor Data Analysis - CMPT353 Project

## Project Description
This project focuses on analyzing phone sensor data to distinguish between different activities, specifically walking and running. The sensor data is manually collected and stored as zip files in the data folder. The analysis involves several steps, including data extraction and refinement, raw data plotting, Fourier transformations, data combination, and final analysis.

## Prerequisites
The project is implemented in Python. Ensure you have Python 3.6 or later installed on your system. 

The following Python libraries are required:
* os
* sys
* zipfile
* pandas
* numpy
* matplotlib
* scipy
* sklearn
---
## Files and Execution Order
The project consists of five Python scripts that should be run in the following order:

This script extracts the zipped data files and refines the data by discarding unnecessary files and columns.
```bash
python3 1_extract_and_refine.py data
```

This script generates raw plots for each activity in the dataset, providing an initial visual analysis of the data.
```bash
python3 2_raw_plotting.py
```

This script combines the refined data into a single CSV file for further analysis.
```bash
python3 3_combine_data.py
```
This script applies the Fast Fourier Transform to the sensor data, converting it to the frequency domain for frequency-based analysis.
```bash
python3 4_fourier.py
```

This script performs the final data analysis, including machine learning models to predict activities based on the dominant frequencies in the data.
```bash
python3 5_analysis.py
```

