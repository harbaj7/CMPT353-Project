# python3 process_data.py

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile
from sklearn.decomposition import FastICA

NUM_SECS_DISPLAY = 10
BUFFER_SECS = 5
RENAME = {'gravity_x':'x', 'gravity_y':'y', 'gravity_z':'z', 'seconds_elapsed_rounded':'seconds_elapsed'}

# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
files = [f for f in listdir('grouped/') if isfile(join('grouped', f))]
print(files)
for file in files:
    try:
        data = pd.read_csv('grouped/' + file)
        data = data[['gravity_x', 'gravity_y', 'gravity_z', 'seconds_elapsed_rounded']].rename(columns=RENAME)
        data = data[data['seconds_elapsed'] > 10]
        data = data[data['seconds_elapsed'] < 12]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(data['x'], data['y'], data['z'])
        plt.show()

        plt.figure()
        plt.plot(data['seconds_elapsed'], data['x'])
        plt.show()
    except:
        print(file)
    # break
    

    



