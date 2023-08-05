# python3 plot3d path/to/csv
# creates 3d plot and 2d plots for gravity vector data in argument
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import scipy.fft

SAMPLE_RATE = 20
NUM_SECS_DISPLAY = 5
BUFFER_SECS = 5
RENAME = {'gravity_x':'x', 'gravity_y':'y', 'gravity_z':'z', 'seconds_elapsed_rounded':'seconds_elapsed'}

def fourier(data, col):
    N = data.shape[0]

    # prevents an unhelpful spike at 0
    mean = np.mean(data[col])
    minus_mean = data[col] - mean

    # the fft, mostly copied from the docs https://docs.scipy.org/doc/scipy/tutorial/fft.html 
    # N is the number of samples, SAMPLE_RATE is in Hertz
    # the x axis
    frequency = np.linspace (0.0, SAMPLE_RATE / 2, int (N/2)) 
    # the y axis
    freq_data = scipy.fft.fft(np.array(minus_mean)) 
    # I'm pretty sure this is to deal with negative and complex values
    # also the N/2 thing is because there's some principle that prevents you from analysing frequencies greater than half your sample rate
    y = 2/N * np.abs (freq_data [0:int (N/2)]) 

    return (frequency, y, minus_mean)

file = sys.argv[1]
data = pd.read_csv(file)
data = data[['gravity_x', 'gravity_y', 'gravity_z', 'seconds_elapsed_rounded']].rename(columns=RENAME)
data = data.dropna()
data = data[data['seconds_elapsed'] > BUFFER_SECS]
data = data[data['seconds_elapsed'] < BUFFER_SECS + NUM_SECS_DISPLAY]

plt.figure()
plt.plot(data['seconds_elapsed'], data['x'], 'r-', label='x')
plt.plot(data['seconds_elapsed'], data['y'], 'g-', label='y')
plt.plot(data['seconds_elapsed'], data['z'], 'b-', label='z')
plt.xlabel('time (seconds)')
plt.ylabel('acceleration (m/s^2)')
plt.legend()
plt.title('Plot of Gravity Components')

frequecy_x, x, minus_mean_x = fourier(data, 'x')
frequecy_y, y, minus_mean_y = fourier(data, 'y')
frequecy_z, z, minus_mean_z = fourier(data, 'z')

plt.figure()
plt.plot(frequecy_x, x, 'r-', label='x')
plt.plot(frequecy_y, y, 'g-', label='y')
plt.plot(frequecy_z, z, 'b-', label='z')
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude (m/s^2)')
plt.legend()
plt.title('Fast Fourier Transform of Gravity Components')

zeros = np.full((data.shape[0], 1), 0)
vec_x = np.c_[data['x'].to_numpy(), zeros]
vec_y = np.c_[data['y'].to_numpy(), zeros]
vec_z = np.c_[data['z'].to_numpy(), zeros]

print(data)
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(0, vec_x.shape[0]):
    ax.plot(vec_x[i], vec_y[i], vec_z[i])
    ax.set_title('3D Plot Of Gravity Vectors')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
plt.show()

    



        