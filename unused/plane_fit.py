from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from zipfile import ZipFile
from sklearn.decomposition import FastICA

# https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
# generate some random test points 
m = 20 # number of points
delta = 0.01 # size of random displacement
origin = np.random.rand(3, 1) # random origin for the plane
basis = np.random.rand(3, 2) # random basis vectors for the plane
coefficients = np.random.rand(2, m) # random coefficients for points on the plane

# generate random points on the plane and add random displacement
points = basis @ coefficients \
         + np.tile(origin, (1, m)) \
         + delta * np.random.rand(3, m)

# now find the best-fitting plane for the test points

# subtract out the centroid and take the SVD
svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))

# Extract the left singular vectors
left = svd[0]
print(left)
normal = left[:, -1]

x = np.arange(-2, 2, 0.25)
y = np.arange(-2, 2, 0.25)
x, y = np.meshgrid(x, y)
z = (normal[0] * x + normal[1] * y) / normal[2] * -1

# x = np.arange(-5, 5, 0.25)
# y = np.arange(-5, 5, 0.25)
# x, y = np.meshgrid(x, y)
# z = np.full((4, 40), 0)
# xyz = np.block([[x], [y], [z]])
# xyz = np.matmul(svd[0], xyz)
# # xyz = xyz.transpose()
# x, y, z = np.meshgrid(xyz[0], xyz[1], xyz[2])
print(x)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, alpha=1)
ax.scatter3D(points[0], points[1], points[2])
plt.show()