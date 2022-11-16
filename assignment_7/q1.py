import numpy as np
from matplotlib import pyplot as plt

rand_trip = np.loadtxt('rand_points.txt')

fig = plt.figure()
ax = plt.axes(projection='3d')
xdata = rand_trip[:,0]
ydata = rand_trip[:,1]
zdata = rand_trip[:,2]

ax.set_proj_type('ortho')
ax.scatter(xdata, ydata, zdata, s = 1, marker=',')

plt.show()

rand_trip_python = np.random.randint(1e8,size=(len(xdata),3))
xdata_p = rand_trip_python[:,0]
ydata_p = rand_trip_python[:,1]
zdata_p = rand_trip_python[:,2]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')
ax.scatter(xdata_p, ydata_p, zdata_p, s = 1, marker=',')
plt.show()