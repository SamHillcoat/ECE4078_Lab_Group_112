import numpy as np
from matplotlib import pyplot as plt
import json
from Obstacle import *
from rrt import *


waypoints_x = []
waypoints_y = []
with open('waypoints.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split()
        waypoints_x.append(line[0])
        waypoints_y.append(line[1])

markers = dict()
with open('M4_true_map.txt', 'r') as f:
    markers = json.load(f)
markers_x = []
markers_y = []
for key in markers:
    markers_x.append(markers[key]['x'])
    markers_y.append(markers[key]['y'])


plt.figure()


plt.scatter(np.array(waypoints_x), np.array(waypoints_y), color='blue' ,marker='o')
plt.scatter(np.array(markers_x), np.array(markers_y), color = 'red', marker = '+')
plt.grid()


# plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
# plt.xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()