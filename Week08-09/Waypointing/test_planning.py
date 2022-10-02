import numpy as np

from rrtc import *
from Obstacle import *

from Practical03_Support.path_animation import *
import meshcat.geometry as g
import meshcat.transformations as tf
from ece4078.Utility import StartMeshcat

import json
from matplotlib import pyplot as plt

class Planning:

    def __init__(self) -> None:
        self.markers = []
        self.waypoints = [(0,0)]
        self.fruit = []

        self.marker_width = 0.07

        self.vis = StartMeshcat()
        
    
    def load(self):
        '''
        Load in map file
        '''
        with open('M4_true_map.txt', 'r') as f:
            markers = json.load(f)

        for key in markers:
            if key[0:5] == 'aruco':
                self.markers.append((markers[key]['x'], markers[key]['y']))
            else:
                self.fruit.append((markers[key]['x'], markers[key]['y']))
        
        with open('waypoints.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            self.waypoints.append(line)
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        self.markers = np.asarray(self.markers, dtype=float)
        self.fruit = np.asarray(self.fruit, dtype=float)

        print('Marker Locs: ', self.markers)
        print('Fruit Locs: ', self.fruit)
        print('Waypoint Locs: ', self.waypoints)

        with open('baseline.txt', 'r') as f:
            self.baseline = np.loadtxt(f, delimiter=',')
        print('Baseline: ', self.baseline)

    def generate_obstacles(self):
        self.all_obstacles = []
        for marker in self.markers:
            width = self.marker_width + self.baseline
            # self.all_obstacles.append(Rectangle([marker[0] - width/2, marker[1]-width/2], width, width))
            self.all_obstacles.append(Circle(marker[0], marker[1], width / 2))

    def generate_path(self, start, end):
        print(start, end)
        rrt = RRTC(start=start, 
                  goal=end, 
                  width=3, 
                  height=3, 
                  obstacle_list=self.all_obstacles,
                  expand_dis=0.6, 
                  path_resolution=0.2)
        path = rrt.planning()
        return path

    def plan(self):
        self.paths = []
        for i in range(len(self.waypoints) - 1):
            self.paths.append(self.generate_path(self.waypoints[i], self.waypoints[i+1]))
        

    def plot(self):
        waypoints_x, waypoints_y = np.split(self.waypoints, 2, axis=1)
        markers_x, markers_y = np.split(self.markers, 2, axis=1)
        fruit_x, fruit_y = np.split(self.fruit, 2, axis = 1)
        self.figure = plt.figure()

        for path in self.paths:
            temp = np.asarray(path)
            x, y = np.split(temp, 2, axis=1)
            plt.plot(x, y, color='green', marker='o')
        
        
        plt.scatter(markers_x, markers_y, color='blue', marker='s')
        plt.scatter(fruit_x, fruit_y, color='yellow', marker='s')
        plt.scatter(waypoints_x.T, waypoints_y.T, color='red', marker='+')
        print(len(self.paths))
        

        plt.show()

    def run(self):
        self.load()
        self.generate_obstacles()
        
        self.plan()
        self.plot()

if __name__ == '__main__':
    planning = Planning()
    planning.run()
