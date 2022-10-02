from lib2to3.pgen2.pgen import generate_grammar
from rrt import *
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

        print(self.markers)
        print(self.fruit)
        print(self.waypoints)

        with open('baseline.txt', 'r') as f:
            self.baseline = np.loadtxt(f, delimiter=',')
        print(self.baseline)

    def generate_obstacles(self):
        self.all_obstacles = []
        for marker in self.markers:
            width = self.marker_width + self.baseline
            self.all_obstacles.append(Rectangle([marker[0] - width/2, marker[1]-width/2], width, width))

    def generate_path(self, start, end):
        print(start, end)
        rrt = RRT(start=start, 
                  goal=end, 
                  width=3, 
                  height=3, 
                  obstacle_list=self.all_obstacles,
                  expand_dis=0.5, 
                  path_resolution=0.1)
        path = rrt.planning()
        print(path)
        return path

    def plan(self):
        self.paths = []
        # path = self.generate_path(self.waypoints[0], self.waypoints[1])
        rrt = RRT(start=self.waypoints[0], 
                  goal=self.waypoints[1], 
                  width=3, 
                  height=3, 
                  obstacle_list=self.all_obstacles,
                  expand_dis=0.5, 
                  path_resolution=0.5)
        self.vis.delete()
        self.vis.Set2DView(20)
        animate_path_rrt(self.vis, rrt)
        self.vis.show_inline(height = 500)
        # for i in range(len(self.waypoints) - 1):
        #     self.paths.append(self.generate_path(self.waypoints[i], self.waypoints[i+1]))
        

    def plot(self):
        waypoints_x, waypoints_y = np.split(self.waypoints, 2, axis=1)
        markers_x, markers_y = np.split(self.markers, 2, axis=1)
        fruit_x, fruit_y = np.split(self.fruit, 2, axis = 1)
        self.figure = plt.figure()
        
        plt.scatter(waypoints_x, waypoints_y, color='red', marker='+')
        plt.scatter(markers_x, markers_y, color='blue', marker='s')
        plt.scatter(fruit_x, fruit_y, color='yellow', marker='s')

        plt.show()

    def run(self):
        self.load()
        self.generate_obstacles()
        
        self.plan()
        self.plot()

if __name__ == '__main__':
    planning = Planning()
    planning.run()
