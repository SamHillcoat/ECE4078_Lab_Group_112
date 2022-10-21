import posix
import pygame
import time
import numpy as np
import sys, os
import json
import argparse
import ast

from operate import Operate
from rrtc import *
from Obstacle import *
from drive_to_waypoint import Controller
from math_functions import *

from matplotlib import pyplot as plt

sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure

class Game:
    '''
    Class for waypoint GUI and planning
    '''

    def __init__(self, args):

        self.map_file = args.map
        self.fruit_map = args.fruit
        self.level = args.level
        self.arena = args.arena
        self.controller = Controller(args, operate, ppi,self.level)
        self.planning_init_flag = True

        if args.arena == 0:
            # sim dimensions
            self.arena_width = 3
            self.scale = 1
        elif args.arena == 1:
            # real dimensions
            self.arena_width = 3
            self.scale = 1

        self.width, self.height = 600, 600
        
        self.waypoints = []

        self.scale_factor = self.width / self.arena_width
        # marker size is 70x70mm
        self.marker_size = 0.07
        self.fruit_r = 5
        self.tolerance = 0.1

        self.pos = (0,0)

        #for reading in map aruco markers as ekflandmarks (see read true map function (sam))
        self.lm_measure = []

        # import images for aruco markers
        self.imgs = [pygame.image.load('pics/8bit/lm_1.png'),
                     pygame.image.load('pics/8bit/lm_2.png'),
                     pygame.image.load('pics/8bit/lm_3.png'),
                     pygame.image.load('pics/8bit/lm_4.png'),
                     pygame.image.load('pics/8bit/lm_5.png'),
                     pygame.image.load('pics/8bit/lm_6.png'),
                     pygame.image.load('pics/8bit/lm_7.png'),
                     pygame.image.load('pics/8bit/lm_8.png'),
                     pygame.image.load('pics/8bit/lm_9.png'),
                     pygame.image.load('pics/8bit/lm_10.png')]


    def read_search_list(self):
        """
        Read the search order of the target fruits
        """
        self.search_list = []
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                self.search_list.append(fruit.strip())
       # self.search_list.append("done")
        print('Search List: ', self.search_list)


    def read_true_map(self):
        """
        Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
        """
        with open(self.map_file, 'r') as fd:
            gt_dict = json.load(fd)
            self.fruit_list = []
            self.fruit_true_pos = []
            self.aruco_true_pos = np.empty([10, 2])
            self.lm_measure = []

            # remove unique id of targets of the same type
            for key in gt_dict:
                x = np.round(gt_dict[key]['x'], 1)
                y = np.round(gt_dict[key]['y'], 1)

                if key.startswith('aruco'):
                    if key.startswith('aruco10'):
                        self.aruco_true_pos[9][0] = x / self.scale
                        self.aruco_true_pos[9][1] = y / self.scale
                        lm = measure.Marker(np.array([x, y]), 10, covariance=0)
                        self.lm_measure.append(lm)
                    else:
                        marker_id = int(key[5])
                        self.aruco_true_pos[marker_id - 1][0] = x / self.scale
                        self.aruco_true_pos[marker_id - 1][1] = y / self.scale
                        lm = measure.Marker(np.array([x, y]), marker_id, covariance=0)
                        self.lm_measure.append(lm)
                else:
                    self.fruit_list.append(key[:-2])
                    if len(self.fruit_true_pos) == 0:
                        self.fruit_true_pos = np.array([[x, y]]) / self.scale
                    else:
                        self.fruit_true_pos = np.append(self.fruit_true_pos, [[x / self.scale, y / self.scale]], axis=0)


    def read_slam_markers(self) -> dict:
        with open(self.map_file, 'r') as f:
            usr_dict = ast.literal_eval(f.read())
            aruco_dict = {}
            for (i, tag) in enumerate(usr_dict["taglist"]):
                x = round(usr_dict["map"][0][i] / self.scale, 1) 
                y = round(usr_dict["map"][1][i] / self.scale, 1)
                aruco_dict[tag] = np.reshape([x,y], (2,1))
                lm = measure.Marker(np.array([x, y]), tag, covariance=0)
                self.lm_measure.append(lm)

        self.slam_markers = aruco_dict
        
    
    def read_slam_fruit(self) -> dict:
        with open(self.fruit_map, 'r') as fd:
            gt_dict = json.load(fd)
            apples, lemons, pears, oranges, strawberries = [], [], [], [], []

            for key in gt_dict:
                if key.startswith('apple'):
                    apples.append(np.array([gt_dict[key]['x'], gt_dict[key]['y']], dtype=float))
                elif key.startswith('lemon'):
                    lemons.append(np.array([gt_dict[key]['x'], gt_dict[key]['y']], dtype=float))
                elif key.startswith('pear'):
                    pears.append(np.array([gt_dict[key]['x'], gt_dict[key]['y']], dtype=float))
                elif key.startswith('orange'):
                    oranges.append(np.array([gt_dict[key]['x'], gt_dict[key]['y']], dtype=float))
                elif key.startswith('strawberry'):
                    strawberries.append(np.array([gt_dict[key]['x'], gt_dict[key]['y']], dtype=float))

        self.slam_fruits = {
            'apples': apples,
            'lemons': lemons,
            'pears': pears,
            'oranges': oranges,
            'strawberries': strawberries
        }


    def convert_to_pygame(self, pos):
        '''
        Convert from world coords to pygame coords
        '''
        x, y = pos
        origin_x, origin_y = self.width/2, self.height/2
        conv_x = origin_x - x * self.width/2 / (self.arena_width / 2)
        conv_y = origin_y + y * self.height/2 / (self.arena_width / 2)

        return conv_x, conv_y


    def convert_to_world(self, pos):
        '''
        Convert from the GUI points to points in the world frame
        '''
        origin_x, origin_y = self.width/2, self.height/2
        x, y = pos
        world_x = (origin_x - x) / (self.width/2 / (self.arena_width / 2))
        world_y = (y - origin_y) / (self.height/2 / (self.arena_width / 2))

    
        return round(world_x / self.scale, 1), round(world_y / self.scale, 1)


    def draw_grid(self):
        '''
        Draw grid for ease of viewing
        '''
        blockSize = int(self.scale_factor / 10)

        for x in range(0, self.width, blockSize):
            for y in range(0, self.height, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(self.canvas, (122,122,122), rect, 1)


    def draw_markers(self):
        '''
        '''
        
        for key in self.slam_markers:
            conv_x, conv_y = self.convert_to_pygame(self.slam_markers[key])
            scale_size = self.marker_size * self.scale_factor
            img_scaled = pygame.transform.scale(self.imgs[int(key)-1], (scale_size, scale_size))
            self.canvas.blit(img_scaled, (conv_x - scale_size/2, conv_y - scale_size/2))
    

        for key in self.slam_fruits:
            for each in self.slam_fruits[key]:
                conv_x, conv_y = self.convert_to_pygame(each)
                if key == 'apples':
                    pygame.draw.circle(self.canvas, (255,0,0), (conv_x, conv_y), self.fruit_r)
                    pygame.draw.circle(self.canvas, (255,0,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
                elif key == 'lemons':
                    pygame.draw.circle(self.canvas, (255,255,0), (conv_x, conv_y), self.fruit_r)
                    pygame.draw.circle(self.canvas, (255,255,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
                elif key == 'oranges':
                    pygame.draw.circle(self.canvas, (255,102,0), (conv_x, conv_y), self.fruit_r)
                    pygame.draw.circle(self.canvas, (255,102,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
                elif key == 'pears':
                    pygame.draw.circle(self.canvas, (0,255,0), (conv_x, conv_y), self.fruit_r)
                    pygame.draw.circle(self.canvas, (0,255,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
                elif key == 'strawberries':
                    pygame.draw.circle(self.canvas, (255,0,255), (conv_x, conv_y), self.fruit_r)
                    pygame.draw.circle(self.canvas, (255,0,255), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
    

    def draw_waypoints(self):
        '''
        Draw waypoints
        '''
        i = 1
        for waypoint in self.waypoints:
            pygame.draw.rect(self.canvas, (235,161,52), pygame.Rect(waypoint.left, waypoint.top, 10, 10))
            self.canvas.blit(self.font.render(f'{i}', True, (0,0,0)), (waypoint.left, waypoint.top))
            i += 1


    def reset_canvas(self):
        # reset canvas and redraw points
        self.canvas.fill((255,255,255))
        self.draw_grid()

        # for obstacle in self.all_obstacles:
        #     pygame.draw.rect(self.canvas, (211,211,211), pygame.Rect(obstacle.origin[0],obstacle.origin[1], obstacle.width, obstacle.height))

        self.draw_markers()
        self.draw_waypoints()
        self.canvas.blit(self.pi_bot, (self.width/2 - self.pi_bot.get_width()/2, self.height/2 - self.pi_bot.get_height()/2))


    def is_over(self, mouse_pos):
        '''
        Check if the mouse click has occurred over an existing marker
        '''
        for waypoint in self.waypoints:
            if waypoint.collidepoint(mouse_pos):
                return waypoint
        return None

    
    def place_waypoint(self, mouse_pos):
        '''
        Place a waypoint on the screen
        '''
        waypoint = pygame.draw.rect(self.canvas, (235,161,52), pygame.Rect(mouse_pos[0]-5, mouse_pos[1]-5, 10, 10) )
        self.waypoints.append(waypoint)

        self.canvas.blit(self.font.render(f'{len(self.waypoints)}', True, (0,0,0)), (waypoint.left, waypoint.top))

        if self.level == 1:
            
            pos = self.convert_to_world(mouse_pos)
            self.controller.drive_to_waypoint(pos)
            # TODO drive(self.pos, pos)
            
            self.pos = pos


    def remove_waypoint(self, waypoint):
        '''
        Remove a waypoint if one has been clicked 
        '''
        ind = self.waypoints.index(waypoint)
        self.waypoints.remove(waypoint)

        self.paths = self.paths[:ind]

        self.reset_canvas()


    def run(self):
        self.read_search_list()
        # self.read_true_map()
        self.read_slam_markers()
        self.read_slam_fruit()
        self.read_search_list()

        self.controller.lms = self.lm_measure
        
        self.current_fruit = self.search_list[0]

        aruco_pos_list = list(self.slam_markers.values())

        arr = np.array(aruco_pos_list).reshape(10,2)
        print(arr)
        print(np.indices(arr.shape, sparse=True))
        self.controller.marker_pos_arr = np.array(aruco_pos_list).reshape(10,2)


        with open('calibration/param/baseline.txt', 'r') as f:
            self.baseline = np.loadtxt(f, delimiter=',')
        print('Baseline: ', self.baseline)

        pygame.init()
    
        self.font = pygame.font.SysFont('Arial', 25)
        self.canvas = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Waypoints')
        self.canvas.fill((255, 255, 255))
        # draw grid and add botty
        self.draw_grid()
        self.pi_bot = pygame.image.load('pics/8bit/pibot_top.png')
        self.canvas.blit(self.pi_bot, (self.width/2 - self.pi_bot.get_width()/2, self.height/2 - self.pi_bot.get_height()/2))
        pygame.display.update()

        self.draw_markers()
        
        if self.level == 2:
            self.relative_point()
            self.generate_obstacles()
            self.best_path(robot_pose=[0,0,0])
        elif self.level == 1:
            self.controller.setup_ekf(self.lm_measure)
            print(self.controller.operate.ekf.robot.state)

        running = True

        while running:
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_presses = pygame.mouse.get_pressed()

                        if mouse_presses[0]:
                            mouse_pos = pygame.mouse.get_pos()
                            waypoint = self.is_over(mouse_pos)
                            if waypoint is None:
                                self.place_waypoint(mouse_pos)
                            else:
                                self.remove_waypoint(waypoint)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.drive()
                    if event.key == pygame.K_BACKSLASH:
                        if self.level == 2:
                            self.plan_paths()
                    if event.key == pygame.K_s:
                        self.controller.full_spin()
                    if event.key == pygame.K_c:
                        self.controller.calibrate(robot_pose=[0,0,0])


    '''
    Driving and planning methods from here on
    '''

    def relative_point(self):
        self.rel_pos = []
        self.true_pos = []
        for key in self.search_list:
            pos = self.slam_fruits[key+'s'][0]
            print(key,pos)
            d = compute_distance_between_points((0,0), pos)
            t = (d - 0.25) / d
            x_t = t * pos[0]
            y_t = t * pos[1]
            conv_pos = self.convert_to_pygame((x_t, y_t))
            self.rel_pos.append((x_t, y_t))
            self.true_pos.append(pos)

            # Test with true fruit pos as rel point
            #self.rel_pos.append(pos)
            pygame.draw.circle(self.canvas, (0,0,0), conv_pos, 5)


    def generate_obstacles(self):
        
        self.all_obstacles = []
        for key in self.slam_markers:
            marker = self.slam_markers[key]
            width = self.marker_size + self.baseline
            # self.all_obstacles.append(Rectangle([marker[0] - width/2, marker[1]-width/2], width, width))
            self.all_obstacles.append(Circle(marker[0], marker[1], width/2 +self.tolerance))
        for key in self.slam_fruits:
            for fruit in self.slam_fruits[key]:
                width = 0.09 + self.baseline
                self.all_obstacles.append(Circle(fruit[0], fruit[1], width/2 + self.tolerance))


    def generate_path(self, start, end):
        rrtc = RRTC(start=start, 
                  goal=end, 
                  width=self.arena_width/2, 
                  height=self.arena_width/2, 
                  obstacle_list=self.all_obstacles,
                  expand_dis=0.5, 
                  path_resolution=0.5)
        path, dist = rrtc.planning()
        return path, dist


    def plan_paths(self):
        self.generate_obstacles()
        self.paths = []

        start = (0,0)
        for end in self.rel_pos:
            path,_ = self.generate_path(start, end)
            self.paths.append(path)
            start = end

        self.reset_canvas()
        for path in self.paths:
            for i in range(len(path) - 1):
                conv_start = self.convert_to_pygame(path[i])
                conv_end = self.convert_to_pygame(path[i+1])
                pygame.draw.circle(self.canvas, (0,0,0), conv_start, 3)
                pygame.draw.line(self.canvas, (0,0,0), conv_start, conv_end, width = 2)

    def plan_to_next(self,robot_pose):
        current_fruit = self.current_fruit
        current_fruit_index = self.search_list.index(current_fruit)
        self.current_fruit_pos = self.rel_pos[current_fruit_index]
        self.current_fruit_true_pos = self.true_pos[current_fruit_index]
        #print('start pos: ', robot_pose)
        
        print('goal pos; ', self.current_fruit_pos)
        
        if(self.planning_init_flag):
            path,dist = self.generate_path((0,0),self.current_fruit_pos)
            
        else:
            robot_pose = [i.item() for i in robot_pose]
            path,dist = self.generate_path(robot_pose[0:2],self.current_fruit_pos)

       
        

      #  self.paths = []
       # self.paths.append(path)

        #self.reset_canvas()
        #for i in range(len(path) - 1):
        #        conv_start = self.convert_to_pygame(path[i])
        #        conv_end = self.convert_to_pygame(path[i+1])
         #       pygame.draw.circle(self.canvas, (0,0,0), conv_start, 3)
         #       pygame.draw.line(self.canvas, (0,0,0), conv_start, conv_end, width = 2)
        #pygame.display.update()
        #time.sleep(0.5)

        return path,dist

    def best_path(self,robot_pose):
        min_dist = 10000 #init to large number
        path = None
        dist = 10000000
        for i in range(20):
            
            try:
                path,dist = self.plan_to_next(robot_pose)
            except Exception as e:
                print("No path found")
                print(e)
            

            # For checking for a backwards path
            if not self.planning_init_flag:
                flat_pose = [robot_pose[0].item(),robot_pose[1].item()]
            else:
                flat_pose = [0,0]

            if ((path is not None) and (not np.allclose(path[0],flat_pose))):
                print("Path Reversed")
                path.reverse()
                

            if ((dist is not None) and (dist <= min_dist)):
                min_dist = dist
               # self.paths = []
                best_path = path
                self.path = path[1:]
               # self.paths.append(path[1:])
                self.reset_canvas()
                print("Path: ", path)
                print("Dist: ", dist)
                for i in range(len(path) - 1):
                        conv_start = self.convert_to_pygame(path[i])
                        conv_end = self.convert_to_pygame(path[i+1])
                        pygame.draw.circle(self.canvas, (0,0,0), conv_start, 3)
                        pygame.draw.line(self.canvas, (0,0,0), conv_start, conv_end, width = 2)
                pygame.display.update()
                time.sleep(0.5)
        self.planning_init_flag = False
        return best_path

        
    
    def drive(self):
        '''
        Drives to waypoints located in self.waypoints
        '''
        print("I'm trying to drive")
        self.controller.setup_ekf(self.lm_measure)
        if self.level == 1:
            # drive to list of waypoints
            start = (0,0)
            for waypoint in self.waypoints:
                pos = waypoint.center
                # drive(start, pos)
                start = pos
        elif self.level == 2:
            at_last_fruit = False
            while (at_last_fruit == False):
                # drive to list of paths
                start = (0,0)
                for node in self.path:
                    print("Node: ",node)
                       
                    robot_pose,new_path = self.controller.drive_to_waypoint(node,self.current_fruit_true_pos)
                        #start = node

                    if (new_path):
                        print("Planning New Path --------------------------------------------- \n\n")
                        path = self.best_path(robot_pose)
                        break

                    

                    if (self.controller.get_distance_robot_to_goal(robot_pose,self.current_fruit_true_pos) < 0.5):
                        #robot is at fruit
                        if (self.current_fruit != self.search_list[-1]):
                            self.current_fruit = self.search_list[self.search_list.index(self.current_fruit)+1]
                            print("At fruit")
                            time.sleep(3)
                            self.controller.full_spin()
                            print("Planning to next")
                            path = self.best_path(robot_pose)
                            
                        else:
                            print("Done all")
                            at_last_fruit = True
                            break

                  #  print("Planning to Next")
                   # self.plan_to_next(robot_pose)
               
            



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", metavar='', type=int, default=0)
    parser.add_argument("--map", metavar='', type=str, default='M4_true_map.txt')
    parser.add_argument("--fruit", metavar='', type=str, default='')
    parser.add_argument("--level", metavar='', type=int, default=2)
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")

    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip, args.port)
    operate = Operate(args)

    game = Game(args)
    game.run()