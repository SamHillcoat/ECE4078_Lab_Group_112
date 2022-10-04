import pygame
import json
import time
import numpy as np

from rrtc import *
from Obstacle import *
from drive_to_waypoint import drive_to_waypoint
from math_functions import *

from matplotlib import pyplot as plt

class Game:
    '''
    Class for waypoint GUI and planning
    '''

    def __init__(self, args):

        self.map_file = args.map
        self.level = args.level

        if args.arena == 0:
            # sim dimensions
            self.arena_width = 3
        elif args.arena == 1:
            # real dimensions
            self.arena_width = 2

        self.width, self.height = 600, 600
        
        self.waypoints = []

        self.scale_factor = self.width / self.arena_width
        # marker size is 70x70mm
        self.marker_size = 0.07
        self.fruit_r = 5
        self.tolerance = 0.1

        self.pos = (0,0)

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
        print('Search List: ', self.search_list)


    def read_true_map(self):
        """
        Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
        """
        with open(self.map_file, 'r') as fd:
            markers = json.load(fd)
            self.fruit_list = []
            self.fruit_true_pos = []
            self.aruco_true_pos = [None] * 10


        for key in markers:
            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    self.aruco_true_pos[9] = (markers[key]['x'], markers[key]['y'])
                else:
                    marker_id = int(key[5]) - 1
                    self.aruco_true_pos[marker_id] = (markers[key]['x'], markers[key]['y'])
            else:
                self.fruit_list.append(key[:-2])
                self.fruit_true_pos.append((markers[key]['x'], markers[key]['y']))

        print('Fruit List: ', self.fruit_list)
        print('Fruit True Positions: ', self.fruit_true_pos)
        print('Aruco True Positions: ', self.aruco_true_pos)


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

        return round(world_x, 1), round(world_y, 1)


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
        i = 0
        for marker in self.aruco_true_pos:
            conv_x, conv_y = self.convert_to_pygame((marker[0], marker[1]))
            scale_size = self.marker_size * self.scale_factor
            img_scaled = pygame.transform.scale(self.imgs[i], (scale_size, scale_size))
            self.canvas.blit(img_scaled, (conv_x - scale_size/2, conv_y - scale_size/2))
            i += 1

        for i in range(len(self.fruit_list)):
            conv_x, conv_y = self.convert_to_pygame(self.fruit_true_pos[i])
            if self.fruit_list[i] == 'apple':
                pygame.draw.circle(self.canvas, (255,0,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (255,0,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif self.fruit_list[i] == 'lemon':
                pygame.draw.circle(self.canvas, (255,255,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (255,255,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif self.fruit_list[i] == 'orange':
                pygame.draw.circle(self.canvas, (255,102,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (255,102,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif self.fruit_list[i] == 'pear':
                pygame.draw.circle(self.canvas, (0,255,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (0,255,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif self.fruit_list[i] == 'strawberry':
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
            drive_to_waypoint(pos)
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
        self.read_true_map()

        with open('baseline.txt', 'r') as f:
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
            self.plan_paths()

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


    '''
    Driving and planning methods from here on
    '''

    def relative_point(self):
        self.rel_pos = []
        for pos in self.fruit_true_pos:
            d = compute_distance_between_points((0,0), pos)
            t = (d - 0.25) / d
            x_t = t * pos[0]
            y_t = t * pos[1]
            conv_pos = self.convert_to_pygame((x_t, y_t))
            self.rel_pos.append((x_t, y_t))
            pygame.draw.circle(self.canvas, (0,0,0), conv_pos, 5)


    def generate_obstacles(self):

        
        self.all_obstacles = []
        for marker in self.aruco_true_pos:
            width = self.marker_size + self.baseline
            # self.all_obstacles.append(Rectangle([marker[0] - width/2, marker[1]-width/2], width, width))
            self.all_obstacles.append(Circle(marker[0], marker[1], width/2 + self.tolerance))

        for fruit in self.fruit_true_pos:
            width = 0.09 + self.baseline
            self.all_obstacles.append(Circle(fruit[0], fruit[1], width/2 + self.tolerance))


    def generate_path(self, start, end):
        rrtc = RRTC(start=start, 
                  goal=end, 
                  width=self.arena_width/2, 
                  height=self.arena_width/2, 
                  obstacle_list=self.all_obstacles,
                  expand_dis=0.6, 
                  path_resolution=0.2)
        path = rrtc.planning()
        return path


    def plan_paths(self):
        self.generate_obstacles()
        self.paths = []

        start = (0,0)
        for fruit in self.search_list:
            idx = self.fruit_list.index(fruit)
            end = self.rel_pos[idx]
            self.paths.append(self.generate_path(start, end))
            start = end

        self.reset_canvas()
        for path in self.paths:
            for i in range(len(path) - 1):
                conv_start = self.convert_to_pygame(path[i])
                conv_end = self.convert_to_pygame(path[i+1])
                pygame.draw.circle(self.canvas, (0,0,0), conv_start, 3)
                pygame.draw.line(self.canvas, (0,0,0), conv_start, conv_end, width = 2)

    
    def drive(self):
        '''
        Drives to waypoints located in self.waypoints
        '''
        print("I'm trying to drive")
        if self.level == 1:
            # drive to list of waypoints
            start = (0,0)
            for waypoint in self.waypoints:
                pos = waypoint.center
                # drive(start, pos)
                start = pos
        elif self.level == 2:
            # drive to list of paths
            start = (0,0)
            for path in self.paths:
                for node in path:
                    '''
                    FOR SAM TO IMPLEMENT
                    '''
                    # TODO drive(start, node)
                    drive_to_waypoint(node)
                    #start = node
                time.sleep(3)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", metavar='', type=int, default=0)
    parser.add_argument("--map", metavar='', type=str, default='M4_true_map.txt')
    parser.add_argument("--level", metavar='', type=int, default=2)
    args, _ = parser.parse_known_args()

    game = Game(args)
    game.run()