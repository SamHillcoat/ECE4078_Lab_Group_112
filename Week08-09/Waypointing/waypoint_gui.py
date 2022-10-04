import pygame
import json
import re
import numpy as np
import math

from Obstacle import *
from rrtc import *

from Practical03_Support.path_animation import *
import meshcat.geometry as g
import meshcat.transformations as tf
#from ece4078.Utility import StartMeshcat

class Game:
    '''
    Class for waypoint GUI and planning
    '''

    def __init__(self, args):
        self.markers = None
        self.width, self.height = 600, 600
        self.waypoints = []
        self.map_file = args.map

        self.marker_locs = []
        self.fruit_locs = []

        self.fruit_r = 5

        if args.arena == 0:
            # sim dimensions
            self.arena_width = 3
        elif args.arena == 1:
            # real dimensions
            self.arena_width = 2

        self.scale_factor = self.width / self.arena_width
        # marker size is 70x70mm
        self.marker_size = 0.07

        self.paths = []

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

        # import images for aruco markers
        self.imgs = {"aruco1_0": pygame.image.load('pics/8bit/lm_1.png'),
                     "aruco2_0": pygame.image.load('pics/8bit/lm_2.png'),
                     "aruco3_0": pygame.image.load('pics/8bit/lm_3.png'),
                     "aruco4_0": pygame.image.load('pics/8bit/lm_4.png'),
                     "aruco5_0": pygame.image.load('pics/8bit/lm_5.png'),
                     "aruco6_0": pygame.image.load('pics/8bit/lm_6.png'),
                     "aruco7_0": pygame.image.load('pics/8bit/lm_7.png'),
                     "aruco8_0": pygame.image.load('pics/8bit/lm_8.png'),
                     "aruco9_0": pygame.image.load('pics/8bit/lm_9.png'),
                     "aruco10_0": pygame.image.load('pics/8bit/lm_10.png'),}


    
    def load(self):
        '''
        Load in map file
        '''
        with open(self.map_file, 'r') as f:
            markers = json.load(f)

        self.markers = markers
        self.draw_markers()


    def draw_grid(self):
        '''
        Draw grid for ease of viewing
        '''
        blockSize = int(self.scale_factor / 10)

        for x in range(0, self.width, blockSize):
            for y in range(0, self.height, blockSize):
                rect = pygame.Rect(x, y, blockSize, blockSize)
                pygame.draw.rect(self.canvas, (122,122,122), rect, 1)


    def draw_coords(self, x, y, img = None, fruit = None):
        '''
        Draw markers on the screen given coordinates
        '''
        origin_x, origin_y = self.width/2, self.height/2
        conv_x = origin_x - x * self.width/2 / (self.arena_width / 2)
        conv_y = origin_y + y * self.height/2 / (self.arena_width / 2)

        if img:
            scale_size = self.marker_size * self.scale_factor
            img_scaled = pygame.transform.scale(img, (scale_size, scale_size))
            self.canvas.blit(img_scaled, (conv_x - scale_size/2, conv_y - scale_size/2))

            self.marker_locs.append((conv_x - scale_size/2, conv_y - scale_size/2))
        else:
            if fruit == 'apple':
                pygame.draw.circle(self.canvas, (255,0,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (255,0,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif fruit == 'lemon':
                pygame.draw.circle(self.canvas, (255,255,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (255,255,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif fruit == 'orange':
                pygame.draw.circle(self.canvas, (255,102,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (255,102,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif fruit == 'pear':
                pygame.draw.circle(self.canvas, (0,255,0), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (0,255,0), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)
            elif fruit == 'strawberry':
                pygame.draw.circle(self.canvas, (255,0,255), (conv_x, conv_y), self.fruit_r)
                pygame.draw.circle(self.canvas, (255,0,255), (conv_x, conv_y), 5 * (self.scale_factor / 10), 1)

            self.fruit_locs.append((conv_x, conv_y))

        pygame.display.update()


    def draw_markers(self):
        '''
        Draw all markers and fruit from the map file
        '''
        for key in self.markers:
            try:
                self.draw_coords(self.markers[key]['x'], self.markers[key]['y'], img=self.imgs[key])
            except:
                self.draw_coords(self.markers[key]['x'], self.markers[key]['y'], fruit=re.sub(r'[^a-zA-Z]', '', key))


    def draw_waypoints(self):
        '''
        Draw waypoints
        '''
        for waypoint in self.waypoints:
            pygame.draw.rect(self.canvas, (235,161,52), pygame.Rect(waypoint.left, waypoint.top, 10, 10))


    def add_text(self):
        '''
        Add text to the waypoints to see the order in which the bot will visit the waypoints
        '''
        i = 1
        for waypoint in self.waypoints:
            self.canvas.blit(self.font.render(f'{i}', True, (0,0,0)), (waypoint.left, waypoint.top))
            i += 1


    def convert_to_world(self, pos):
        '''
        Convert from the GUI points to points in the world frame
        '''
        origin_x, origin_y = self.width/2, self.height/2
        x, y = pos
        world_x = (origin_x - x) / (self.width/2 / (self.arena_width / 2))
        world_y = (y - origin_y) / (self.height/2 / (self.arena_width / 2))

        return round(world_x, 1), round(world_y, 1)


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
        self.add_text()

        self.path_planning()


    def remove_waypoint(self, waypoint):
        '''
        Remove a waypoint if one has been clicked 
        '''
        ind = self.waypoints.index(waypoint)
        self.waypoints.remove(waypoint)

        self.paths = self.paths[:ind]

        if ind == 0:
            self.paths.append(self.generate_path((self.width/2, self.height/2), self.waypoints[0]))
            for i in range(0, len(self.waypoints)-1):
                self.paths.append(self.generate_path(self.waypoints[i], self.waypoints[i+1]))
        else:
            for i in range(ind-1, len(self.waypoints)-1):
                self.paths.append(self.generate_path(self.waypoints[i], self.waypoints[i+1]))
        self.draw_paths()
        

    def reset_canvas(self):
        # reset canvas and redraw points
        self.canvas.fill((255,255,255))
        self.draw_grid()

        for obstacle in self.all_obstacles:
            pygame.draw.rect(self.canvas, (211,211,211), pygame.Rect(obstacle.origin[0],obstacle.origin[1], obstacle.width, obstacle.height))

        self.draw_markers()
        self.draw_waypoints()
        self.add_text()
        self.canvas.blit(self.pi_bot, (self.width/2 - self.pi_bot.get_width()/2, self.height/2 - self.pi_bot.get_height()/2))


    def write_waypoints(self):
        '''
        Write waypoints to a file (testing purposes)
        '''
        with open('waypoints.txt', 'w+') as f:
            for waypoint in self.waypoints:
                x, y = self.convert_to_world(waypoint.center)
                f.write(f'{x} {y}\n')


    def run(self):
        '''
        Run the GUI
        '''

        self.load()

        with open('baseline.txt', 'r') as f:
            self.baseline = np.loadtxt(f, delimiter=',')
        print(self.baseline)

        self.all_obstacles = []
        
        for marker in self.marker_locs:
            width = (self.marker_size + self.baseline) * self.scale_factor
            marker_width = self.marker_size * self.scale_factor
    
            self.all_obstacles.append(Rectangle([marker[0] - width/2, marker[1]-width/2], width+marker_width, width+marker_width))

            
        self.reset_canvas()
        
        running = True

        while running:
            pygame.display.update()

            for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_presses = pygame.mouse.get_pressed()

                        if mouse_presses[0]:
                            mouse_pos = pygame.mouse.get_pos()
                            waypoint = self.is_over(mouse_pos)
                            if waypoint is None:
                                self.place_waypoint(mouse_pos)
                            else:
                                self.remove_waypoint(waypoint)
                            self.write_waypoints()

                            pygame.display.set_caption(f'{mouse_pos[0]}, {mouse_pos[1]}')
                    elif event.type == pygame.QUIT:
                        running = False


    '''
    Functions for RRT planning from now on
    '''
    def generate_path(self, start, end):
        rrt = RRTC(start=start, 
                  goal=end, 
                  width=self.width, 
                  height=self.height, 
                  obstacle_list=self.all_obstacles,
                  expand_dis=100, 
                  path_resolution=0.1)
        path = rrt.planning()
        # vis = StartMeshcat()
        # vis.delete()
        # vis.Set2DView(scale = 20, center = [-1, 16, 12, 0])
        # animate_path_rrt(vis, rrt)
        # vis.show_inline(height = 500)

        return path


    def draw_paths(self):

        self.reset_canvas()
        for path in self.paths:
            for i in range(len(path) - 1):
                pygame.draw.circle(self.canvas, (0,0,0), path[i], 3)
                pygame.draw.line(self.canvas, (0,0,0), path[i], path[i+1], width = 2)
            pygame.draw.circle(self.canvas, (0,0,0), path[-1], 3)


    def path_planning(self):
        '''
        Function for RRT planning
        '''
        
        if len(self.paths) == 0:
            self.paths.append(self.generate_path((self.width/2, self.height/2), self.waypoints[0]))
            
        else:
            self.paths.append(self.generate_path(self.waypoints[-2], self.waypoints[-1]))

        self.draw_paths()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", metavar='', type=int, default=0)
    parser.add_argument("--map", metavar='', type=str, default='M4_true_map.txt')
    args, _ = parser.parse_known_args()

    game = Game(args)
    game.run()
    

