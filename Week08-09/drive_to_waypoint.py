# M4 - Autonomous fruit searching

# basic python packages
import optparse
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
from operate import Operate
import pygame

"""
# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
"""
#import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi
import util.measure as measure


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints

class Controller:
    def __init__(self, args, operate, ppi):
        self.operate = operate
        self.ppi = ppi
        self.args = args


        # P gains (MAYBE CHANGE FOR REAL ROBOT)
        self.turnK = 0.25
        self.driveKv = 0.2 #linear
        self.driveKw = 0.02 #angular (want to be very low)
        

        #Real
        #self.turnK = 1
      #  self.driveKv = 0.7
       # self.driveKw = 0.02
        

        pygame.font.init() 
        TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
        TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
        width, height = 700, 660
        self.canvas = pygame.display.set_mode((width, height))
        pygame.display.set_caption('ECE4078 2021 Lab')
        pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
        self.canvas.fill((0, 0, 0))
        splash = pygame.image.load('pics/loading.png')
        pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
        pygame.display.update()

        
    def setup_ekf(self, lm_measure):
        drive_meas = self.operate.control()

        self.operate.request_recover_robot = True
        self.operate.update_slam(drive_meas,lms=lm_measure)


    def drive_to_waypoint(self, waypoint):
        #Setup EKF
        start = False
        #Add true map landmarks to ekf
        # The following code is only a skeleton code the semi-auto fruit searching task
      #  while True:
            # estimate the robot's pose
                # Run SLAM with true marker positions (modied from self.operate.py)
            #self.operate.update_keyboard()
        self.operate.take_pic()
        drive_meas = self.operate.control()

        self.operate.request_recover_robot = False
        self.operate.update_slam(drive_meas)
        robot_pose = self.operate.ekf.robot.state
        #draw(canvas)
        #pygame.display.update()
        print(robot_pose)

        # robot drives to the waypoint
        # waypoint = [x,y]
        robot_pose = self.turn_to_point(waypoint, robot_pose)
        robot_pose = self.drive_to_point(waypoint, robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(
            waypoint, robot_pose))

        # exit
        self.ppi.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        #  if uInput == 'N':
        #      break

    def drive_to_point_simple(self, waypoint, robot_pose):
        delta_time = 1

        initial_state = robot_pose
        distance_to_goal = self.get_distance_robot_to_goal(initial_state,waypoint)

        stop_criteria_met = False

        threshold_dist = 0.1
        
        while not stop_criteria_met:
            v = 3

            wheel_vel = self.operate.ekf.robot.convert_to_wheel_v(v,0)

            drive_meas = measure.Drive(wheel_vel[0]*2,wheel_vel[1]*2,dt=delta_time,left_cov = 0.2,right_cov = 0.2)
            self.operate.take_pic()
            
            self.operate.update_slam(drive_meas)
            robot_pose = self.operate.ekf.robot.state
            new_state = robot_pose

            distance_to_goal = self.get_distance_robot_to_goal(
                new_state,waypoint)
            print("Distance Error:", distance_to_goal)

            if (distance_to_goal < threshold_dist):
                #ENDTODO -----------------------------------------------------------------
                stop_criteria_met = True

        return robot_pose

            








    def drive_to_point(self, waypoint, robot_pose):
        drive_time = time.time()
        delta_time = 0.8
    
        #PID controler
        threshold_dist = 0.08
        threshold_angle = 0.23

        initial_state = robot_pose

        stop_criteria_met = False

        K_pw = self.driveKw
        K_pv = self.driveKv
        
        distance_to_goal = self.get_distance_robot_to_goal(initial_state,waypoint)
        desired_heading = self.get_angle_robot_to_goal(initial_state,waypoint)


        while not stop_criteria_met:

            v_k = K_pv*distance_to_goal
            w_k = K_pw*desired_heading

            # Apply control to robot
            # wheel_vel = (lv,rv)
            wheel_vel = self.operate.ekf.robot.convert_to_wheel_v(v_k,w_k)
            self.operate.pibot.set_velocity(wheel_vel=wheel_vel, time=delta_time)
            #time.sleep(0.1)
            drive_meas = measure.Drive(wheel_vel[0]*2,wheel_vel[1]*2,dt=delta_time,left_cov = 0.2,right_cov = 0.2)
            self.operate.take_pic()
            
            self.operate.update_slam(drive_meas)
            robot_pose = self.operate.ekf.robot.state
           # self.draw()
          #  pygame.display.update()
            new_state = robot_pose
            print(robot_pose)


            #TODO 4: Update errors ---------------------------------------------------
            distance_to_goal = self.get_distance_robot_to_goal(
                new_state,waypoint)
            print("Distance Error:", distance_to_goal)
            print("Wheel Vel:", wheel_vel)
            desired_heading = self.get_angle_robot_to_goal(new_state,waypoint)

            #ENDTODO -----------------------------------------------------------------
            #TODO 5: Check for stopping criteria -------------------------------------
            if (distance_to_goal < threshold_dist):
                #ENDTODO -----------------------------------------------------------------
                stop_criteria_met = True

        return robot_pose


    def turn_to_point(self,waypoint,robot_pose):
        initial_state = robot_pose

        drive_time = time.time()
        delta_time = 0.1

        #PID controler
        threshold_dist = 0.015
        threshold_angle = 0.15

        initial_state = robot_pose

        stop_criteria_met = False

        K_pw = self.turnK
        #K_pv = 0.1
        
     # distance_to_goal = get_distance_robot_to_goal(initial_state,waypoint)
        desired_heading_error = self.clamp_angle(self.get_angle_robot_to_goal(initial_state,waypoint))


        while not stop_criteria_met:

        # v_k = K_pv*distance_to_goal
            w_k = K_pw*desired_heading_error

            # Apply control to robot
            # wheel_vel = (lv,rv)
            wheel_vel = self.operate.ekf.robot.convert_to_wheel_v(0,w_k)
            print("Wheel Vel: ")
            print(wheel_vel)
            self.operate.pibot.set_velocity(wheel_vel=wheel_vel, time=delta_time)
            time.sleep(0.1)
            drive_meas = measure.Drive(wheel_vel[0],wheel_vel[1],dt=delta_time,left_cov = 0.2,right_cov = 0.2)
            self.operate.take_pic()
            self.operate.update_slam(drive_meas)
            #self.draw()
          #  pygame.display.update()
            robot_pose = self.operate.ekf.robot.state
            new_state = robot_pose
            print(robot_pose)


            #TODO 4: Update errors ---------------------------------------------------
        # distance_to_goal = get_distance_robot_to_goal(
            # new_state,waypoint)
            desired_heading_error = self.clamp_angle(self.get_angle_robot_to_goal(new_state,waypoint))
            print("heading Error")
            print(desired_heading_error)
            #ENDTODO -----------------------------------------------------------------
            #TODO 5: Check for stopping criteria -------------------------------------
            if (abs(desired_heading_error) < threshold_angle):
                #ENDTODO -----------------------------------------------------------------
                stop_criteria_met = True

        return robot_pose

    def get_distance_robot_to_goal(self,robot_pose, goal=np.zeros(2)):
        """
        Compute Euclidean distance between the robot and the goal location
        :param robot_pose: 3D vector (x, y, theta) representing the current state of the robot
        :param goal: 2D Cartesian coordinates of goal location
        """

        #if goal.shape[0] < 3:
        #	goal = np.hstack((goal, np.array([0])))

        x_goal, y_goal = goal
        x, y,_ = robot_pose
        x_diff = x_goal - x
        y_diff = y_goal - y

        rho = np.hypot(x_diff, y_diff)

        return rho


    def get_angle_robot_to_goal(self, robot_state, goal=np.zeros(2)):
        """
        Compute angle to the goal relative to the heading of the robot.
        Angle is restricted to the [-pi, pi] interval
        :param robot_state: 3D vector (x, y, theta) representing the current state of the robot
        :param goal: 3D Cartesian coordinates of goal location
        """

        #if goal.shape[0] < 3:
        #	goal = np.hstack((goal, np.array([0])))

        x_goal, y_goal = goal
        x, y, theta = robot_state
        x_diff = x_goal - x
        y_diff = y_goal - y

        alpha = self.clamp_angle(np.arctan2(y_diff, x_diff) - theta)

        return alpha


    def clamp_angle(self,rad_angle=0, min_value=-np.pi, max_value=np.pi):
        """
        Restrict angle to the range [min, max]
        :param rad_angle: angle in radians
        :param min_value: min angle value
        :param max_value: max angle value
        """

        if min_value > 0:
            min_value *= -1

        angle = (rad_angle + max_value) % (2 * np.pi) + min_value

        return angle

    

    def draw(self):
        bg = pygame.image.load('pics/gui_mask.jpg')
        self.canvas.blit(bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.operate.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.operate.ekf_on)
        self.canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.operate.aruco_img, (320, 240))
        self.operate.draw_pygame_window(self.canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )
        #return canvas