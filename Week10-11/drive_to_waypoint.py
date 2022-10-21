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
    def __init__(self, args, operate, ppi,level):
        self.operate = operate
        self.ppi = ppi
        self.args = args
        self.level = level
        self.lms = None

        self.new_path = False
        # P gains (MAYBE CHANGE FOR REAL ROBOT)
        self.turnK = 0.4 #angular gain in turn loop
        self.driveKv = 0.6 #linear
        self.driveKw = 0 #angular gain while in drive loop(want to be very low)

        self.driveCorrFac = 1.25 # correction factor applied to slam predict wheel vel determined using calibration func

        #For Robot
        #self.turnK = 2
       # self.driveKv = 2 #linear
        #self.driveKw = 0.1 #angular gain while in drive loop(want to be very low)
        
        self.heading_correct_ang_thresh = 0.4
        self.heading_correct_dist_thresh = 0.25

        self.spin_time = 7.5 # Based on calibration values, time taken to do a full 360 in sec

        #Turn loop heading error threshold
        self.threshold_angle = 0.08

        #Covariance and uncertainty
        self.XY_uncertainty = 0
        self.XY_uncertainty_thresh = 0.6 #needs tuning TODO
        


        self.debug = False
        self.level = level
        #Real
        #self.turnK = 1
      #  self.driveKv = 0.7
       # self.driveKw = 0.02
        
        if self.level == 2 or self.debug == True:
            pygame.font.init() 
            TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
            TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
        
            width, height = 700, 660
            #self.canvas = canvas
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


    def drive_to_waypoint(self, waypoint,fruit_pos=None):
        self.fruit_pos = fruit_pos
        #Setup EKF
        start = False
        self.new_path = False
        #Add true map landmarks to ekf
        # The following code is only a skeleton code the semi-auto fruit searching task
      #  while True:
            # estimate the robot's pose
                # Run SLAM with true marker positions (modied from self.operate.py)
            #self.operate.update_keyboard()
        #self.operate.take_pic()
        drive_meas = self.operate.control()

        self.operate.request_recover_robot = False
        self.operate.update_slam(drive_meas)
        robot_pose = self.operate.ekf.robot.state
       # self.draw(canvas)
        #pygame.display.update()
        print(robot_pose)

        # robot drives to the waypoint
        # waypoint = [x,y]
        robot_pose = self.full_spin()
        robot_pose = self.turn_to_point(waypoint, robot_pose)
        robot_pose = self.drive_to_point(waypoint, robot_pose)
        print("At waypoint")
        
        print("Finished driving to waypoint: {}; New robot pose: {}".format(
            waypoint, robot_pose))

        # exit
        self.ppi.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        #  if uInput == 'N':
        #      break

        return robot_pose,self.new_path



    def drive_to_point(self, waypoint, robot_pose):
        drive_time = time.time()
        delta_time = 0.1
    
        #PID controler
        threshold_dist = 0.08
        threshold_angle = 0.23

        initial_state = robot_pose

        stop_criteria_met = False

        K_pw = self.driveKw
        K_pv = self.driveKv

        last_distances_to_goal = [] 
        
        distance_to_goal = self.get_distance_robot_to_goal(initial_state,waypoint)
        desired_heading_error = self.clamp_angle(self.get_angle_robot_to_goal(initial_state,waypoint))

        last_distances_to_goal.append(distance_to_goal)
        

       

        while not stop_criteria_met:
            pygame.display.update() #maybe remove if causing issues


            v_k = K_pv*distance_to_goal

            if v_k > 0.08:
                v_k = 0.08


            w_k = K_pw*desired_heading_error

         #   print("Contorl Input:",v_k)

            # Apply control to robot
            # wheel_vel = (lv,rv)
            wheel_vel = self.operate.ekf.robot.convert_to_wheel_v(v_k,w_k)
            self.operate.pibot.set_velocity(wheel_vel=wheel_vel, time=delta_time)
            print("Wheel Vel: ", wheel_vel)
            # time.sleep(0.1)
            drive_meas = measure.Drive(wheel_vel[0]*2*self.driveCorrFac,wheel_vel[1]*2*self.driveCorrFac,dt=delta_time,left_cov = 0.001,right_cov = 0.001)
            self.operate.take_pic()
            
            self.operate.update_slam(drive_meas)
            last_state = robot_pose
            robot_pose = self.operate.ekf.robot.state
            if self.level == 2 or self.debug:
                self.draw()
                pygame.display.update()
            new_state = robot_pose
           # print("Pose:", robot_pose)

            # Check if pose has changed by too much in one update (tries to see if slam updates) and make new path
            if (self.get_distance_robot_to_goal(last_state,new_state[0:2]) > 0.2):
                self.new_path = True
                stop_criteria_met = True


            #check uncertainty in slam position, if too big do a spin to relocate robot (only x,y for now)
            P = self.operate.ekf.P[0:2,0:2]
            axes,_ = self.operate.ekf.make_ellipse(P)
            self.XY_uncertainty = self.ellipse_area(*axes)
            print('uncertainty: ',self.XY_uncertainty)

            if (self.XY_uncertainty > self.XY_uncertainty_thresh):
                robot_pose = self.full_spin()
                self.new_path = True
                print("Uncertainty High: new path ---------------------------------- \n")
                stop_criteria_met = True
                
            


            #TODO 4: Update errors ---------------------------------------------------
            
            distance_to_goal = self.get_distance_robot_to_goal(
                new_state,waypoint)
            #desired_heading = self.get_angle_robot_to_goal(new_state,waypoint)
            desired_heading_error = self.clamp_angle(self.get_angle_robot_to_goal(new_state,waypoint))

            last_distances_to_goal.append(distance_to_goal)
            
            print("Distance:", distance_to_goal)
            print("Heading Error:", desired_heading_error)
            

            if self.fruit_pos is not None:
                dist_to_fruit = self.get_distance_robot_to_goal(robot_pose,self.fruit_pos)
                if(dist_to_fruit < 0.4):
                    stop_criteria_met = True
                    print("breaking: at fruit ---------------------------------\n")
                    break

            #if heading error too high, stop and correct
            if ((not stop_criteria_met) and (abs(desired_heading_error) > self.heading_correct_ang_thresh) and (distance_to_goal > self.heading_correct_dist_thresh)):
                robot_pose = self.turn_to_point(waypoint,new_state)


            # If distance to waypoint is increasing, stop loop
            if (self.level != 1):
                if ((not stop_criteria_met) and (len(last_distances_to_goal) > 4)):
                    # Last 3 distance measurements are increasing
                    if(all(i < j for i, j in zip(last_distances_to_goal[-4:], last_distances_to_goal[-3:]))):
                        print("Distance Increasing: Stopped -----------------------------------------\n\n")
                        self.new_path = True
                        stop_criteria_met = True


            #ENDTODO -----------------------------------------------------------------
            #TODO 5: Check for stopping criteria -------------------------------------
            if (distance_to_goal < threshold_dist):
                #ENDTODO -----------------------------------------------------------------
                stop_criteria_met = True

        return robot_pose

    def calibrate(self,robot_pose):
        startTime = time.time()
        v_k = 0.08
        w_k = 0
        delta_time = 0.1
        delLoopTime = 0
        while True:
            corrFac = float(input("CorrFac: "))
            startTime = time.time()
            delLoopTime = 0

            self.setup_ekf(self.lms)

            while (delLoopTime < 4):
                wheel_vel = self.operate.ekf.robot.convert_to_wheel_v(v_k,w_k)
                print("Wheel Vel: ", wheel_vel)
                self.operate.pibot.set_velocity(wheel_vel=wheel_vel, time=delta_time)
                # time.sleep(0.1)
                drive_meas = measure.Drive(wheel_vel[0]*2*corrFac,wheel_vel[1]*2*corrFac,dt=delta_time,left_cov = 0.1,right_cov = 0.1)
                self.operate.take_pic()
                    
                self.operate.update_slam(drive_meas)
                last_state = robot_pose
                robot_pose = self.operate.ekf.robot.state

                print("Pose: ",robot_pose)
                delLoopTime = time.time() - startTime

    def turn_to_point(self,waypoint,robot_pose):
        initial_state = robot_pose

        drive_time = time.time()
        delta_time = 0.1

        #PID controler
        threshold_dist = 0.01
        

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
           # print("Wheel Vel:")
            print(wheel_vel)
            self.operate.pibot.set_velocity(wheel_vel=wheel_vel, time=delta_time)
           #time.sleep(0.1)
            drive_meas = measure.Drive(wheel_vel[0]*2,wheel_vel[1]*2,dt=delta_time,left_cov = 0.1,right_cov = 0.1)
            self.operate.take_pic()
            self.operate.update_slam(drive_meas)
            robot_pose = self.operate.ekf.robot.state
            if self.level == 2 or self.debug:
                self.draw()
                pygame.display.update()
            
            
            new_state = robot_pose
           # print("pose: ",robot_pose)


            #TODO 4: Update errors ---------------------------------------------------
        # distance_to_goal = get_distance_robot_to_goal(
            # new_state,waypoint)
            desired_heading_error = self.clamp_angle(self.get_angle_robot_to_goal(new_state,waypoint))
            print("heading Error: ",desired_heading_error)
            print("Desired angle: ",self.get_angle_robot_to_goal(new_state,waypoint))
            #ENDTODO -----------------------------------------------------------------
            #TODO 5: Check for stopping criteria -------------------------------------
            if (abs(desired_heading_error) < self.threshold_angle):
                #ENDTODO -----------------------------------------------------------------
                stop_criteria_met = True

        

        return robot_pose

    def full_spin(self):
        startTime = time.time()
        dt = 0.1
        #lv,rv = self.operate.pibot.set_velocity([0,1],turning_tick=30,time=self.spin_time)
        deltaTime = 0
        while (deltaTime < self.spin_time):
            print("spinning")
            lv,rv = self.operate.pibot.set_velocity([0,1],turning_tick=20,time=dt)
            drive_meas = measure.Drive(lv*2,rv*2,dt=dt,left_cov = 0.06,right_cov = 0.06)
            self.operate.take_pic()
            self.operate.update_slam(drive_meas)
            robot_pose = self.operate.ekf.robot.state
            if self.level == 2 or self.debug == True:
                self.draw()
                pygame.display.update()
          #  print("pose:", robot_pose)
            deltaTime = time.time() - startTime

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
    @staticmethod
    def ellipse_area(ax1,ax2):
        return np.pi*ax1*ax2
