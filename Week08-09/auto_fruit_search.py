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


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])
        lm_measure = []

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                    lm = measure.Marker(np.array([x,y]),10,covariance=0)
                    lm_measure.append(lm)
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
                    lm = measure.Marker(np.array([x,y]),marker_id,covariance=0)
                    lm_measure.append(lm)
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos, lm_measure


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    drive_time = time.time()
    delta_time = 0.25
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

   

    #PID controler
    threshold_dist = 0.1
    threshold_angle = 0.23

    initial_state = robot_pose

    stop_criteria_met = False

    K_pw = 0.02
    K_pv = 0.07
    
    distance_to_goal = get_distance_robot_to_goal(initial_state,waypoint)
    desired_heading = get_angle_robot_to_goal(initial_state,waypoint)


    while not stop_criteria_met:

        v_k = K_pv*distance_to_goal
        w_k = K_pw*desired_heading

        # Apply control to robot
        # wheel_vel = (lv,rv)
        wheel_vel = operate.ekf.robot.convert_to_wheel_v(v_k,w_k)
        operate.pibot.set_velocity(wheel_vel=wheel_vel, time=delta_time)
       # time.sleep(0.1)
        drive_meas = measure.Drive(wheel_vel[0]*2,wheel_vel[1]*2,dt=delta_time,left_cov = 0.2,right_cov = 0.2)
        operate.take_pic()
        
        operate.update_slam(drive_meas)
        robot_pose = operate.ekf.robot.state
        draw(canvas)
        pygame.display.update()
        new_state = robot_pose
        print(robot_pose)


        #TODO 4: Update errors ---------------------------------------------------
        distance_to_goal = get_distance_robot_to_goal(
            new_state,waypoint)
        desired_heading = get_angle_robot_to_goal(new_state,waypoint)

        #ENDTODO -----------------------------------------------------------------
        #TODO 5: Check for stopping criteria -------------------------------------
        if (distance_to_goal < threshold_dist):
            #ENDTODO -----------------------------------------------------------------
            stop_criteria_met = True



    '''
    # turn towards the waypoint
    turn_time = 0.0 # replace with your calculation
    print("Turning for {:.2f} seconds".format(turn_time))
    lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    dt = time.time() - drive_time   
    drive_meas = measure.Drive(lv, rv, dt,0.4,0.4)
    
    # after turning, drive straight to the waypoint
    drive_time = 0.0 # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))'''



def turn_to_point(waypoint,robot_pose):
    initial_state = robot_pose

    drive_time = time.time()
    delta_time = 0.1
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point



    #PID controler
    threshold_dist = 0.01
    threshold_angle = 0.1

    initial_state = robot_pose

    stop_criteria_met = False

    #K_pw = 0.25
    K_pw = 1 
    #K_pv = 0.1
    
   # distance_to_goal = get_distance_robot_to_goal(initial_state,waypoint)
    desired_heading_error = clamp_angle(get_angle_robot_to_goal(initial_state,waypoint))


    while not stop_criteria_met:

       # v_k = K_pv*distance_to_goal
        w_k = K_pw*desired_heading_error

        # Apply control to robot
        # wheel_vel = (lv,rv)
        wheel_vel = operate.ekf.robot.convert_to_wheel_v(0,w_k)
        print("Wheel Vel:")
        print(wheel_vel)
        operate.pibot.set_velocity(wheel_vel=wheel_vel, time=delta_time)
        time.sleep(0.1)
        drive_meas = measure.Drive(wheel_vel[0],wheel_vel[1],dt=delta_time,left_cov = 0.2,right_cov = 0.2)
        operate.take_pic()
        operate.update_slam(drive_meas)
        draw(canvas)
        pygame.display.update()
        robot_pose = operate.ekf.robot.state
        new_state = robot_pose
        print(robot_pose)


        #TODO 4: Update errors ---------------------------------------------------
       # distance_to_goal = get_distance_robot_to_goal(
           # new_state,waypoint)
        desired_heading_error = clamp_angle(get_angle_robot_to_goal(new_state,waypoint))
        print("heading Error")
        print(desired_heading_error)
        #ENDTODO -----------------------------------------------------------------
        #TODO 5: Check for stopping criteria -------------------------------------
        if (abs(desired_heading_error) < threshold_angle):
            #ENDTODO -----------------------------------------------------------------
            stop_criteria_met = True

    
   

def get_robot_pose(drive_meas):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    #robot_pose = [0.0,0.0,0.0] # replace with your calculation
   # print(aruco_true_pos)
  #  print(lm_measure[0].position.shape)

    operate.ekf.add_landmarks(lm_measure)
    operate.ekf.recover_from_pause(lm_measure)
    ####################################################
   # operate.ekf.predict(drive_meas)
   # operate.ekf.update(lm_measure)

    robot_pose = operate.ekf.get_state_vector()[0:3,:]
    print(robot_pose)

    return robot_pose


def get_distance_robot_to_goal(robot_pose, goal=np.zeros(2)):
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


def get_angle_robot_to_goal(robot_state, goal=np.zeros(2)):
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

	alpha = clamp_angle(np.arctan2(y_diff, x_diff) - theta)

	return alpha


def clamp_angle(rad_angle=0, min_value=-np.pi, max_value=np.pi):
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

def draw(canvas):
        bg = pygame.image.load('pics/gui_mask.jpg')
        canvas.blit(bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = operate.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = operate.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(operate.aruco_img, (320, 240))
        operate.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )
        return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")

    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    #Setup EKF
    operate = Operate(args)
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False


    
    
    


    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos, lm_measure = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
  
    waypoint = [0.0,0.0]
    #robot_pose = [0.0,0.0,0.0]

    #Add true map landmarks to ekf
    drive_meas = operate.control()

    operate.request_recover_robot = True
    operate.update_slam(drive_meas,lms=lm_measure)


    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:
        # enter the waypoints
        
        # instead of manually enter waypoints, you can get coordinates by clicking on a map, see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue 

        # estimate the robot's pose
            # Run SLAM with true marker positions (modied from operate.py)
        #operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        

        operate.request_recover_robot = False
        operate.update_slam(drive_meas)
        robot_pose = operate.ekf.robot.state
        draw(canvas)
        pygame.display.update()
        print(robot_pose)
        
        

        # robot drives to the waypoint
        waypoint = [x,y]
        turn_to_point(waypoint,robot_pose)
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break