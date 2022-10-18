# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import json
from pathlib import Path
import ast
import math
import matplotlib.pyplot as plt
import PIL
from sklearn.cluster import KMeans

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
from yolov3 import fruit_detection
import slam.aruco_detector as aruco




class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # fruit detection
        self.fruit_poses = {"apple": [], "lemon": [], "orange": [], "pear": [], "strawberry": []}
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            #self.detector = Detector(args.ckpt, use_gpu=False)
            #self.network_vis = np.ones((240, 320,3))* 100
            pass
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.args = args
        self.search_list = []


    # wheel control
    def control(self):       
        if self.args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt,0.4,0.4)
        self.control_clock = time.time()
        return drive_meas
        
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas,lms=None):
        map_flag = True
        if lms is None:
            map_flag = False
            # If the robot is in the inital state [0,0] then the we pass that to the marker function to reduce covariacne for markers seen from this state
            if (np.allclose(self.ekf.robot.state[:3,0],np.zeros((3,)), atol=0.0005) ):
                #print('init state')
                lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img,init_pos=True)
            elif ((np.allclose(self.ekf.robot.state[:2,0],np.zeros((2,)), atol=0.005) )):
                #print('init spin state')
                lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img,init_pos=False,init_spin=True)
            else:
               # print('operate update lms')
                lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)


        if self.request_recover_robot:
            print("operate: recover from pause")
            if map_flag:
                #for lm in lms:
                   # self.ekf.taglist.append(int(lm.tag))
                self.ekf.add_landmarks(lms,true_pos=True)
                

            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                print('Robot pose is successfuly recovered')
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                print('Recover failed, need >2 landmarks!')
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            #print('operate lms:')
           # print(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
           # self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
       # detector_view = cv2.resize(self.network_vis,
                              #     (320, 240), cv2.INTER_NEAREST)
      #  self.draw_pygame_window(canvas, detector_view, 
      #                          position=(h_pad, 240+2*v_pad)
       #                         )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    def read_search_list(self):
        """
        Read the search order of the target fruits
        """
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()
            for fruit in fruits:
                self.search_list.append(fruit.strip())
       # self.search_list.append("done")
        #print('Search List: ', self.search_list)

    def merge_estimations(self):
        apple_est = self.fruit_poses["apple"]
        lemon_est = self.fruit_poses["lemon"]
        pear_est = self.fruit_poses["pear"]
        orange_est = self.fruit_poses["orange"]
        strawberry_est = self.fruit_poses["strawberry"]
        self.read_search_list()

        n = {"apple": 2, "lemon": 2, "pear": 2, "orange": 2, "strawberry": 2}
        for i in self.search_list:
            n[i] = 1

        if len(apple_est) > n["apple"]:
            apple_est_sort = np.sort(apple_est).reshape(-1, 2)
            apple_est = []
            kmeans = KMeans(n_clusters=n["apple"], random_state=0).fit(apple_est_sort)
            apple_est.append(kmeans.cluster_centers_)

        if len(lemon_est) > n["lemon"]:
            lemon_est_sort = np.sort(lemon_est).reshape(-1, 2)
            lemon_est = []
            kmeans = KMeans(n_clusters=n["lemon"], random_state=0).fit(lemon_est_sort)
            lemon_est.append(kmeans.cluster_centers_)

        if len(pear_est) > n["pear"]:
            pear_est_sort = np.sort(pear_est).reshape(-1, 2)
            pear_est = []
            kmeans = KMeans(n_clusters=n["pear"], random_state=0).fit(pear_est_sort)
            pear_est.append(kmeans.cluster_centers_)
        if len(orange_est) > n["orange"]:
            orange_est_sort = np.sort(orange_est).reshape(-1, 2)
            orange_est = []
            kmeans = KMeans(n_clusters=n["orange"], random_state=0).fit(orange_est_sort)
            orange_est.append(kmeans.cluster_centers_)
        if len(strawberry_est) > n["strawberry"]:
            strawberry_est_sort = np.sort(strawberry_est).reshape(-1, 2)
            strawberry_est = []
            kmeans = KMeans(n_clusters=n["strawberry"], random_state=0).fit(strawberry_est_sort)
            strawberry_est.append(kmeans.cluster_centers_)
        target_est = {}
        for i in range(2):
            try:
                target_est['apple_' + str(i)] = {'y': apple_est[0][i][0].tolist(), 'x': apple_est[0][i][1].tolist()}
            except:
                pass
            try:
                target_est['lemon_' + str(i)] = {'y': lemon_est[0][i][0].tolist(), 'x': lemon_est[0][i][1].tolist()}
            except:
                pass
            try:
                target_est['pear_' + str(i)] = {'y': pear_est[0][i][0].tolist(), 'x': pear_est[0][i][1].tolist()}
            except:
                pass
            try:
                target_est['orange_' + str(i)] = {'y': orange_est[0][i][0].tolist(), 'x': orange_est[0][i][1].tolist()}
            except:
                pass
            try:
                target_est['strawberry_' + str(i)] = {'y': strawberry_est[0][i][0].tolist(),
                                                      'x': strawberry_est[0][i][1].tolist()}
            except:
                pass

        with open('targets.txt', 'w') as fo:
            json.dump(target_est, fo)

        print('Estimations saved!')

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [1,0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-1, 0]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 5]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -5]
            # only move when key is cont pressed
            elif event.type == pygame.KEYUP and (event.key == pygame.K_UP or event.key == pygame.K_DOWN or event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT):
                self.command['motion'] = [0, 0]
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # run target_pose est
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                self.command['save_image'] = True
                robot_pose = self.ekf.robot.state
                guesses = fruit_detection(robot_pose)
                for i in guesses:
                    self.fruit_poses[i[0]].append([i[1], i[2]])
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.merge_estimations()
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='')
    args, _ = parser.parse_known_args()
    
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

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
       # operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()




