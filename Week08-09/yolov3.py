import cv2 as cv
import numpy as np
import json
import os
from pathlib import Path
import ast
import math
import matplotlib.pyplot as plt
import PIL
from sklearn.cluster import KMeans

im_width = 416
fruit_poses = []


def get_darknet_bbox(image_path):
    net = cv.dnn_DetectionModel('yolo-obj.cfg', 'yolo-obj_final.weights')  # change path
    net.setInputSize(416, 416)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    with open('obj.names', 'rt') as f:  # change path
        names = f.read().rstrip('\n').split('\n')

    bounding_boxes = []
    frame = cv.imread(image_path)
    classes, confidences, boxes = net.detect(frame, confThreshold=0.25, nmsThreshold=0.4)
    if len(classes) == 0:
        return "No detections"
    else:
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            label = '%s: %s' % (names[classId], box)

            labelSize, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            left, top, width, height = box
            top = max(top, labelSize[1])
            cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
            cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseline), (255, 255, 255),
                         cv.FILLED)
            cv.putText(frame, names[classId], (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))

            bb_label = [names[classId], left, top, width, height]
            bounding_boxes.append(bb_label)

        # cv.imshow('out', frame)
        # cv.imwrite('/mnt/c/Users/prakr/Documents/GitHub/ECE4078_Lab_Group_112/Week08-09/result.jpg', frame)
        return bounding_boxes


def estimate_pose(camera_matrix, detections, robot_pose, maptype='sim'):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    target_dimensions = {'apple': [],
                         'lemon': [],
                         'pear': [],
                         'orange': [],
                         'strawberry': []}
    if (maptype == "sim"):
        apple_dimensions = [0.075448, 0.074871, 0.071889]
        target_dimensions['apple'] = apple_dimensions
        lemon_dimensions = [0.060588, 0.059299, 0.053017]
        target_dimensions['lemon'] = lemon_dimensions
        pear_dimensions = [0.0946, 0.0948, 0.135]
        target_dimensions['pear'] = pear_dimensions
        orange_dimensions = [0.0721, 0.0771, 0.0739]
        target_dimensions['orange'] = orange_dimensions
        strawberry_dimensions = [0.052, 0.0346, 0.0376]
        target_dimensions['strawberry'] = strawberry_dimensions
    elif (maptype == "robot"):
        apple_dimensions = [0.075448, 0.074871, 0.083]
        target_dimensions['apple'] = apple_dimensions
        lemon_dimensions = [0.060588, 0.059299, 0.050]
        target_dimensions['lemon'] = lemon_dimensions
        pear_dimensions = [0.0946, 0.0948, 0.085]
        target_dimensions['pear'] = pear_dimensions
        orange_dimensions = [0.0721, 0.0771, 0.070]
        target_dimensions['orange'] = orange_dimensions
        strawberry_dimensions = [0.052, 0.0346, 0.040]
        target_dimensions['strawberry'] = strawberry_dimensions

    target_list = ['apple', 'lemon', 'pear', 'orange', 'strawberry']

    fruit_type = detections[0]
    true_height = target_dimensions[detections[0]][2]

    d1 = focal_length * true_height / detections[4]  # focal_length*true_height/boundingbox_height # depth from camera

    bb_cx = detections[1] + (detections[3] / 2)
    d2_hat = -camera_matrix[0][2] + bb_cx
    d2 = (d2_hat / focal_length) * d1
    fruit_x = robot_pose[0] + d1 * np.cos(robot_pose[2] + d2 * np.sin(robot_pose[2]))
    fruit_y = robot_pose[1] + d1 * np.sin(robot_pose[2] + d2 * np.cos(robot_pose[2]))

    return [fruit_type, fruit_y, fruit_x]


def fruit_detection(robot_pose):
    fileK = "{}intrinsic.txt".format('calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')

    import argparse

    parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    parser.add_argument("--type", type=str, default='sim')
    args, _ = parser.parse_known_args()

    if (args.type == "sim"):
        fileK = "{}intrinsic_sim.txt".format('./calibration/param/')
    elif (args.type == "robot"):
        fileK = "{}intrinsic_robot.txt".format('./calibration/param/')
    else:
        fileK = "{}intrinsic_sim.txt".format('./calibration/param/')

    # estimate pose of targets in each detector output
    image_path = 'pibot_dataset/img_0.png'

    estimates = []
    detections = get_darknet_bbox(image_path)
    if detections == "No detections":
        return None
    else:
        for detection in detections:
            estimates.append(estimate_pose(camera_matrix, detection, robot_pose, maptype=args.type))
        return(estimates)
