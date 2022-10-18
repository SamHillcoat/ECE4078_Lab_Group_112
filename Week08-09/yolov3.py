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


def merge_estimations(target_pose_dict):
    target_pose_dict = target_pose_dict
    apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
    target_est = {}

    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('pear'):
                pear_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('strawberry'):
                strawberry_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    if len(apple_est) > 2:
        apple_est_sort = np.sort(apple_est).reshape(-1, 2)
        apple_est = []
        kmeans = KMeans(n_clusters=2, random_state=0).fit(apple_est_sort)
        apple_est.append(kmeans.cluster_centers_)

    if len(lemon_est) > 2:
        lemon_est_sort = np.sort(lemon_est).reshape(-1, 2)
        lemon_est = []
        kmeans = KMeans(n_clusters=2, random_state=0).fit(lemon_est_sort)
        lemon_est.append(kmeans.cluster_centers_)

    if len(pear_est) > 2:
        pear_est_sort = np.sort(pear_est).reshape(-1, 2)
        pear_est = []
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pear_est_sort)
        pear_est.append(kmeans.cluster_centers_)
    if len(orange_est) > 2:
        orange_est_sort = np.sort(orange_est).reshape(-1, 2)
        orange_est = []
        kmeans = KMeans(n_clusters=2, random_state=0).fit(orange_est_sort)
        orange_est.append(kmeans.cluster_centers_)
    if len(strawberry_est) > 2:
        strawberry_est_sort = np.sort(strawberry_est).reshape(-1, 2)
        strawberry_est = []
        kmeans = KMeans(n_clusters=2, random_state=0).fit(strawberry_est_sort)
        strawberry_est.append(kmeans.cluster_centers_)

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

    return target_est


def fruit_detection(robot_pose):
    fileK = "{}intrinsic_sim.txt".format('./calibration/param/')
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
    image_path = base_dir / 'pibot_dataset/img_0.png'

    estimates = []
    for detections in get_darknet_bbox(image_path):
        estimates.append(estimate_pose(camera_matrix, detections, robot_pose, maptype=args.type))

    return(estimates)
