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
image_path= '/mnt'
fruit_poses = []
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
    print("No detections")
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
    print(bounding_boxes)