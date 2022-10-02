# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
import cv2
import math
#from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

#darknet
import darknet4078 as darknet
from sklearn.cluster import KMeans 

im_width = 416

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.NEAREST)
    target = Image(image)==target_number
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box


def get_darknet_bbox(image_path,network,class_names,class_colors,thresh=0.25):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    bounding_boxes = []
    im_width = darknet.network_width(network)
    print(im_width)
    im_height = darknet.network_height(network)
    darknet_image = darknet.make_image(im_width, im_height, 3)

    image = cv2.imread(image_path)
 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (im_width, im_height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        bbox_data = [label,x,y,w,h]
        bounding_boxes.append(bbox_data)    
        #print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
   # print(detections)
    darknet.print_detections(detections)
    #image = darknet.draw_boxes(detections, image_resized, class_colors)
    
    return bounding_boxes

class ImageInfo():
    def __init__(self,bbox_list,robot_pose):
        self.num_targets = len(bbox_list)
        self.robot_pose = robot_pose
        self.bboxes = {'apple': [],
                        'lemon': [],
                        'pear': [],
                        'orange': [],
                        'strawberry': []}
        for i in range(self.num_targets):
            target = bbox_list[i][0]
            self.bboxes[target].append(bbox_list[i][1:])
    def getDetections(self):
        detections = []

        for t in self.bboxes.keys():
            if len(self.bboxes[t]) > 0:
                detections.append(t)
        return detections

        



# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = []
    target_lst_pose = []
    completed_img_dict = {}

   

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = pear, 4 = orange, 5 = strawberry, 0 = not_a_target

    #bboxes for all targets in this image
    bboxes = get_darknet_bbox(file_path,network,class_names,class_colors)
    num_targets = len(bboxes)

    imageInfo = ImageInfo(bboxes,image_poses[file_path])
    '''
    for t in range(num_targets):
        target_lst_box[t] = bboxes[t]
        robot_pose = image_poses[file_path]
        target_lst_pose[t] = np.array(robot_pose).reshape(3,)

   
    for target_num in img_vals:
        if target_num > 0:
            try:
                box = get_darknet_bbox(target_num, base_dir/file_path) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass
    '''
    # if there are more than one objects of the same type, combine them
    #for i in range(5):
   #     if len(target_lst_box[i])>0:
    #        box = np.stack(target_lst_box[i], axis=1)
    #        pose = np.stack(target_lst_pose[i], axis=1)
    #        completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return imageInfo

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, Info,maptype='sim'):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = {'apple': [],
                        'lemon': [],
                        'pear': [],
                        'orange': [],
                        'strawberry': []}
    if (maptype=="sim"):
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
    elif (maptype=="robot"):
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

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    targets = Info.getDetections()

    for target in targets:
        box = Info.bboxes[target] # [[x],[y],[width],[height]], can be as many boxes as same object in image
        robot_pose = Info.robot_pose # [[x], [y], [theta]]
        true_height = target_dimensions[target][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        for i in range(len(box)):
            target_pose = {'y': 0.0, 'x': 0.0}
            d1 = focal_length*true_height/box[i][3] #focal_length*true_height/boundingbox_height # depth from camera
            im_cx = im_width/2
            
            bb_cx = box[i][0] + box[i][2]/2
            d2_hat = -camera_matrix[0][2] + bb_cx
            d2 = (d2_hat/focal_length) * d1
            target_pose['x'] = robot_pose[0] + d1*np.cos(robot_pose[2] + d2*np.sin(robot_pose[2]))
            target_pose['y'] = robot_pose[1] + d1*np.sin(robot_pose[2] + d2*np.cos(robot_pose[2]))
            
            target_pose_dict[target] = target_pose
        ###########################################
    
    return target_pose_dict

# merge the estimations of the targets so that there are at most 3 estimations of each target type
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





    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    if len(apple_est) > 2:
        apple_est_sort = np.sort(apple_est).reshape(-1,2)
       
        apple_est = []
        kmeans = KMeans(n_clusters = 2,random_state=0).fit(apple_est_sort)
        apple_est.append(kmeans.cluster_centers_)
        
    if len(lemon_est) > 2:
        lemon_est_sort = np.sort(lemon_est).reshape(-1,2)
       # lemon_est_sort = np.array(lemon_est_sort)
        lemon_est = []
        kmeans = KMeans(n_clusters = 2,random_state=0).fit(lemon_est_sort)
        lemon_est.append(kmeans.cluster_centers_)
        print(lemon_est)
       # print(lemon_est.shape)
    if len(pear_est) > 2:
        pear_est_sort = np.sort(pear_est).reshape(-1,2)
        
        pear_est = []
        kmeans = KMeans(n_clusters = 2,random_state=0).fit(pear_est_sort)
        pear_est.append(kmeans.cluster_centers_)
    if len(orange_est) > 2:
        orange_est_sort = np.sort(orange_est).reshape(-1,2)
       
        orange_est = []
        kmeans = KMeans(n_clusters = 2,random_state=0).fit(orange_est_sort)
        orange_est.append(kmeans.cluster_centers_)
    if len(strawberry_est) > 2:
        strawberry_est_sort = np.sort(strawberry_est).reshape(-1,2)
        
        strawberry_est = []
        kmeans = KMeans(n_clusters = 2,random_state=0).fit(strawberry_est_sort)
        strawberry_est.append(kmeans.cluster_centers_)

    for i in range(2):
        try:
            target_est['apple_'+str(i)] = {'y':apple_est[0][i][0].tolist(), 'x':apple_est[0][i][1].tolist()}
        except:
            pass
        try:
            target_est['lemon_'+str(i)] = {'y':lemon_est[0][i][0].tolist(), 'x':lemon_est[0][i][1].tolist()}
        except:
            pass
        try:
            target_est['pear_'+str(i)] = {'y':pear_est[0][i][0].tolist(), 'x':pear_est[0][i][1].tolist()}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'y':orange_est[0][i][0].tolist(), 'x':orange_est[0][i][1].tolist()}
        except:
            pass
        try:
            target_est['strawberry_'+str(i)] = {'y':strawberry_est[0][i][0].tolist(), 'x':strawberry_est[0][i][1].tolist()}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic_sim.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    config_file = "yolo-obj.cfg"
    data_file = "obj.data"
    weights = "yolo-obj_final.weights"

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
    
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict,maptype=args.type)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
    print(target_est)
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')



