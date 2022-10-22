import cv2
import cvzone
import numpy as np
import shutil
import random

import os
from os import walk

import re

def generate_images(num_images, num_bg, num_combo):
    '''
    params:     num_images, number of images to generate from per fruit
                num_bg, number of backgrounds to generate images with
                num_combo, number of images to generate with two or three fruits

    output:     path: generated_images/
                containing generated images and associated YOLO formatted label files

    YOLO Format:
        - one row per object
        - Each row is class x_center y_center width height format.
        - Box coordinates must be in normalized xywh format (from 0 - 1). 
            If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
        - Class numbers are zero-indexed (start from 0).

    Label Definition:
    0 - Apple
    1 - Lemon
    2 - Orange
    3 - Pear
    4 - Strawberry
    '''

    # create dictionary of labels
    label_dict = {
        'apple' : '0',
        'lemon' : '1',
        'orange' : '2',
        'pear' : '3',
        'strawberry' : '4'
    }

    # create output directory, if it exists then wipe it
    dir = 'generated_images'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    # get list of folders of fruit
    fruit_dirs = []
    for (dirpath, dirnames, filenames) in walk('cut_images'):
        fruit_dirs.extend(dirnames)

    # get list of background images
    bg_f = []
    for (dirpath, dirnames, filenames) in walk('backgrounds'):
        bg_f.extend(filenames)

    # iterate through fruit folders
    for fruit_folder in fruit_dirs:

        # get list of images of fruit in current fruit folder
        fruit_imgs = []
        for (dirpath, dirnames, filenames) in walk('cut_images/' + fruit_folder):
            fruit_imgs.extend(filenames)

        # choose random images from the current folder
        rand_imgs = np.random.choice(fruit_imgs, num_images, replace = False)

        # for each randomly selected fruit image
        for fruit_1 in rand_imgs:

            # select random backgrounds
            bg_samples = np.random.choice(bg_f, num_bg, replace = False)
            
            # read in current images
            img_fruit_1 = cv2.imread('cut_images/' + fruit_folder + '/' + fruit_1, cv2.IMREAD_UNCHANGED)
            img_fruit_1 = cv2.resize(img_fruit_1, (0, 0), None, 0.25, 0.25)

            scale_factor1 = (round(random.uniform(0.6, 1.3), 1))
            print(scale_factor1)
            img_fruit_1 = cv2.resize(img_fruit_1, None, fx = scale_factor1, fy = scale_factor1)

            fruit_h, fruit_w = img_fruit_1.shape[0:2]
            fruit = fruit_1.split('.')[0]

            # generate list of second fruits
            
            for fruit_folder_2 in fruit_dirs:
                curr_folder = []
                for (dirpath, dirnames, filenames) in walk('cut_images/' + fruit_folder_2):
                    curr_folder.extend(filenames)

                # randomly select images from the second folder
                fruit_imgs_2 = np.random.choice(curr_folder, num_combo, replace = False)

                # iterate through random images from second folder
                for fruit_2 in fruit_imgs_2:

                    # read in second image
                    # print('cut_images/' + fruit_folder_2 + '/' + fruit_2)
                    img_fruit_2 = cv2.imread('cut_images/' + fruit_folder_2 + '/' + fruit_2, cv2.IMREAD_UNCHANGED)
                    img_fruit_2 = cv2.resize(img_fruit_2, (0, 0), None, 0.25, 0.25)

                    scale_factor2 = (round(random.uniform(0.6, 1.3), 1))
                    img_fruit_2 = cv2.resize(img_fruit_2, None, fx = scale_factor2, fy = scale_factor2)

                    fruit_h2, fruit_w2 = img_fruit_2.shape[0:2]
                    fruit2 = fruit_2.split('.')[0]

                    # generate list of third fruits
                    for fruit_folder_3 in fruit_dirs:
                        curr_folder = []
                        for (dirpath, dirnames, filenames) in walk('cut_images/' + fruit_folder_3):
                            curr_folder.extend(filenames)

                        # randomly select images from the third folder
                        fruit_imgs_3 = np.random.choice(curr_folder, num_combo, replace = False)

                        # iterate through random images from second folder
                        for fruit_3 in fruit_imgs_3:

                            img_fruit_3 = cv2.imread('cut_images/' + fruit_folder_3 + '/' + fruit_3, cv2.IMREAD_UNCHANGED)
                            img_fruit_3 = cv2.resize(img_fruit_3, (0, 0), None, 0.25, 0.25)

                            scale_factor3 = (round(random.uniform(0.6, 1.3), 1))
                            img_fruit_3 = cv2.resize(img_fruit_3, None, fx = scale_factor3, fy = scale_factor3)

                            fruit_h3, fruit_w3 = img_fruit_3.shape[0:2]
                            fruit3 = fruit_3.split('.')[0]

                            # iterate through each background
                            for bg in bg_samples:

                                path = 'generated_images/'

                                # read in background
                                img_bg = cv2.imread('backgrounds/' + bg)

                                # YOLOv3 requires 416x416 images
                                img_bg = cv2.resize(img_bg, (416, 416))
                                bg_h, bg_w = img_bg.shape[0:2]

                                # define a random position for first fruit
                                pos_w = np.random.randint(0, 416 - fruit_w)
                                pos_h = np.random.randint(0, 416 - fruit_h)

                                # generate image with one fruit
                                output = cvzone.overlayPNG(img_bg, img_fruit_1, [pos_w, pos_h])
                                cv2.imwrite('generated_images/' + fruit + bg, output)
                                
                                path += fruit
                                label_1 = label_dict[re.sub(r'[^a-zA-Z]', '', fruit)]

                                x_centre_1 = (pos_w + fruit_w / 2) / bg_w
                                y_centre_1 = (pos_h + fruit_h / 2) / bg_h

                                bbox_w_1 = fruit_w / bg_w
                                bbox_h_1 = fruit_h / bg_h

                                with open(path + bg.split('.')[0] + '.txt', 'w+') as f:
                                    f.write(f"{label_1} {str(x_centre_1)} {str(y_centre_1)} {str(bbox_w_1)} {str(bbox_h_1)}")

                                # define a random position for the second fruit
                                pos_w2 = np.random.randint(0, 416 - fruit_w2)
                                pos_h2 = np.random.randint(0, 416 - fruit_h2)

                                # generate image with two fruits
                                output = cvzone.overlayPNG(output, img_fruit_2, [pos_w2, pos_h2])
                                cv2.imwrite('generated_images/' + fruit + fruit2 + bg, output)

                                path += fruit2
                                label_2 = label_dict[re.sub(r'[^a-zA-Z]', '', fruit2)]
                                x_centre_2 = (pos_w2 + fruit_w2 / 2) / bg_w
                                y_centre_2 = (pos_h2 + fruit_h2 / 2) / bg_h

                                bbox_w_2 = fruit_w2 / bg_w
                                bbox_h_2 = fruit_h2 / bg_h
                                
                                with open(path + bg.split('.')[0] + '.txt', 'w+') as f:
                                    f.write(f"{label_1} {str(x_centre_1)} {str(y_centre_1)} {str(bbox_w_1)} {str(bbox_h_1)}\n")
                                    f.write(f"{label_2} {str(x_centre_2)} {str(y_centre_2)} {str(bbox_w_2)} {str(bbox_h_2)}")

                                # define a random position for the third fruit
                                pos_w3 = np.random.randint(0, 416 - fruit_w3)
                                pos_h3 = np.random.randint(0, 416 - fruit_h3)

                                # generate image with three fruits
                                output = cvzone.overlayPNG(output, img_fruit_3, [pos_w3, pos_h3])
                                cv2.imwrite('generated_images/' + fruit + fruit2 + fruit3 + bg, output)

                                path += fruit3
                                label_3 = label_dict[re.sub(r'[^a-zA-Z]', '', fruit3)]
                                x_centre_3 = (pos_w3 + fruit_w3 / 2) / bg_w
                                y_centre_3 = (pos_h3 + fruit_h3 / 2) / bg_h

                                bbox_w_3 = fruit_w3 / bg_w
                                bbox_h_3 = fruit_h3 / bg_h

                                with open(path + bg.split('.')[0] + '.txt', 'w+') as f:
                                    f.write(f"{label_1} {str(x_centre_1)} {str(y_centre_1)} {str(bbox_w_1)} {str(bbox_h_1)}\n")
                                    f.write(f"{label_2} {str(x_centre_2)} {str(y_centre_2)} {str(bbox_w_2)} {str(bbox_h_2)}\n")
                                    f.write(f"{label_3} {str(x_centre_3)} {str(y_centre_3)} {str(bbox_w_3)} {str(bbox_h_3)}")

if __name__ == '__main__':

    num_images = 7
    num_combos = 3
    num_bg = 5
    generate_images(num_images, num_bg, num_combos)
