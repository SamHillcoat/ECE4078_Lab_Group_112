import cv2
import cvzone
import numpy as np

from os import walk

def generate_images(num_images, num_bg, num_combo):
    '''
    params:     num_images, number of images to generate from per fruit
                num_bg, number of backgrounds to generate images with
                num_combo, number of images to generate with two fruits

    output:     path: generated_images/
                containing num_images * num_bg * num_combo images
    '''

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
            img_fruit_1 = cv2.resize(img_fruit_1, (0, 0), None, 0.5, 0.5)

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
                    img_fruit_2 = cv2.resize(img_fruit_2, (0, 0), None, 0.5, 0.5)

                    fruit_h2, fruit_w2 = img_fruit_2.shape[0:2]
                    fruit2 = fruit_2.split('.')[0]

                    # iterate through each background
                    for bg in bg_samples:

                        # read in background
                        img_bg = cv2.imread('backgrounds/' + bg)

                        # YOLOv3 requires 416x416 images
                        img_bg = cv2.resize(img_bg, (416, 416))

                        # define a random position for first fruit
                        pos_h = np.random.randint(0, 416 - fruit_w)
                        pos_w = np.random.randint(0, 416 - fruit_h)

                        # generate image with one fruit
                        output = cvzone.overlayPNG(img_bg, img_fruit_1, [pos_h, pos_w])
                        cv2.imwrite('generated_images/' + fruit + bg, output)

                        # define a random position for the second fruit
                        pos_h2 = np.random.randint(0, 416 - fruit_w2)
                        pos_w2 = np.random.randint(0, 416 - fruit_h2)

                        # generate image with two fruits
                        output = cvzone.overlayPNG(output, img_fruit_2, [pos_h2, pos_w2])
                        cv2.imwrite('generated_images/' + fruit + fruit2 + bg, output)



if __name__ == '__main__':

    num_images = 10
    num_combos = 3
    num_bg = 5
    generate_images(num_images, num_bg, num_combos)