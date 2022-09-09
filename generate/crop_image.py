import sys
import os
import numpy as np
from PIL import Image

def crop_image(path):
    output_path = 'cut_images/'
    # for f in os.listdir(path):
    #     # image = cv2.imread(path + '/' + f)
    #     image = Image.open(path + '/' + f)
    #     positions = np.nonzero(image)

    #     top = positions[0].min()
    #     bottom = positions[0].max()
    #     left = positions[1].min()
    #     right = positions[1].max()

    #     image = image.crop((left, top, right, bottom))
    #     print(path.split)
    #     image.save(output_path + path.split('/')[1] + '/' + f)
    # return

    for filename in os.listdir(path):
        try:
            image = Image.open(path + '/' + filename)
            max_x = 0
            min_x = image.width
            max_y = 0
            min_y = image.height
            i = 0
            for px in image.getdata():
                if(px[3] > 100):
                    y = i / image.width
                    x = i % image.width
                    if(x < min_x):
                        min_x = x
                    if(x > max_x):
                        max_x = x
                    if(y < min_y):
                        min_y = y
                    if(y > max_y):
                        max_y = y
                i = i + 1
            image = image.crop((min_x,min_y,max_x,max_y))
            image.save(output_path + path.split('/')[1] + '/' + filename)
            # print(fruit + " " + str(j))
        except:
            continue

if __name__ == '__main__':

    if len(sys.argv) > 1:
        path = sys.argv[1]
        crop_image(path)
