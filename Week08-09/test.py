import torch
import numpy

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#print(model)
# Images
img = '/mnt/c/Users/prakr/Documents/GitHub/ECE4078_Lab_Group_112/Week08-09/test1.jpg'  # image path
# Inference
results = model(img)
# Results
detections= results.xyxy[0].numpy()

bounding_boxes = []
for i in range(numpy.shape(detections)[0]):
    label = detections[i][5]
    xmin = detections[i][0]
    ymin = detections[i][1]
    xmax = detections[i][2]
    ymax = detections[i][3]
    width= xmax- xmin
    height= ymax- ymin

    if label == 0:
        name= 'apple'
    elif label == 1:
        name= 'lemon'
    elif label == 2:
        name= 'orange'
    elif label == 3:
        name= 'pear'
    elif label == 1:
        name= 'strawberry'
    else:
        name= 'none'

    bb_label = [name, xmin, ymin, width, height]
    bounding_boxes.append(bb_label)
print(bounding_boxes)
