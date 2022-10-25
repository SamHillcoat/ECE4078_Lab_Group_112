import torch
import numpy

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Week10-11_robot/yolov5s.pt')  # local model
#print(model)
# Images
img = '/mnt/c/Users/prakr/Documents/GitHub/ECE4078_Lab_Group_112/Week08-09/test.jpg'  # image path
# Inference
results = model(img)
results.save()  # or .show()

# Results
detections= results.xyxy[0].numpy()
print(numpy.shape(detections)[0])
bounding_boxes = []
for i in range(numpy.shape(detections)[0]):
    print(numpy.shape(detections)[0])
    print(i)
    label = detections[i][5]
    xmin = detections[i][0]
    ymin = detections[i][1]
    xmax = detections[i][2]
    ymax = detections[i][3]
    width= xmax- xmin
    height= ymax- ymin
    if detections[i][4]>0.1:
        if label == 0:
            name= 'apple'
            bb_label = [name, xmin, ymin, width, height]
        elif label == 1:
            name= 'lemon'
            bb_label = [name, xmin, ymin, width, height]
        elif label == 2:
            name= 'orange'
            bb_label = [name, xmin, ymin, width, height]
        elif label == 3:
            name= 'pear'
            bb_label = [name, xmin, ymin, width, height]
        elif label == 1:
            name= 'strawberry'
            bb_label = [name, xmin, ymin, width, height]
        else:
            print("none")
        bounding_boxes.append(bb_label)
print(bounding_boxes)
