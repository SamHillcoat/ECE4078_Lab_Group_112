import torch
import numpy as np
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print(model)
# Images
img = '/mnt/c/Users/prakr/Documents/GitHub/ECE4078_Lab_Group_112/Week08-09/test2.jpg'  # image path
# Inference
results = model(img)
# Results
#results.print()
#results.show()
print(results.xyxy[0].np())