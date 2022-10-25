import torch
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print(model)
# Images
img = '/mnt/c/Users/prakr/Documents/GitHub/ECE4078_Lab_Group_112/Week08-09/test1.jpg'  # image path
# Inference
results = model(img)
# Results
results.print()
results.save()  # or .show()