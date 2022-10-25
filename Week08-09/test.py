import torch
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/mnt/c/Users/prakr/Desktop/ECE4078/Training101/yolo5s.pt')
print(model)
# Images
img = '/mnt/c/Users/prakr/Documents/GitHub/ECE4078_Lab_Group_112/Week08-09/test1.jpg'  # image path
# Inference
results = model(img)
# Results
results.print()
results.save()  # or .show()