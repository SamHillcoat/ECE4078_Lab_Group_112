Steps for training:

0. git clone https://github.com/AlexeyAB/darknet.git

1. Generate dataset 

- go to the generate folder in Training101
- run generate.py (this may take a while)
- generated images and labels get stored in folder
- copy generated images to darknet->data->images (create new images folder for this)
- copy generated labels to darknet->data->labels (create new labels folder for this)


2. Prep

- Open Training101->split.py
- change current_dir to the correct path to the generated images dir  (for me it is: '/home/prakrati/darknet/data/images')
- copy split.py to darknet->data and run it
- two files in data directory are created, train.txt and valid.txt
- make sure the files aren't empty and have paths


3. Config

- copy the following files to the darknet->data directory from Training101:
 	obj.names
 	obj.data


4. Training 
- edit the Makefile in darknet, add your gpu info
   	Compile with make command
- copy the yolo-obj.cfg file from Training101 to darknet->cfg directory 
- To train run the following command:

./darknet detector train data/obj.data cfg/yolo-obj.cfg yolo.weights -clear 1

To use GPU for training, add the -gpus flag (for example, -gpus 0,1,2,3)
Weights are stored after every 1000 iterations 
Wait for final weights to be generated 


Good luck!


Refer to https://medium.com/@anirudh.s.chakravarthy/training-yolov3-on-your-custom-dataset-19a1abbdaf09 if confused
