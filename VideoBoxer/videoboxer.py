# Author: Andre Rosa
# Originals: VIDEO TO FRAMES from: https://www.linkedin.com/pulse/fun-opencv-video-frames-arun-das
#            FRAMES TO VIDEO from: https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
#            RETINANET from: https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb

# VIDEO BOXER
# This program uses Retinanet with COCO Dataset to create 
# a video with bounding boxes for identified objects in video
# The program does the following:
# 1. CONVERT VIDEO TO FRAMES
# 2. USES RETINANET TO BOX OBJECTS IN FRAMES
# 3. CONVERT FRAMES TO VIDEO
# line 236 at the bottom to change the video name

#check ATTENTION 1 above. Disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import *

# import miscellaneous modules
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os
from os.path import isfile, join
from os import walk

# avoid recursion problem
import sys
sys.setrecursionlimit(10000)

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

#----------------------------------------------------------------
# VIDEO TO FRAMES
#----------------------------------------------------------------
def video_to_frames(input_loc, output_loc, interval):

    # create the output folder for frames
    try:
        os.mkdir(output_loc)
    except OSError:
        pass

    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    video_seconds = int(video_length/video_fps) 
    print ("Number of frames:   ", video_length)
    print ("Frames per Seconds: ", video_fps)
    print ("Length in Seconds : ", video_seconds)
    count = 0 # to count the frames
    saved = 0 # to count the frames saved

    interval = interval * video_fps

    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        
        if (interval == 0): # if interval = 0 saves all frames
            cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        else:
            if (count % interval) == 0 and count < (video_length-1):
                cv2.imwrite(output_loc + "/%#05d.jpg" % (saved+1), frame)
                saved += 1

        count += 1
        # If there are no more frames left
        if (count > (video_length-5)): # ignoring the last 5 frames
            # Release the feed
            cap.release()
            # Print stats
            break
#-------------------------------------------------------------------
# END OF VIDEO_TO_FRAMES
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# FRAMES_TO_VIDEOS
#-------------------------------------------------------------------
def frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    
    #for sorting the file names properly
    files.sort()

    i = 0
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    
#--------------------------------------------------------------------
# END OF FRAMES_TO_VIDEOS
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# IDENTIFY OBJECTS IN FRAMES
#--------------------------------------------------------------------
def identify_objects(pathIn, pathOut):

    # create the output folder for frames
    try:
        os.mkdir(pathOut)
    except OSError:
        pass

    # 1. get all frames
    frames = []
    for (dirpath, dirnames, filenames) in walk(pathIn):
        frames.extend(filenames)
        break
    frames.sort() #order the files otherwise the output is a mess....
    #loop the frames to create boxes
    i = 0
    while i < len(frames):
        # load image
        image = read_image_bgr(pathIn + frames[i])
        # create a copy of the image to draw the identification boxes
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB) # Note here the use of OpenCV library
        # Preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # Process image in the neural network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        # correct for image scale
        boxes /= scale
        # visualize detections
        #-------------------------------
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
             # scores are sorted so we can break
            if score < 0.5: #shows only objects with more than 50% correct detection probability
                 break   
            color = label_color(label) 
            b = box.astype(int)
            draw_box(draw, b, color=color)  
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        # end of for loop --------------    

        #plt.figure(frameon=False)
        plt.figure(figsize= (6.4,3.6)) #640 x360 pixels

        plt.subplots_adjust(bottom = 0)
        plt.subplots_adjust(top = 1)
        plt.subplots_adjust(right = 1)
        plt.subplots_adjust(left = 0)

        #plt.axis('off')
        plt.imshow(draw)
        name = str("/%#05d.jpg" % (i+1))
        #os.remove(pathIn + frames[i]) #delete the original frame
        plt.savefig (pathOut + name)
        plt.close()
        i += 1 # Add the loop index 
        if (i%20 == 0):
            print('.', end="", flush=True)
    print('')
        
        
# end of while loop -------------------------
#--------------------------------------------------------------------
# END OF IDENTIFY OBJECTS
#--------------------------------------------------------------------

#----------------------------------------------------------------------------
# FUNCTION TO GET TENSORFLOW SESSION
#----------------------------------------------------------------------------
def get_session():
    config = tf.ConfigProto()
    # ATTENTION: UNCOMMENT THE LINE BELLOW FOR GPU !!!!
    #config.gpu_options.allow_growth = true 
    return tf.Session(config=config)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# ATTENTION: uncomment this line to select CUDA GPU to use
#----------------------------------------------------------------------------
# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#----------------------------------------------------------------------------
# Set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# SETUP COCO DATASET FOR USE
#----------------------------------------------------------------------------
# Point to downloaded/trained model (in this case COCO Pretrained Dataset)
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('resnet50_coco_best_v2.1.0.h5') # the file is in the same folde rof the code

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# THE COCO CATEGORIES' LABELS
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
                   6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
                   11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
                   16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
                   22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
                   27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
                   32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
                   36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
                   41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 
                   48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 
                   54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
                   66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
                   72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
                   78: 'hair drier', 79: 'toothbrush'}
#---------------------------------------------------------------------------------

#--------------------------------------------------------------------
# MAIN ()
#--------------------------------------------------------------------
def main():

    videoIn = 'myVideo.mp4'
    frames1= './frames/'
    frames2= './frout/'
    pathOut = 'output.mp4'

    # Log the time
    time_start = time.time()
    
    #@param 1-video file name,2-destination folder of frames, 3-number of seconds between frames
    video_to_frames(videoIn, frames1, 0) # if third param is 0 saves all frames
    
    identify_objects(frames1, frames2)

    cap = cv2.VideoCapture(videoIn)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    frames_to_video(frames2, pathOut, fps)

    shutil.rmtree(frames1) # delete the frame folder
    shutil.rmtree(frames2) # delete the frame folder

    time_end = time.time()
    print ("It took %d seconds forconversion." % (time_end-time_start))
    #----------------------------------------------------------------
    # END OF MAIN
    #----------------------------------------------------------------

    # Call main
main()