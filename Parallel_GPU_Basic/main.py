
#------------------------------------------------------------------------------------------------
# Author: Andre Rosa
# Date: 22 NOV 2018
# 
# Objective: THIS PROGRAM USES GPU in PARALLEL TO CATEGORIZE IMAGES 
# This code was tested with OpenCV 3.4.3, TensorFlow 1.11.0 and Keras 2.2.4
# This code was inspired by https://github.com/yuanyuanli85/Keras-Multiple-Process-Prediction
# The Pretrained Dataset is InceptionV3, downloaded from:
# https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
#------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------
# THE PROGRAM RECEIVES TWO PARAMETERS 
# - folder name of images
# - gpus to be used in processing
# use as: python main.py images 0         # gpu 0 folder name 'images'
# or as: python main.py images 0,1,2,3    # with 4 gpus from 0 to 3
#----------------------------------------------------------------------

from multiprocessing import Process, Queue
import os

from threadCtrl import threadCtrl

#-------------------------------------------------------------------------------------------------------- 
# 2. RUN: Get the list of files and call the function to start processess     
#--------------------------------------------------------------------------------------------------------
def run(imgPath, gpuids):

    #get all files from the image folder
    filelist = list()
    for any in os.listdir(imgPath):
        filelist.append(os.path.join(imgPath, any))
    
    #initialize scheduler
    x = threadCtrl(gpuids) # create threads for each gpu
    
    #start processing and wait to complete 
    x.start(filelist)
#--------------------------------------------------------------------------------------------------------

#def get_available_gpus():
#   from tensorflow.python.client import device_lib
#    print(len(device_lib.list_local_devices()))

#--------------------------------------------------------------------------------------------------------
# 1. MAIN FUNCTION
#--------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    import sys
    imgPath = sys.argv[1] # folder with images
    gpuids = sys.argv[2]  # gpu to use 0 or 0,1,2,3

    GPUs = [int(x) for x in gpuids.split(',')]  # just split the gpu ids list

    run(imgPath, GPUs)
#--------------------------------------------------------------------------------------------------------