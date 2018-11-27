
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
import argparse
from worker import Worker

#----------------------------------------------------------------------
class Scheduler:
    #------------------------------------------------------------------
    def __init__(self, gpuids):
        self._queue = Queue()
        self._gpuids = gpuids

        self.__init_workers()
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(Worker(gpuid, self._queue))
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    def start(self, xfilelst):

        # put all of files into queue
        for xfile in xfilelst:
            self._queue.put(xfile)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print("all of workers have been done")
    #------------------------------------------------------------------
#----------------------------------------------------------------------

#----------------------------------------------------------------------
def run(img_path, gpuids):
    #scan all files under img_path
    xlist = list()
    for xfile in os.listdir(img_path):
        xlist.append(os.path.join(img_path, xfile))
    
    #init scheduler
    x = Scheduler(gpuids)
    
    #start processing and wait for complete 
    x.start(xlist)
#----------------------------------------------------------------------

#----------------------------------------------------------------------
if __name__ == "__main__":

    import time
    # Log the time
    time_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", help="path to your images to be proceed")
    parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )

    args = parser.parse_args()

    gpuids = [int(x) for x in args.gpuids.strip().split(',')]

    print(args.imgpath)
    print(gpuids)

    run(args.imgpath, gpuids)

    time_end = time.time()
    print ("Done in %d "  %(time_end-time_start) + "seconds \n")
#----------------------------------------------------------------------
