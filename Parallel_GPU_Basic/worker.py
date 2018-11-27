from multiprocessing import Queue, Process
import cv2
import numpy as np
import os

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
# import keras
import keras

# import inception_v3, these lines must be under the the import tensorflow as tf line
# ATTENTION: there is a false positive 'E0611: no name python in module tensorflow', just ignore
from tensorflow.python.keras.preprocessing import image  
from tensorflow.python.keras.applications.inception_v3 import *

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
class Worker(Process):
    def __init__(self, gpuid, queue):
        Process.__init__(self, name='ModelProcessor') # this is necessary to allow this class work as a thread
        self._gpuid = gpuid #assigined gpu id for this thread
        self._queue = queue

    # ----------------------------------------------------------------------------
    # GET TENSORFLOW SESSION
    # ----------------------------------------------------------------------------
    def get_session(self):
        config = tf.ConfigProto()
        # ATTENTION: UNCOMMENT THE LINE BELLOW FOR GPU !!!!
        #config.gpu_options.allow_growth = true
        return tf.Session(config=config)
    # ----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # RUN - initialize the keras for a specific gpu, then categorize each picture
    #-----------------------------------------------------------------------------
    def run(self):

        print ('run', self._gpuid)
        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)  #defines which gpu is assigned for this thread

        print ('initialize Keras Inception for gpu -', self._gpuid)
        keras.backend.tensorflow_backend.set_session(self.get_session())
        model = InceptionV3(weights='imagenet')
        keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

        print('initialize done with gpu', self._gpuid)

        while True:
            img = self._queue.get()
            if img == None:
                self._queue.put(None)
                break
            label = self.categorize(model, img)
            #print('GPU ', self._gpuid, ' image ', img, " categorized as ", label)

        print('inception done ', self._gpuid)

    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # CATEGORIZE
    #-----------------------------------------------------------------------------
    def categorize(self, lmodel, imgName):
        #BGR
        img = image.load_img(imgName, target_size=(299, 299)) 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y = lmodel.predict(x) #

        return np.argmax(y)
     #-----------------------------------------------------------------------------

#------------------------------------------------------------------------------------------