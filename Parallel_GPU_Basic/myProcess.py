
import cv2
import numpy as np
import os
from multiprocessing import Queue, Process

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
# import keras
import keras

# import inception_v3, these lines must be under the the import tensorflow as tf line
# ATTENTION: there is a false positive 'E0611: no name python in module tensorflow', just ignore
from tensorflow.python.keras.preprocessing import image  
from tensorflow.python.keras.applications.inception_v3 import *

#------------------------------------------------------------------------------------------
# CLASS MYPROCESS 
#------------------------------------------------------------------------------------------
class myProcess(Process): 
    def __init__(self, gpuid, queue):
        Process.__init__(self, name='ModelProcessor') # inheritance from Process
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
        #set GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)  #defines which gpu is assigned for this thread

        print ('initialize Keras Inception for gpu -', self._gpuid)
        # SETUP INCEPTION V3 DATASET
        # Download the file from:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
        keras.backend.tensorflow_backend.set_session(self.get_session())
        model = InceptionV3(weights='imagenet')
        keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        print('initialize done with gpu -', self._gpuid)

        while True:
            img = self._queue.get()
            if img == None:
                self._queue.put(None)
                break
            labels = self.categorize(model, img)
            #print (decode_predictions(labels))
            for res in decode_predictions(labels)[0]:
                print(res[1] + ': ' +  str(100 * res[2]) + '%') #res[1] category, res[2] probability


        print('inception done gpu -', self._gpuid)

    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # CATEGORIZE - Uses inception to categorize images
    #-----------------------------------------------------------------------------
    def categorize(self, lmodel, imgName):
        #BGR
        img = image.load_img(imgName, target_size=(299, 299)) 
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        y = lmodel.predict(x) #

        return y #np.argmax(y)
     #-----------------------------------------------------------------------------

#------------------------------------------------------------------------------------------