#---------------------------------------------------------------------------------------------------
# AUTHOR: ANDRE ROSA
# Definition of myInception class
# --------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# ATTENTION: INCEPTION IS A CLASSIFICATION NETWORK 
# IF YOU ARE LOOKING FOR DETECTION BOXS TRY THE OTHER EXAMPLE IN: 
# https://github.com/Cadesh/TensorFlow/tree/master/RetinaNetTest
#---------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# ATTENTION 1: If you getting the message:
# "Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA"
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# --------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# 1. LOAD MODULES
# ---------------------------------------------------------------------
# check ATTENTION 1 above. Disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import keras
import keras

# import miscellaneous modules
import matplotlib.pyplot as plt
import numpy as np

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# import inception_v3, these lines must be under the the import tensorflow as tf line
# ATTENTION: there is a false positive 'E0611: no name python in module tensorflow', just ignore
from tensorflow.python.keras.preprocessing import image  
from tensorflow.python.keras.applications.inception_v3 import *

# import the list of categories from the file categories.py
from categories import labels_to_names
# ----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# CLASS REPORTUNIT UDED FROM INSIDE CLASS MYINCEPTION
#-----------------------------------------------------------------------------
class reportUnit:
    def __init__(self, cat, perc):
        self.quantity = 0        #number of times a category appeared
        self.category = cat      #name of category
        self.percentage = perc   #percentage result of categorization
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# CLASS MYINCEPTION
#-----------------------------------------------------------------------------
class myInception:
      
    #-------------------------------------------------------------------------
    # CLASS CONSTRUCTOR
    #-------------------------------------------------------------------------
    def __init__(self,frameFolder):

        print('using TensorFlow', tf.VERSION) 
        print('using Keras', keras.__version__) #shows the version of the TensorFlow

        self.folder = frameFolder # path of the frame folder
        self.repUnits = [] # list of categories for the video

        #ATTENTION: uncomment this line to select CUDA GPU to use
        # use this environment flag to change which GPU to use
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        # Set the modified tf session as backend in keras
        keras.backend.tensorflow_backend.set_session(self.get_session())

        # SETUP KERAS DATASET INCEPTION
        # Download the file from:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
        self.model_path = os.path.join('/dataset/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
        # load retinanet model
        self.model = InceptionV3(weights='imagenet')
        # Check https://keras.io/applications/#inceptionresnetv2
        keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    #-----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    # GET TENSORFLOW SESSION
    # ----------------------------------------------------------------------------
    def get_session(self):
        config = tf.ConfigProto()
        # ATTENTION: UNCOMMENT THE LINE BELLOW FOR GPU !!!!
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)
    # ----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    #
    #-----------------------------------------------------------------------------
    def insert_report(self, report):
        found = False
        i = 0
        while i < len(self.repUnits):
            if (self.repUnits[i].category == report.category):
                self.repUnits[i].percentage += report.percentage
                self.repUnits[i].quantity += 1
                i = len(self.repUnits)
                found = True
            i+=1

        if (found == False):
            self.repUnits.append(report)
            self.repUnits[i].quantity = 1

    #-----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------
    # SAVE THE REPORT IN A CSV FILE
    # ----------------------------------------------------------------------------------------
    def print_report(self):

        file = open('report.csv', 'w') # open file for output
        file.write('category, percentage, quantity \n')
        #file.write('----------------\n')

        i = 0
        while i < len(self.repUnits):
            tStr = '{}, {:.5f}, {}'.format(self.repUnits[i].category, self.repUnits[i].percentage, self.repUnits[i].quantity)
            file.write(tStr + '\n')
            i += 1
        file.close()
    #-----------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------
    # CATEGORIZE A VECTOR OF IMAGES
    # ----------------------------------------------------------------------------------------
    def categorize (self, images, framesFolder):

        i = 0
        numFrames = len(images) #used for tlater calculations

        while i < numFrames:
            # load image
            img = images[i]
            #img = image.load_img(framesFolder + images[i], target_size=(299, 299)) #keep the size 299x299????

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            y = self.model.predict(x) #  predict(x, batch_size=None, verbose=0, steps=None)

            for _, res in enumerate(decode_predictions(y)[0]):                
                report = reportUnit(res[1], 100*res[2])
                #print ('{}: {}%'.format(report.category, report.percentage))
                self.insert_report(report)

            i += 1
        # end of while loop -------------------------

        # order the list by percentage, note the lambda function
        self.repUnits.sort(key=lambda x: x.percentage, reverse = True)
    #--------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------