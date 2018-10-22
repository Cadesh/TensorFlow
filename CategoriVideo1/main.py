#-------------------------------------------------------------
# AUTHOR: ANDRE ROSA
# Categorize Videos - test 1
# This code was tested with OpenCV 3.4.3, TensorFlow 1.11.0 and Keras 2.2.4
# The Pretrained Dataset is InceptionV3 downloaded from:
# https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
#-------------------------------------------------------------

# DESCRIPTION OF THE CODE
# The code frame a video and selects some frames to categorize.
# Later the code categorize each frame and creates a list with: 
# categories and sum of percentages and number of times each category appeared in video

import time

from frames import myImgMani
from inception import myInception

#-----------------------------------------------------------------------
def categorizeVideo(video, videoFolder, framesFolder, interFrame):

    imgMan = myImgMani(videoFolder, framesFolder) #declare object that uses OpenCV
    incep = myInception(framesFolder) #declare object that uses TensorFLow

    # 1. CREATE FRAMES FROM VIDEO
    imgMan.video_to_frames(video, interFrame)

    # 2. SELECT THE MOST MEANINGFUL FRAMES
    frames = []
    frames = imgMan.select_frames()

    # 3. CATEGORIZE EACH SELECTED FRAME
    incep.categorize(frames, framesFolder)
    incep.print_report()
#------------------------------------------------------------------------

#------------------------------------------------------------------------
def main ():
    # Log the time
    time_start = time.time()

    # setup variables
    interFrame = 5 # interval between frames to be saved, in seconds
    video = 'test1.mp4'
    videoFolder = './video/'
    framesFolder= './frames/'

    # CATEGORIZE VIDEO
    categorizeVideo(video, videoFolder, framesFolder, interFrame)

    # finish
    time_end = time.time()
    print ("Done in %d seconds." % (time_end-time_start))
# ----------------------------------------------------------------------

main()
