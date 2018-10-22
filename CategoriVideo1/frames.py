#------------------------------------------------------------------
# AUTHOR: ANDRE ROSA
# Class myOpenCV to create frames 
#------------------------------------------------------------------

import cv2
from PIL import Image
#------------------------------------------------------------------
# CLASS MY IMAGE MANIPULATOR
#------------------------------------------------------------------
class myImgMani:

    #------------------------------------------------------
    def __init__(self, videoFolder, frameFolder):

        print('using OpenCV', cv2.__version__)
        self.vidFolder = videoFolder     # folder to find the video
        self.fraFolder = frameFolder     # folder where frames will be stored
        self.mFrames = []                # list to save frames in memory 
        self.mSelected = []              # selected frames
    #------------------------------------------------------

    #------------------------------------------------------
    # CONVERT VIDEO TO FRAMES 
    # vidName: name of the video file
    # interval: interval in seconds between frames
    #------------------------------------------------------
    def video_to_frames(self, vidName, interval):
      
        # Start capturing the feed
        cap = cv2.VideoCapture(self.vidFolder + vidName)
        # Find the number of frames
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        video_fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        #video_seconds = int(video_length/video_fps) 

        count = 0 # to count the frames
        saved = 0 # to count the frames saved
        interval = interval * video_fps

        # Start converting the video
        while cap.isOpened():
            # Extract the frame
            _, cv2Img = cap.read()
            # Write the results back to output location.
            if (count % interval) == 0 and count < (video_length-1):

                # CONVERT THE CV2 FRAME TO A PIL IMAGE FOR FUTURE CATEGORIZE
                cv2Img = cv2.cvtColor(cv2Img,cv2.COLOR_BGR2RGB)
                pilImg = Image.fromarray(cv2Img)
                self.mFrames.append(pilImg)
                saved += 1

            count += 1
            # If there are no more frames left
            if (count > (video_length-1)):
                # Release the feed
                cap.release()
                break
    #------------------------------------------------------

    #------------------------------------------------------
    # DELETE FRAMES IN MEMORY
    #------------------------------------------------------
    def delete_frames (self):
        self.mFrames.clear()  # erase the list in memory
        self.mSelected.clear()
    #------------------------------------------------------

    #------------------------------------------------------
    # CHECK THE DIFFERENCE BETWEEN TWO IMAGES
    # returns the percentage of difference between two images
    # modified from: https://rosettacode.org/wiki/Percentage_difference_between_images#Python
    #------------------------------------------------------
    def difference_in_frames (self, i1, i2):
 
        #i1 = img1 #Image.open(img1)
        #i2 = img2 #Image.open(img2)
        #assert i1.mode == i2.mode, "Different kinds of images."
        #assert i1.size == i2.size, "Different sizes."
 
        pairs = zip(i1.getdata(), i2.getdata())
        if len(i1.getbands()) == 1:
            # for gray-scale jpegs
            dif = sum(abs(p1-p2) for p1,p2 in pairs)
        else:
            dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
 
        ncomponents = i1.size[0] * i1.size[1] * 3
        difference = ((dif / 255.0 * 100) / ncomponents)
        #print ('difference:', difference)
        return difference 
    #-------------------------------------------------

    #-------------------------------------------------
    # SELECT FRAMES WITH DIFFERENCES
    # returns the list of frames filtered
    #-------------------------------------------------
    def select_frames(self):

        i = 1
        j = 0
        self.mSelected.append(self.mFrames[0])

        while i < len(self.mFrames):
            #compare current  
            #diff = self.difference_in_frames(self.mSelected[j], self.mFrames[i])
            diff = 20
            if (diff > 15.0):  #if the difference is bigger than 15 percent
                #add next folder to filtered
                self.mSelected.append(self.mFrames[i])
                j += 1
            i += 1

        #self.mFrames.clear()
        return self.mSelected
    #-------------------------------------------------

    #-------------------------------------------------
    # SELECT ALL FRAMES
    # returns the list of frames selected
    #-------------------------------------------------
    def select_frames_all(self):
        return self.mFrames
     #-------------------------------------------------