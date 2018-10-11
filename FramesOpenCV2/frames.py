# author: Andre 
# original from: https://www.linkedin.com/pulse/fun-opencv-video-frames-arun-das
# added option to save frames every X seconds

def video_to_frames(input_loc, output_loc, interval):
    import time
    import cv2
    import os
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) - 1
    video_fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS)) 
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
        
        if (count % interval) == 0 and count < (video_length-1):
            cv2.imwrite(output_loc + "/%#05d.jpg" % (saved+1), frame)
            saved += 1

        count += 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames saved" % saved)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

#@param video file name, destination folder of frames, number of seconds between frames
video_to_frames('video.mp4', 'frames', 2)
