#--------------------------------------------------------------------------------------------------
# Andre Rosa in 08 OCT 2018
# Modified from original: https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb
# The objective of this code is to test RETINANET installation 
# For a better visual result run the code in Jupyter Qt Console. 
#--------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------
# ATTENTION 1: If you getting the message: 
# "Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA" 
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
#--------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------
# 1. LOAD MODULES
#---------------------------------------------------------------------
#show images inline
#%matplotlib inline

# automatically reload modules when they have changed
#%load_ext autoreload
#%autoreload 2

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

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# 2. FUNCTION TO GET TENSORFLOW SESSION
#----------------------------------------------------------------------------
def get_session():
    config = tf.ConfigProto()
    # ATTENTION: UNCOMMENT THE LINE BELLOW FOR GPU !!!!
    #config.gpu_options.allow_growth = true 
    return tf.Session(config=config)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# 3. ATTENTION: uncomment this line to select CUDA GPU to use
#----------------------------------------------------------------------------
# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#----------------------------------------------------------------------------
# Set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# 4. SETUP COCO DATASET FOR USE
#----------------------------------------------------------------------------
# Point to downloaded/trained model (in this case COCO Pretrained Dataset)
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
#model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
model_path = os.path.join('resnet50_coco_best_v2.1.0.h5') # the file is in the same folde rof the code

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.load_model(model_path, backbone_name='resnet50', convert=True)

#print(model.summary())

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

#----------------------------------------------------------------------------------------
# 9. CATEGORIZE A VECTOR OF IMAGES
#----------------------------------------------------------------------------------------
images = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg']  
i = 0
while i < len(images):
  # load image
  image = read_image_bgr(images[i])
  # create a copy of the image to draw the identification boxes
  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB) # Note here the use of OpenCV library
  # Preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image)
  # Process image in the neural network
  start = time.time()
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  print("processing time: ", time.time() - start)
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
  plt.figure(figsize=(12, 12))
  plt.axis('off')
  plt.imshow(draw)
  plt.show()
  i += 1 # Add the loop index 
# end of while loop -------------------------