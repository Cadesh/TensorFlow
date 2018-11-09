#--------------------------------------------------------------------------------------------------
# Andre Rosa in 08 OCT 2018
# Modified from original: https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb
# The objective of this code is to download and test the Inception v3 dataset. 
#--------------------------------------------------------------------------------------------------

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import *
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"   #here you can select the GPU you will use

from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())


model = InceptionV3(weights='imagenet')


img = image.load_img('test5.jpg', target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

y = model.predict(x)
for index, res in enumerate(decode_predictions(y)[0]):
    print('{}. {}: {:.3f}%'.format(index + 1, res[1], 100 * res[2]))
