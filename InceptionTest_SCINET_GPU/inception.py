#--------------------------------------------------------------------------------------------------
# Andre Rosa in 08 OCT 2018
# Modified from original: https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb
# The objective of this code is to download and test the Inception v3 dataset. 
# Download the dataset from: https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5
#--------------------------------------------------------------------------------------------------

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import *
import numpy as np

model = InceptionV3(weights='imagenet')

img = image.load_img('test5.jpg', target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

y = model.predict(x)
for index, res in enumerate(decode_predictions(y)[0]):
    print('{}. {}: {:.3f}%'.format(index + 1, res[1], 100 * res[2]))
