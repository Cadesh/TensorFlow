
import tensorflow as tf
#from tensorflow.python.client import device_lib

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #test with value 1 and 2 for more GPU
tf.Session()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def main ():
    tf.Session()
    print ('Prints GPU name: ' + str(tf.test.gpu_device_name())) 


main()