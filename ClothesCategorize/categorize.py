
# Andre Rosa in 05 OCT 2018
# Modified from: https://www.tensorflow.org/tutorials/keras/basic_classification 
# Github: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb
# The objective of this code is to run a clothes classifier
# We have 10 categories as follows:
# 0 T-shirt, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 
# 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.

# I had to install the libray matplotlib with Pip to be able to run this code. 
# matplotlib is a plotting library used to plot graphics in GUI systems as Qt and wxPython

#------------------------------------------------------------------------------------
# Import TensorFlow and tf.keras libraries
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt 
# ATTENTION: if matplotlib is not installed use in terminal: pip install matlotlib
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# 1. IMPORT THE TRAINING DATA
#----------------------------------------------------------------------------------
# Download the dataset to be trained
# The 'train_images' and 'train_labels' arrays are the training set the data the model uses to learn.
# The model is tested against the 'test_images', and 'test_labels' arrays.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Define categories names in an array
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# 2. PRINT SOME FUN STUFF (YOU CAN COMMENT THIS)
#----------------------------------------------------------------------------------
print ("--------------------------------------------------------------")
print ("                   Clothes Categorize                         ")
print ("--------------------------------------------------------------")

# To know which TensorFLow version we are using
print("TensorFlow version: " + tf.__version__) 
print ("Trainset - quantity and size of images in train set: ", len(train_images))
print ("Testset - quantity of images in the test set: ", len(test_labels))
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# 3. PREPARE IMAGES FOR TRAINING
#---------------------------------------------------------------------------------------
# We scale these values to a range of 0 to 1 before feeding to the neural network model.
# For this, cast the datatype of the image components from an integer to a float, and divide by 255. 
# Here's the function to preprocess the images:
train_images = train_images / 255.0
test_images = test_images / 255.0

#---------------------------------------------------------------------------------------------
# 4. BUILD THE MODEL
#---------------------------------------------------------------------------------------------
# 4.1 SETUP THE LAYERS
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),      # FIRST LAYER: Turn the 28x28 image to an array of 784 pixels
    keras.layers.Dense(128, activation=tf.nn.relu),  # SECOND LAYER: 128 nodes
    keras.layers.Dense(10, activation=tf.nn.softmax) # THIRD LAYER: 10 nodes
])

# 4.2 SETUP NEURAL NETWORK PARAMETERS FOR TRAINING
# Optimizer — This is how the model is updated based on the data it sees and its loss function.
# Loss function — This measures how accurate the model is during training. We want to minimize this function to "steer" the model in the right direction.
# Metrics — Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4.3 TRAIN THE MODEL
model.fit(train_images, train_labels, epochs=5)
# the model will reach an accuracy of 88% of the training data

# 4.4 USE THE TEST DATASET TO TEST OUR ACCURACY
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:" , test_acc) # The test data accuracy will be a little lower. 
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# 5. FUNTIONS TO BE CALLED LATER TO PRINT RESULTS ON SCREEN
#-----------------------------------------------------------------------------------------
# 5.1 Print image of result --------------------------------------------------------------
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
#-----------------------------------------------------------------------------------------
# 5.2 Print array of results -------------------------------------------------------------
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
# 6. FUNCTION TO PRINT ON SCREEN CATEGORIZED OBJECT
#----------------------------------------------------------------------------------------
def plot_obj (obj):
  i = obj
  plt.figure(figsize=(9,3))
  plt.subplot(1,2,1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(1,2,2)
  plot_value_array(i, predictions,  test_labels)
  _ = plt.xticks(range(10), class_names, rotation=45)
  plt.show()  # show the image
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
# 7. PRINTS THE FIRST 25 OBJECTS OF THE TRAIN ARRAY AS EXAMPLE OF OBJECTS
#----------------------------------------------------------------------------------------
# Display the first 25 images from the training set and display the class name below each image. 
# Verify that the data is in the correct format and we're ready to build and train the network.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# 8. MAKE PREDICTIONS
#-----------------------------------------------------------------------------------------
predictions = model.predict(test_images)
#-----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------
# 9. CATEGORIZE A VECTOR OF IMAGES
#----------------------------------------------------------------------------------------
objects = [26,35,128,557,669,2001,5002,8971] # Randomly chosen indexes 
i = 0
while i < len(objects):
    plot_obj(objects[i]) # Calls the categorizer function
    i += 1
#----------------------------------------------------------------------------------------