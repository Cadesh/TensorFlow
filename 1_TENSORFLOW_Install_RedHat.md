TENSORFLOW IN RED HAT VM

||
||
||
||
||
||

Introduction

The objective of this document is to explain the steps taken to install and run TensorFlow in the Red Hat VM. TensorFlow is a Python open-source machine learning library for research and production.

Development with TensorFlow demands Python and VS Code. There are different ways to install TensorFlow, we will use the Pip installation as explained below.

1 – Install Anaconda-Navigator (Python)

Install as explained in page: <http://docs.anaconda.com/anaconda/install/linux/>

At the end of installation you will be asked to install VSCode, if you already have VSCode installed there is not problem to answer ‘yes’ as the installer will check any prior installation.

ATTENTION: After installation close and open the terminal window again to run Anaconda. To run Anaconda just type anaconda-navigator in the terminal window.

2 – Check dependencies

Tensorflow demands the use of Python, Pip and VirtualEnv. For more detail check: <https://www.tensorflow.org/install/pip?lang=python2>

3 – Install TensorFlow with Conda

There are several ways to install TensorFlow, CPU version, the best way for those using Anaconda is with conda command line after Anaconda installation:

******conda install -c conda-forge tensorflow****

********

To install TensorFlow with Conda, GPU version, the command is (before choosing a version, look the comments bellow):

******conda install -c conda-forge tensorflow******-gpu******

********

**ATTENTION: **y**ou should not install TensorFlow as explained in the TensorFlow home page (with Pip or Docker). **Use **Conda **because it** will install all dependencies as required by TensorFlow.**

ATTENTION: For the Red Hat VM Conda will *install the TensorFlow CPU version*. The Red Hat VM does not have GPU. The tests for CUDA were made as described in:

<https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions>

4 – Create and Activate Environment

Now we will create an Anaconda environment that will allow us to develop Python in VS Code. Tp proceed we will create an environment named ‘tensorflow\_env’ with the command:

******conda create -n tensorflow\_env tensorflow****

Answer “yes” to the qeustions and when done activate the environment with:

******conda activate tensorflow\_env****

********

5 – Anaconda Navigator

Open a new terminal window and launch Anaconda with the command:

**anaconda-navigator**

In Anaconda-Navigator select the environment ‘tensorflow\_env’ in “Application on” option, as shown in the picture bellow.

6 – Launch VS Code

From Anaconda launch VS Code. Three extensions will be already installed in VS Code to facilitate your Python development: Python, YAML and Anaconda.

If your VS Code asks for Lint installation proceed. Lint analyzes Python code for errors.

To be sure that everything is right check the bottom of the VS Code environment, you shall see a status bar like this:

7 – Test if TensorFlow is running

Open terminal window in VS Code in menu *View &gt; Terminal*

Create a file and paste the following code:

Save and select with a ‘Right-Click’ the option ‘Run Python File in Terminal’.

Your code will output a series of tests of NN layers in the terminal area like this:

Congratulations your TensorFlow installation with Anaconda and Python is successful.

8 – Using Jupyter Qt Console

Albeit you can run your code and get output with VS Code one interesting feature of Python is the capacity to generate visual output with libraries as mapplotlib. Qt Console is also part of the Anaconda Navigator and works as a terminal capable of generating colorful output unlike VS Code.

The code in the example below is found in GitHub and run in Qt Console with the command: ***run categorize.py***

<https://github.com/Cadesh/TensorFlow/tree/master/ClothesCategorize>

Sources:

Anaconda Installation:

<http://docs.anaconda.com/anaconda/install/linux/>

TensorFlow Installation with Conda:

<https://anaconda.org/conda-forge/tensorflow>

<https://www.anaconda.com/blog/developer-blog/tensorflow-in-anaconda/>

TesorFlow Installation page:

<https://www.tensorflow.org/install/pip>

nVidia Testing CUDA;

<https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions>

Python with VS Code

<https://code.visualstudio.com/docs/languages/python>

Versions Used:

****- ******Python ****2.7.5 **from terminal with command: python -V**

****- ******Python**** 3.6.6 **from tensorflow\_env with terminal: python -V**** ****

**- Anaconda **5.6

**- Anaconda-Navigator **1.9.2

**- VS Code** 1.27.2

**- TensorFlow **1.11.0 check with terminal: **python -c 'import tensorflow as tf; print(tf.\_\_version\_\_)'**

****- Qt Console**** 4.3.1 **

****

****

****
