#!/bin/bash
# author: Andre Rosa
# 21 SEP 2020
# objective: Install and creates a virtual environment for TensorFlow test
#---------------------------------------------------------------------------

# GETS ARGUMENTS DOMAIN MUST BE IP, SECOND ARGUMENT IS CREATOR EMAIL
REPO='https://AndreVR@bitbucket.org/vubble/vubble-categorizer.git'  # git bitbucket repository origin of REAL code
DOMAIN='andretestbot.vubblepop.com'
EMAIL='andre@vubblepop.com'

IPNUM=$(hostname -I | awk '{print $1}') # captures the machine ip

echo -e "\e[96mINITIATE SCRIPT FOR MACHINE: $IPNUM\e[39m"
echo -e "\e[96mGIT REPOSITORY LOCATION: $REPO\e[39m"

#---------------------------------------------------------------------------
# CAPTURES THE NAME OF THE REPOSITORY FROM THE GIT URL (to use at the bottom)
basename=$(basename $REPO)
echo $basename
GITFOLDER=${basename%.*}
echo $GITFOLDER
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
# INSTALL ESSENTIALS - C++ MAKE
echo -e "\e[96mINSTALL UBUNTU ESSENTIALS (MAKE)\e[39m"
apt-get install build-essential -y
#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
# INSTALL PYTHON3
echo -e "\e[96mINSTALL PYTHON3\e[39m"
apt-get update
apt-get -y upgrade
apt-get install -y python3-pip

apt-get install -y build-essential libssl-dev libffi-dev
apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev
#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
echo -e "\e[96mINSTALL PYTHON VIRTUAL ENV\e[39m"
#CREATE PYTHON ENVIRONMENT
sudo -H pip3 install --upgrade pip
sudo -H pip3 install virtualenv
cd pyCategorizer
echo -e "\e[96mCREATE PYTHON ENVIRONMENT\e[39m"
virtualenv catenv
source catenv/bin/activate
#INSTALL PYTHON MODULES
echo -e "\e[96mINSTALL PYTHON MODULES\e[39m"
pip install Pillow==6.0.0
pip install tensorflow==1.10
pip install Keras==2.1.6

#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
echo -e "\e[96mNOW ACTIVATE THE VIRTUAL ENV WITH COMMAND source ./catenv/bin/activate \e[39m"
#---------------------------------------------------------------------------

# #---------------------------------------------------------------------------
# #---------------------------------------------------------------------------
