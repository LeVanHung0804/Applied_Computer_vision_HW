#========================================================================================#
#=====================================how to run code====================================#
#========================================================================================#

#####################
## install library ##
#####################
pip install the below libraries.

cycler                0.10.0
decorator             4.4.2
imageio               2.9.0
kiwisolver            1.2.0
matplotlib            3.3.2
networkx              2.5
numpy                 1.19.2
opencv-contrib-python 4.4.0.44
opencv-python         4.4.0.44
Pillow                7.2.0
pip                   20.1.1
pyparsing             2.4.7
python-dateutil       2.8.1
PyWavelets            1.1.1
scikit-image          0.17.2
scipy                 1.5.2
setuptools            47.1.0
six                   1.15.0
tifffile              2020.10.1
wheel                 0.35.1

######################
## prepare all file ##
######################

put your both all images, libraryV2.py and 0860831.py in the same directory.

######################
##    run file      ##
######################

python 0860831.py argv1 argv2

where: 
argv1 is the name of the set two images that you want to test ( argv1 = {bamboo_fox, mountain, tree, my_test})
argv2 is the DoG layer that you want to plot. (argv2 = {0,1,2,3})

For example: python 0860831.py bamboo_fox 0

