=======================================================
===================HOW TO RUN CODE=====================
=======================================================

1. Setup env
python 3.7
torch 1.7.0+cu101
torchvision 0.2.1

2. Setup directory
There are 4 files with extension .py in this project: datasets.py, main.py, model.py, utils.py
There is 1 file with extension .txt: datadx.txt
Put all 5 files above in the same directory

3. Command to run code

a. Run code without oversampling and data augmentation
python 0860831.py --augmentation 0

b. Run code with oversampling and data augmentation
python 0860831.py --augmentation 1
