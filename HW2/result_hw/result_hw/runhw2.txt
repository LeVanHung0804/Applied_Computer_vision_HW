#!/bin/bash

#create folder for save image
mkdir result_hw

#cd to darknet folder
cd darknet

#first image
./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/1.jpg -thresh 0.02
cp predictions.jpg ../result_hw/image1_thresh002.jpg

./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/1.jpg -thresh 0.4
cp predictions.jpg ../result_hw/image1_thresh04.jpg

#second image
./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/2.jpg -thresh 0.02
cp predictions.jpg ../result_hw/image2_thresh002.jpg

./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/2.jpg -thresh 0.4
cp predictions.jpg ../result_hw/image2_thresh04.jpg

#thrid image
./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/3.jpg -thresh 0.02
cp predictions.jpg ../result_hw/image3_thresh002.jpg

./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/3.jpg -thresh 0.4
cp predictions.jpg ../result_hw/image3_thresh04.jpg

#forth image
./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/4.jpg -thresh 0.02
cp predictions.jpg ../result_hw/image4_thresh002.jpg

./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/4.jpg -thresh 0.4
cp predictions.jpg ../result_hw/image4_thresh04.jpg

#fifth
./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/5.jpg -thresh 0.02
cp predictions.jpg ../result_hw/image5_thresh002.jpg

./darknet detector test data/obj.data yolo-acv.cfg yolo-acv.weights data/5.jpg -thresh 0.4
cp predictions.jpg ../result_hw/image5_thresh04.jpg












