# Face Mask Detection Application

## Problem

The problem we will be investigating is individuals neglecting to wear face masks during the COVID-19 pandemic and how they can be identified by an automated system. This problem is interesting because it is a very critical issue that has been affecting the entire world for almost a year. Despite urgings from the government and health officials, many people still either refuse to wear masks in public, or wear masks incorrectly. Our project will seek to remedy this by providing a computer vision solution to identify those who are not wearing masks properly based on video analysis.


## Collecting Dataset

To collect the dataset to train the mask detection model 3 different approaches were taken. 

1. First, we have gathered images of people wearing masks _other public data sources_ from various projects. 
There are various sources available online which have already gathered the required datasets to train the mask detection model.
The public dataset from this [__repo__][1] contains images of people with masks and no masks. The quality of the image and the types of masks and people vary greatly producing a wide range of data for training.

2. Second, despite the availability of public datasets, we wanted to create our own to promote a more diverse set of images through customizable data generation. 
Therefore, in the first data generation approach we utilized [__OpenCV’s Cascade Classifiers__][2] to detect facial landmarks and 
perform bitwise operations to overlay the mask onto the lower half of the face.
This approach requires one to extract the ROI and overlay the mask. 
The visual process and github link of generating the masked image is referenced below (see Figure 2).

[1]: https://github.com/balajisrinivas/Face-Mask-Detection/tree/master/dataset/with_mask
[2]: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
