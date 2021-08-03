# Face Mask Detection Application

## Problem

The problem we will be investigating is individuals neglecting to wear face masks during the COVID-19 pandemic and how they can be identified by an automated system. This problem is interesting because it is a very critical issue that has been affecting the entire world for almost a year. Despite urgings from the government and health officials, many people still either refuse to wear masks in public, or wear masks incorrectly. Our project will seek to remedy this by providing a computer vision solution to identify those who are not wearing masks properly based on video analysis.

## Tech/Framework Used
* OpenCV
* Keras
* Tensorflow
* Python


## Collecting Dataset

To collect the dataset to train the mask detection model 3 different approaches were taken. 

### Public Data set
First, we have gathered images of people wearing masks _other public data sources_ from various projects. 
There are various sources available online which have already gathered the required datasets to train the mask detection model.
The public dataset from this [__repo__][1] contains images of people with masks and no masks. The quality of the image and the types of masks and people vary greatly producing a wide range of data for training.

### Facial Landmarks via Haar Cascade Crossifier
Second, despite the availability of public datasets, we wanted to create our own to promote a more diverse set of images through customizable data generation. 
Therefore, in the first data generation approach we utilized [__OpenCV’s Cascade Classifiers__][2] to detect facial landmarks and 
perform bitwise operations to overlay the mask onto the lower half of the face.
This approach requires one to extract the ROI and overlay the mask. 
The visual process and github link of generating the masked image is referenced below.

<p align="center"><img src="https://github.com/rsmpark/mask-detection/blob/master/readme_res/haar_cascade_diagram.png" width="700" height="400"></p>

### Facial Landmarks via dlib
Third approach we have decided to use was to leverage OpenCV and dlib’s library to superimpose face masks on the images. 
Unlike the previous approach, we are using dlib’s pre-trained face detector that is based on a modification to the 
standard [__Histogram of Oriented Gradients and Linear SVM method__][3]. The dlib’s detector will return the bounding box of faces in our images. 
Then we will extract the coordinates for each component of the face using dlib’s facial landmark predictor. By using the OpenCV’s fillPoly functions we can draw a mask shaped polygon to cover the appropriate portions of the face.

<p align="center"><img src="https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg" width="400" height="300"></p>


## Face Mask Detection Models
4 models were created using each dataset discussed in the previous section. The models were created using TensorFlow with the MobileNetV2 pre-trained model. When training the models we used 20 epochs, with a batch size of 32 and an initial learning rate of 1e-4. Figure 10 shows the training loss and accuracy when creating the blue dlib model.

<p align="center"><img src="https://github.com/rsmpark/mask-detection/blob/master/readme_res/model_result.png" width="500" height="450"></p>

For face mask detection, another method of face detection was utilized. This time, a deep learning model was used to detect faces. A prototext file was used for the architecture and serialization of the model while a caffe model file was used to for the weights of the layer. Both these files were loaded into the program to create the model. CV2’s deep neural network module, “dnn”, was used to load the face detection model along with it’s method readnet. The mask detection model was loaded into the program using tensorflow’s load_model function.

Multiple mask detection models were created using different datasets, and the user is given the choice, when they run the program, which model they want to use on the video. 

## Result

To evaluate the effectiveness of each mask detector model, we recorded a video of us doing various tests then processing that video with each model.

* Top left corner: default dataset model
* Top right corner: cascade dataset model
* Bottom left corner: blue dlib dataset model
* Bottom right corner: black dlib dataset model

### Notebook test

We attempted using a notebook to mimic a mask by placing it over the bottom half of the face. The default and cascade models both recognized the notebook as a false positive, while the dlib models were very effective at recognizing it as a true negative with over *95% accuracy*.

<p align="center"><img src="https://github.com/rsmpark/mask-detection/blob/master/readme_res/result_notebook.png" width="450" height="450"></p>

## Head Tilt Test
For this test the head was tilted down vertically. The default model was very effective at recognizing my blue mask with a 98.3% accuracy. The blue dlib model was the 2nd best with 89.7% accuracy, followed by the cascade model with *78.5% accuracy*. The only model to fail this test was the black dlib model which registered a false negative with *59.5% accuracy*.

<p align="center"><img src="https://github.com/rsmpark/mask-detection/blob/master/readme_res/result_face_down.png" width="450" height="450"></p>

## Head Movement Test
The head quickly moved from side to side to test the video capture speed. The model that performed the worst in this test was the cascade model which had a success rate of only *56.7%*. The default model did a decent job at recognizing that I was wearing a mask with *95.2% accuracy*. However, the dlib models showed great improvement over both the other models by demonstrating consistently over 90% accuracy while moving the head.

<p align="center"><img src="https://github.com/rsmpark/mask-detection/blob/master/readme_res/result_head_movement.png" width="450" height="450"></p>


## Measuring time spent with mask on vs mask off 
For this test, different models were used (Cascade, Default, Dlib with black mask and Dlib with Blue mask) on the same video to check the differences in mask detection. Pie charts were made using matplotlib with measurements of time with the mask on and off in seconds taken from the test video for each model. 

<p align="center"><img src="https://github.com/rsmpark/mask-detection/blob/master/readme_res/result_dataset_chart.png" width="650" height="500"></p>




[1]: https://github.com/balajisrinivas/Face-Mask-Detection/tree/master/dataset/with_mask
[2]: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
[3]: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
