#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[camera_center]: ./examples/center_camera.jpg "Center Camera"
[camera_left]: ./examples/left_camera.jpg "Left Camera"
[camera_right]: ./examples/right_camera.jpg "Right Camera"
[mse_plot]: ./examples/mse_plot.png "MSE for training and validation loss"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 12 and 60 (model.py lines 103-107) 

The model includes RELU layers to introduce nonlinearity (model.py lines 103-107), and the data is normalized in the model using a Keras lambda layer (code line 95). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 108). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 118). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, as well as multiple camera-angles (center, left and right) for generating training data. I augmented the left and right steering data with a little offset for better normalization.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off with a simple neural network model as reference, and then trying to tune parameters so as to lower MSE of both training and validation data sets.

My first step was to use a convolution neural network model similar to the Nvidia architecture. I thought this model might be appropriate because it's recommended in this [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I made use of the Dropout approach so that the MSE decreased.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like the place where there's a sharp turn, and also at the bridge. To improve the driving behavior in these cases, I augmented the training data by mirroring the images and angles, which kept the vehicle on the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 94-115) consisted of a convolution neural network with the following layers and layer sizes:

* Input image size - 160 x 320 x 3
* Normalization by 255
* Conv layer: RELU, 12 5x5 filters
* Conv layer: RELU, 24 5x5 filters
* Conv layer: RELU, 36 5x5 filters
* Conv layer: RELU, 48 3x3 filters
* Conv layer: RELU, 60 3x3 filters
* Dropout: 20%
* Flattening: 540 nodes
* Fully connected layer: 540 x 50
* Fully connected layer: 50 x 25
* Fully connected layer: 25 x 10
* Fully connected layer: 10 * 1

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving with different camera angles:

![An example showing a centered camera][camera_center]
![An example showing a left camera][camera_left]
![An example showing a right camera][camera_right]


To augment the data sat, I also flipped images and angles thinking and preprocessed these images by cropping out the sky and the hood of the vehicle [model.py lines 102].
I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. The MSE for the training and validation loss is shown:
![MSE plot][mse_plot]
