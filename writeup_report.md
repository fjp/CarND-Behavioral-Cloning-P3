# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/cnn-architecture.png "Model Visualization"
[image2]: ./report_images/center_2017_06_29_21_39_36_811.jpg "Grayscaling"
[image3]: ./report_images/center_2017_06_29_21_42_04_448.jpg "Recovery Image"
[image4]: ./report_images/center_2017_06_29_21_42_06_423.jpg "Recovery Image"
[image5]: ./report_images/center_2017_06_29_21_42_07_469.jpg "Recovery Image"
[image6]: ./report_images/center_2017_06_28_19_41_05_577.jpg "Normal Image"
[image7]: ./report_images/center_2017_06_28_19_41_05_577_flipped.jpg "Flipped Image"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md and writeup_report.ipynb summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network (NVIDIA architecture) with 5x5 and 3x3 filter sizes and depths between 24 and 64. The convolution layers are followed by a flatten layer and four fully connected layers with decreasing sizes from 100 to 1. The single output neuron is required to predict the steering angle (model.py lines 58-76).

The model includes RELU layers to introduce nonlinearity (code lines 67-69), and the data is normalized in the model using a Keras lambda layer (code line 63). Furthermore, the images were cropped to include only useful data of the track (code line 65).

[Link to the work of NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers. Instead a large data set was collected covering the "trouble zones":
- bridge
- turns
Additional data was collected where the vehicle drives from the side of the road back to the center.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving the first track in both directions.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVIDIA state of the art network.
To avoid overfitting a dropout layer could be included.

My first step was to use a convolution neural network model similar to the LeNet architecute. I thought this model might be appropriate because it used to work well for traffic sign classification.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to the NVIDIA architecture so that both mean squared errors of training and validation set kept decreasing.

Then I reduced the number of epochs from 5 to 3 in order to avoid overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially in curves. To improve the driving behavior in these cases, I collected more data in curves and steering the car back from the edges to the center of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-76) consisted of a convolution neural network with the following layers and layer sizes
- Normalization and mean shift layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
- Cropping layer
model.add(Cropping2D(cropping=((70,25), (0,0))))
- Convolutional layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
- Flatten layer
model.add(Flatten())
- Fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay in the middle of the track. These images show what a recovery looks like starting from the right side of the track (inside a curve) and moving back to the center of the lane:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles because it is a fast way to augment the data.  For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 22024 number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by decreasing mean squared errors. I used an adam optimizer so that manually training the learning rate wasn't necessary.
