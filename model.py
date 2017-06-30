import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    # load images
    image = cv2.imread(current_path)

    # store images
    images.append(image)

    # get steering wheel input
    measurement = float(line[3])
    measurements.append(measurement)

# Data augmentation
# to get more data to train the newtork (twice as many) and
# and the used data is more comprehensive
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    # flip the image using opencv
    augmented_images.append(cv2.flip(image,1))
    # also flip the steering wheel angle
    augmented_measurements.append(measurement*-1.0)

# create training data (images) and corresponding labels (measurements)
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# build basic neural network to verify data
# flattened image connected to a singel output node
# which will predict the steering angle, which makes this a regression network
# For a classification we would apply a softmax activation function to the
# output layer. But in this case (regression) we use a singel output node
# to directly predict the steering measurement. So we wont apply an activation function
# import lambda layers to to create a function that operates on each image as
# it passes through the layer. This will be used to normalize the images
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import cv2


model = Sequential()
# normalization of images
# divide by maximium image value 255 (to normalize range to 0 to 1)
# and then shift the mean to zero by subtracting 0.5
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# set up cropping2D layer
model.add(Cropping2D(cropping=((70,25), (0,0))))
# NVIDIA architecture
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# #model.add(Dropout(0.5))
# model.add(Convolution2D(16, 5, 5, activation='relu'))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# For the loss function we use mean squer error instead of cross-entropy funciton
# because this is a regression model instead of a classification network.
# We want to minimize the error between the steering measurement that the model
# predicts and the ground truth steering measurement.
# the next line compiles and trains the model with the feater and label arrays
# built previously
model.compile(loss='mse', optimizer='adam')

# Here we shuffle the data and split off 20% of the data to use for a validation set.
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# Save the trained model for later use
model.save('model.h5')
