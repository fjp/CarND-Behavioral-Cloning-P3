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

X_train = np.array(images)
y_train = np.array(measurements)

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

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

# For the loss function we use mean squer error instead of cross-entropy funciton
# because this is a regression model instead of a classification network.
# We want to minimize the error between the steering measurement that the model
# predicts and the ground truth steering measurement.
# the next line compiles and trains the model with the feater and label arrays
# built previously
model.compile(loss='mse', optimizer='adam')

# Here we shuffle the data and split off 20% of the data to use for a validation set.
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

# Save the trained model for later use
model.save('model.h5')
