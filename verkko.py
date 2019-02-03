import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv1D, Dense, Dropout, Flatten, MaxPooling2D, MaxPooling1D
from keras.regularizers import l1, l2
import numpy as np


classes = 9

def model_2D_definition():
    # Setup model
    model = Sequential()


    # Data params
    imgw = 10
    imgh = 384
    channels = 1

    # Sample count
    N = 1703

    input_shape = (channels, imgw, imgh)
    filters = 25
    filters2 = 35
    kernel_size = (5,5)

    # Add layers
    model.add(Conv2D(filters, kernel_size, activation='relu', input_shape = input_shape, kernel_regularizer=l2(0.005))) #32 filtteria
    model.add(MaxPooling2D((2,2), strides = 2))
    model.add(Conv2D(filters, kernel_size, activation='relu', kernel_regularizer=l2(0.005))) #32 filtteria
    model.add(MaxPooling2D((2,2), strides = 2))
    model.add(Conv2D(filters2, kernel_size, activation='relu', kernel_regularizer=l2(0.005))) #64 filtteria
    model.add(MaxPooling2D((2,2), strides = 2))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax', kernel_regularizer = l2(0.05)))

    return model

def model_1D_definition():
    model = Sequential()

    array_size = (384, 1)

    filters = 16
    filters2 = 16
    kernel_size = 16

    model.add(Conv1D(filters, kernel_size, activation='relu', input_shape = array_size, kernel_regularizer=l2(0.01)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Conv1D(filters2, kernel_size, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    return model
