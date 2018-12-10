
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# import cv2
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
IMG_SIZE = 64


def load_from_url(link):
    img = io.imread(link)
    img = rgb2gray(img)
    img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    img = tf.keras.utils.normalize(img, axis=1)
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return(img)

def create_model():
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(64,64,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))

    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return(model)
