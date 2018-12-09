import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle

IMG_SIZE = 64

### Preparing the Data ###
def load_image(img_path):
    try:
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        return(img)
    except:
        print(img_path)
        return(None)

def create_training_set(path, label):
    images_ = []
    for subdir, dirs, files in os.walk(path):
        for i,file in enumerate(tqdm(files)):
            image_filepath = os.path.join(os.getcwd(),subdir,file)
            img = load_image(image_filepath)
            if img is not None:
                images_.append(img)
    labels_ = [label] * len(images_)
    return(np.array(images_), np.array(labels_))

path1 = "Data/Cats"
path2 = "Data/Dogs"

path1 = "../Data/Cats"
path2 = "../Data/Dogs"

example_file = "Data/sample_image2.jpg"
img = cv2.imread(example_file)
print(img.shape)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

imgs_cats, labels_cats = create_training_set(path1,0)
imgs_dogs, labels_dogs = create_training_set(path2,1)

imgs = np.concatenate((imgs_cats, imgs_dogs))
labels = np.concatenate((labels_cats, labels_dogs))

x_train, x_test, y_train, y_test = train_test_split(imgs,
                                                    labels,
                                                    train_size=0.8,
                                                    test_size=0.2)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1





### Creating the Model template ###
def create_model():
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
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


# Training Model and Saving weights
model = create_model()
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.3)
model.summary()
model.save_weights("Model-Checkpoints/model-checkpoints3")


# Create second version of model and load weights of previous training
new_model = create_model()
new_model.load_weights("Model-Checkpoints/model-checkpoints2")


# Test our results
import sklearn.metrics as metrics
preds_test = new_model.predict_classes(x_test)
confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=preds_test)


### Flask Prep ###
from skimage import io

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_from_url(link):
    img = io.imread(link)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = tf.keras.utils.normalize(img, axis=1)
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return(img)

img_url2 = "https://cdn8.bigcommerce.com/s-rj0z9yqukq/images/stencil/original/uploaded_images/sayslove.jpg?t=1518480203"
image = load_from_url(img_url2)
# plt.imshow(image, cmap = 'gray')

new_model = create_model()
new_model.load_weights("Model-Checkpoints/model-checkpoints2")
new_model.predict_classes(image)


#
