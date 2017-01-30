import pickle
import csv
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Input, Lambda, SpatialDropout2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering("tf")
import cv2
import numpy as np
import pandas as pd
import h5py
import sys
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
from scipy.stats import norm


BATCH_SIZE = 64
EPOCHS = 50
DATA_PATH = "/home/octo/Desktop/simulator-linux/data/data"
LABEL_PATH = "{}/driving_log.csv".format(DATA_PATH)

# Reading and showing images if Flag is TRUE. By default enabled
def read_img_file(img):
    img = "{}/{}".format(DATA_PATH, img)
    img = plt.imread(img)[60:135, : ]
    return img
# Randomly selected batch
def make_batch(data):
    indices = np.random.choice(len(data), BATCH_SIZE)
    return data.sample(n=BATCH_SIZE)
def randomise_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # brightness
    bv = .3 + np.random.random()
    hsv[::2] = hsv[::2]*bv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#    Translation to augment the steering angles and images randomly and avoid overfitting
def image_translation(image,steer,trans_range = 100):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(320,75))
    return image_tr,steer_ang
def resize(img):
    import tensorflow as tf
    img = tf.image.resize_images(img, (66, 200))
    return img
def togray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)

def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return imgs / (255.0 / 2) - 1

#Randomization between left, center and right image and add a shift
def image_random(data, value):
    random = np.random.randint(4)
    if (random == 0):
        path_file = data['left'][value].strip()
        shift_ang = .25
    if (random == 1 or random == 3):
        # Twice as much center images
        path_file = data['center'][value].strip()
        shift_ang = 0.
    if (random == 2):
        path_file = data['right'][value].strip()
        shift_ang = -.25

    return path_file,shift_ang

#Remove about 70% of steering values below 0.05
def remove_low_steering(data):
    ind = data[abs(data['steer'])<.05].index.tolist()
    rows = []
    for i in ind:
        random = np.random.randint(10)
        if random < 8:
            rows.append(i)

    data = data.drop(data.index[rows])
    print("Dropped {} rows with low steering".format(len(rows)))
    return data

def generate_data(data):
    obs = 0
    while 1:
        batch = make_batch(data)
        features = np.empty([BATCH_SIZE, 75, 320, 3])
        labels = np.empty([BATCH_SIZE, 1])

        for i, value in enumerate(batch.index.values):
            x, shift = image_random(data, value)
            x = read_img_file(x)

            x = x.reshape(x.shape[0], x.shape[1], 3)

            # Add shift to steer
            y = float(data['steer'][value]) + shift

            x, y = image_translation(x,y)

            random = np.random.randint(1)
            if (random == 0):
                x = np.fliplr(x)
                y = -y

            labels[i] = y
            features[i] = x

        x = np.array(features)
        y = np.array(labels)
        obs += len(x)
        yield x, y

data = pd.read_csv(LABEL_PATH, index_col=False)
data.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
img = "{}/{}".format(DATA_PATH,data['center'][100].strip())
img = plt.imread(img)[60:135, : ]
img=randomise_image_brightness(img)
img=normalize(img)

def nvidia(img):
    """
    Model based on Nvidia paper
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """

    shape = (img[0], img[1], 3)

    model = Sequential()
    model.add(Lambda(resize, input_shape=shape))
    model.add(Lambda(lambda x: x/255.-0.5))
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model

# 0 = center,1 = left,2 = right,3 = steering angle
for i in range(1):
    # Train the network x times
    # Load data
    model = nvidia(img.shape)
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
    # Shuffle data
    shuffled_data = data.reindex(np.random.permutation(data.index))
    # Split data on a multiple of BATCH SIZE
    split = (int(len(shuffled_data) * 0.9) // BATCH_SIZE) * BATCH_SIZE
    train_data = data[:split]
    train_data = remove_low_steering(train_data)
    val_data = data[split:]
    new_val = (len(val_data) // BATCH_SIZE) * BATCH_SIZE
    val_data = val_data[:new_val]
    samples_per_epoch = len(train_data) - BATCH_SIZE
    values = model.fit_generator(generate_data(train_data), samples_per_epoch=samples_per_epoch, nb_epoch=EPOCHS, validation_data=generate_data(val_data), nb_val_samples=len(val_data))
    model_rep = model.to_json()
