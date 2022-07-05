import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import math

import cv2
import keras_preprocessing.image
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tensorflow import keras
from keras import datasets,models,layers,callbacks
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
EPOCHS = 100

BATCH_SIZE = 64
MAX_SEQ_LENGTH = 50
NUM_FEATURES = 2048

sample_nums = 0

with open('train_data_encoded.pickle','rb') as handle:
    train_data = pickle.load(handle)

with open('train_value.pickle','rb') as handle:
    train_value = pickle.load(handle)

train_data = np.squeeze(train_data)

x_train, x_test, y_train, y_test = train_test_split(
    train_data,train_value,test_size=0.15
)

print(x_train.shape)
print(x_test.shape)

model = models.Sequential(
    [
        layers.Input(shape=(40,20)),
        layers.Masking(0.,input_shape=(40,20)),
        layers.LSTM(64),
        layers.Dense(2, activation="sigmoid"),
    ]
)

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])