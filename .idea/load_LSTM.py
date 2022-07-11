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
from sklearn.metrics import confusion_matrix

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

# print(x_train.shape)
# print(x_test.shape)

# print(y_test)


# model = models.Sequential(
#     [
#         layers.Input(shape=(40,20)),
#         layers.Masking(0.,input_shape=(40,20)),
#         layers.LSTM(64,return_sequences=True),
#         layers.LSTM(32,return_sequences=True),
#         layers.Dropout(0.2),
#         layers.LSTM(16),
#         layers.Dense(2, activation="sigmoid"),
#     ]
# )
#
# model.summary()
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(
#     x_train,
#     y_train,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_split=0.1
# )
#
# model.save('LSTM_model')

model = keras.models.load_model('LSTM_model')

excitement_labels = y_test[:,0]
funny_labels = y_test[:,1]

print(sum(excitement_labels))
print(sum(funny_labels))

score = model.evaluate(x_test, y_test, verbose=0)
y_score = model.predict(x_test)

max_prob=np.argmax(y_score, axis=1)
excitement_labels = y_test[:,0]
funny_labels = y_test[:,1]
print(confusion_matrix(max_prob,excitement_labels))
print(confusion_matrix(max_prob,funny_labels))

print("Test loss:", score[0])
print("Test accuracy:", score[1])