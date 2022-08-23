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

IMG_SIZE = 216
EPOCHS = 400

BATCH_SIZE = 4
NUM_FEATURES = 512

sample_nums = 0

with open('train_data_encoded_CLIP_SD.pickle','rb') as handle:
    train_data = pickle.load(handle)

with open('train_value_encoded_CLIP_SD.pickle','rb') as handle:
    train_value = pickle.load(handle)

train_data = np.asarray(train_data).astype('float32')
train_data = np.squeeze(train_data,axis=-2)

# model = keras.models.load_model('LSTM_model')
# print(model.predict(train_data))
# quit(0)

excitement_labels = train_value[:,0]
funny_labels = train_value[:,1]

print(sum(excitement_labels))
print(sum(funny_labels))

x_train, x_test, y_train, y_test = train_test_split(
    train_data,train_value,test_size=0.2
)

print(x_train.shape)
MAX_SEQ_LENGTH = x_train.shape[1]
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# print(y_test)


model = models.Sequential(
    [
        layers.Input(shape=(MAX_SEQ_LENGTH,NUM_FEATURES)),
        layers.Masking(0.),
        layers.Dropout(0.15),
        layers.LSTM(256,return_sequences=True),
        layers.LSTM(128,return_sequences=True),
        layers.LSTM(64,return_sequences=True),
        layers.LSTM(32),
        layers.Dropout(0.15),
        layers.Dense(2, activation="sigmoid"),
    ]
)

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

model.save('LSTM_model')

# model = keras.models.load_model('LSTM_model')

excitement_labels = y_train[:,0]
funny_labels = y_train[:,1]

print(sum(excitement_labels))
print(sum(funny_labels))

excitement_labels = y_test[:,0]
funny_labels = y_test[:,1]

print(sum(excitement_labels))
print(sum(funny_labels))

score = model.evaluate(x_test, y_test, verbose=0)
y_score = model.predict(x_test)

# max_prob=np.argmax(y_score, axis=1)
# excitement_labels = y_test[:,0]
# funny_labels = y_test[:,1]

exciteTT = 0
exciteTF = 0
exciteFT = 0
exciteFF = 0
funnyTT = 0
funnyTF = 0
funnyFT = 0
funnyFF = 0
excitepre = True
funnypre = True
ALLright = 0

print(y_test[0])
print(y_score[0])

for x in range(y_test.shape[0]):
    if y_score[x][0]>0.5:
        y_score[x][0]=1
    else:
        y_score[x][0]=0
    if y_score[x][1]>0.5:
        y_score[x][1]=1
    else:
        y_score[x][1]=0

for x in range(y_test.shape[0]):
    if y_test[x][0] == 1 and y_score[x][0] == 1:
        exciteTT=exciteTT+1
        excitepre = True
    if y_test[x][0] == 1 and y_score[x][0] == 0:
        exciteTF=exciteTF+1
        excitepre = False
    if y_test[x][0] == 0 and y_score[x][0] == 1:
        exciteFT=exciteFT+1
        excitepre = False
    if y_test[x][0] == 0 and y_score[x][0] == 0:
        exciteFF=exciteFF+1
        excitepre = True
    if y_test[x][1] == 1 and y_score[x][1] == 1:
        funnyTT=funnyTT+1
        funnypre = True
    if y_test[x][1] == 1 and y_score[x][1] == 0:
        funnyTF=funnyTF+1
        funnypre = False
    if y_test[x][1] == 0 and y_score[x][1] == 1:
        funnyFT=funnyFT+1
        funnypre = False
    if y_test[x][1] == 0 and y_score[x][1] == 0:
        funnyFF=funnyFF+1
        funnypre = True
    if excitepre == True and funnypre == True:
        ALLright=ALLright+1




print("exciteTT,exciteTF,exciteFT,exciteFF,funnyTT,funnyTF,funnyFT,funnyFF")
print(exciteTT,exciteTF,exciteFT,exciteFF,funnyTT,funnyTF,funnyFT,funnyFF)

exciteRecall = exciteTT/(exciteFT+exciteTT)
excitePrecision = exciteTT/(exciteTT+exciteTF)
exciteF1 = 2*exciteRecall*excitePrecision/(exciteRecall+excitePrecision)

funnyRecall = funnyTT/(funnyFT+funnyTT)
funnyPrecision = funnyTT/(funnyTT+funnyTF)
funnyF1 = 2*funnyRecall*funnyPrecision/(funnyRecall+funnyPrecision)

print('excite F1:'+str(exciteF1))
print('funny F1:'+str(funnyF1))

print('ALLright:'+str(ALLright))

print("Exciting accuracy:"+str((exciteTT+exciteFF)/(exciteTT+exciteTF+exciteFT+exciteFF)))
print("Funny accuracy:"+str((funnyTT+funnyFF)/(funnyTT+funnyTF+funnyFT+funnyFF)))
print("Total accuracy:"+str((exciteTT+exciteFF+funnyTT+funnyFF)/(exciteTT+exciteTF+exciteFT+exciteFF+funnyTT+funnyTF+funnyFT+funnyFF)))


# print(confusion_matrix(max_prob,excitement_labels))
# print(confusion_matrix(max_prob,funny_labels))

print("Test loss:", score[0])
print("Test accuracy:", score[1])