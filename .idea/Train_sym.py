import math
import cv2
import pandas as pd
from pathlib import Path
import io
import os
import random
import torch
import numpy as np
from tensorflow import keras
from keras import layers,Model
from PIL import Image
import pytesseract
import clip
from keras_preprocessing import sequence
import keras_preprocessing.image
from pathlib import Path
import pickle
from keras import datasets,models,layers,callbacks
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from scenedetect import detect, ContentDetector


# train variables
EPOCHS = 120
BATCH_SIZE = 16
NUM_FEATURES = 512
FRAME_SIZE = 216
# model type = 'SVM' or 'LSTM'

def run_model(train_data,avg_frames,n,ModelType,NUM_FEATURES=NUM_FEATURES,BATCH_SIZE=BATCH_SIZE,EPOCHS=EPOCHS,Label='Combined'):

    with open('train_value_CLIP_ALL.pickle','rb') as handle:
        train_value = pickle.load(handle)

    train_data = np.asarray(train_data).astype('float32')
    train_data = np.squeeze(train_data,axis=-2)

    Target_nums = 2

    if Label == 'Excitement':
        train_value,_ = np.hsplit(train_value,2)
        Target_nums = 1
    if Label == 'Funny':
        _,train_value = np.hsplit(train_value,2)
        Target_nums = 1

    x_train, x_test, y_train, y_test = train_test_split(
        train_data,train_value[0:n],test_size=0.2
    )

    seq_length = x_train.shape[1]

    AUTOTUNE=tf.data.AUTOTUNE
    # x_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    # x_train = x_train.batch(BATCH_SIZE).cache('x_train_cache').prefetch(buffer_size=AUTOTUNE)


    model = get_model(ModelType,seq_length,NUM_FEATURES,Target_nums)
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    y_score = model.predict(x_test)

    if Label == 'Excitement':
        exciteRecall,excitePrecision,exciteF1 = get_result(label,y_score,y_test)
        funnyRecall = 0
        funnyPrecision = 0
        funnyF1 = 0
    if Label == 'Funny':
        funnyRecall,funnyPrecision,funnyF1 = get_result(label,y_score,y_test)
        exciteRecall = 0
        excitePrecision = 0
        exciteF1 = 0
    if Label == 'Combined':
        y_score_e,y_score_f = np.hsplit(y_score,2)
        y_test_e,y_test_f = np.hsplit(y_test,2)
        exciteRecall,excitePrecision,exciteF1 = get_result(label,y_score_e,y_test_e)
        funnyRecall,funnyPrecision,funnyF1 = get_result(label,y_score_f,y_test_f)
    results = [avg_frames,exciteRecall,excitePrecision,exciteF1,funnyRecall,funnyPrecision,funnyF1]
    return results

def get_result(label,y_score,y_test):
    exciteTT = 0
    exciteTF = 0
    exciteFT = 0
    exciteFF = 0
    funnyTT = 0
    funnyTF = 0
    funnyFT = 0
    funnyFF = 0

    for x in range(y_test.shape[0]):
        if y_score[x]>0.5:
            y_score[x]=1
        else:
            y_score[x]=0

    for x in range(y_test.shape[0]):
        if y_test[x] == 1 and y_score[x] == 1:
            exciteTT=exciteTT+1
        if y_test[x] == 1 and y_score[x] == 0:
            exciteTF=exciteTF+1
        if y_test[x] == 0 and y_score[x] == 1:
            exciteFT=exciteFT+1
        if y_test[x] == 0 and y_score[x] == 0:
            exciteFF=exciteFF+1

    if (exciteTF+exciteTT == 0 ):
        exciteTT = 1
    if (exciteFT+exciteTT == 0 ):
        exciteTT = 1
    exciteRecall = exciteTT/(exciteFT+exciteTT)
    excitePrecision = exciteTT/(exciteTT+exciteTF)
    exciteF1 = 2*exciteRecall*excitePrecision/(exciteRecall+excitePrecision)

    return exciteRecall,excitePrecision,exciteF1

# load video
def load_sample(SampleType,SampleNumbers,SampleRate,SampleContext,FrameSize=FRAME_SIZE,vae=False):


    # CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # read video list
    folder_path = "./videos/"
    sample_path = "./images/"
    data_path = "./datas/"
    if vae:
       data_path = "./datas_vae/"
    pd_reader = pd.read_csv("./videos/video_list.csv",header=None)

    VIDEO_NUMS = pd_reader.shape[0]
    # VIDEO_NUMS = 1024

    # initialize variables
    train_data = []
    avg_sampled_frames = 0
    for x in range(VIDEO_NUMS):
        video_samples = []
        video_tail = pd_reader.loc[x][0]
        video_tail = str(video_tail).replace("'",'')
        video_path = folder_path+video_tail+".mp4"
        video = cv2.VideoCapture(video_path)
        # get frames in the video
        FrameNumber = video.get(7)
        FrameNumber = int(FrameNumber)
        data_file_path = data_path+str(x)+'.pickle'
        with open(data_file_path,'rb') as handle:
            video_clip = pickle.load(handle)

        if SampleType == 'TIME':
            sample_place = []
            n = SampleContext
            while n<FrameNumber:
                sample_place.append(n)
                n = n+SampleRate
            for i in sample_place:
                clip_samples = np.zeros((1,512))
                j = i-SampleContext
                while j<i+SampleContext+1:
                    encoded_image_array = video_clip[FrameNumber-1]
                    if j<FrameNumber:
                        encoded_image_array = video_clip[j]
                    clip_samples = clip_samples+encoded_image_array
                    j = j+1
                aver_sample = clip_samples/(2*SampleContext+1)
                video_samples.append(aver_sample)

        if SampleType == 'FIXED':
            rate = math.floor((FrameNumber-1)/SampleNumbers)
            if rate<=0:
                rate = 1
            for i in range(SampleNumbers):
                frame = video.set(cv2.CAP_PROP_POS_FRAMES,i*rate)
                encoded_image_array = video_clip[i*rate]
                video_samples.append(encoded_image_array)

        if SampleType == 'SCENE_DETECT':
            scene_place = []
            scene_list = detect(video_path, ContentDetector())
            for i, scene in enumerate(scene_list):
                scene_place.append(scene[0].get_frames())
            for k in scene_place:
                clip_samples = np.zeros((1,512))
                n = k-SampleContext
                j = k
                if k+SampleContext>=FrameNumber-1:
                    j = FrameNumber-SampleContext-1
                    n = j-SampleContext
                if n<0:
                    j = k+(0-n)
                    n = 0
                while n<j+SampleContext+1:
                    encoded_image_array = video_clip[n]
                    clip_samples = clip_samples+encoded_image_array
                    n = n+1
                aver_sample = clip_samples/(2*SampleContext+1)
                video_samples.append(aver_sample)

        avg_sampled_frames = avg_sampled_frames+np.shape(video_samples)[0]
        train_data.append(video_samples)
        video.release()

    print("video sample Completed.")
    avg_sampled_frames = float(avg_sampled_frames/VIDEO_NUMS)
    train_data = sequence.pad_sequences(train_data,padding='post',dtype='float')
    train_data = np.array(train_data,dtype=object)

    return train_data,avg_sampled_frames,VIDEO_NUMS

# build model
def get_model(type,seq_length,NUM_FEATURES,targets):
    if type=='SVM':
        model = models.Sequential(
            [
                layers.Input(shape=(seq_length,NUM_FEATURES)),
                layers.Dropout(0.15),
                layers.Flatten(),
                RandomFourierFeatures(
                    output_dim=2048*targets, scale=10.0, kernel_initializer="gaussian"
                ),
                layers.Dense(targets,activation="sigmoid")
            ]
        )
        model.compile(loss='hinge',
                      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=['accuracy'])
        return model
    else:
        model = models.Sequential(
            [
                layers.Input(shape=(seq_length,NUM_FEATURES)),
                layers.Masking(0.),
                layers.Dropout(0.15),
                layers.LSTM(512*targets),
                layers.Dense(targets, activation="sigmoid"),
            ]
        )
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model



# train variables
MODEL_TYPE_LIST = ['SVM','LSTM']
SAMPLE_RATE_LIST = [30,60,90,150,300,450,600]
# SAMPLE_RATE_LIST = [1]
SAMPLE_CONTEXT_LIST = [0,1,3]
LABEL_LIST = ['Combined']
# SAMPLE_NUM_LIST = [10,30,50,100]
# sample type = 'TIME', 'FIXED' or 'SCENE_DETECT'
# LABEL_LIST = ['Funny','Excitement','Combined']
# results = [avg_frames,exciteRecall,excitePrecision,exciteF1,funnyRecall,funnyPrecision,funnyF1]
sample_type = 'TIME'
x = 0
vae = True
for rate in SAMPLE_RATE_LIST:
    for context in SAMPLE_CONTEXT_LIST:
        for type in MODEL_TYPE_LIST:
            for label in LABEL_LIST:
                train_data, avg_frames, n = load_sample(sample_type,100,rate,context,vae=vae)
                result = run_model(train_data, avg_frames, n, type, Label=label)
                resultString1 = 'SampleType: ' + sample_type + '\tSample rate: ' + str(rate) + '\tSample Context: ' + str(context) + '\tModel Type: ' + type
                resultString4 = 'Avg samples: ' + str(result[0]) + '\tLabel: ' + label + '\tVAE:' + str(vae)
                resultString2 = 'Excite Recall: '+ str(result[1]) + '\tExcite Prec: '+ str(result[2]) + '\t Excite F1:' + str(result[3])
                resultString3 = 'Funny Recall: '+ str(result[4]) + '\tFunny Prec: '+ str(result[5]) + '\t Funny F1:' + str(result[6])
                resultString = resultString1 + '\n' + resultString4 +'\n'+ resultString2 + '\n' + resultString3 + '\n\n'
                file = open('result.txt','a+')
                file.write(resultString)
                file.close()
                print('Model '+str(x)+' Completed')
                x = x+1