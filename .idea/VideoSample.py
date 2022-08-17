import math
import cv2
import pandas as pd
from pathlib import Path
import pickle
import io
import os
import random
import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers,Model
from PIL import Image
import pytesseract
import clip
from keras_preprocessing import sequence

# read video list
folder_path = "./videos/"
sample_path = "./images/"

# sample_path = "./smallimages/"

pd_reader = pd.read_csv("./videos/video_list.csv",header=None)
pd_reader_exciting = pd.read_json("./video_Exciting_clean.json",orient='index')
pd_reader_funny = pd.read_json("./video_Funny_clean.json",orient='index')
SamplesForEachVideo = 100
FrameSize = 216
SampleType = 'CLIP'
SampleRate = 30
SampleLength = 1

# CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

Y_list = []
samples = []
sample_sum = 0

for x in range(pd_reader.shape[0]):
    video_tail = pd_reader.loc[x][0]
    video_tail = str(video_tail).replace("'",'')
    video_path = folder_path+video_tail+".mp4"
    video = cv2.VideoCapture(video_path)

    exciting_value = pd_reader_exciting.loc[video_tail][0]
    funny_value = pd_reader_funny.loc[video_tail][0]

    if exciting_value > 0.5:
        exciting_value = 1
    else:
        exciting_value = 0
    if funny_value > 0.5:
        funny_value = 1
    else:
        funny_value = 0
    Y_list.append([exciting_value,funny_value])

    if video.isOpened():
        rval, frame = video.read()
    else:
        rval = False

    FrameNumber = video.get(7)
    count = 0
    Samplecount = 0

    # video_sample_folder = sample_path + video_tail
    # if not os.path.exists(video_sample_folder):
    #     os.mkdir(video_sample_folder)

    video_frames = []
    IsSampling = False
    sampletime = []
    raw_frames = np.zeros((1,512))

    n = math.floor(FrameNumber/SampleRate)+1

    for k in range(n):
        sampletime.append(k*SampleRate)

    while rval:
        if (count in sampletime):
            IsSampling = True
        if IsSampling:
            frame = cv2.resize(frame,(FrameSize,FrameSize),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame)
            img_pred = preprocess(frame_pil).unsqueeze(0).to(device)
            encoded_image = model.encode_image(img_pred)
            encoded_image_array = encoded_image.detach().numpy()
            raw_frames = raw_frames+encoded_image_array
            # image_name = video_sample_folder + "/" + str(samplecount) + ".jpg"
            # cv2.imwrite(image_name,frame)
            Samplecount += 1
        if Samplecount>SampleLength*2:
            Samplecount = 0
            IsSampling = False
            averaged_image_array = raw_frames/(SampleLength*2)
            video_frames.append(averaged_image_array)
            raw_frames = np.zeros((1,512))

        count+=1
        rval, frame = video.read()
        # if not rval and samplecount<SamplesForEachVideo:
        #     while samplecount<SamplesForEachVideo:
        #         image_name = video_sample_folder + "/" + str(samplecount) + ".jpg"
        #         cv2.imwrite(image_name,temp_frame)
        #         samplecount+=1
    samples.append(video_frames)
    sample_sum = sample_sum+np.shape(video_frames)[0]
    # photos[video_tail] = samples
    print("video No:" + str(x) +" Completed.")
    video.release()

samples = sequence.pad_sequences(samples,padding='post',dtype='float')
samples = np.array(samples,dtype=object)
Y_list = np.array(Y_list)

print(samples[0:1])
print(Y_list[0:1])

print(samples.shape)
print(Y_list.shape)

print(sample_sum/pd_reader.shape[0])

with open('train_data_encoded_CLIP_30.pickle','wb') as handle:
    pickle.dump(samples,handle)

with open('train_value_CLIP_30.pickle','wb') as handle:
    pickle.dump(Y_list,handle)

print('Encoded_sample saved.')



