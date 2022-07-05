import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers,Model
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import pickle

IMG_SIZE = 224
MAX_SEQ_LENGTH = 40
BATCH_SIZE = 50
sample_path = "./images/"
pd_reader_video = pd.read_csv("./videos/video_list.csv",header=None)
encoder = keras.models.load_model('vae_encoder')
decoder = keras.models.load_model('vae_decoder')
pd_reader_exciting = pd.read_json("./video_Exciting_clean.json",orient='index')
pd_reader_funny = pd.read_json("./video_Funny_clean.json",orient='index')

AUTOTUNE=tf.data.AUTOTUNE
Y_list = []

samples = []
for x in range(pd_reader_video.shape[0]):
    video_tail = pd_reader_video.loc[x][0]
    video_tail = str(video_tail).replace("'",'')
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
    image_folder = sample_path+video_tail + "/"
    video_frames = []
    for i in range(MAX_SEQ_LENGTH):
        image_path = image_folder + str(i) + ".jpg"
        img_PIL = Image.open(image_path)
        image = tf.keras.preprocessing.image.img_to_array(img_PIL)
        image = image/255
        image = np.expand_dims(image, 0)
        _, _, encoded_image = encoder(image)
        encoded_image_array = np.array(encoded_image)
        video_frames.append(encoded_image_array)
        # img_PIL.show('Original')
        # img_decoded_array = decoder(encoded_image)
        # img_decoded = tf.keras.preprocessing.image.array_to_img(img_decoded_array[0]*255)
        # img_decoded.show('decoded')

    samples.append(video_frames)
    print("Video "+ str(x) +" Completed")

samples = np.array(samples)
Y_list = np.array(Y_list)
print(samples.shape)
print(Y_list.shape)

with open('train_data_encoded.pickle','wb') as handle:
    pickle.dump(samples,handle)

with open('train_value.pickle','wb') as handle:
    pickle.dump(Y_list,handle)

print('Encoded_sample saved.')

