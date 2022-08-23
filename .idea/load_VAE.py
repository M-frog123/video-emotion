import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers,Model
import pandas as pd
from pathlib import Path
import cv2

IMG_SIZE = 216
MAX_SEQ_LENGTH = 100
EPOCHES = 50
BATCH_SIZE = 4
latent_dim = 512
sample_path = "./images/"
random_seed = random.randint(1,1000000)

pd_reader = pd.read_csv("./videos/video_list.csv",header=None)

samples = tf.keras.preprocessing.image_dataset_from_directory(
    sample_path,
    image_size=(216,216),
    seed=random_seed,
    subset="training",
    validation_split = 0.8,
    batch_size=BATCH_SIZE,
    label_mode = None,
    labels = None
)

AUTOTUNE=tf.data.AUTOTUNE
samples = samples.cache("vae_train_cache").shuffle(1000).prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_samples = samples.map(lambda x : normalization_layer(x))

# samples = []
# for x in range(pd_reader.shape[0]):
# for x in range(50):
#     video_tail = pd_reader.loc[x][0]
#     video_tail = str(video_tail).replace("'",'')
#     image_folder = sample_path+video_tail + "/"
#     for i in range(MAX_SEQ_LENGTH):
#         image_path = image_folder + str(i) + ".jpg"
#         image = cv2.imread(image_path)
#         image = np.array(image).astype("float32")/255
#         samples.append(image)

# samples = np.array(samples)
# print(samples.shape)
class Sampling(layers.Layer):
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(216, 216, 3))
    x = layers.Conv2D(16, 3, activation="relu", strides=3, padding="same")(encoder_inputs)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim+latent_dim)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

def build_decoder(latent_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(9 * 9 * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((9, 9, 128))(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=3, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, strides=1, padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder

class VAE(Model):
    def __init__(self,encoder,decoder,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.acc_tracker = tf.keras.metrics.BinaryAccuracy()

    def call(self,inputs):
        z_mean, log_var, z = self.encoder(inputs)
        decoder_outputs = self.decoder(z)
        return z_mean, log_var, decoder_outputs

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.acc_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        self.acc_tracker.update_state(data, reconstruction)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
        }


# encoder = build_encoder(latent_dim)
# decoder = build_decoder(latent_dim)


encoder = keras.models.load_model('vae_encoder')
decoder = keras.models.load_model('vae_decoder')

encoder.summary()
decoder.summary()

def get_model(encoder,decoder):
    vae = VAE(encoder,decoder)
    vae.compile(optimizer='adam')
    return vae

vae = get_model(encoder,decoder)
vae.fit(normalized_samples,epochs=EPOCHES,batch_size=BATCH_SIZE)
vae.build((None,216,216,3))

encoder.compile(optimizer='adam')
decoder.compile(optimizer='adam')

vae.encoder.save('vae_encoder')
vae.decoder.save('vae_decoder')