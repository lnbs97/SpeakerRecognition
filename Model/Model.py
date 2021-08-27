import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


class Model:

    def __init__(self):
        # The factor to multiply the noise with according to:
        #   noisy_sample = sample + noise * prop * scale
        #      where prop = sample_amplitude / noise_amplitude
        self.SCALE = 0.5
        self.BATCH_SIZE = 128
        self.EPOCHS = 100
        self.model = tf.keras.models.load_model('Model/model.h5')

    def predict(self, ffts):
        # Predict
        return self.model.predict(ffts)
