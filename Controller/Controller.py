import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


class Controller:
    def __init__(self):
        self.input_audio_path = None
        self.folder = None
        self.noises = []
        self.model = tf.keras.models.load_model('Model/model.h5')
        self.DATASET_ROOT = "dataset"
        self.AUDIO_SUBFOLDER = "audio"
        self.NOISE_SUBFOLDER = "noise"
        self.FIRST_NAME = ""
        self.LAST_NAME = ""
        self.DATASET_AUDIO_PATH = os.path.join(self.DATASET_ROOT, self.AUDIO_SUBFOLDER)
        self.DATASET_NOISE_PATH = os.path.join(self.DATASET_ROOT, self.NOISE_SUBFOLDER)
        # Percentage of samples to use for validation
        self.VALID_SPLIT = 0.1
        # Seed to use when shuffling the dataset and the noise
        self.SHUFFLE_SEED = 43
        self.SAMPLING_RATE = 16000
        # The factor to multiply the noise with according to:
        #   noisy_sample = sample + noise * prop * scale
        #      where prop = sample_amplitude / noise_amplitude
        self.SCALE = 0.5
        self.BATCH_SIZE = 128
        self.EPOCHS = 1
        self.init_noise()
        self.class_names = []
        self.update_class_names()

    def validate_speaker(self):
        test_ds = self.paths_to_dataset([self.input_audio_path])
        test_ds = test_ds.shuffle(buffer_size=self.BATCH_SIZE * 8, seed=self.SHUFFLE_SEED).batch(
            self.BATCH_SIZE
        )

        test_ds = test_ds.map(lambda x: (self.add_noise(x, self.noises, scale=self.SCALE)))

        audio = next(iter(test_ds.take(1)))
        # Get the signal FFT
        ffts = audio_to_fft(audio)
        # Predict
        y_pred = self.model.predict(ffts)
        y_pred = np.argmax(y_pred, axis=-1)[[0]]
        # sd.play(audios[0, :, :].squeeze(), 16000)
        # sd.stop()
        return self.class_names[y_pred[0]]

    # Split noise into chunks of 16,000 steps each
    def load_noise_sample(self, path):
        sample, sampling_rate = tf.audio.decode_wav(
            tf.io.read_file(path), desired_channels=1
        )
        if sampling_rate == self.SAMPLING_RATE:
            # Number of slices of 16000 each that can be generated from the noise sample
            slices = int(sample.shape[0] / self.SAMPLING_RATE)
            sample = tf.split(sample[: slices * self.SAMPLING_RATE], slices)
            return sample
        else:
            print("Sampling rate for {} is incorrect. Ignoring it".format(path))
            return None

    def update_class_names(self):
        self.class_names = os.listdir(self.DATASET_AUDIO_PATH)

    def path_to_audio(self, path):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, self.SAMPLING_RATE)
        return audio

    def add_noise(self, audio, noises=None, scale=0.5):
        if noises is not None:
            # Create a random tensor of the same size as audio ranging from
            # 0 to the number of noise stream samples that we have.
            tf_rnd = tf.random.uniform(
                (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
            )
            noise = tf.gather(noises, tf_rnd, axis=0)

            # Get the amplitude proportion between the audio and the noise
            prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
            prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

            # Adding the rescaled noise to audio
            audio = audio + noise * prop * scale

        return audio

    def paths_to_dataset(self, audio_paths):
        """Constructs a dataset of audios"""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(lambda x: self.path_to_audio(x))
        return audio_ds

    def residual_block(self, x, filters, conv_num=3, activation="relu"):
        # Shortcut
        s = keras.layers.Conv1D(filters, 1, padding="same")(x)
        for i in range(conv_num - 1):
            x = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x = keras.layers.Activation(activation)(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)

    def build_model(self, input_shape, num_classes):
        inputs = keras.layers.Input(shape=input_shape, name="input")

        x = self.residual_block(inputs, 16, 2)
        x = self.residual_block(x, 32, 2)
        x = self.residual_block(x, 64, 3)
        x = self.residual_block(x, 128, 3)
        x = self.residual_block(x, 128, 3)

        x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

        return keras.models.Model(inputs=inputs, outputs=outputs)

    def add_speaker(self):
        # Copy speaker files to speaker directory
        shutil.copytree(self.folder, self.DATASET_AUDIO_PATH + '/' + self.FIRST_NAME + '_' + self.LAST_NAME)
        self.update_class_names()

        audio_paths = []
        labels = []
        for label, name in enumerate(self.class_names):
            print("Processing speaker {}".format(name, ))
            dir_path = Path(self.DATASET_AUDIO_PATH) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            audio_paths += speaker_sample_paths
            labels += [label] * len(speaker_sample_paths)

        print(
            "Found {} files belonging to {} classes.".format(len(audio_paths), len(self.class_names))
        )

        # Shuffle
        rng = np.random.RandomState(self.SHUFFLE_SEED)
        rng.shuffle(audio_paths)
        rng = np.random.RandomState(self.SHUFFLE_SEED)
        rng.shuffle(labels)

        # Split into training and validation
        num_val_samples = int(self.VALID_SPLIT * len(audio_paths))
        print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
        train_audio_paths = audio_paths[:-num_val_samples]
        train_labels = labels[:-num_val_samples]

        print("Using {} files for validation.".format(num_val_samples))
        valid_audio_paths = audio_paths[-num_val_samples:]
        valid_labels = labels[-num_val_samples:]

        # Create 2 datasets, one for training and the other for validation
        train_ds = self.paths_to_dataset(train_audio_paths, train_labels)
        train_ds = train_ds.shuffle(buffer_size=self.BATCH_SIZE * 8, seed=self.SHUFFLE_SEED).batch(
            self.BATCH_SIZE
        )

        valid_ds = self.paths_to_dataset(valid_audio_paths, valid_labels)
        valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=self.SHUFFLE_SEED).batch(32)

        # Add noise to the training set
        train_ds = train_ds.map(
            lambda x, y: (self.add_noise(x, self.noises, scale=self.SCALE), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        # Transform audio wave to the frequency domain using `audio_to_fft`
        train_ds = train_ds.map(
            lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        valid_ds = valid_ds.map(
            lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

        model = self.build_model((self.SAMPLING_RATE // 2, 1), len(self.class_names))

        model.summary()

        # Compile the model using Adam's default learning rate
        model.compile(
            optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        # Add callbacks:
        # 'EarlyStopping' to stop training when the model is not enhancing anymore
        # 'ModelCheckPoint' to always keep the model that has the best val_accuracy
        model_save_filename = "../Model/model2.h5"

        earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
            model_save_filename, monitor="val_accuracy", save_best_only=True
        )

        """
        ## Training
        """
        model.fit(
            train_ds,
            epochs=self.EPOCHS,
            validation_data=valid_ds,
            callbacks=[earlystopping_cb, mdlcheckpoint_cb],
        )

        """
        ## Evaluation
        """
        print(model.evaluate(valid_ds))


    def init_noise(self):
        # Get the list of all noise files
        noise_paths = []
        for subdir in os.listdir(self.DATASET_NOISE_PATH):
            subdir_path = Path(self.DATASET_NOISE_PATH) / subdir
            if os.path.isdir(subdir_path):
                noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(".wav")
                ]

        noises = []
        for path in noise_paths:
            sample = self.load_noise_sample(path)
            if sample:
                noises.extend(sample)
        self.noises = tf.stack(noises)
