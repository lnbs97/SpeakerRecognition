import os
import shutil
import numpy as np
import playsound as playsound
from playsound import playsound

import sounddevice as sd

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from IPython.display import display, Audio


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
        self.DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")
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
        self.SAMPLES_TO_DISPLAY = 20
        self.create_folder_structure()

    def validate_speaker(self):
        model = tf.keras.models.load_model('../Model/model.h5')

        # Get the list of audio file paths along with their corresponding labels
        class_names = os.listdir(self.DATASET_AUDIO_PATH)
        print("Our class names: {}".format(class_names, ))

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
        noises = tf.stack(noises)

        test_ds = self.paths_and_labels_to_dataset([self.input_audio_path], [3])
        test_ds = test_ds.shuffle(buffer_size=self.BATCH_SIZE * 8, seed=self.SHUFFLE_SEED).batch(
            self.BATCH_SIZE
        )

        test_ds = test_ds.map(lambda x, y: (self.add_noise(x, noises, scale=self.SCALE), y))

        for audios, labels in test_ds.take(1):
            # Get the signal FFT
            ffts = audio_to_fft(audios)
            # Predict
            y_pred = model.predict(ffts)
            # Take random samples
            rnd = np.random.randint(0, 1, self.SAMPLES_TO_DISPLAY)
            audios = audios.numpy()[rnd, :, :]
            labels = labels.numpy()[rnd]
            y_pred = np.argmax(y_pred, axis=-1)[rnd]


            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            print(
                "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                    "[92m" if labels[0] == y_pred[0] else "[91m",
                    class_names[labels[0]],
                    "[92m" if labels[0] == y_pred[0] else "[91m",
                    class_names[y_pred[0]],
                )
            )
            sd.play(audios[0, :, :].squeeze(), 16000)
            sd.stop()

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

    def paths_and_labels_to_dataset(self, audio_paths, labels):
        """Constructs a dataset of audios and labels."""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(lambda x: self.path_to_audio(x))
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((audio_ds, label_ds))

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

    def create_folder_structure(self):
        # If folder `audio`, does not exist, create it, otherwise do nothing
        if os.path.exists(self.DATASET_AUDIO_PATH) is False:
            os.makedirs(self.DATASET_AUDIO_PATH)

        # If folder `noise`, does not exist, create it, otherwise do nothing
        if os.path.exists(self.DATASET_NOISE_PATH) is False:
            os.makedirs(self.DATASET_NOISE_PATH)

        for folder in os.listdir(self.DATASET_ROOT):
            if os.path.isdir(os.path.join(self.DATASET_ROOT, folder)):
                if folder in [self.AUDIO_SUBFOLDER, self.NOISE_SUBFOLDER]:
                    # If folder is `audio` or `noise`, do nothing
                    continue
                elif folder in ["other", "_background_noise_"]:
                    # If folder is one of the folders that contains noise samples,
                    # move it to the `noise` folder
                    shutil.move(
                        os.path.join(self.DATASET_ROOT, folder),
                        os.path.join(self.DATASET_NOISE_PATH, folder),
                    )
                else:
                    # Otherwise, it should be a speaker folder, then move it to
                    # `audio` folder
                    shutil.move(
                        os.path.join(self.DATASET_ROOT, folder),
                        os.path.join(self.DATASET_AUDIO_PATH, folder),
                    )

    def add_speaker(self):
        # Copy speaker files to speaker directory
        shutil.copytree(self.folder, self.DATASET_AUDIO_PATH + '/' + self.FIRST_NAME + '_' + self.LAST_NAME)

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
        print(
            "Found {} files belonging to {} directories".format(
                len(noise_paths), len(os.listdir(self.DATASET_NOISE_PATH))
            )
        )
        noises = []
        for path in noise_paths:
            sample = self.load_noise_sample(path)
            if sample:
                noises.extend(sample)
        noises = tf.stack(noises)

        print(
            "{} noise files were split into {} noise samples where each is {} sec. long".format(
                len(noise_paths), noises.shape[0], noises.shape[1] // self.SAMPLING_RATE
            )
        )

        # Get the list of audio file paths along with their corresponding labels

        class_names = os.listdir(self.DATASET_AUDIO_PATH)
        print("Our class names: {}".format(class_names, ))

        audio_paths = []
        labels = []
        for label, name in enumerate(class_names):
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
            "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
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
        train_ds = self.paths_and_labels_to_dataset(train_audio_paths, train_labels)
        train_ds = train_ds.shuffle(buffer_size=self.BATCH_SIZE * 8, seed=self.SHUFFLE_SEED).batch(
            self.BATCH_SIZE
        )

        valid_ds = self.paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
        valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=self.SHUFFLE_SEED).batch(32)

        # Add noise to the training set
        train_ds = train_ds.map(
            lambda x, y: (self.add_noise(x, noises, scale=self.SCALE), y),
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

        model = self.build_model((self.SAMPLING_RATE // 2, 1), len(class_names))
        # model = tf.keras.models.load_model('../Model/model.h5')

        print("summary")
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
        history = model.fit(
            train_ds,
            epochs=self.EPOCHS,
            validation_data=valid_ds,
            callbacks=[earlystopping_cb, mdlcheckpoint_cb],
        )

        """
        ## Evaluation
        """
        print(model.evaluate(valid_ds))

        """
        ## Demonstration
        """
        test_ds = self.paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
        test_ds = test_ds.shuffle(buffer_size=self.BATCH_SIZE * 8, seed=self.SHUFFLE_SEED).batch(
            self.BATCH_SIZE
        )

        test_ds = test_ds.map(lambda x, y: (self.add_noise(x, noises, scale=self.SCALE), y))

        for audios, labels in test_ds.take(1):
            # Get the signal FFT
            ffts = audio_to_fft(audios)
            # Predict
            y_pred = model.predict(ffts)
            # Take random samples
            rnd = np.random.randint(0, self.BATCH_SIZE, self.SAMPLES_TO_DISPLAY)
            audios = audios.numpy()[rnd, :, :]
            labels = labels.numpy()[rnd]
            y_pred = np.argmax(y_pred, axis=-1)[rnd]

            for index in range(self.SAMPLES_TO_DISPLAY):
                # For every sample, print the true and predicted label
                # as well as run the voice with the noise
                print(
                    "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                        "[92m" if labels[index] == y_pred[index] else "[91m",
                        class_names[labels[index]],
                        "[92m" if labels[index] == y_pred[index] else "[91m",
                        class_names[y_pred[index]],
                    )
                )
                display(Audio(audios[index, :, :].squeeze(), rate=self.SAMPLING_RATE))

