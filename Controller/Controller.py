from Model.Model import Model
import os
from pathlib import Path
import numpy as np

import tensorflow as tf


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
        self.input_audio = None
        # self.database = Database()

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
        self.EPOCHS = 100
        self.dataset_root = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")
        self.dataset_audio_path = os.path.join(self.dataset_root, "audio")
        self.noise_files = self.get_chunks_of_noise_files()
        self.audio_path, self.audio_labels = self.get_list_with_path_and_labels()
        self.model = Model()

    # reads and decodes audio file and returns audio tensor
    def compile_audio_file(self, input_audio_path):
        valid_ds = self.path_to_dataset(input_audio_path)
        valid_ds = valid_ds.map(
            lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)
        for audios, labels in valid_ds.take(1):
            # Get the signal FFT
            ffts = audio_to_fft(audios)
            # Predict
            y_pred = self.model.predict(ffts)
            # Take random samples
            rnd = np.random.randint(0, self.BATCH_SIZE)
            y_pred = np.argmax(y_pred, axis=-1)[rnd]
            print(y_pred)


    # get the list of all chunks noise files
    def get_chunks_of_noise_files(self):
        dataset_noise_path = os.path.join(self.dataset_root, "noise")
        noise_paths = []
        for subdir in os.listdir(dataset_noise_path):
            subdir_path = Path(dataset_noise_path) / subdir
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
        return noises

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

    def get_list_with_path_and_labels(self):
        class_names = os.listdir(self.dataset_audio_path)
        audio_paths = []
        labels = []
        for label, name in enumerate(class_names):
            print("Processing speaker {}".format(name,))
            dir_path = Path(self.dataset_audio_path) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            audio_paths += speaker_sample_paths
            labels += [label] * len(speaker_sample_paths)
            return audio_paths, labels

    def shuffle_path_and_labels(self):
        # Shuffle pat and labels
        rng = np.random.RandomState(self.SHUFFLE_SEED)
        rng.shuffle(self.audio_path)
        rng = np.random.RandomState(self.SHUFFLE_SEED)
        rng.shuffle(self.audio_labels)

    def predict_audio_file(self):
        if self.input_audio is not None:
            y_predict = self.model.predict_audio_file(audio_to_fft(self.input_audio))
            print(y_predict)
        else:
            print("first choose an audio file")

    def path_to_dataset(self, audio_path):
        """Constructs a dataset of audios and labels."""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_path)
        audio_ds = path_ds.map(lambda x: self.path_to_audio(x))
        return tf.data.Dataset.zip(audio_ds)

    def path_to_audio(self, path):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, self.SAMPLING_RATE)
        return audio