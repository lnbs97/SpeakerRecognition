# SpeakerRecognition
Speaker Recognition is a tool written in python with tensorflow that allows to detect a speaker in a one second sound clip with high accuracy.
For detection it uses a Convoluted Neural Network (CNN).
It comes with a pre-trained model for 6 speakers.

Based on the keras Speaker Recognition example: https://keras.io/examples/audio/speaker_recognition_using_cnn/

## Features
- add new speakers and re-train the model
- analyze audio files
- list trained speakers

## Usage
To add new speakers you need to prepare about 1.000 different audio clips with one second length each. This can be done easily with a 30 minute recording of the speaker that can be trimmed and cut with audacity. Then select the folder containing the sound clips and train the model. This will take 30+ minutes based on your computer.

## Screenshots
![image](https://user-images.githubusercontent.com/45437638/155401333-f7111857-0c9e-459b-9333-390dbb9db8ff.png)
![image](https://user-images.githubusercontent.com/45437638/155401900-0dd4bb1a-2952-43b8-9ac2-5c4e5b3fef0b.png)
![image](https://user-images.githubusercontent.com/45437638/155401870-120ddb0d-4026-4009-a6fd-4d843514758c.png)
