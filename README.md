# Acoustic Scenes Classification

This project shares the python code used for the challenge [DCASE 2018 task 1](http://dcase.community/challenge2018/index).

The goal of Acoustic Scene Classification is to classify a test recording into one of the provided predefined classes that characterizes the environment in which it was recorded

![ASC schema](https://raw.githubusercontent.com/Splumecocq/AcousticScenesClassification/master/Image/Schema-acoustic-scene-classification.jpg)

In this project the audio signal in input is preprocessed into a spectrogram by short-time Fourier transform [SFTF](https://en.wikipedia.org/wiki/Short-time_Fourier_transform).
A methodology of **supervised machine learning** based on Convolutional Neuronal Network ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) classified the signal.

## Struture
The program **main.py** launches the training.

The directory **Models** contains the CNN modules:
* simpleCNN: simple module to test the process
* simpleCNN_v2: same with a standard conv: conv2D + batchNorm + Relu
* CNN_Dorfer: CNN proposed by CP-JKU team to the DCASE Challenge 2018 Task 1 Subtask A (without the Gaussian Noise)
* CNN_Dorfer2: CNN proposed by CP-JKU team to the DCASE Challenge 2018 Task 1 Subtask A (with the Gaussian Noise)
* Xception: Xception modele with variable number of Middle Flow and simplified Exit Flow (one conv2d and average pooling)


The directory **Data** contains:
* the split of train and validation/test as in the challenge DCASE 2018 (fold2_train.txt and fold2_evaluate.txt)
* the split of initial train in train and validation (fold3_train.txt and fold3_val.txt)
* mean and standard deviation on train data (mean_train.npy and std_train.npy)

The directory **Utils** contains:
* audio_to_spectrogram: Convert audio signal to spectrogram with Librosa librairy
* progressbar.py: a simple progress to follow the evolution of training process
* save_model: Save a model in a file
* unzip: extract the zip file of input data



