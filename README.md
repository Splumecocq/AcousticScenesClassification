# Acoustic Scenes Classification

This project shares the python code used for the challenge [DCASE 2018](http://dcase.community/challenge2018/index).

## Struture
The directory **Models** contains the CNN modules.

The directory **Data** contains:
* the split of train and validation as in the challenge DCASE 2018 (fold2_train.txt and fold2_evaluate.txt)
* mean and standard deviation on train data (mean_train.npy and std_train.npy)

The directory **Utils** contains:
* audio_to_spectrogram: Convert audio signal to spectrogram with Librosa librairy
* progressbar.py: a simple progress to follow the evolution of training process



