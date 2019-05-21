
"""
Project of Acoustic Scene Classification

Started on 17 Feb 2019

@author: Plumecocq Simon

Use Librosa to transform Audio signal to spectogram

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display as display

main_path = "C:\\Users\\Plumecocq\\Documents\\Python Scripts\\ProjetAcousticSceneClassification"
os.chdir(main_path)

data_path = "data"
unzip_path = "unzip"
save1_path = "transform_1"

#First 
path_data = data_path+"\\"+unzip_path+"\\TUT-urban-acoustic-scenes-2018-development\\audio\\"
y, sr = librosa.load(path_data+"airport-barcelona-0-0-a.wav",mono=False,sr=22050)
print(sr)
print(y.shape)




plt.figure()
plt.subplot(2, 1, 1)
display.waveplot(y[0], sr=sr)
plt.subplot(2, 1, 2)
display.waveplot(y[1], sr=sr)



#Resample if needed
y_22 = librosa.resample(y, sr, 22050)
print(y_22.shape)

#Short-time Fourier transform (STFT)
#Window size
n_fft=2048
hop_length = 512
ft_left = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
ft_right = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length)
print(ft_left.shape)
print(ft_right.shape)


#np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
#np.angle(D[f, t]) is the phase of frequency bin f at frame t
#Rmq: ft = magnitude * phase
Magnitude_l = np.abs(ft_left)
Magnitude_r = np.abs(ft_left)
#Phase = np.angle(ft_left)

Power_l = Magnitude_l**2
Power_r = Magnitude_r**2
print(Power_l.shape)

#Remove of the boucle ?
fft_frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
print(fft_frequencies.shape)

#Perceptual weighting of a power spectrogram
pw_l = librosa.perceptual_weighting(S=Magnitude_l**2,frequencies=fft_frequencies)
pw_r = librosa.perceptual_weighting(S=Magnitude_r**2,frequencies=fft_frequencies)
#more option as power_to_db: ref=1.0, amin=1e-10, top_db=80.0
print(pw_l.shape)

ms_l = librosa.feature.melspectrogram(S=pw_l, n_mels=256)
ms_r = librosa.feature.melspectrogram(S=pw_r, n_mels=256)
#by default n_mels=128
print(ms_l.shape)

tranform = np.empty((2,256,431))
tranform[0] = ms_l
tranform[1] = ms_r


path_save = data_path+"\\"+save1_path+"\\"
np.save(path_save+'airport-barcelona-0-0-a.npy', tranform)




#CQT
Cqt = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1'))
print(Cqt.size)
C = np.abs(Cqt)
freqs = librosa.cqt_frequencies(C.shape[0], fmin=librosa.note_to_hz('A1'))
print(freqs.size)
perceptual_Cqt = librosa.perceptual_weighting(C**2,freqs,ref=np.max)

plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(Cqt,ref=np.max),fmin=librosa.note_to_hz('A1'),y_axis='cqt_hz')
plt.title('Log CQT power')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 1, 2)
librosa.display.specshow(perceptual_Cqt, y_axis='cqt_hz',fmin=librosa.note_to_hz('A1'),x_axis='time')
plt.title('Perceptually weighted log CQT')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

#Spectrogam
librosa.display.specshow(librosa.amplitude_to_db(ft,ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()


  
