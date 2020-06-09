import os

import numpy as np

import librosa
import librosa.display

import IPython.display as ipd
import matplotlib.pyplot as plt

#
# takes numpy as input
# 
def showAudio(data = None, sr = 8000):
    ipd.display(ipd.Audio(data, rate=sr))

def showSpectogram(sample, title="", sr = 8000):
    plt.figure(figsize=(10,5))
    S = librosa.feature.melspectrogram(sample, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title(title if title else 'mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout
    
def showWaveplot(sample, title="", sr = 8000):
    plt.figure(figsize=(10,5))
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(sample, sr=sr)
    plt.title(title if title else 'Monophonic ')
    

def drawPlot(x = [1,2,3], data = [{"data": [0.1, 0.33, 0.5], "color" : "r", "label": "test"}], x_label = "", y_label = "", title = ""):
    
    # create an index for each tick position
    xi = list(range(len(x)))

    # create plot
    plt.figure(figsize=(14,8))

    # y-axis: 
    plt.ylim(0.,1.19)
    plt.yticks(np.arange(0, 1.01, 0.1))

    for d in data:
        plt.plot(xi, d["data"], marker='o', linestyle='--', color=d["color"], label=d["label"]) 

    plt.xlabel(x_label)
    plt.ylabel(y_label) 
    plt.xticks(xi, x)
    plt.title(title)
    plt.legend() 
    plt.xticks(xi)
    plt.show()
    
#
# further helpers
#
def save_audio(np_sample, sr = 8000, title = "exported_sample"):
    
    directory = "./export"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_name = title + ".wav"
    path = os.path.join(directory, file_name)
    
    librosa.output.write_wav(path, np_sample, sr)