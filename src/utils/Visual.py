import os

import numpy as np

import librosa
import librosa.display

import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns

#
# takes numpy as input
# 
def show_audio(data = None, sr = 8000):
    ipd.display(ipd.Audio(data, rate=sr))

def show_spectogram(sample, title="", sr = 8000):
    plt.figure(figsize=(10,5))
    S = librosa.feature.melspectrogram(sample, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title(title if title else 'mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout
    
def show_waveplot(sample, title="", sr = 8000):
    plt.figure(figsize=(10,5))
    plt.subplot(3, 1, 1)
    librosa.display.waveplot(sample, sr=sr)
    plt.title(title if title else 'Monophonic ')
    

def draw_plot(x = [1,2,3], data = [{"data": [0.1, 0.33, 0.5], "color" : "r", "label": "test"}], x_label = "x-axis", y_label = "y-axis", title = "Title", save_path=None,legend_title="Legend",legend_x_offset=1.5, context="poster"):
    
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context(context)
    
    if y_label == "acc": y_label = "Accuracy"
    if x_label == "epsilon": x_label = "Epsilon"
    
    # create an index for each tick position
    xi = list(range(len(x)))

    # create plot
    f = plt.figure(figsize=(12,8))

    # y-axis: 
    plt.ylim(0.,1)
    plt.yticks(np.arange(0, 1.01, 0.1))

    for d in data:
        if 'color' in d:
            plt.plot(xi, d["data"], marker='o', linestyle='--', color=d["color"], label=d["label"]) 
        else:
            plt.plot(xi, d["data"], marker='o', linestyle='--', label=d["label"]) 

    plt.xlabel(x_label)
    plt.ylabel(y_label) 
    plt.xticks(xi, x)
    plt.title(title,fontsize=35)
    plt.legend() 
    plt.xticks(xi)
    leg = plt.legend(loc='center right', bbox_to_anchor=(legend_x_offset, 0.5), ncol=1, frameon=False, title=legend_title)
    leg._legend_box.align = "left"
    plt.show()
    
    if save_path:
        f.savefig(save_path, bbox_inches='tight')

    
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
    
    
    
#
#
#
def speak(text):
    from IPython.display import Javascript as js, clear_output
    # Escape single quotes
    text = text.replace("'", r"\'")
    display(js('''
    if(window.speechSynthesis) {{
        var synth = window.speechSynthesis;
        synth.speak(new window.SpeechSynthesisUtterance('{text}'));
    }}
    '''.format(text=text)))
    # Clear the JS so that the notebook doesn't speak again when reopened/refreshed
    clear_output(False)
    
def notify(language = "DE"):
    if language == "DE":
        speak("Hallo! Hier ist Jupyter Notebook! Ihre Berechnungen wurden soeben fertig gestellt. Ich wünsche viel Spaß und einen angenehmen Tag")
    else:
        speak("Your Jupyter Notebook Operation has just been finished!")