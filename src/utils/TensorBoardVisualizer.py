'''
wrapper("/nfs/students/summer-term-2020/project-4/logs/CRNNPLModule/version_53/events.out.tfevents.1594571337.gpu11.23231.42")
wrapper("/nfs/students/summer-term-2020/project-4/logs/DeepRecursiveCNN8k/version_29/events.out.tfevents.1594857464.gpu11.11584.18")
wrapper("/nfs/students/summer-term-2020/project-4/logs/SpectrogramCNN_8KPLModule/version_47/events.out.tfevents.1594477366.gpu10.2750.6")
wrapper("/nfs/students/summer-term-2020/project-4/logs/M5PLModule/version_54/events.out.tfevents.1594930683.gpu11.31059.2")
wrapper("/nfs/students/summer-term-2020/project-4/logs/CRNN8k/version_74/events.out.tfevents.1594586861.gpu11.31198.73")
'''

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")
sns.set_context("talk")

def extract_branch_name(path):
    start_str = "/run/"
    end_str = "/events.out.tfevents."
    start = path.find(start_str)
    end = path.find(end_str)
    subs1 = path[start:end]
    return subs1

def extract_loss_name(loss_key):
    prefixes = ["Loss/scores/", "Loss/"]
    
    for p in prefixes:
        start = loss_key.find(p)
        if start > -1: 
            return loss_key[start+len(p):]
    return loss_key

def get_logs_from_tfevent(path):
    logs = {}
    for event in tf.train.summary_iterator(path):
        for value in event.summary.value:        
            if value.HasField('simple_value'):
                if value.tag not in logs: logs[value.tag] = []
                logs[value.tag].append(value.simple_value)
    return logs

def invert_logs(logs, key):
    logs[key] = [element * -1 for element in logs["Loss/scores/fake"] ]

def draw_plot(logs, keys, title, save_path = None):

    f = plt.figure(figsize=(7,4))

    for i, key in enumerate(keys):
        plt.plot(logs[key], color = "brgyk"[i], label = extract_loss_name(key))

    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('')
    plt.xlabel('iterations')
    plt.legend()
    plt.grid(True)
    
    xticks = np.arange(0, 1601, 200)
    x_labels = [(str(int(element * 50 / 1000)) + "K") for element in xticks]
    plt.xticks(xticks, labels=x_labels)
    plt.show()
    
    if save_path:
        f.savefig(save_path, bbox_inches='tight')
    
    
''' 
###
### Example 
###
'''

def wrapper(path):


    def draw_plot(logs, keys, title, save_path = None, ax=None):
        f = plt.figure(figsize=(7,4))

        plot = plt
        if ax:
            plot = ax

        for i, key in enumerate(keys):
            plot.plot(logs[key], color = "brgyk"[i], label = extract_loss_name(key))



        plot.legend()
        plot.grid(True)

        #xticks = np.arange(0, 1601, 200)
        #x_labels = [(str(int(element * 50 / 1000)) + "K") for element in xticks]

        if not ax:
            #plot.xticks(xticks, labels=x_labels)
            plot.title(title)
            plot.ylabel('')
            plot.xlabel('epochs')
            plot.show()
        else:
            plot.title.set_text(title)
            plot.set_xlabel("epochs")
            #plot.set_xticks(xticks)
            #plot.set_xticklabels(x_labels)




        if save_path:
            f.savefig(save_path, bbox_inches='tight')        
        

    print(extract_branch_name(path))
    logs = get_logs_from_tfevent(path)
    
    print("J",len(logs["training_loss"]))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    draw_plot(logs, ["val_acc", "training_acc"], title="Accuracy\n")
    draw_plot(logs, ["val_loss", "training_loss"], title="Loss\n")

    fig.suptitle(extract_branch_name(path), y=0)
    fig.tight_layout()
    fig.savefig(extract_branch_name(path)[5:]+"_export.pdf", bbox_inches='tight')   
    fig.savefig(extract_branch_name(path)[5:]+"_export.png", bbox_inches='tight')