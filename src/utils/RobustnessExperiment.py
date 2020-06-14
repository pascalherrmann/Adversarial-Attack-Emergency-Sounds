import os
import sys
import json
import pickle
from itertools import product

from tqdm import tqdm
import torch

from utils.Visual import drawPlot

import config

#### 
# Utils - can also be used from other files.
####

def create_dir(path, exist_ok=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=exist_ok)
        print("Created Dir '{}'".format(path))
        
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_highest_number():
    list_of_subdirs = [dI for dI in os.listdir(config.EXPERIMENTS_RESULTS_DIR) if os.path.isdir(os.path.join(config.EXPERIMENTS_RESULTS_DIR,dI))]
    
    numbers = [ int(name[:4]) for name in list_of_subdirs if (len(name) > 4 and name[4] == "_" and is_number(name[:4]))] 
    
    if len(numbers) > 0: return max(numbers)
    else: return 0      
    
def check_experiment_configs(experiment_configs):
    assert type(experiment_configs) == list
    assert len(experiment_configs) > 0
    for c in experiment_configs:
        assert "attack_fn" in c.keys()
        assert "attack_arg" in c.keys()
        assert "meta" in c.keys()
        
def save_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=4)  
        
def write_to_file(content, path):
    file_object = open(path, 'a')
    file_object.write(content)
    file_object.close()
    
def save_pickle(obj, path):
    filehandler = open(path, 'wb')
    pickle.dump(obj, filehandler)
    
def load_pickle(path):
    obj = pickle.load( open( path, "rb" ) )
    return obj

def load_experiment(pickle_path = None, exp_folder = None):
    assert pickle_path or exp_folder
    assert not pickle_path and exp_folder
    if exp_folder:
        pickle_path = os.path.join(config.EXPERIMENTS_RESULTS_DIR, exp_folder, "backup.pickle")
    exp = load_pickle(pickle_path)
    print("Loaded model {}".format(exp.folder_name))
    return exp

####

class RobustnessExperiment():
    
    def __init__(self, experiment_configs, title="Experiment", description=""):
        
        check_experiment_configs(experiment_configs)
        
        # creating new directory
        create_dir(config.EXPERIMENTS_RESULTS_DIR, exist_ok=True)
        self.id = get_highest_number() + 1
        self.title = title
        self.folder_name = str(self.id).zfill(4) + "_" + self.title
        self.dir = os.path.join(config.EXPERIMENTS_RESULTS_DIR, self.folder_name)
        self.experiment_configs = experiment_configs
        self.all_results = {}
        create_dir(self.dir)
        
        # back up experiment_configs
        save_json(str(experiment_configs), os.path.join(self.dir, "experiment_configs_#{}.json".format(self.id)))
        
        # save as pickle
        #filehandler = open(os.path.join(self.dir, "experiment_#{}.pickle".format(self.id)), 'wb') 
        #pickle.dump(self, filehandler)
        
        info = "EXPERIMENT {} ".format(self.id)
        info += "\n\nExperiment-Configs {}\n\n".format(str(self.experiment_configs))
        write_to_file(info, os.path.join(self.dir, "summary.txt"))
        
    #
    # evaluates ONE attack for ONE model for a list of attack args.
    # shows plot and saves results
    #
    def evaluate_attack(self, model, loader, attack_class, list_attack_args, meta=None, 
                        results_dir = ".", model_name = "model", title = "title"):
        
        current_attack_report = []
        
        # run attack on model for all attack_args
        for i in tqdm(range(len(list_attack_args))):
            # result = model.report(attack = current_attack, attack_args = configs[c], log=False)
            att = attack_class(model, loader,  list_attack_args[i],
                                  early_stopping=-1, device='cuda', save_samples=False)
            att.attack()
            current_attack_report.append( att.report(log=False) )
        # 
        if meta:

            # draw & save plot
            key_x = meta["key_config"]
            key_y = meta["key_result"]
            xs = [ res[key_x] for res in list_attack_args ] 
            ys = [ res[key_y] for res in current_attack_report ]
            vis_object = [{"data": ys, "color" : "rbgycmk"[0], "label": key_y}]
            drawPlot(x = xs, data = vis_object, x_label = key_x, y_label = key_y, 
                     title = meta["title"] + " | model:" + model_name, 
                     save_path = os.path.join(results_dir, "plot_{}_{}.pdf".format(title,model_name)))

            # write summary file
            info = "Attack: \t" + attack_class.__name__ + "\n"
            info += key_x + "\t\t" + str(xs) + "\n"
            info += key_y + "\t\t" + str(ys) + "\n\n"
            write_to_file(info, os.path.join(self.dir, "summary.txt"))

            # save json
            save_json(current_attack_report, os.path.join(results_dir, "json_{}_{}.json".format(title,model_name)))

        if title not in self.all_results: self.all_results[title] = {"CONFIGS": list_attack_args}
        self.all_results[title][model_name] = current_attack_report
        save_json(self.all_results, os.path.join(self.dir, "exp#{}_all_results.json".format(self.id)))
        self.backup()

        return current_attack_report

    def run(self, model_path, module_class, model_nickname=None):
                
        # load model
        loaded_dict = torch.load(model_path)
        model = module_class(loaded_dict["hparams"])
        model.model.load_state_dict(loaded_dict["state_dict"])
        model.prepare_data()
        
        # create sub directory
        model_name = os.path.basename(os.path.normpath(model_path)) if not model_nickname else model_nickname
        print("\n\n" + "="*60 + "\nRunning experiment on model {}\n".format(model_name) + "="*60)
        results_dir = os.path.join(self.dir, model_name)
        create_dir(results_dir)
        info = ("="*50 + "\n") + "Model: {}\n".format(model_name) + "="*50 + "\n"
        write_to_file(info, os.path.join(self.dir, "summary.txt"))

        
        # evaluate through all attacks
        for a in range(len(self.experiment_configs)):
            
            # get experiment_config for the current attack
            exp_conf = self.experiment_configs[a]
            current_attack = exp_conf["attack_fn"]
            current_attack_arg_search_space = exp_conf["attack_arg"]
            meta = exp_conf["meta"]
            title = current_attack.__name__ + "_(Atk#{}_Exp#{})".format(a, self.id)  if not meta["title"] else meta["title"]
            print("\nPerform Attack #{}/{}: {}".format(a+1, len(self.experiment_configs), title))
            if (title in self.all_results.keys()) and (model_name in self.all_results[title].keys()):
                print("Attack {} has already been performed for model {} - skipping.".format(title, model_name))
                continue

            # create all possible attack args and run them
            configs = []
            for instance in product(*current_attack_arg_search_space.values()):
                configs.append(dict(zip(current_attack_arg_search_space.keys(), instance)))

            current_attack_report = self.evaluate_attack(model.model, model.val_dataloader(), 
                                                         current_attack, configs, meta=meta, 
                                                         results_dir = results_dir, model_name = model_name, title=title)
    def backup(self):
        path = os.path.join(self.dir, "backup.pickle")
        save_pickle(self, path)
        print("Backup created at \"{}\"".format(path))
        
    # 
    # ANALYTICS
    #
    def compare(self, config_key = "epsilon", results_key = "success_rate", models = None):

        if not models:
            print("Only showing the first 7 models! Specify a model string to show specific models!")
            models = list(list(self.all_results.values())[0].keys())[1:8] #first item = CONFIGS > don't plot

        for i, attack in enumerate(self.all_results.keys()):
            xs = [ res[config_key] for res in self.all_results[attack]["CONFIGS"]] 
            vis_objects = []
            for m, model in enumerate(models):
                ys = [ res[results_key] for res in self.all_results[attack][model]]
                vis_object = {"data": ys, "color" : "rbgycmk"[m], "label": model}
                vis_objects.append(vis_object)

            drawPlot(x = xs, data = vis_objects, x_label = config_key, y_label = results_key, 
                     title = attack, 
                     save_path = os.path.join(self.dir, "plot_comparison_{}.pdf".format(attack)))


    def show_evaluated_models(self):
        models = list(list(self.all_results.values())[0].keys())[1:] #first item = CONFIGS > don't plot
        return models
            
    def show_best_models(self, metric = "success_rate", best_n = 1):

        for i, attack in enumerate(self.all_results.keys()):
            print("\nAttack = {}:".format(attack))
            losses = []
            for m, model in enumerate(list(self.all_results[attack].keys())[1:]): #1: 1st item is CONFIGS
                ys = [ res[metric] for res in self.all_results[attack][model]]
                losses.append((sum(ys), model))
            print(sorted(losses)[:best_n])