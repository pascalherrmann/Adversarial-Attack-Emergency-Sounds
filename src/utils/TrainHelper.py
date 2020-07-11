import os
import torch
from itertools import product
import pytorch_lightning as pl
from pytorch_lightning import loggers

from utils.RobustnessExperiment import create_dir, save_json, write_to_file, save_pickle, load_pickle, load_module
from attacks.attack import Attack
from datasets.datasethandler import DatasetHandler
import config
import datetime

from pytorch_lightning.callbacks import Callback

class SaveCallback(Callback): 
    def __init__(self, save_epochs, model_name):
        super().__init__()
        self.save_epochs = save_epochs
        self.model_name = model_name

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) in self.save_epochs:
            save_path = self.model_name + "_v{}_epoch_{}.p".format(trainer.logger.version, trainer.current_epoch + 1)
            pl_module.save(save_path)
            print("Saved checkpoint at epoch {} at \"{}\"".format((trainer.current_epoch + 1), save_path))

def check_attack_args(attack_args):
    assert type(attack_args) == list
    assert len(attack_args) > 0
    for a in attack_args:
        assert "ATTACK_CLASS" in a.keys()
        assert issubclass(a["ATTACK_CLASS"], Attack)
        
        assert "META" in a.keys()
        
        assert "ATTACK_ARGS" in a.keys()
        #assert type(a["ATTACK_ARGS"]) == list
        
        '''
        for aa in a["ATTACK_ARGS"]:
            for val in aa.values():
                assert type(val) == list
        '''
'''
Idea:
* we have one shared directory ~/saved_models, that contains ALL models that have been trained 
* this directory has a subfolder for each architecture
* by calling TrainHelper.run() you run adversrial training for ONE model for different configurations (e.g., attack mode)
    * because training hparams are usually specific to model type
    
* all results will be stored in a central json: all_models = {"M5": { M5... } }
    * Model-Architecture
        * Arc_[adv]_ModelID__e123
            * Results (at least val_acc)
            * 
'''

def get_all_models_data():
    global_all_results_path = os.path.join(config.SAVED_MODELS_DIR, "GLOBAL_ALL_RESULTS.p")
    return load_pickle(global_all_results_path)

def get_best_models():
    global_models = get_all_models_data()
    
def get_model_info(model_name):
    
    if model_name[-2:] != ".p":
        model_name = model_name + ".p"
    
    global_models = get_all_models_data()
    for model_results in global_models.values():
        if model_name in model_results:
            return model_results[model_name]
    print("No information for model \"{}\" found.".format(model_name))
    

class TrainHelper():
    
    def __init__(self):
        create_dir(config.SAVED_MODELS_DIR)
        self.log_dir = os.path.join(config.SAVED_MODELS_DIR, "logs")
        create_dir(self.log_dir)
        self.datasetHandler = DatasetHandler()
         
    # run adversarial training for one model for a list of attacks with different arguments
    def run(self, model_class, hparams, attack_configs, save_epochs = [50, 100, 150, 200, 250]):
        
        # sanity check of arguments
        check_attack_args(attack_configs)
        
        new_model_paths = []
        
        datasetHandler = self.datasetHandler 
        print("loaded!")
        
        for a, attack_config in enumerate(attack_configs):
            
            # get experiment_config for the current attack
            current_attack = attack_config["ATTACK_CLASS"]
            current_attack_arg_search_space = attack_config["ATTACK_ARGS"]
            meta = attack_config["META"]
            
            print(("="*60 + "\nTraining Models with Attack {} ({} of {})\n" + "="*60).format(current_attack.__name__, a+1, len(attack_configs)))

            # all possible attack args
            list_attack_args = []
            for instance in product(*current_attack_arg_search_space.values()):
                list_attack_args.append(dict(zip(current_attack_arg_search_space.keys(), instance)))
                   
            # train model
            for a_index, attack_args in enumerate(list_attack_args):
                print("-"*60 + "\nTrainig model {}/{}\n".format(a_index+1, len(list_attack_args)) + "-"*60)

                model = model_class(hparams)
                model.prepare_data()
                datasetHandler.load(model, 'training')
                datasetHandler.load(model, 'validation') 
                model.setAttack(current_attack, attack_args)
                
                model_title = type(model.model).__name__ + "_attack_" + (current_attack.__name__ if not meta["TITLE"] else meta["TITLE"])
                current_dir = os.path.join(config.SAVED_MODELS_DIR, type(model.model).__name__)
                create_dir(current_dir)
                cb = SaveCallback(save_epochs, os.path.join(current_dir, model_title))

                trainer = pl.Trainer(
                    max_epochs=hparams["epochs"],
                    logger = loggers.TensorBoardLogger(self.log_dir, name=type(model.model).__name__),
                    gpus=1 if torch.cuda.is_available() else None,
                    log_gpu_memory='all',
                    callbacks = [cb]
                )
                
                trainer.fit(model)
                
                model_class_name = type(model.model).__name__
                model_name = model_title + "_v" + str(trainer.logger.version) + ".p"
                model_path = os.path.join(current_dir, model_name)
                model.save(model_path)
                
                # do evaluation
                result_dict = {"path": model_path,
                               "hparams": hparams, 
                               "attack_args": attack_args,
                               "final_val_acc": model.val_results_history[-1]["val_acc"]}
   
                # load existing global history
                global_all_results_path = os.path.join(config.SAVED_MODELS_DIR, "GLOBAL_ALL_RESULTS.p")
                
                if os.path.isfile(global_all_results_path):
                    GLOBAL_ALL_RESULTS = load_pickle(global_all_results_path)
                else: GLOBAL_ALL_RESULTS = {}
                if model_class_name not in GLOBAL_ALL_RESULTS: GLOBAL_ALL_RESULTS[model_class_name] = {}
                GLOBAL_ALL_RESULTS[model_class_name][model_name] = result_dict
                save_pickle(GLOBAL_ALL_RESULTS, global_all_results_path)

                new_model_paths.append(model_path)
                
        print("="*60 + "\nTrained Models:\n" +"="*60)
        print(new_model_paths)
        return new_model_paths