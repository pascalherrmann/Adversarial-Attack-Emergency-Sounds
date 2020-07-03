# Emergency Dataset
DATASET1_DATA_8K_DIR = "/nfs/students/summer-term-2020/project-4/data/dataset1/dataset_8k"
DATASET1_DATA_48K_DIR = "/nfs/students/summer-term-2020/project-4/data/dataset1/dataset_48k"

# Control dataset
DATASET2_DATA_8K_DIR = "" # TODO
DATASET2_DATA_48K_DIR = "" # TODO

# default (emergency dataset)
DATA_8K_DIR = DATASET1_DATA_8K_DIR
DATA_48K_DIR = DATASET1_DATA_48K_DIR

# dataset paths as dict 
DATASETS_DIR = {
		'0': {'8000': DATASET1_DATA_8K_DIR, '48000': DATASET1_DATA_48K_DIR},
	 	'1': {'8000': DATASET2_DATA_8K_DIR, '48000': DATASET2_DATA_48K_DIR}
	}

DATASET_EMERGENCY_ID = '0'
DATASET_CONTROL_ID = '1'

# Further paths
LOG_DIR = "/nfs/students/summer-term-2020/project-4/logs"
EXPERIMENTS_RESULTS_DIR = "/nfs/students/summer-term-2020/project-4/EXPERIMENTS"
SAVED_MODELS_DIR = "/nfs/students/summer-term-2020/project-4/SAVED_MODELS"
