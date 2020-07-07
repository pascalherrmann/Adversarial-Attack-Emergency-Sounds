# Project 4: Adversarial Examples for Emergency Vehicle Detection

## Structure
```
.
├── README.md
├── experiments  # Experiment result and Jupyter notebooks 
│   └── notebooks # Distinct folders for each member of the group
│       ├── omer
│       ├── pascal
│       └── yan
│           └── 1.3-yan-e2e-approach.ipynb
└── src # Library
    ├── attacks # Attack module
    ├── classification
    │   ├── eval # Inference and evaluation scripts
    │   ├── h_parameters.json  # Config file for experiment
    │   ├── models # PyTorch model files
    │   ├── trainer # Training loop, optimization, loss, logger, etc.
    │   └── utils # Others
    └── datasets # Extends PyTorch dataset class
```