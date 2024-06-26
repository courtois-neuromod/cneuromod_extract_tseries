cneuromod_extract_tseries
==============================

Timeseries extraction for Courtois-Neuromod fMRI datasets.

Documentation on how to use this tool can be found [here](https://github.com/courtois-neuromod/cneuromod_extract_tseries/blob/main/docs/overview.rst)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── atlases            <- Where the atlases are saved
    ├── data               <- Where the CNeuroMod fMRI datasets are installed
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── output             <- Where extracted timeseries are saved
    ├── setup.py           <- makes project pip installable (pip install -e .) so timeseries can be imported
    ├── timeseries         <- Scripts to denoise and extract fMRI timeseries.
    │   ├── __init__.py    <- Makes timeseries a Python module
    │   │
    │   ├── config         <- Where hydra config files (.yaml) are saved
    │   │   ├── dataset
    │   │   │   ├── friends.yaml
    │   │   │   ├── mario3.yaml
    │   │   │   ├── movie10.yaml
    │   │   │   └── shinobi.yaml
    │   │   │   
    │   │   ├── denoise
    │   │   │   ├── simple.yaml
    │   │   │   ├── simple+gsr.yaml
    │   │   │   ├── scrubbing.2.yaml
    │   │   │   └── scrubbing.2+gsr.yaml
    │   │   │       
    │   │   ├── parcellation
    │   │   │   ├── fLocFFA.yaml
    │   │   │   ├── langToneva_AngularG.yaml
    │   │   │   ├── npythyV1.yaml
    │   │   │   ├── ward10k_T1w.yaml
    │   │   │   ├── yeo7nets_DMN.yaml
    │   │   │   ├── schaefer1000.yaml
    │   │   │   └── mist444.yaml
    │   │   │
    │   │   └── base.yaml
    │   │
    │   ├── extract.py
    │   ├── run.py
    │   └── utils.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
