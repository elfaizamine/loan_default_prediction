Loan Default Risk
==============================

This project is a kaggle Data Science competition where the goal is to predict if a client will succeed or fail in repaying their loans : 
https://www.kaggle.com/c/home-credit-default-risk

Project Organization
------------

    ├── LICENSE
    ├── Mabrrbkefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── clean          <- clean data that will be used in feature builder and model training
    │   ├── prediction     <- submission csv  file to kaggle competition.
    │   └── raw            <- The original data.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    │
    └── src                                 <- Source code for use in this project.
    │   ├── __init__.py                     <- Makes src a Python module
    │   │
    │   ├── config                          
    │   │   ├── Consts.py                   <- file contains feature, model training parameters, paths, ...
    │   │   └── selectedFeatures.json       <- best selected features
    │   │
    │   ├── data                            
    │   │   ├── DataLoader.py               <- load clean data to be processed by feature builder
    │   │   └── RawDataCleaner.py           <- clean raw data and save files
    │   │
    │   ├── features                        
    │   │   ├── feature files (...).py      <- each data table has a file that contains the features related to the table
    │   |   └── features_selection          <- feature selection algorithms that saves the best optimal features in config
    │   |       └── multiples files.py
    │   │
    │   ├── models                          
    │   │   └── Model.py                    <- Scripts that load train save model and make predictions
    │   │
    │   ├── utils                           
    │   │   ├── Evaluator.py                <- evaluate trained model bu calculating auc
    │   │   └── utils.py                    <- functions that will be used across different features
    |   |
    │   ├── visualization                   
    │   |   └── visualize.py                <- Scripts to create exploratory and results oriented visualizations
    |   |
    │   └── FeatureBuilder                  <- Scripts that create master table (features plus application ID)
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
