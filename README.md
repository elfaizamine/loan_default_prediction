Loan Default Risk
==============================

This project is a kaggle Data Science competition where the goal is to predict if a client will succeed or fail in repaying their loans : 
https://www.kaggle.com/c/home-credit-default-risk.

To download data, you must enter the competition and go to https://www.kaggle.com/c/home-credit-default-risk/data.

With this version of Project, current kaggle score is 0.7422

Project Organization
------------

    ├── main.py                             <- Script to run to clean data, generate features, train model, predict on test data and generate kaggle submission file
    │
    ├── src                                 <- Source code for use in this project. (All important code is in this file)
    │   ├── __init__.py                     <- Makes src a Python module
    │   │
    │   ├── config                          
    │   │   ├── Consts.py                   <- File contains feature, model training parameters, paths, ...
    │   │   └── selectedFeatures.json       <- Best selected features
    │   │
    │   ├── data                            
    │   │   ├── DataLoader.py               <- Load clean data to be processed by feature builder
    │   │   └── RawDataCleaner.py           <- Clean raw data and save files
    │   │
    │   ├── features                        <- Features code 
    │   │   ├── feature files (...).py      <- Each data table has a file that contains the features related to the table
    │   |   └── features_selection          <- Feature selection algorithms that saves the best optimal features in config
    │   |       └── multiples files.py
    │   │
    │   ├── models                          
    │   │   └── Model.py                    <- Scripts that load train save model and make predictions
    │   │
    │   ├── utils                           
    │   │   ├── Evaluator.py                <- evaluate trained model bu calculating auc
    │   │   └── utils.py                    <- Functions that will be used across different features
    |   |
    │   ├── visualization                   
    │   |   └── visualize.py                <- Scripts to create exploratory and results oriented visualizations
    |   |
    │   └── FeatureBuilder                  <- Scripts that create master table (features plus application ID)
    │
    ├── data_sample                         <- Only one row of data is used
    │   ├── clean                           <- Clean data that will be used in feature builder and model training
    │   ├── prediction                      <- Submission csv  file to kaggle competition.
    │   └── raw                             <- The original data.
    │
    ├── notebooks                           <- Jupyter notebooks
    │
    ├── models                              <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports                             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                         <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt                    <- The requirements file for reproducing the analysis environment
    │
    ├── Makefile                            <- Makefile with commands like `make data` or `make train`
    │
    ├── README.md                           <- The top-level README for developers using this project.
    │
    └── LICENSE

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
