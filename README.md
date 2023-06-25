===========================================================================
# Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron

A predictive model is constructed demonstrating the possibilities of adapting a Multi-Layered 
Perceptron to envisage the compressive strength of high-performance concrete. 

![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/img/git_img%20(1).jpg)

### Objective:
The primary incentive of this research is to:
* Initiate an exploratory analysis of data to find the patterns of the feature that makes up the data.
* Conduct a comparative analysis of feature transformation algorithms to find the most suitable for utilization.
* Experiment with different Hyper-parameters to obtain a well-organized tuning for the MLP model.
* Locate a viable approach to solve this problem to develop an efficient model capable of correctly predicting 
the strength of the concrete, given certain parameter exists.

### Approach:
This research is classified into 5 steps:
1.	Identifying the problem and its data sources.
2.	Constructing raw data into clean processed data and analyzing it using both Jupyter Notebooks and IDE.
3.	Scaling the data with 3 different transformation algorithms for comparisons and us the best-suited one for this problem.
4.	Experiment and Diagnose in order to achieve the best Hyper-parameters for building an efficient model.
5.	Result Analysis for both training and test data.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
