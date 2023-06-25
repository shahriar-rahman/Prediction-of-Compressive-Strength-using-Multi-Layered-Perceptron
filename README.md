===========================================================================
# Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron

A predictive model is constructed demonstrating the possibilities of adapting a Multi-Layered 
Perceptron to envisage the compressive strength of high-performance concrete. 

![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/img/git_img%20(1).jpg)

### Introduction:
Several types of research have indicated that concrete strength development is mostly determined by the water-cement ratio 
(w/c ratio) with the amalgamation of other ingredients. Despite displaying a pattern of practical acceptability of this theory, 
there have been some deviations from the norm. Codes consist of various empirical equations that can be applied to achieve 
a proper prediction of compressive strengths, which are usually based on experiments without using supplementary 
cementitious materials such as fly ash, blast furnace slag, and so forth. Therefore, it is crucial to investigate the validity of 
the relationships with the aforementioned materials in order to get better interpretability in circumventing this particular problem. 

### Objective:
The primary incentive of this research is to:
* Initiate an exploratory analysis of data to find the patterns of the feature that makes up the data.
* Conduct a comparative analysis of feature transformation algorithms to find the most suitable for utilization.
* Experiment with different Hyper-parameters to obtain a well-organized tuning for the MLP model.
* Locate a viable approach to solve this problem to develop an efficient model capable of correctly predicting 
the strength of the concrete, given certain parameter exists.

![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/img/git_img%20(2).jpg)

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
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   └── interim        <- Intermediate data that has been transformed.
    │   └── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries  
    │    └── mlp.pkl
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         		initials, and a short `-` delimited description
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── figures            <- Generated graphics and figures to be used in reporting
    │   └── ide_graphs           <- Generated using PyCharm IDE
    │   └── notebook_graphs    <- Generated using Jupyter Notebooks
    │
    ├── img            <- Project related files
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable, so that src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── construct_features.py
    │   │   └──  feature_analysis.py
    │   │   └── feature_transformation.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions         
    │   │   └── mlp_test.py
    │   │   └── mlp_train.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

### Study Flowchart
![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/img/flow_chart.JPG)

### Results
| Data Type | Mean Squared Error | R-Squared Error |
|----------- | --------------------- | ------------------ |
| Training | 0.077 | 0.86 |
| Test     | 0.077 | 0.85 |

* Training Loss Curve:
![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/figures/ide_graphs/training_loss_curve.png)

* Training set Ground truth vs Estimated values:
![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/figures/ide_graphs/training_fit_data.png)

* Training Error Density Plot:
![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/figures/ide_graphs/training_error_distribution.png)

* Test Loss Curve:
![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/figures/ide_graphs/test_loss_curve.png)

* Test set Ground truth vs Estimated values:
![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/figures/ide_graphs/test_fit_data.png)

* Test Error Density Plot:
![alt text](https://github.com/shahriar-rahman/Prediction-of-Compressive-Strength-using-Multi-Layered-Perceptron/blob/main/figures/ide_graphs/test_error_distribution.png)


### Packages and Modules used:
* os
* sys
* math
* pickle
* random
* pandas
* sklearn
* seaborn
* matplotlib
* missingno

===========================================================================

