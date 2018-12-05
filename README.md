### energy-prices-forecasting

Aim of the project is forecasting market energy prices (on Polish Power Exchange) using machine-learning methods like autoencoders and recurrent neural networks. 

Project consists of files:

1. dataset.csv - data used in forecasting
2. feature_processing_and_arimax.R - code generating features and running ARIMAX model with forecasts in a loop
3. featue_engineergin.py - dimensionality reduction of data using autoencoders
4. model_xtb.py - modeling and forecasting energy prices in a loop using variety of ML algorithms and features transformed in previous steps
5. eda_and_eval_report.ipynb - Jupyter notebook generating exploratory analysis and cross-validation in time of forecasts
6. eda_and_eval_report.HTML - generated report

Below there are descriptions of files mentioned above:
 
### dataset.csv 

file consists of features:

RDN - energy prices in Polish Power Exchange
CRO - energy prices in balancing markets
SE - energy prices in Sweden
demand - predicted demand on energy
supply - predicted production of powerplants
wind_prod - predicted production of windparks
reserve - predisvted system energy reserve

those features will be further called "fundamental features"


### feature_processing_and_arimax.R 

this code is responsible for two tasks:

Firstly, it generates any new features based on prices time-series : lags of prices in time, moving average, moving standard deviations, some financial ratios etc. Those features will be further called "time-series features". In next step data is divided into train (first 600 observations) and test (144 observations) sets.

Secondly, ARIMAX model of energy prices is built and forecasts are made in a loop for every day of test datase (updating train dataset of every day from test data with every iteration).


### featue_engineergin.py 
Because "time-series features" are highly correlated (with prices and between themselves), dimensionality reduction were used. Previously separated train dataset were divided into new train and new test datasets to train and evaluate PCA and few type of autoencoders (evaluation was define as building linear regression forecasting prices on reduced and original features, and comparing models accuracy using R^2 measure). After that, all data were transformed using most efficient metods of dimensionality reduction.


# model_xtb.py 
Data transformed in previous step were used to built models forecasting energy prices - random forests, linear regression with different penalties, multi-layer perceptrons and recurrent neural networks with different parameters. Again, all forecasts were calculated in a loop, using the same method as in (2)
