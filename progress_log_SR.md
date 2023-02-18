## 2023-02-05

### Hyperparameter sweeps
- Parameters to sweep
    alpha
    kappa
    lags
- Strategy
    - Setup an array job that takes in the value of the hyperparameter of interest and passes that into a the python script that calls fit_model with the specific setting of said hyperparameter
    - fit the sLDS part with the setting of this new hyperparameter
    - figure out the number of jobs to run (one per setting of the hyperparameter)
    - generate a pandas csv file with the necessary info
    - use the array ID parameter to index into the pandas csv/dataframe to read out the settings of the hyperparameter
    


### Simulated timeseries

### Cluster states
hierarchically cluster each state A, b, Q 
similarity metric between different dynamics parameters

### Per syllable syllable duration


### Per frame probability distribution over states 