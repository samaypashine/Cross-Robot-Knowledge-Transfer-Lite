# 
Cross-Robot-Knowledge-Transfer-Lite
Kinova Gen3 Lite Dataset experiments

## Installation

`Python 3.10` and `MATLAB R2022b` are used for development.

```
git clone https://github.com/gtatiya/paper5.git
cd paper5
pip install -e .
```

### MATLAB

[Install MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) <br>
MATLAB Dependencies: Statistics and Machine Learning Toolbox

## Dataset

- Download the dataset and create a symbolic link to it as `data`: <br>

Windows: `mklink "data" "<path to dataset>"` <br>
Linux: `ln -s "<path to dataset>" "data"`

- Create dataset: `python data_processing/create_dataset.py`
- Discretize data: `python data_processing/discretize_data.py`
- Autoencode data: `python data_processing/autoencode_data.py`
- Plot data: `python data_processing/plot_data.py`
- Plot features: `python data_processing/plot_features.py`

## Learn

- Learn object recognition: `python learn/classify_objects.py`
- Learn object recognition: `python learn/classify_objects_v2.py`
- Learn tool recognition: `python learn/classify_tools.py`

## Knowledge Transfer

- Transfer robot knowledge:
```
python transfer/robot_knowledge.py -increment-train-objects -num-folds 10 -augment-trials 10
python transfer/robot_knowledge.py -increment-train-objects -augment-trials 10 -across tools -feature autoencoder-linear-tl
```

## Analyze Results and Plots

- Analyze results and plots: `python analyze/transfer_results.py`
- Plot KEMA features: `python transfer/plot_kema_features.py`
