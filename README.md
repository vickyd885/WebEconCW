# WebEconCW

Repo for WebEconomics Coursework

## Installation & Set up

Download the dataset (link in PDF) and save it to `datasets/we_data/`

Create a virtual env for dependency management and activate it

Install dependencies from the requirements file using pip. (it includes some useful libraries pandas, numpys, etc..)

```python
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Once installed, you can launch Jupyter Notebook calling `$ jupyter notebook` and select the `Bidding Models` notebook for part 2, 3, 4 and 5.

Initial data exploration is in `/part1`

## Part 1: Data exploration

Found in '/part1'

```python
python explore.py
```

## Part 2: Constant and Random Bidding

Found in '/part2'

## Part 3: Linear Bidding

```
Logistic regression: linear_bidding.py
Feature engineering: feature_engineering_tests.py
Feature importance: feature-importance.py
Logistic regression fine tuning: lr_fine_tuning.py
```

### Running the Neural Network
To run the bidding strategies with Deep Neural Network pCTR estimation, run as follows:

```python
python run_neural_network.py
```

**NOTE** : Ensure the *h5py* Python package is installed to load the Neural Network checkpoint *neural_bid_model.h5* from the current working directory and avoid re-training the model. Also ensure the datasets *validation.csv* and *test.csv* are within the current working directory.

# Part 4: Non-linear bidding

### ORTB

Found in `ortb.py` (and also in `linear_bidding.py`)

### Lift Bidding

Found in `lift_bidding.py`

### GDBT

Found in `XGB.py`

# Part 5:
Found in `combined_strategy.py`
