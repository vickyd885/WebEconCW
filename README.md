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

```python
python explore.py
```

Includes:
- Basic stats (num Imps, num Clicks, Cost, CTR, avg CPM, eCPC )
- CTR/Weekday Graph
- CTR/Hour Graph
- CTR/Browser Graph
- CTR/Region Graph
- CTR/Ad Exchange Graph

Graphs are saved in `part1/`

## Running the Neural Network
To run the bidding strategies with Deep Neural Network pCTR estimation, run as follows:

```python
python run_neural_network.py
```

**NOTE** : Ensure the *h5py* Python package is installed to load the Neural Network checkpoint *neural_bid_model.h5* from the current working directory and avoid re-training the model. Also ensure the datasets *validation.csv* and *test.csv* are within the current working directory.