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
