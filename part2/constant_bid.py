import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv("small_train.csv")
test_data = pd.read_csv("small_test.csv")
validate_data = pd.read_csv("validation.csv")
# validate_data = pd.read_csv("small_validation.csv")

def constant_bidding(const):
	impressions = 0
	clicks = 0
	cost = 0
	budget = 6250000

	for index, row in validate_data.iterrows():
		if const > row['payprice']:
			impressions += 1
			clicks += row['click']
			cost += row['payprice']
		if cost >= budget:
			# print("Elapsed budget")
			break

	return impressions, cost, clicks

num_valid_impressions = train_data.shape[0]

min_value = train_data['payprice'].min()
if min_value % 2 == 1:
	min_value = min_value - 1

max_value = train_data['bidprice'].max()

results = pd.DataFrame()
results['constants'] = np.arange(min_value, max_value, 2)

impressions = []
cost = []
clicks = []

for const in results['constants']:
	print(const)
	const_impressions, const_cost, const_clicks = constant_bidding(const)
	impressions.append(const_impressions)
	cost.append(const_cost)
	clicks.append(const_clicks)

results['impressions'] = impressions
results['cost'] = cost
results['clicks'] = clicks

total_num_clicks = len(train_data.groupby('click').get_group(1))

results['CTR'] = (results['clicks']/results['impressions'])*100
results['CVR'] = (results['clicks']/total_num_clicks)*100
results['CPM'] = (results['cost']/results['impressions'])
results['eCPC'] = (results['cost']/results['clicks'])*100 

results.to_csv("file.csv")