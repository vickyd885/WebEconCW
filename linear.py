import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn import linear_model, neighbors
from sklearn.metrics import accuracy_score

from random import randint

# Datasets
# validate_data = pd.read_csv("small_validation_100.csv")
# training_data = pd.read_csv("small_train_100.csv")
# testing_data = pd.read_csv("small_test_100.csv")

validate_data = pd.read_csv("validation.csv")
training_data = pd.read_csv("datasets/we_data/train.csv")
testing_data = pd.read_csv("datasets/we_data/test.csv")

min_value = validate_data['payprice'].min()
if min_value % 2 == 1:
    min_value = min_value - 1

max_value = validate_data['bidprice'].max()

"""

### Constant Bidding
def constant_bidding(const):
    impressions = 0
    clicks = 0
    total_spend = 0
    budget = 6250000

    for index, row in validate_data.iterrows():
        # Don't exceed budget
        if total_spend >= budget:
        # print("Elapsed budget")
            break

        if const > row['payprice']:
            impressions += 1
            clicks += row['click']
            total_spend += row['payprice']


    return impressions, total_spend, clicks

num_valid_impressions = validate_data.shape[0]

min_value = validate_data['payprice'].min()
if min_value % 2 == 1:
    min_value = min_value - 1

max_value = validate_data['bidprice'].max()

results = pd.DataFrame()
results['constants'] = np.arange(min_value, max_value, 2)

impressions = []
cost = []
clicks = []

print("Starting constant bidding process")

for const in results['constants']:
    #print(const)
    const_impressions, const_cost, const_clicks = constant_bidding(const)
    impressions.append(const_impressions)
    cost.append(const_cost)
    clicks.append(const_clicks)

results['impressions'] = impressions
results['total_spend'] = cost
results['clicks'] = clicks

total_num_clicks = len(validate_data.groupby('click').get_group(1))

results['CTR'] = (results['clicks']/results['impressions'])*100
results['CVR'] = (results['clicks']/total_num_clicks)*100
results['CPM'] = (results['total_spend']/results['impressions'])
results['CPC'] = (results['total_spend']/results['clicks'])*100

print("Finished constant bidding process")
#print(results)
#results.to_csv("output/constant_bid_output.csv")
constant_bidding_results = results

results = []


###### Random Bidding

def random_bidding(min_bound, max_bound):
    impressions = 0
    total_spend = 0
    clicks = 0
    budget = 6250000

    for index, row in validate_data.iterrows():
        bid = randint(min_bound, max_bound)

        # Don't exceed budget
        if total_spend >= budget:
        # print("Elapsed budget")
            break
        if bid > row['payprice']:
            impressions += 1
            clicks += row['click']
            total_spend += row['payprice']

    return impressions, total_spend, clicks

num_valid_impressions = validate_data.shape[0]

min_value = validate_data['payprice'].min()
if min_value % 2 == 1:
    min_value = min_value - 1

max_value = validate_data['bidprice'].max()

results = pd.DataFrame()

results['constants'] = np.arange(min_value, max_value, 2)

impressions = []
cost = []
clicks = []

print("Starting random bidding process")

for const in results['constants']:
    #print(const)
    const_impressions, const_cost, const_clicks = random_bidding(min_value, const)
    impressions.append(const_impressions)
    cost.append(const_cost)
    clicks.append(const_clicks)

results['impressions'] = impressions
results['total_spend'] = cost
results['clicks'] = clicks

total_num_clicks = len(validate_data.groupby('click').get_group(1))

results['CTR'] = (results['clicks']/results['impressions'])*100
results['CVR'] = (results['clicks']/total_num_clicks)*100
results['CPM'] = (results['total_spend']/results['impressions'])
results['CPC'] = (results['total_spend']/results['clicks'])*100


random_bidding_results = results
results = []

print("Finishing random bidding process")
#print(results)
#results.to_csv("random_bid_output.csv")

"""

# We select fields we are interested in
features = ['click', 'weekday', 'hour', 'useragent',
'region', 'city',
 'slotwidth', 'slotheight', 'slotvisibility',
'slotformat']

# Get a list of columns to delete from the dataset
unused_fields = [x for x in list(training_data) if x not in features]


# print(unused_fields)

def delete_unused_columns(raw_df):
    for field in unused_fields:
        if not field in raw_df:
            continue
        raw_df = raw_df.drop(field,axis=1)
    return raw_df


def encode_df(raw_df):
    encoded_df = delete_unused_columns(raw_df) # removes unused fields
    label_encoder = LabelBinarizer() # Uses SKLearn Label Binary Encoder
    for field in features:
        if not field in encoded_df:
            continue
        encoded_df[field] = label_encoder.fit_transform(encoded_df[field])

    return encoded_df



# Encode the DFs
X_train = encode_df(training_data)
X_validate = encode_df(validate_data)
X_test = encode_df(testing_data)

Y_train = training_data.click
Y_validate = validate_data.click

print("Finish feature encoding")

################################################################################################################################

# Declare the logistic model
logistic = linear_model.LogisticRegression()

# Fit the data
logistic.fit(X_train, Y_train)

# Print accuracy by testing against validation set
print('Logistic Regression Accuracy: %f'
      % logistic.score(X_validate, Y_validate))

test_set_predictions_logistic = logistic.predict(X_validate)

print(test_set_predictions_logistic)

################################################################################################################################

# knn = neighbors.KNeighborsClassifier(n_neighbors=2)
#
# knn.fit(X_train, Y_train)
#
# test_set_predictions_knn = knn.predict(X_validate)
#
# #print(test_set_predictions_knn)
# print("KNN Accuracy: ",knn.score(X_validate, Y_validate))

################################################################################################################################

avg_ctr = training_data['click'].sum() / len(training_data.groupby('click').get_group(1))

def linear_bidding(lower_limit, upper_limit, increment, predictions):
    base_bids = np.arange(lower_limit, upper_limit, increment)
    bids = []

    for base_bid in base_bids:
        for i in range(0, len(predictions)):
            bid = base_bid * (predictions[i] / avg_ctr)
            bids.append(bid)

    bid_groups = [bids[x:x+len(predictions)] for x in range(0, len(bids), len(predictions))]
    return bid_groups, base_bids

def placing_bids(bids):
    impressions = 0
    clicks = 0
    cost = 0
    budget = 6250000

    for i in range(0, len(bids)):
        # Don't exceed budget
        if cost >= budget:
        # print("Elapsed budget")
            break

        if bids[i] >= validate_data['payprice'][i]:
            impressions += 1
            clicks += validate_data['click'][i]
            cost += validate_data['payprice'][i]

    return impressions, clicks, cost

def evaluate_bid_strategy(strategy, prediction_Set):
    # min_value = 2
    # max_value = 302
    # increment = 2
    bid_groups, base_bids = linear_bidding(min_value,max_value,2,prediction_Set)

    impressions = []
    total_clicks = []
    total_spent = []

    results_df = pd.DataFrame()

    results_df['bid'] = base_bids
    #results_df['strategy'] = strategy

    print(results_df)

    for bids in bid_groups:
        [imps, clicks, cost] = placing_bids(bids)
        impressions.append(imps)
        total_clicks.append(clicks)
        total_spent.append(cost)

    # Immediate details
    print(len(total_clicks))
    results_df['clicks'] = total_clicks
    results_df['total_spend'] = total_spent
    results_df['impressions'] = impressions

    total_num_of_clicks = len(validate_data.groupby('click').get_group(1))
    # Other metrics
    results_df['CTR'] = (results_df.clicks/results_df.impressions * 100)
    results_df['CPM'] = (results_df.clicks/results_df.impressions)
    results_df['CPC'] = (results_df.total_spend/results_df.clicks * 100)

    return results_df


print("starting linear bidding with LR")
linear_bidding_results_df = evaluate_bid_strategy("linear", test_set_predictions_logistic)

print(linear_bidding_results_df)


################################################################################################################################

# Create DFs with best bid results
# # best_constant_bidding_df = constant_bidding_results.sort_values(by=['clicks'] , ascending=False).iloc[0]
# best_random_bidding_df = random_bidding_results.sort_values(by=['clicks'] , ascending=False).iloc[0]
best_linear_bid_df = linear_bidding_results_df.sort_values(by=['clicks'] , ascending=False).iloc[0]


# # best_constant_bidding_df = best_constant_bidding_df.drop("constants")
# best_random_bidding_df = best_random_bidding_df.drop("constants")
best_linear_bid_df = best_linear_bid_df.drop("bid")

table_df = pd.concat([best_linear_bid_df],1)
#table_df.columns = ['constant', 'random', 'linear']
table_df.columns = ['linear']
table_df = table_df.T
#table_df.column = ['linear']
print(table_df)

table_df.to_csv("initial_results.csv")
