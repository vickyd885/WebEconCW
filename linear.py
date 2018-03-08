import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn import linear_model, neighbors
from sklearn.metrics import accuracy_score


# Datasets
validate_data = pd.read_csv("small_validation.csv")
training_data = pd.read_csv("small_train.csv")
testing_data = pd.read_csv("small_test.csv")

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

knn = neighbors.KNeighborsClassifier(n_neighbors=2) 

knn.fit(X_train, Y_train)

test_set_predictions_knn = knn.predict(X_validate)

#print(test_set_predictions_knn)
print("KNN Accuracy: ",knn.score(X_validate, Y_validate))

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

    valid = bids >= validate_data['payprice']
    for i in range(0, len(valid)):
        # Don't exceed budget
        if cost >= budget:
        # print("Elapsed budget")
            break
            
        if valid[i] == True:
            impressions += 1
            clicks += validate_data['click'][i]
            cost += validate_data['payprice'][i]

    return impressions, cost, clicks

def evaluate_bid_strategy(strategy):
    min_value = 2
    max_value = 302
    increment = 2
    bid_groups, base_bids = linear_bidding(min_value,max_value,2,test_set_predictions_knn)

    impressions = []
    total_clicks = []
    total_spent = []
    
    results_df = pd.DataFrame()
    
    results_df['bid'] = base_bids
    results_df['strategy'] = strategy
    
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
    results_df['CVR'] = (results_df.clicks/total_num_of_clicks * 100)
    results_df['CPM'] = (results_df.clicks/results_df.impressions)
    results_df['CPC'] = (results_df.clicks/results_df.clicks * 100)
    results_df['eCPC'] = (results_df.total_spend/results_df.clicks * 100)
    
    return results_df

linear_bidding_results_df = evaluate_bid_strategy("linear")

print(linear_bidding_results_df)
