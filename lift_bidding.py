import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn import linear_model, neighbors
from sklearn.metrics import accuracy_score

from random import randint
import random
from xgboost import XGBClassifier, XGBRegressor


### Data Imports

# validate_data = pd.read_csv("small_validation_100.csv")
# training_data = pd.read_csv("small_train_100.csv")
# testing_data = pd.read_csv("small_test_100.csv")

validate_data = pd.read_csv("validation.csv")
training_data = pd.read_csv("datasets/we_data/train.csv")
testing_data = pd.read_csv("datasets/we_data/test.csv")

# Random function with weights
def random_weighted(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

AW_LENGTH = 3 # Action Window in DAYS

# Sets the Action Window accounting for wrap around
def set_aw(hour, weekday):
    if weekday + AW_LENGTH >= 7:
        return (hour, (weekday + AW_LENGTH) % 7)
    else:
        return (hour, weekday + AW_LENGTH)


# We select fields we are interested in
features = [ 'weekday', 'hour',
'region', 'city',
 'slotwidth', 'slotheight', 'slotvisibility',
'slotformat']

# Get a list of columns to delete from the dataset
unused_fields = [x for x in list(training_data) if x not in features]

# Removes fields not needed for encoding
def delete_unused_columns(raw_df):
    for field in unused_fields:
        if not field in raw_df:
            continue
        raw_df = raw_df.drop(field,axis=1)
    return raw_df

# Encodings the fields
def encode_df(raw_df):
    encoded_df = delete_unused_columns(raw_df) # removes unused fields
    label_encoder = LabelBinarizer() # Uses SKLearn Label Binary Encoder
    for field in features:
        if not field in encoded_df:
            print(field, "not found")
            continue
        encoded_df[field] = label_encoder.fit_transform(encoded_df[field])
    return encoded_df

# most frequent user = 5ac7ec84bb700b7a6bd1c57b1ae7c269af65850b
"""
First split the dataset into single users (initial training) and groupable users

"""

# Group by users
grouped_by_users = training_data.groupby("userid").size().reset_index(name='count').sort_values(by=['count'] , ascending=False)


single_users = grouped_by_users[grouped_by_users['count'] == 1]

print("Single user count: ", len(single_users))


list_of_single_users = list(single_users['userid'])

single_data = training_data[training_data['userid'].isin(list_of_single_users)]

groupable_users = grouped_by_users[grouped_by_users['count'] > 1]

list_of_group_users = list(groupable_users['userid'])

group_data = training_data[training_data['userid'].isin(list_of_group_users)]

print("Group user count: ", len(list_of_group_users))
print("Number of clicks:", group_data.click.sum())

#print(groupable_users.head())



"""
Train on the single users initially
"""

## Initial Decision Tree - training on the single users subset
model = XGBRegressor()

X_train = encode_df(single_data)
Y_train = single_data.click

model.fit(X_train, Y_train)

X_validate = encode_df(validate_data)

y_pred = model.predict(X_validate)

Y_validate = validate_data.click



predictions_t = [round(value) for value in y_pred]
print(y_pred)
accuracy = accuracy_score(Y_validate, predictions_t)
print("Initial accuracy: %.2f%%" % (accuracy * 100.0))


pd.DataFrame(y_pred).to_csv("initial_xgb.csv")


print("INITIAL MODEL", model)



#sys.exit(0)
"""
Now lets use the groupable users using the lift bidding method
"""
users_list = list(groupable_users['userid'])
weightings = list(groupable_users['count'])

total_users = len(groupable_users['userid'])
total_clicks = group_data['click'].sum()
print("Termination condition: ", total_clicks)

samples = 0

while(False and samples != total_clicks):

    # Select randomly weighted by ad request frequency
    selected_user_id = users_list[random_weighted(weightings)]

    # Filter df by the userid
    user_df = training_data[training_data['userid'] == selected_user_id]

    # Randomly pick a time stamp
    ts = user_df.sample(1)

    # Extract row number, use this to look at all future entries
    ts_index = ts.index[0]

    # Extract hour and weekday
    ts_hour = ts.hour.values[0]
    ts_weekday = ts.weekday.values[0]

    # TS + AW ,
    aw = set_aw(ts_hour, ts_weekday)

    # Get remaining entries past the row index
    #print(user_df)
    #print(ts_index)
    remaining_hours = user_df
    #print(remaining_hours)
    # Greater than ts, but less than aw
    query_string = "weekday < " + str(aw[1]) + " and hour < " + str(aw[0])

    windowed_df = remaining_hours.query(query_string)

    # If the query returns a non-empty DF, train the model with it!
    if windowed_df.size > 0:
        X_local_train = encode_df(windowed_df)
        Y_local_train = windowed_df.click

        model.fit(X_local_train, Y_local_train)

        samples += 1

    #print(samples)


y_pred = model.predict(X_validate)
print(y_pred)
predictions_t = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_validate, predictions_t)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


pCTR = pd.DataFrame(y_pred)


print("FINAL MODEL", model)
# a = len(training_data) / 2 * np.bincount(training_data.click)
# w = a[1] / a[0]
#
# for p in pCTR[1]:
#     predictions.append( p / (p + ((1-p)/w)))

pCTR.to_csv("lift_bidding_raw_predictions.csv")


#####

avg_ctr = training_data['click'].sum() / len(training_data['bidid'])
avg_cpc = training_data['payprice'].sum() /  training_data['click'].sum()

def lift_bidding(lower_limit, upper_limit, increment, predictions):
    base_bids = np.arange(lower_limit, upper_limit, increment)
    bids = []

    for base_bid in base_bids:
        for i in range(0, len(predictions)):
            #print("Base bid: %f, pCTR: %f, avg_ctr: %f",base_bid, predictions[i], avg_ctr)
            bid = base_bid * (predictions[i] / avg_ctr)
            bids.append(bid)
            #print("Prediction: ", predictions[i])
            #print("New bid rn: ", bid)
            #print("* CPC ", bid * avg_cpc )
            #print("/ avg CTR", bid / avg_ctr)

    bid_groups = [bids[x:x+len(predictions)] for x in range(0, len(bids), len(predictions))]
    return bid_groups, base_bids


def placing_bids(bids):
    impressions = 0
    clicks = 0
    cost = 0
    budget = 6250000
    bool_check = bids >= validate_data.payprice
    for i in range(0, len(bids)):
        # Don't exceed budget
        if cost >= budget:
        # print("Elapsed budget")
            break

        if bool_check[i] == True:
            impressions += 1
            clicks += validate_data['click'][i]
            cost += validate_data['payprice'][i]
            #print(validate_data['payprice'][i])

    return impressions, clicks, cost

def evaluate_bid_strategy(strategy, prediction_Set):
    min_value = 2
    max_value = 302
    increment = 2
    print("Generating bids")
    bid_groups, base_bids = lift_bidding(min_value,max_value,2,prediction_Set)
    print("Finished generating bids")

    impressions = []
    total_clicks = []
    total_spent = []

    results_df = pd.DataFrame()

    results_df['bid'] = base_bids
    #results_df['strategy'] = strategy

    # print(results_df)

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

results_df = evaluate_bid_strategy("lift bidding", y_pred)


best_lift_bid_df = results_df.sort_values(by=['clicks'] , ascending=False).iloc[0]


# # best_constant_bidding_df = best_constant_bidding_df.drop("constants")
# best_random_bidding_df = best_random_bidding_df.drop("constants")
best_lift_bid_df = best_lift_bid_df.drop("bid")

table_df = pd.concat([best_lift_bid_df],1)
#table_df.columns = ['constant', 'random', 'linear']
table_df.columns = ['lift bidding']
table_df = table_df.T
#table_df.column = ['linear']
print(table_df)
