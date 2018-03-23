import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn import linear_model, neighbors
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin

from sklearn import metrics
from random import randint
import random
import xgboost as xgb


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

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


# Group by users
grouped_by_users = training_data.groupby("userid").size().reset_index(name='count').sort_values(by=['count'] , ascending=False)

groupable_users = grouped_by_users[grouped_by_users['count'] > 1]

list_of_group_users = list(groupable_users['userid'])

group_data = training_data[training_data['userid'].isin(list_of_group_users)]

#training_data = group_data

# print(training_data)
# Initial Preprocessing to make the datasets feature encodable

# Splits useragent feild to OS and Browser
def extract_useragent_data(raw_df):
    raw_df['os'] = raw_df['useragent'].apply(lambda x: x.split('_')[0])
    raw_df['browser'] = raw_df['useragent'].apply(lambda x: x.split('_')[1])
    return raw_df

# Groups the continous slot prices for easier encoding
def group_slot_prices(raw_df):
    price_brackets = pd.DataFrame()
    price_brackets['slotprice_brackets'] = pd.cut(raw_df.slotprice.values,5, labels=[1,2,3,4,5])
    raw_df = pd.concat([raw_df,price_brackets],axis=1)
    raw_df = raw_df.drop('slotprice', axis=1)
    return raw_df

def preprocess(raw_df):
    p_df = extract_useragent_data(raw_df)
    #p_df = group_slot_prices(p_df)
    #print(list(p_df))
    return p_df

training_data = preprocess(training_data)
validate_data = preprocess(validate_data)
testing_data = preprocess(testing_data)

print("FINISHED PREPROCESSING")


# We select fields we are interested in
# Note that we do infact encode the useragent, because it's first mapped
# to OS and Browser fields. The useragent field itself then gets deleted
features = [ 'weekday', 'hour',
'region', 'city',
 'slotwidth', 'slotheight', 'slotvisibility',
'slotformat','advertiser', 'adexchange', 'slotprice', 'os', 'browser',
'usertag']


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

    # print("Fields in encoded df:")
    # print(list(encoded_df))
     # Uses SKLearn Label Binary Encoder

    for field in features:
        encoded_df[field] = label_encoder.fit_transform(encoded_df[field])
    return encoded_df


nonnumeric_columns = ['os','browser',  'slotvisibility', 'slotformat', 'usertag']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = training_data[features].append(validate_data[features])

#print(list(big_X))

big_X_imputed = DataFrameImputer().fit_transform(big_X)
#print(big_X_imputed.head(100))

label_encoder = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = label_encoder.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
X_train = big_X_imputed[0:training_data.shape[0]].as_matrix()
X_validate = big_X_imputed[training_data.shape[0]::].as_matrix()

Y_train = training_data['click']
Y_validate = validate_data.click


print("Finish feature encoding")
# most frequent user = 5ac7ec84bb700b7a6bd1c57b1ae7c269af65850b

model = xgb.XGBClassifier()

model = model.fit(X_train, Y_train)

y_pred = model.predict_proba(X_validate)
fpr, tpr , thresholds = metrics.roc_curve(Y_validate,y_pred[1])
metrics.auc(fpr, tpr)

# predictions_t = [round(value) for value in y_pred]
# accuracy = accuracy_score(Y_validate, predictions_t)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))



a = len(training_data) / 2 * np.bincount(training_data.click)
w = a[1] / a[0]
#
predictions = []
for p in pd_df[1]:
    predictions.append( p / (p + ((1-p)/w)))

print(predictions)

#####

avg_ctr = training_data['click'].sum() / len(training_data['bidid'])
avg_cpc = training_data['payprice'].sum() /  training_data['click'].sum()
print("Avg CPC: ", avg_cpc)
print("Avg CTR: ", avg_ctr)

def lift_bidding(lower_limit, upper_limit, increment, predictions):
    base_bids = np.arange(lower_limit, upper_limit, increment)
    bids = []

    for base_bid in base_bids:
        for i in range(0, len(predictions)):
            #print("Base bid: %f, pCTR: %f, avg_ctr: %f",base_bid, predictions[i], avg_ctr)
            bid = base_bid * (predictions[i] * avg_cpc )
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

results_df = evaluate_bid_strategy("lift bidding", predictions)


best_lift_bid_df = results_df.sort_values(by=['clicks'] , ascending=False).iloc[0]


# # best_constant_bidding_df = best_constant_bidding_df.drop("constants")
# best_random_bidding_df = best_random_bidding_df.drop("constants")
#best_lift_bid_df = best_lift_bid_df.drop("bid")

table_df = pd.concat([best_lift_bid_df],1)
#table_df.columns = ['constant', 'random', 'linear']
table_df.columns = ['lift bidding']
table_df = table_df.T
#table_df.column = ['linear']
print(table_df)


def create_test_file_lift_bidding(base_bid, testing_predictions):
    new_df = pd.DataFrame()
    bids = []

    for i in range(0,len(testing_predictions)):
        bid = base_bid * (testing_predictions[i] / avg_ctr)
        bids.append(bid)

    new_df['bidid'] = testing_data['bidid']
    new_df['bidprice'] = bids

    new_df.to_csv("testing_bidding_price_lift_bidding.csv", index=False)

big_X_2 = training_data[features].append(testing_data[features])


big_X_imputed_2 = DataFrameImputer().fit_transform(big_X_2)


for feature in nonnumeric_columns:
    big_X_imputed_2[feature] = label_encoder.fit_transform(big_X_imputed_2[feature])

# Prepare the inputs for the model
X_train = big_X_imputed_2[0:training_data.shape[0]].as_matrix()
X_test = big_X_imputed_2[training_data.shape[0]::].as_matrix()


testing_predictions = model.predict(X_test)

best_base_bid = best_lift_bid_df['bid']

create_test_file_lift_bidding(best_base_bid, testing_predictions)
