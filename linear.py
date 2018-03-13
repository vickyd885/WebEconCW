import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn import linear_model, neighbors
from sklearn.metrics import accuracy_score

from random import randint

#Datasets
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
features = [ 'weekday', 'hour',
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
            print(field, "not found")
            continue
        encoded_df[field] = label_encoder.fit_transform(encoded_df[field])


    return encoded_df


# # Encode the DFs
X_train = encode_df(training_data)
X_validate = encode_df(validate_data)
X_test = encode_df(testing_data)
#
Y_train = training_data.click
Y_validate = validate_data.click

print("Finish feature encoding")
################################################################################################################################


####
# Declare the logistic model
logistic = linear_model.LogisticRegression(class_weight='balanced', C = 0.001)

# Fit the data
ysef = [x for x in list(X_train) if x not in X_validate]
print(ysef)

logistic.fit(X_train, Y_train)

# Print accuracy by testing against validation set
print('Logistic Regression Accuracy: %f'
      % logistic.score(X_validate, Y_validate))

test_set_predictions_logistic = logistic.predict_proba(X_validate)

pCTR = pd.DataFrame(test_set_predictions_logistic)

predictions = []

a = len(training_data) / 2 * np.bincount(training_data.click)
w = float(a[1]) / float(a[0])

for p in pCTR[1]:
    predictions.append( p / (p + ((1-p)/w)))

pCTR.to_csv("predicted_values_lr.csv")
pd.DataFrame(predictions).to_csv("adjusted_pred_lr.csv")


#print(test_set_predictions_logistic)

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

avg_ctr = float(training_data['click'].sum()) / float(len(training_data['bidid']))


print("Total impressions: ", len(training_data['bidid']), " Total clicks: ",training_data['click'].sum() )
print("Avg ctr: ", avg_ctr)

def linear_bidding(lower_limit, upper_limit, increment, predictions):
    base_bids = np.arange(lower_limit, upper_limit, increment)
    bids = []

    for base_bid in base_bids:
        for i in range(0, len(predictions)):
            #print("Base bid: %f, pCTR: %f, avg_ctr: %f",base_bid, predictions[i], avg_ctr)
            bid = base_bid * (predictions[i] / avg_ctr)
            bids.append(bid)
            #print("New bid: ", bid)

    bid_groups = [bids[x:x+len(predictions)] for x in range(0, len(bids), len(predictions))]
    return bid_groups, base_bids

def ortb_bid_1(predictions):
    bids_ortb = []
    c_lambda = []

    lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    c_range = np.arange(10, 90, 10)

    for c in c_range:
        print(c)
        for l in lambdas:
            print(l)
            c_lambda.append((c, l))
            bid = np.sqrt((np.multiply(np.divide(c, l), predictions)) + np.square(c) - c)
            bids_ortb.append(bid.tolist())

    return bids_ortb, c_lambda

def ortb_bid_2(predictions):
    bids_ortb = []
    c_lambda = []

    lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    c_range = np.arange(10, 90, 10)

    for c in c_range:
        print(c)
        for l in lambdas:
            print(l)
            c_lambda.append((c, l))

            ################################################
            # ORTB Winning Formula 2 Formula Breakdown #
            ################################################
            part1 = np.cbrt(np.divide(predictions + np.sqrt(np.multiply(np.square(c), np.square(l)) + np.square(predictions)), np.multiply(c, l)))
            part2 = np.cbrt(np.divide(np.multiply(c, l), predictions + np.sqrt(np.multiply(np.square(c),np.square(l)) + np.square(predictions))))
            form = part1 - part2
            bid = np.dot(c, form)

            bids_ortb.append(bid.tolist())

    return bids_ortb, c_lambda

def placing_bids(bids):
    impressions = 0
    clicks = 0
    cost = 0
    budget = 6250000
    bool_check = bids >= validate_data.payprice
    for i in range(0, len(bids)):
        # Don't exceed budget
        if (cost + validate_data['payprice'][i]) >= budget:
        # print("Elapsed budget")
            break

        if bool_check[i] == True:
            impressions += 1
            clicks += validate_data['click'][i]
            cost += validate_data['payprice'][i]
            #print(validate_data['payprice'][i])

    return impressions, clicks, cost

def evaluate_bid_strategy(strategy, prediction_Set):
    bid_groups = None
    results_df = pd.DataFrame()

    if(strategy == "linear"):
        min_value = 2
        max_value = 302
        increment = 2
        print("Generating bids")
        linear_bid_groups, base_bids = linear_bidding(min_value,max_value,2,prediction_Set)
        print("Finished generating bids")
        results_df['bid'] = base_bids
        bid_groups = linear_bid_groups
    elif (strategy == "ortb"):
        print("Generating ORTB bids")
        ortb_bids, c_lambda = ortb_bid_1(prediction_Set)
        print("Finished generating ORTB bids")
        results_df['c', 'lambda'] = c_lambda
        bid_groups = ortb_bids
    elif (strategy == "ortb2"):
        print("Generating ORTB_2 bids")
        ortb_bids, c_lambda = ortb_bid_2(prediction_Set)
        print("Finished generating ORTB_2 bids")
        results_df['c', 'lambda'] = c_lambda
        bid_groups = ortb_bids

    impressions = []
    total_clicks = []
    total_spent = []

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


print("Starting linear bidding with LR")
linear_bidding_results_df = evaluate_bid_strategy("linear", predictions)
ortb_bidding_results_df = evaluate_bid_strategy("ortb", predictions)
ortb2_bidding_results_df = evaluate_bid_strategy("ortb2", predictions)

# print(linear_bidding_results_df)


################################################################################################################################

# Create DFs with best bid results
# # best_constant_bidding_df = constant_bidding_results.sort_values(by=['clicks'] , ascending=False).iloc[0]
# best_random_bidding_df = random_bidding_results.sort_values(by=['clicks'] , ascending=False).iloc[0]
best_linear_bid_df = linear_bidding_results_df.sort_values(by=['clicks'] , ascending=False).iloc[0]
best_ortb_bid_df = ortb_bidding_results_df.sort_values(by=['clicks'], ascending=False).iloc[0]
best_ortb2_bid_df = ortb2_bidding_results_df.sort_values(by=['clicks'], ascending=False).iloc[0]

# # best_constant_bidding_df = best_constant_bidding_df.drop("constants")
# best_random_bidding_df = best_random_bidding_df.drop("constants")
#best_linear_bid_df = best_linear_bid_df.drop("bid")

table_df = pd.concat([best_linear_bid_df],1)
#table_df.columns = ['constant', 'random', 'linear']
table_df.columns = ['linear']
table_df = table_df.T
#table_df.column = ['linear']
print(table_df)

table_df.to_csv("initial_results.csv")
best_ortb_bid_df.to_csv("ortb_result.csv")
best_ortb2_bid_df.to_csv("ortb2_result.csv")
# best_ortb_bid_df.to_csv("ortb_result.csv")

#### Creating the results file
def create_test_file(base_bid, testing_predictions):
    new_df = pd.DataFrame()
    bids = []

    for i in range(0,len(testing_predictions)):
        bid = base_bid * (testing_predictions[i] / avg_ctr)
        bids.append(bid)

    new_df['bidid'] = testing_data['bidid']
    new_df['bidprice'] = bids

    new_df.to_csv("testing_bidding_price.csv", index=False)

# predict for the test data
testing_predictions = logistic.predict_proba(X_test)

test_pCTR = pd.DataFrame(testing_predictions)

# Normalise predictions
testing_pred = []
for p in test_pCTR[1]:
    testing_pred.append( p / (p + ((1-p)/w)))

print("Using base bid...", best_linear_bid_df['bid'])
create_test_file(best_linear_bid_df['bid'], testing_pred)
