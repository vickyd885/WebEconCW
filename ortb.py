import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn import linear_model, neighbors
from sklearn.metrics import accuracy_score


#Datasets
validate_data = pd.read_csv("validation.csv")
training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("small_test.csv")

# We select fields we are interested in
features = [ 'weekday', 'hour', 'region', 'city', 
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
logistic = linear_model.LogisticRegression(class_weight='balanced', C = 0.001)

# Fit the data
ysef = [x for x in list(X_train) if x not in X_validate]

logistic.fit(X_train, Y_train)

# Print accuracy by testing against validation set
print('Logistic Regression Accuracy: %f'
      % logistic.score(X_validate, Y_validate))

test_set_predictions_logistic = logistic.predict_proba(X_validate)

pCTR = pd.DataFrame(test_set_predictions_logistic)

predictions = []

a = len(training_data) / 2 * np.bincount(training_data.click)
w = np.divide(a[1],a[0])
print(a[1])
print(a[0])
print(w)

for p in pCTR[1]:
	predictions.append( p / (p + ((1-p)/w)))

print(test_set_predictions_logistic)

################################################################################################################################

# knn = neighbors.KNeighborsClassifier(n_neighbors=2) 

# knn.fit(X_train, Y_train)

# test_set_predictions_knn = knn.predict(X_validate)

# #print(test_set_predictions_knn)
# print("KNN Accuracy: ",knn.score(X_validate, Y_validate))

################################################################################################################################
avg_ctr = training_data['click'].sum() / len(training_data['bidid'])

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

	print(lambdas)
	print(c_range)

def placing_ORTB_bids(bids):
    impressions = 0
    clicks = 0
    cost = 0
    budget = 6250000

    valid = bids >= validate_data['payprice']
    for i in range(0, len(valid)):
        # Don't exceed budget
        if (cost + validate_data['payprice'][i]) >= budget:
        # print("Elapsed budget")
            break

        if valid[i] == True:
            impressions += 1
            clicks += validate_data['click'][i]
            cost += validate_data['payprice'][i]

    return impressions, clicks, cost

def evaluate_ortb_bid_strategy():
	ortb_bids, c_lambda = ortb_bid_1(test_set_predictions_logistic)

	impressions = []
	total_clicks = []
	total_spent = []
    
	results_df = pd.DataFrame()
	
	results_df['c', 'lambda'] = c_lambda
    
	results_df['strategy'] = 'ORTB'

	print(results_df)

	for bid in ortb_bids:
		print(bid)
		[imps, clicks, cost] = placing_ORTB_bids(bid)
		impressions.append(imps)
		total_clicks.append(clicks)
		total_spent.append(cost)

	# Immediate details
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

ortb_linear_strategy = evaluate_ortb_bid_strategy()
ortb_linear_strategy.to_csv("ortb_results.csv")