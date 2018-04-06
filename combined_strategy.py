import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input, BatchNormalization

from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn import linear_model, neighbors
from sklearn.metrics import accuracy_score

from random import randint
import random
from xgboost import XGBClassifier, XGBRegressor

### Data Imports

# validate_data = pd.read_csv("small_validation.csv")
# training_data = pd.read_csv("small_train.csv")
# testing_data = pd.read_csv("small_test.csv")

validate_data = pd.read_csv("datasets/we_data/validation.csv")
training_data = pd.read_csv("datasets/we_data/train.csv")
testing_data = pd.read_csv("datasets/we_data/test.csv")


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

def add_grouped_user_tags(encoded_df):
    tags_df = pd.DataFrame(encoded_df.usertag.astype(str).str.split(',').tolist())
    print(tags_df)
    usertag_df = pd.DataFrame(tags_df)
    print(usertag_df)
    usertag_df2 = pd.get_dummies(usertag_df,prefix='usertag')
    usertag_df2 = usertag_df2.groupby(usertag_df2.columns, axis=1).sum()
    encoded_df = pd.concat([encoded_df, usertag_df2], axis=1)
    encoded_df = encoded_df.drop('usertag', axis=1)
    return encoded_df

def preprocess(raw_df):
    p_df = extract_useragent_data(raw_df)
    p_df = group_slot_prices(p_df)
    print(list(p_df))
    return p_df

training_data = preprocess(training_data)
validate_data = preprocess(validate_data)
testing_data = preprocess(testing_data)



# We select fields we are interested in
# Note that we do infact encode the useragent, because it's first mapped
# to OS and Browser fields. The useragent field itself then gets deleted
features = [ 'weekday', 'hour',
'region', 'city',
 'slotwidth', 'slotheight', 'slotvisibility',
'slotformat','advertiser', 'adexchange', 'slotprice_brackets', 'os', 'browser',
'usertag']

## Fails on
# usertag


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
    #raw_df = raw_df.fillna(0)


    encoded_df = delete_unused_columns(raw_df) # removes unused fields
    label_encoder = LabelBinarizer() # Uses SKLearn Label Binary Encoder

    print("LIST HERE")
    print(list(encoded_df))
    for field in features:
        if field == 'usertag':
            encoded_df = add_grouped_user_tags(encoded_df)
            continue
        else:
            encoded_df = pd.concat([encoded_df,pd.get_dummies(encoded_df[field],prefix=field)],axis=1)
            encoded_df = encoded_df.drop(field,axis=1)

    return encoded_df

# # Encode the DFs
X_train = encode_df(training_data)
X_validate = encode_df(validate_data)
X_test = encode_df(testing_data)
#
Y_train = training_data.click
Y_validate = validate_data.click

print("Finish feature encoding")

print(list(X_train))

####
# Declare the logistic model
logistic = linear_model.LogisticRegression(class_weight='balanced', C = 0.001)

# # Fit the data
# ysef = [x for x in list(X_train) if x not in X_validate]
# print(ysef)

logistic.fit(X_train, Y_train)

# # Print accuracy by testing against validation set
print('Logistic Regression Accuracy: %f'       % logistic.score(X_validate, Y_validate))


test_set_predictions_logistic = logistic.predict_proba(X_validate)

lr_pCTRs = pd.DataFrame(test_set_predictions_logistic)

lr_predictions = []

a = len(training_data) / 2 * np.bincount(training_data.click)
w = float(a[1]) / float(a[0])

for p in lr_pCTRs[1]:
     lr_predictions.append( p / (p + ((1-p)/w)))



#print(test_set_predictions_logistic)

################################################################################################################################

################################################################################################################################

                                                ## Keras Neural Network Model ##

################################################################################################################################

model_type = "standard"

only_1_Y = Y_train[Y_train == 1].index
only_1_Y_validate = Y_validate[Y_validate == 1].index

print(len(Y_train[Y_train == 0].index))
print(len(only_1_Y))

batch_size = 10
num_input_nodes = 519
num_hidden_nodes = 512 
num_output_nodes = 2

class DataGenerator:
    def __init__(self, batch_size = 64, shuffle = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, features, labels, one_labels):
        while True:
            batch_features = []
            batch_labels = []

            index1 = random.choice(one_labels)
            batch_features.append(features.ix[index1].tolist())
            batch_labels.append(labels[index1])

            for i in range(batch_size - 1):
                index0 = random.choice(np.arange(len(features)))
                batch_features.append(features.ix[index0].tolist())
                batch_labels.append(labels[index0])

            if(model_type=="convolution"):
                batch_features = np.expand_dims(batch_features, axis = 2)
            yield np.array(batch_features), np.array(batch_labels)

    def generate_test(self, features):
        while True:
            batch_features = []

            for i in range(batch_size):
                index = random.choice(np.arange(len(features)))
                batch_features.append(features.ix[index].tolist())
        
        yield np.array(batch_features)

model = Sequential()

if('neural_bid_model.h5' in os.listdir(os.getcwd())):
    print("Found existing Keras model.")
    model = load_model('neural_bid_model.h5')
else:
    print("Did not find existing Keras Model. Creating/Training new model")
    Y_train = keras.utils.to_categorical(Y_train, num_output_nodes)
    Y_validate = keras.utils.to_categorical(Y_validate, num_output_nodes)

    training_generator = DataGenerator(batch_size=batch_size).generate(X_train, Y_train, only_1_Y)
    validation_generator = DataGenerator(batch_size=batch_size).generate(X_validate, Y_validate, only_1_Y_validate)
    test_generator = DataGenerator(batch_size = batch_size).generate_test(X_test)

    if(model_type == "standard"):
        model.add(Dense(num_hidden_nodes, activation='sigmoid', input_dim = num_input_nodes, kernel_initializer='random_normal'))
        # model.add(BatchNormalization(epsilon=1e-3))
        model.add(Dropout(0.5))
        model.add(Dense(num_hidden_nodes, activation='sigmoid', input_dim = num_input_nodes, kernel_initializer='random_normal'))
        model.add(Dropout(0.5))
        model.add(Dense(num_output_nodes, activation='softmax', kernel_initializer='random_normal'))
    elif(model_type == "convolution"):
        print("Building Convolution Model")
        input_shape = (num_input_nodes,1)
        conv_filters = 16

        inp = Input(shape = input_shape)
        conv_1 = Conv1D(conv_filters, kernel_size = 2, input_shape = (input_shape), activation='sigmoid', padding='valid')(inp)
        conv_2 = Conv1D(conv_filters, kernel_size = 2, activation='sigmoid', padding='valid')(conv_1)
        pooling_1 = MaxPooling1D(pool_size=2, padding="valid")(conv_2)

        flat = Flatten()(pooling_1)

        hidden1 = Dense(num_hidden_nodes, activation='sigmoid', kernel_initializer='random_normal')(flat)
        dropout1 = Dropout(0.5)(hidden1)
        hidden2 = Dense(num_hidden_nodes, activation='sigmoid', kernel_initializer='random_normal')(dropout1)
        dropout2 = Dropout(0.5)(hidden2)

        out = Dense(num_output_nodes, activation='softmax', kernel_initializer='random_normal')(dropout2)
        model = Model(inputs=inp, outputs=out)

    model.summary()

    epochs = 1

    click_class_weight = {0:0.1, 1:0.9}

    model.compile(loss='categorical_crossentropy',
                  optimizer= keras.optimizers.Adagrad(),
                  metrics=['accuracy'])

    H = model.fit_generator(generator = training_generator, 
                            steps_per_epoch = 10000, 
                            epochs=epochs,
                            verbose=1,
                            validation_data = validation_generator,
                            validation_steps = 10000,
                            class_weight = click_class_weight)

    model.save('neural_bid_model.h5')

if(model_type=="convolution"):
    X_validate = np.expand_dims(X_validate, axis = 2)
predictions = model.predict(np.array(X_validate))
# predictions = model.predict_generator(test_generator,verbose = 1, steps = len(X_test)//batch_size)

print(predictions)

adjust_pred = []

a = len(training_data) / 2 * np.bincount(training_data.click)
w = float(a[1]) / float(a[0])

for p in predictions:
    adjust_pred.append( p[1] / (p[1] + ((1-p[1])/w)))

nn_predictions = adjust_pred


## Combined strategy: maximising pctr

predictions = []

for i in range(0, len(nn_predictions)):
 predictions.append(min(lr_predictions[i], nn_predictions[i]))

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
        #print(c)
        for l in lambdas:
            #print(l)
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
        #print(c)
        for l in lambdas:
            #print(l)
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
#ortb_bidding_results_df = evaluate_bid_strategy("ortb", predictions)

#ortb2_bidding_results_df = evaluate_bid_strategy("ortb2", predictions)

print(linear_bidding_results_df)
#print(ortb_bidding_results_df)
#print(ortb2_bidding_results_df)


################################################################################################################################

# Create DFs with best bid results
# # best_constant_bidding_df = constant_bidding_results.sort_values(by=['clicks'] , ascending=False).iloc[0]
# best_random_bidding_df = random_bidding_results.sort_values(by=['clicks'] , ascending=False).iloc[0]
best_linear_bid_df = linear_bidding_results_df.sort_values(by=['clicks'] , ascending=False).iloc[0]
#best_ortb_bid_df = ortb_bidding_results_df.sort_values(by=['clicks'], ascending=False).iloc[0]
#best_ortb2_bid_df = ortb2_bidding_results_df.sort_values(by=['clicks'], ascending=False).iloc[0]

# # best_constant_bidding_df = best_constant_bidding_df.drop("constants")
# best_random_bidding_df = best_random_bidding_df.drop("constants")
#best_linear_bid_df = best_linear_bid_df.drop("bid")

table_df = pd.concat([best_linear_bid_df],1)
#table_df.columns = ['constant', 'random', 'linear']
# table_df.columns = ['ortb2']
table_df = table_df.T
table_df.column = ['linear']
print(table_df)
sys.exit(0)
# table_df.to_csv("initial_results.csv")
best_ortb_bid_df.to_csv("ortb_result.csv")
best_ortb2_bid_df.to_csv("ortb2_result.csv")
# best_linear_bid_df.to_csv("linear_bidding_new.csv")
# best_ortb_bid_df.to_csv("ortb_result.csv")

sys.exit(0)
#### Creating the results file
def create_test_file_linear(base_bid, testing_predictions):
    new_df = pd.DataFrame()
    bids = []

    for i in range(0,len(testing_predictions)):
        bid = base_bid * (testing_predictions[i] / avg_ctr)
        bids.append(bid)

    new_df['bidid'] = testing_data['bidid']
    new_df['bidprice'] = bids

    new_df.to_csv("testing_bidding_price.csv", index=False)


def create_test_file_ortb(c, l, testing_predictions, version):

    new_df = pd.DataFrame()
    bids = []

    if version == 1: # use ortb1
        bid = np.sqrt((np.multiply(np.divide(c, l), testing_predictions)) + np.square(c) - c)
        bids = bid.tolist()
    else: #use ortb2
        part1 = np.cbrt(np.divide(testing_predictions + np.sqrt(np.multiply(np.square(c), np.square(l)) + np.square(testing_predictions)), np.multiply(c, l)))
        part2 = np.cbrt(np.divide(np.multiply(c, l), testing_predictions + np.sqrt(np.multiply(np.square(c),np.square(l)) + np.square(testing_predictions))))
        form = part1 - part2
        bid = np.dot(c, form)
        bids = bid.tolist()

    new_df['bidid'] = testing_data['bidid']
    new_df['bidprice'] = bids

    new_df.to_csv("testing_bidding_price_ortb" + str(version) + "_new.csv", index=False)


# predict for the test data
testing_predictions = logistic.predict_proba(X_test)

test_pCTR = pd.DataFrame(testing_predictions)

# Normalise predictions
testing_pred = []
for p in test_pCTR[1]:
    testing_pred.append( p / (p + ((1-p)/w)))

#print("Using base bid...", best_linear_bid_df['bid'])
#create_test_file_linear(best_linear_bid_df['bid'], testing_pred)

c = best_ortb2_bid_df['c','lambda'][0]
l = best_ortb2_bid_df['c','lambda'][1]
create_test_file_ortb(c, l, testing_pred, 2)
