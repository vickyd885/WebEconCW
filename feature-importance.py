from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

validate_data = pd.read_csv("small_validation_100.csv")
training_data = pd.read_csv("small_train_100.csv")
testing_data = pd.read_csv("small_test_100.csv")

# validate_data = pd.read_csv("datasets/we_data/validation.csv")
# training_data = pd.read_csv("datasets/we_data/train.csv")
# testing_data = pd.read_csv("datasets/we_data/test.csv")



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
# testing_data = preprocess(testing_data)



# We select fields we are interested in
# Note that we do infact encode the useragent, because it's first mapped
# to OS and Browser fields. The useragent field itself then gets deleted
features = [ 'weekday', 'hour',
'region', 'city',
 'slotwidth', 'slotheight', 'slotvisibility',
'slotformat','advertiser', 'adexchange', 'slotprice_brackets', 'os', 'browser', 'usertag']

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
# X_test = encode_df(testing_data)
#
Y_train = training_data.click
Y_validate = validate_data.click



forest = RandomForestClassifier()
forest.fit(X_train,Y_train)

important_features = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

indices = np.argsort(important_features)[::-1]


# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], important_features[indices[f]]))
