import pandas as pd
import matplotlib.pyplot as plt

'''
## Import training data
Headers:
['click', 'weekday', 'hour', 'bidid', 'userid', 'useragent',
'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid',
'slotid', 'slotwidth', 'slotheight', 'slotvisibility',
'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice',
 'keypage', 'advertiser', 'usertag']
'''

df = pd.read_csv("../datasets/we_data/small.csv")

'''
BASIC STATS
'''

# Number of impressions
num_of_impressions = df['click'].count()
print("Number of impressions: ", num_of_impressions)

# Number of clicks
grouping_of_clicks = df.groupby('click').size()
num_of_clicks = grouping_of_clicks[1]
print("Number of clicks: ", num_of_clicks)

# Click/Impression percentage
percentage_of_clicks = num_of_clicks*100/num_of_impressions
print("Percentage of clicks in entire dataset", percentage_of_clicks)

# Avg bidding cost
avg_bid_cost = df['bidprice'].mean()
print("Average cost paid for impression", avg_bid_cost)

# Avg cost
avg_pay_cost = df['payprice'].mean()
print("Average cost paid for impression", avg_pay_cost)

# Average difference in bid/pay
difference_in_cost = df['bidprice'] - df['payprice']
avg_difference = difference_in_cost.mean()
print("Average difference in bidding and paid cost", avg_difference)
