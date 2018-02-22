import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable

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
Gets stats for each advertiser id
'''
def get_basic_stats_for_advertiser(advertiser):

    advertiser_df = df[df['advertiser'] == advertiser]

    # Number of impressions
    num_of_impressions = advertiser_df['click'].count()
    #print("Number of impressions: ", num_of_impressions)

    # Number of clicks
    grouping_of_clicks = advertiser_df.groupby('click').size()
    num_of_clicks = num_of_impressions - grouping_of_clicks[0]
    #print("Number of clicks: ", num_of_clicks)

    # CTR
    ctr = num_of_clicks / num_of_impressions
    #print("CTR: ", ctr)

    # Avg bidding cost
    avg_bid_cost = advertiser_df['bidprice'].mean()
    #print("Average highest paid for impression", avg_bid_cost)

    # Avg cost
    avg_pay_cost = advertiser_df['payprice'].mean()
    #print("Average price paid for impression", avg_pay_cost)

    # Total cost
    total_cost = advertiser_df['payprice'].sum()

    # CPM
    cpm = total_cost / 1000

    # CPC
    cpc = 0
    if num_of_clicks != 0:
        cpc = total_cost / num_of_clicks

    # # Win ratio
    # print(advertiser_df['bidprice'].count())
    # print(advertiser_df['payprice'].count())

    # Average difference in bid/pay
    difference_in_cost = advertiser_df['bidprice'] - advertiser_df['payprice']
    avg_difference = difference_in_cost.mean()
    #print("Average difference in bidding and paid cost", avg_difference)

    row_data = [advertiser, num_of_impressions, num_of_clicks, total_cost, ctr, cpm, cpc]
    return row_data

"""
Returns stat table with headers
"""
def initialise_stats_table():
    # Basic stats
    stats_table = PrettyTable()
    stats_table.field_names = ['Advertiser','Impressions', 'Clicks', 'Cost',
    'CTR', 'avg CPM', 'CPC']
    return stats_table

"""
Prints the basic stat tablle
"""
def create_basic_stats_table():

    stats_table = initialise_stats_table()
    # Get advertiser keys
    advertisers = dict(df.groupby('advertiser').apply(list))
    for advertiser in advertisers:
        print(advertiser)
        stats_table.add_row(get_basic_stats_for_advertiser(advertiser))

    print(stats_table)

create_basic_stats_table()
