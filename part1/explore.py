import pandas as pd
import numpy as np
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

df = pd.read_csv("../datasets/we_data/train.csv")

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
def print_basic_stats_table():

    stats_table = initialise_stats_table()
    # Get advertiser keys
    advertisers = dict(df.groupby('advertiser').apply(list))
    for advertiser in advertisers:
        print(advertiser)
        stats_table.add_row(get_basic_stats_for_advertiser(advertiser))

    print(stats_table)

"""
Gets CTR info given an advertiser_id and a topic
Returns a tuple (x,y) where x and y are axis data for the graph
"""
def get_ctr_info_for_advertiser(advertiser_id, grouping_topic):
    df_1 = df[df['advertiser'] == advertiser_id]
    keys = dict(df_1.groupby(grouping_topic).apply(list))
    x = []
    y = []
    for k in keys:

        k_df = df_1[df_1[grouping_topic] == k]

        if k_df.empty:
            continue

        ctr = get_ctr(k_df)

        y.append(ctr)
        x.append(k)

    return (x,y)



"""
Create and save ctr graph given a list of advertiser id, topic,
axis labels and file names
"""
def create_and_save_ctr_graph(advertisers, topic, xlabel, ylabel, filename):

    advertiser_data = []

    for advertiser in advertisers:
        advertiser_data.append(get_ctr_info_for_advertiser(advertiser, topic))

    for data in advertiser_data:
        plt.plot(data[0], data[1])

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename)

    plt.gcf().clear()

"""
Handle grouping_of_clicks edge cases
"""
def get_click_count(grouping_of_clicks):
    #print(grouping_of_clicks)
    num_of_clicks = 0
    if 0 in grouping_of_clicks:
        return grouping_of_clicks[0]
    else:
        return grouping_of_clicks[1]

"""
Get CTR given a filtered df
"""
def get_ctr(partial_df):
    num_of_impressions = partial_df['click'].count()
    grouping_of_clicks = partial_df.groupby('click').size()

    num_of_clicks = get_click_count(grouping_of_clicks)
    ctr = num_of_clicks / num_of_impressions
    return ctr



print_basic_stats_table()

advertiser_list = [ 1458, 2259 ]

# Create advertiser graphs
create_and_save_ctr_graph(advertiser_list,'useragent','user agents', 'CTR', 'ctr_usergent.png')
create_and_save_ctr_graph(advertiser_list,'region','Region', 'CTR', 'ctr_region.png')
create_and_save_ctr_graph(advertiser_list,'adexchange','Adexchange', 'CTR', 'ctr_adexchange.png')
create_and_save_ctr_graph(advertiser_list,'hour','Hour', 'CTR', 'ctr_hour.png')
create_and_save_ctr_graph(advertiser_list,'weekday','Weekday', 'CTR', 'ctr_weekday.png')
