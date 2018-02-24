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
Create and save ctr/weekday graph
"""
def create_ctr_weekday_graph():

    df_1 = df[df['advertiser'] == 2259]

    weekdays = list(range(0,7))
    y = []

    for day in weekdays:
        day_df = df_1[df_1['weekday'] == day]

        if day_df.empty:
            y.append(0)
            continue

        ctr = get_ctr(day_df)
        y.append(ctr)

    print(weekdays, y)

    plt.plot(weekdays,y)
    plt.ylabel("CTR")
    plt.xlabel("Weekday")
    plt.savefig("ctr_weekday.png")

"""
Create and save ctr/hourly graph
"""
def create_ctr_hourly_graph():
    df_1 = df[df['advertiser'] == 2259]

    hours = list(range(0,25))
    y = []

    for hour in hours:
        day_df = df_1[df_1['hour'] == hour]

        if day_df.empty:
            y.append(0)
            continue

        ctr = get_ctr(day_df)
        y.append(ctr)

    print(hours, y)

    plt.plot(hours,y)
    plt.ylabel("CTR")
    plt.xlabel("Hours")
    plt.savefig("ctr_hours.png")

"""
Create and save ctr/ad exchange graph
"""
def create_ctr_ad_exchange_graph():
    df_1 = df[df['advertiser'] == 2259]

    exchange_list = list(range(0,4))
    y = []

    for ad_exchange in exchange_list:
        day_df = df_1[df_1['adexchange'] == ad_exchange]

        if day_df.empty:
            y.append(0)
            continue

        ctr = get_ctr(day_df)
        y.append(ctr)

    plt.bar(exchange_list,y)
    plt.ylabel("CTR")
    plt.xlabel("Ad exchange")
    plt.savefig("ctr_adexchange.png")

"""
Create and save ctr/region graph for 1 advertiser
too many tags, dont use this!
"""
def create_ctr_region_graph():
    df_1 = df[df['advertiser'] == 1458]

    regions = dict(df_1.groupby('region').apply(list))
    x = []
    y = []
    for r in regions:

        region_df = df_1[df_1['region'] == r]

        if region_df.empty:
            continue

        ctr = get_ctr(region_df)

        y.append(ctr)
        x.append(r)

    plt.bar(x, y)
    plt.ylabel("CTR")
    plt.xlabel("Region")
    plt.savefig("ctr_region.png")

"""
Create and save ctr/tag graph for 1 advertiser
"""
def create_ctr_tag_graph():
    df_1 = df[df['advertiser'] == 1458]

    tags = dict(df_1.groupby('usertag').apply(list))
    x = []
    y = []
    print(len(tags))
    for t in tags:

        tag_df = df_1[df_1['usertag'] == t]

        if tag_df.empty:
            continue

        ctr = get_ctr(tag_df)

        y.append(ctr)
        x.append(t)

    plt.bar(x, y)
    plt.ylabel("CTR")
    plt.xlabel("Tag")
    plt.savefig("ctr_tag.png")


"""
Create and Save Browser Graph
"""
def create_ctr_browser_graph():
    df_1 = df[df['advertiser'] == 1458]

    agents = dict(df_1.groupby('useragent').apply(list))
    x = []
    y = []
    for agent in agents:

        agent_df = df_1[df_1['useragent'] == agent]

        if agent_df.empty:
            continue

        ctr = get_ctr(agent_df)

        y.append(ctr)
        x.append(agent)

    plt.bar(x, y)
    plt.ylabel("CTR")
    plt.xlabel("Browser")
    plt.savefig("ctr_browser.png")


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

#print_basic_stats_table()
#create_ctr_weekday_graph()
#create_ctr_hourly_graph()
#create_ctr_ad_exchange_graph()
#create_ctr_region_graph()
#create_ctr_tag_graph()
create_ctr_browser_graph()
