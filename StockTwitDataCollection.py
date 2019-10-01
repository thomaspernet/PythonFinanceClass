#!/usr/bin/env python
# coding: utf-8

# # Collect Stocktwit messages
# 
# The objective of this notebook is to:
# 
# - Iteratively collect stock-related messages using Stocktwit API
# - Store the data in a MongoDB 
# 
# ## Ressources for the project
# 
# Slide availables here: [Project](https://drive.google.com/open?id=0B0rdK44Elj9RZzdYMnJzaTREaHlhOUlmNGd2Qzg3RFJTSDBn)
# 
# The online codes are available [here](https://repl.it/@trenault/StockTwits101)
# 
# ## Note about Stocktwit API
# 
# The API returns 30 messages at a time.
# 
# To get older messages need the specify the `max` arguments: https://api.stocktwits.com/developers/docs/api#streams-symbol-docs
# 
# - https://api.stocktwits.com/api/2/streams/symbol/max=177617138/BTC.X.json
# 
# Example
# 
# -  base URL: https://api.stocktwits.com/api/2/streams/symbol/
# 
# Paramteres:
# 
# -  1) ticker
# - 2) max id
# 
# - Only latest 30 messages
# https://api.stocktwits.com/api/2/streams/symbol/BTC.X.json
# 
# 
# 
# 
# We need the following information in a dataframe:
# 
# - `messages`
#   - `id`
#   - `body`
#   - `created_at`
#   - `user`
#     - `id`
#     - `username`
#     - `name`
#     - `avatar_url`
#     - `join_date` 
# 
# ## Stocktwit API
# 
# [API](https://api.stocktwits.com/developers/docs)
# 
# For next week, scrap  the data for bitcoin and messages about bitcoins
# 
# ## MongoDB Documentation (MacOnly)
# 
# Installation process: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/
# 
# To open the database, please first open the terminal and paste the following code 
# 
# ```
# mongod --config /usr/local/etc/mongod.conf
# ```
# 
# Then open another terminal and launch mongoDB with `mongo` to use MongoDB with the command line
# 
# You are all set!
# 
# ## Sentiment
# 
# If sentiment is Bullish, label +1, if sentiment is Bearish, lable -1 else 0
# 

# # Data Collection
# 
# We define a function to collect the data. The steps performed in the function are the following:
# 
# - Step 1: Send a Get-request to the API
# - Step 2: Evaluate the status. If the connection is refused, wait `n` amount of time
# - Step 3: Extract the following information from the JSON file:
#     - ID, Body, UserName, Created At, sentiment
# - Step 4: Insert to mongoDB
# - Step 5: Get the latest ID of the JSON file: allow iterative data collection

# In[ ]:


import urllib3
import json
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
import time
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def get_json_ticker(
        nameDatabase,
        collectionName,
        ticker, max_twit=False,
        to_mongodb=False):
    """
    ### Note that 
    - nameDatabase is the clienty
    - collectionName is the client and collection
    
    fonction to scrap message in stocktwit

    ## Step 1:
    - Extract single ticker

    ## Step 2:
    - Format to list

    ## Step 3:
    - Save to MongoDB

    """

    http = urllib3.PoolManager()

    if max_twit:
        url = "https://api.stocktwits.com/api/2/streams/symbol/{0}.json?max={1}".format(
            ticker,
            max_twit)
    else:
        url = "https://api.stocktwits.com/api/2/streams/symbol/{0}.json".format(
            ticker)

     # get the data
    r = http.request('GET', url)
    data = json.loads(r.data)

    if data["response"]["status"] == 200:
        
        
        for i, element in enumerate(data["messages"]):

            id_message = element["id"]
            body_message = element["body"]
            user_name = element["user"]["username"]
            created_at = element["created_at"]
            
            ### Store sentiment
            sentiment =  element['entities']['sentiment']
            
            if sentiment:
                to_store_sentiment = sentiment['basic']
            else:
                to_store_sentiment = 'Neutral'
                
            ### Symobls
            list_symbols = []
            
            ### itertate over the dictionnary to store the symbol

            list_symbols = []
            for sym in element['symbols']:
                list_symbols.append(sym['title'])                                    
                        ### Get different symbols

            if to_mongodb:

                # Client is the database
                db = nameDatabase

                # Collection name
                collection = collectionName
                
                if to_store_sentiment == 'Bullish':
                    to_store_sentiment_ = 1
                elif to_store_sentiment == 'Bearish':
                    to_store_sentiment_ = -1
                else:
                    to_store_sentiment_ = 0
                    
                    

                insert_element = {"id": id_message,
                                  "created_at": element["created_at"],
                                  "user": user_name,
                                  "sentiment": to_store_sentiment,
                                  "sentiment_":to_store_sentiment_, 
                                  "body": body_message,
                                  "symbols": list_symbols}

                result = collection.insert_one(insert_element)
                
            if i == 29:
                latest_id = element["id"]

                dic_ = {
                    'id': latest_id,
                    'url': url
                }
                
        return dic_
                
    else:
        #print(data["response"]["status"])
        status = 429
        
        return status


# # Call API in loop
# 
# We need to loop to get historical data. Since the API allows to get 6.000 messages per hour (i.e 200 request), we implement a time sleep.
# 
# The logic is simple, we check the time the loop begins, then compute the time it took to collect the 6.000 messages. Finaly, we compute the time the next batch should start (ie time begin + 1hour - time to collect)
# 
# For instance, if the collection starts at 7:05am, last 1 minute, then the next batch will be triggered at 8:04am.

# In[ ]:


# Client is the database
db = client['StockTwitClass101']

# Collection name
collection = db.messages

# tickers
ticker = 'BTC.X'

# Get the first 30 messages and store the ID
output = get_json_ticker(
    nameDatabase=db,
    collectionName=collection,
    ticker=ticker,
    max_twit=False,
    to_mongodb=True)

# Loop to get the messages

timesecond = 0
lastid = output['id']

for h in range(0,8):
    time.sleep(timesecond)
    begin = datetime.now()
    for i in tqdm(range(0, 250)):

        output = get_json_ticker(
                nameDatabase=db,
                collectionName=collection,
                ticker=ticker,
                max_twit=lastid,
                to_mongodb=True)
        
        if output == 429:
            end = datetime.now()
            time_code = end - begin
    
            time_next_batch = begin + timedelta(hours=1)
            time_end_batch = begin + timedelta(seconds=time_code.seconds)
            timesecond = (time_next_batch - time_end_batch).seconds
            
            break
            
        else:
            
            lastid = output['id']
    
    print('Next batch in {} minutes. It will happen at {}'.format(timesecond/60, 
                                           time_next_batch.strftime("%H:%M:%S"))
         )

