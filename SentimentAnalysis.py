#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# In[ ]:


#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')


# In[ ]:





# ## Create connection with Mongo

# In[3]:


client = MongoClient('localhost', 27017)

### Client is the database
db = client['StockTwitClass101']


# ## Pipeline To create Sentiment
# 
# A) Create a Function to prepare the data
#     
#     1. Keep only twit with sentiment either `Bullish` or `Bearish` and remove multiple stock twits
#    
#     2. take negation into account, we add the prefix "negtag_" to all words following "not","no","none","neither","never" or “nobody”
#     
#     3. Convert digit to "_digit"
#     
#     4. Remove when mention a user
#     
#     5. lemmatize corpus
#     
#     6. Prepare train/test set
#     
# B) Build the Vectorization
# C) Construct the Naive classifier
# D) Predict out of sample
# 

# ### Keep only twit with sentiment either Bullish or Bearish and remove multiple stock twits

# In[ ]:


#df = pd.DataFrame(list(db.messages.find(query)))

#df['count_stock'] = df['symbols'].apply(lambda x: len(x))

#df_toanalyse = df.copy()

#df_toanalyse = df_toanalyse[df_toanalyse['count_stock'].isin([1])]

#df_toanalyse.shape

#df_toanalyse.groupby('sentiment')['sentiment'].count()


# ### Create a Function to prepare the data
# 
# Step : 1
#        
#        - Exclude multi tickers
# 
# Step : 2
#        
#        - take negation into account:
#        
#        - "not","no","none","neither","never" or “nobody”
# 
# Step : 3
#        
#        - Convert digit to "_digit"
# 
# Step : 4
#         
#        - Remove @USER
# 
# Step : 5
#        
#        - Remove unicode issue
#         
# Step 6: Lemmanize
# 

# In[9]:


def metatransformation(query, to_train = True):
    """
    Step : 1
        - Exclude multi tickers

    Step : 2
        - take negation into account:
        - "not","no","none","neither","never" or “nobody”

    Step : 3
        - Convert digit to "_digit"

    Step : 4
        - Remove @USER

    Step : 5
        - Remove unicode issue
        
    Step 6: Lemmanize


    """
    
    text = pd.DataFrame(list(db.messages.find(query)))
    
    ### Count stock
    
    text['count_stock'] = text['symbols'].apply(lambda x: len(x))
    
    ### Extract single count
    
    text = text[text['count_stock'].isin([1])]

    #text = df.copy()

    # take negation into account
    text['body_transform'] = text['body'].replace(regex={r"\bnothing\b": 'nothing_negword',
                                                         r"\bno\b": 'no_negword',
                                                         r"\bnone\b": 'none_negword',
                                                         r"\bneither\b": 'neither_negword',
                                                         r"\bnever\b": 'never_negword',
                                                         r"\bnobody\b": 'nobody_negword'
                                                         })

    # Convert digit to "_digit"

    text['body_transform'] = text['body_transform'].replace(regex={r"\d+": 'isDigit'})

    ### Remove @USER

    text['body_transform'] = text['body_transform'].replace(
        regex={r"([@?])(\w+)\b": 'user'})

    # Remove unicode issue

    text['body_transform'] = text['body_transform'].replace(regex={r"\b&#\b": ' '})

    # Lemmatize

    lemmatizer = WordNetLemmatizer()
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

    text['body_transform'] = text['body_transform'].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)]))
    
    ### Split the dataset 

    
    X_ = text['body_transform']
    y_ = text['sentiment_']
    
    count_ = text.groupby('sentiment')['sentiment'].count()
    
    print("The shape of the data is {}, and {}".format(text.shape,
                                                       count_
                                                      ))
    
    if to_train:
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, 
                                                        test_size=0.1,
                                                        random_state=0)

        return X_train, X_test, y_train, y_test
    
    else:
        
        return X_


# ## Pipeline step
# 
# This step includes:
# 
# - Build the Vectorization
# - Construct the Naive classifier

# In[12]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[14]:


stopwords.words('english')[:10]


# In[15]:


from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=10,
                             max_df=0.7,
                             stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
 ])


# Create the first transformation of the data

# In[5]:


query ={
    "sentiment":{ "$ne": "Neutral" }
}


# In[10]:


X_train, X_test, y_train, y_test = metatransformation(query = query)


# In[16]:


text_clf.fit(X_train, y_train)


# In[17]:


y_pred = text_clf.predict(X_train)
y_pred[:10]


# In[18]:


from sklearn import metrics


# In[19]:


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test,
                                    predicted))


# In[20]:


metrics.confusion_matrix(y_test, predicted)


# ## Predict out of sample

# In[22]:


query ={
    "sentiment":"Neutral" 
}
X_predict = metatransformation(query = query,
                               to_train = False)


# In[26]:


predicted = text_clf.predict(X_predict)


# In[28]:


pd.concat([pd.Series(X_predict, name = 'body').reset_index(),
          pd.Series(predicted, name = 'predict')], axis = 1)



