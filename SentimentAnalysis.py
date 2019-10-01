#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis for the Bitcoins 
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/c/c5/Bitcoin_logo.svg)
# 
# In the first notebook [StockTwit](https://nbviewer.jupyter.org/github/thomaspernet/PythonFinanceClass/blob/master/FinancialProject.ipynb), we created a robot using Stocktwit API to automatically collect the messages related to the Bitcoins.
# 
# In this notebook, we are going to use those data to run a sentiment analysis. More specifically, the aim is to answer the question:
# 
# - What is the relation between investor sentiment on social media and stock market returns? Any predictive power?
# 
# ## Agenda
# 
# We will proceed as follow:
# 
# 1. Create a function to clean the corpus
# 2. Create a brief text analysis
# 2. Evaluate the sentiment using the L2 method
# 3. Evaluate the sentiment using the L1 method
# 4. Evaluate the sentiment using a machine learning method
# 5. Predicg out of sample
# 6. Get Bitcoin data
# 7. Run the Granger test on Bitcoin's return and sentiment
# 
# Related paper: [Intraday online investor sentiment and return patterns in the U.S.
# stock market](https://docs.google.com/file/d/1L8bS8vNTXS-HWToP4zMpqfT308n-LxZb/edit)
# 
# L1/L2 Lexicon: [here](http://www.thomas-renault.com/data.php)

# In[ ]:


import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[ ]:


#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')


# # Clean the corps
# 
# We create a function named `metatransformation` to prepare the data.
# 
# Note that, we only care about the sentiment either `Bullish` or `Bearish` and remove multiple stock twits
# 
# ## Workflow of the function
# 
# Step 1: 
# 
#     - Extract the data from MonGoDb using the `query` argument of the function
#     
# Step :2
#        
#     - Exclude multi tickers
# 
# Step :3
#        
#     - take negation into account and add prefix `negtag_:
#        
#        - "not","no","none","neither","never" or “nobody”
# 
# Step :3
#     
#     - Further clean up as follow:
#         - Convert digit to "numbertag"        
#         - Remove @USER
#         - Remove ticker
#         - Remove special characters
#         - lower the text
# 
# Step 4: 
# 
#     - Remove stop words
# 
# Below some example of stop words

# In[ ]:


stopwords.words('english')[:10]


# Step 4: 
# 
#     - Lemmanize
#     
# Step 5: 
# 
#     - Create train/test set
#     
# ### note about Lemmatization
# 
# Lemmatization reduces words to their base word, which is linguistically correct lemmas. It transforms root word with the use of vocabulary and morphological analysis. Lemmatization is usually more sophisticated than stemming. Stemmer works on an individual word without knowledge of the context. For example, The word "better" has "good" as its lemma. This thing will miss by stemming because it requires a dictionary look-up

# ### Define MetaFunction
# 
# The function below clean and lemmanize the corpus and return a train/test set

# In[ ]:


def metatransformation(client, db, query, to_train=True):
    """
    Arguments:
    Query: MongoDB query 
    to_train:  True: return a train and test dataset
    False: return only data to predict out of sample
    
    Step :1
        - Extract the data from MonGoDb
        
    Step 2:
        - Exclude multi tickers

    Step :3
        - take negation into account:
            - "not","no","none","neither","never" or “nobody”
        - Convert digit to "numbertag"        
        - Remove @USER
        - Remove ticker
        - Remove special characters
        - Lower test

    Step 4: Remove stop words
    Step 5: Lemmanize
    Step 6: Train/test set

    """

    text = pd.DataFrame(list(db.messages.find(query)))

    # Count stock

    text["count_stock"] = text["symbols"].apply(lambda x: len(x))

    # Extract single count

    text = text[text["count_stock"].isin([1])]

    # take negation into account
    # Convert digit to "_digit"
    # Remove @USER
    # Remove unicode issue
    # Remove ticker
    # Remove all the special characters
    # remove all single characters
    # Remove Ya
    # Remove bitcoin
    # remove btc
    text["body_transform"] = text["body"].replace(
        regex={
            r"\bnothing\b": "negtag_nothing",
            r"\bno\b": "negtag_no",
            r"\bnone\b": "negtag_none",
            r"\bneither\b": "negtag_neither",
            r"\bnever\b": "negtag_never",
            r"\bnobody\b": "negtag_nobody",
            r"\d+": "numbertag ",
            r"([@?])(\w+)\b": "user",
            r"\b&#\b": " ",
            r"[$][A-Za-z][\S]*": "",
            r"\W": " ",
            r"\s+[a-zA-Z]\s+": " ",
            r"\^[a-zA-Z]\s+": " ",
            r"\s+": " ",
            r"^b\s+": "",
            r"\bya\b": "",
            r"\bbitcoin\b": "",
            r"\bBitcoin\b": "",
            r"\bbtc\b": "",
        }
    )
    # Lower

    text["body_transform"] = text["body_transform"].str.lower()

    # Remove stop words

    stop = stopwords.words('english')

    text["body_transform"] = text["body_transform"].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # Lemmatize

    lemmatizer = WordNetLemmatizer()
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

    text["body_transform"] = text["body_transform"].apply(
        lambda x: " ".join([lemmatizer.lemmatize(w)
                            for w in w_tokenizer.tokenize(x)])
    )

    # Split the dataset

    X_ = text["body_transform"]
    y_ = text["sentiment_"]

    count_ = text.groupby("sentiment")["sentiment"].count()

    print("The shape of the data is {}, and {}".format(text.shape, count_))

    if to_train:
        X_train, X_test, y_train, y_test = train_test_split(
            X_, y_, test_size=0.1, random_state=0
        )

        return X_train, X_test, y_train, y_test

    else:

        return X_


# # Preliminary analysis
# 
# Make brief plots of the frequency count, distribution of keywords by sentiment and bigrams frequency

# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt


# In[ ]:


client = MongoClient('localhost', 27017)

### Client is the database
db = client['StockTwitClass101']


# In[ ]:


query ={
    "sentiment":{ "$ne": "Neutral" }
}
X_train, X_test, y_train, y_test = metatransformation(client, db,
    query = query)


# Example of message

# In[ ]:


X_train.iloc[0]


# In[ ]:


y_train.iloc[0]


# Sentiments distribtion in the train set. Highly unbalanced

# In[ ]:


y_train.reset_index().groupby('sentiment_')['sentiment_'].count()


# In[ ]:


Word_tokenize = X_train.apply(word_tokenize) 
### Need to flatten the list
flattened_list = [y for x in Word_tokenize.tolist() for y in x]
fdist = FreqDist(flattened_list)
fdist.plot(30,cumulative=False)
plt.show()


# In[ ]:


def plot_keyword_sentiment(df, nbKeyword= 10):
    """
    Plot the distribution of sentiments by keyword
    """
    
    df_fdist = pd.DataFrame.from_dict(df, orient='index')
    df_fdist.columns = ['Frequency']
    df_fdist.index.name = 'Term'
    df_fdist =df_fdist.sort_values(by = 'Frequency', ascending = False)
    
    ### 
    
    df_top_sent = pd.DataFrame()
    for key in df_fdist.head(nbKeyword).index:

        count_sentiment = (
            pd.concat([X_train[X_train.str.contains(key)],
                             y_train], axis = 1, join = 'inner')
            .groupby('sentiment_')['body_transform']
            .count()
            .reset_index()
        )
        count_sentiment['keyword'] = key
        df_top_sent = df_top_sent.append(count_sentiment)
    df_top_sent = df_top_sent.pivot(index='keyword',
                  columns='sentiment_',
                  values='body_transform')
    df_top_sent['sum'] = df_top_sent.apply(lambda x: x.sum(), axis = 1)
    df_top_sent.sort_values(by = 'sum').drop(columns = 'sum').plot.barh(stacked=True)


# In[ ]:


plot_keyword_sentiment(df = fdist, nbKeyword= 10)


# ## Bigrams
# 
# Definitelly needs to clean more the corpus..

# In[ ]:


bgs = nltk.ngrams(flattened_list, 2)

fdist = nltk.FreqDist(bgs)

fdist.plot(30,cumulative=False)
plt.show()


# # L2 Approach

# In[ ]:


L2 = pd.read_csv('http://www.thomas-renault.com/l2_lexicon.csv',
           sep = ";")
L2.head(3)


# In[ ]:


list_pos = L2[L2['sentiment'].isin(['positive'])]['keyword'].tolist()
list_neg = L2[L2['sentiment'].isin(['negative'])]['keyword'].tolist()


# In[ ]:


def sentiment_L2(x):

    """
    Compute the average sentiment,
    given a list L2
    """
    sentiment = 0
    text = word_tokenize(x)
    for x in text:
        if x in list_pos:
            sentiment += 1
        elif x in list_neg:
            sentiment += -1
            
    try:
        avg_sentiment = sentiment / len(text)
    except:
        avg_sentiment = 0
            
    return avg_sentiment


# Example of message

# In[ ]:


X_train.iloc[0]


# The L2 lexicon correctly classified it as Bearish

# In[ ]:


sentiment_L2(x = X_train.iloc[0])


# Issue with the 0, classify as bearish. The cleaned corpus does not 100% match the L2 lexicon. For instance, open the second message in the test set. 

# In[ ]:


X_test.iloc[1]


# There is no `got` in the lexicon. 

# In[ ]:


y_pred = X_test.apply(sentiment_L2)
y_pred.head()


# Remove unclassified messages 

# In[ ]:


y_pred = y_pred[y_pred !=0]
y_pred.shape


# In[ ]:


y_pred_ = np.where(y_pred > 0, 1, -1)
print(metrics.classification_report(y_test[y_test.index.isin(y_pred.index)],
                                    y_pred_))


# In[ ]:


metrics.confusion_matrix(y_test[y_test.index.isin(y_pred.index)],
                         y_pred_)


# # L1 Approach

# In[ ]:


L1 = pd.read_csv('http://www.thomas-renault.com/l1_lexicon.csv',
           sep = ";")
L1.head(3)


# In[ ]:


L1_keywords = L1['keyword'].tolist()
L1_weights = L1['sw'].tolist()


# In[ ]:


def sentiment_L1(x):
    """
    Compute the average sentiment,
    given a list L1
    """


    sentiment = 0
    text = word_tokenize(x)
    for x in text:
        if x in L1_keywords:
        # find index
            index = L1_keywords.index(x)
            wg = L1_weights[index]
            sentiment += wg

    try:
        avg_sentiment = sentiment / len(text)
    except:
        avg_sentiment = 0

    return avg_sentiment


# In[ ]:


X_train.iloc[0]


# In[ ]:


sentiment_L1(X_train.iloc[0])


# In[ ]:


y_pred = X_test.apply(sentiment_L1)
y_pred.head()


# In[ ]:


y_pred = y_pred[y_pred !=0]
y_pred.shape


# In[ ]:


y_pred_ = np.where(y_pred > 0, 1, -1)
print(metrics.classification_report(y_test[y_test.index.isin(y_pred.index)],
                                    y_pred_))


# In[ ]:


metrics.confusion_matrix(y_test[y_test.index.isin(y_pred.index)], y_pred_)


# # Machine Learning Approach
# 
# This step includes:
# 
# - Build the Vectorization
# - Construct the Naive classifier

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[ ]:


text_clf = Pipeline([
    ('vect', CountVectorizer(max_features=1500,
                             min_df=10,
                             max_df=0.7)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
 ])


# Create the first transformation of the data

# ## Fit the model

# In[ ]:


text_clf.fit(X_train, y_train)


# Predict first message

# In[ ]:


X_train.iloc[0]


# In[ ]:


text_clf.predict(X_train)[0]


# Test the accuracy of the model

# In[ ]:


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test,
                                    predicted))


# In[ ]:


metrics.confusion_matrix(y_test, predicted)


# # Predict out of sample
# 
# Using ML method

# In[ ]:


query ={
    "sentiment":"Neutral" 
}
X_predict = metatransformation(client, db,
    query = query,
                               to_train = False)


# In[ ]:


predicted = text_clf.predict(X_predict)


# In[ ]:


outofsample = pd.concat([pd.Series(X_predict, name = 'body').reset_index(),
          pd.Series(predicted, name = 'predict')], axis = 1)


# The model really struggles to classify the bearish

# In[ ]:


outofsample.groupby('predict')['predict'].count()


# Example

# In[ ]:


outofsample[outofsample['predict'].isin([1])].head(4)


# In[ ]:


outofsample[outofsample['predict'].isin([-1])].head(4)


# ## Get Bitcoins Data
# 
# Extracted from [Quandl](https://www.quandl.com/data/BCHAIN/MKPRU-Bitcoin-Market-Price-USD)

# In[ ]:


import quandl
quandl.ApiConfig.api_key = "gs_J3domJb8kT6WjLz9s"


# In[ ]:


bitcoin = quandl.get("BCHAIN/MKPRU")
bitcoin['returns'] = bitcoin.pct_change(1)
bitcoin.tail()


# In[ ]:


bitcoin['Value'].plot(title='Values of Bitcoins')


# In[ ]:


### Looks stationary
bitcoin['returns'].dropna().plot(title='Returns of Bitcoins')


# ## Daily aggregated sentiment
# 
# Compute the daily average: Only on messages we have the sentiments since our classifier is not very much reliable

# In[ ]:


query = {"sentiment": {"$ne": "Neutral"}}
text = pd.DataFrame(list(db.messages.find(query)))
text["created_at"] = pd.to_datetime(text["created_at"], infer_datetime_format=True)
text = (text
        .set_index("created_at")
        .drop(columns="id")
        .resample("D")
        .mean()
       )


# In[ ]:


timeseries = pd.concat([text, bitcoin], axis = 1, join="inner")

timeseries.head()


# In[ ]:


axw = timeseries[['sentiment_', 'returns']].plot(secondary_y = 'returns',
                      title='Relationship betxeen sentiments and Bitcoins return')
figw = axw.get_figure()


# ## Granger test
# 
# Test the Granger Causality between sentiment on social media and stock returns
# 
# ### How does Granger causality test work?
# 
# It is based on the idea that if X causes Y, then the forecast of Y based on previous values of Y AND the previous values of X should outperform the forecast of Y based on previous values of Y alone.
# 
# According to Statsmodels 
# 
# The Null hypothesis for `grangercausalitytests` is that the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1. Grange causality means that past values of x2 have a statistically significant effect on the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.
# 
# The null hypothesis for all four test is that the coefficients corresponding to past values of the second time series are zero.

# In[ ]:


from statsmodels.tsa.stattools import grangercausalitytests


# According to the results, we believe there is no granger causality from sentiment to returns, ie all p-values are all above .05, accepting the null hypothesis, the time series in the second column, x2, does NOT Granger cause the time series in the first column, x1.

# In[ ]:


grangercausalitytests(timeseries[['returns', 'sentiment_']], maxlag=4)


# ## Regress
# 
# $$
# r_{i, t}=\alpha+\beta_{1} \Delta s_{1, t}+\beta_{2} \Delta s_{i, t-1}+\epsilon_{t}
# $$
# 

# In[ ]:


timeseries['sentiment_lag'] = timeseries['sentiment_'].shift(1)


# In[ ]:


timeseries['L_s1'] = timeseries['sentiment_'].pct_change(1)
timeseries['L_s2'] = timeseries['sentiment_lag'].pct_change(1)
timeseries.head()


# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:


mod1 = smf.ols(formula='returns ~ L_s1 + L_s2', 
               data=timeseries).fit()
mod1.summary()


# $$
# r_{i, t}=\alpha+ \beta_{1} \Delta s_{1, t}+ \beta_{2} \Delta s_{i, t-1}+ 
# +\beta_{3} \Delta r_{1, t-1}+\beta_{4} \Delta r_{i, t-2}
# +  \epsilon_{t}
# $$

# In[ ]:


timeseries['returns_lag'] = timeseries['returns'].shift(1)
timeseries['L_r1'] = timeseries['returns'].pct_change(1)
timeseries['L_r2'] = timeseries['returns_lag'].pct_change(1)
timeseries.head()


# In[ ]:


mod1 = smf.ols(formula='returns ~ L_s1 + L_s2 +L_r1 + L_r2', 
               data=timeseries).fit()
mod1.summary()


# # Overall
# 
# ## Shortcoming of the analysis
# 
# - Lack of data: 1 month data is not enough
# - Lack of variance in the data: The data collected shows a biais toward bullish sentiment, even though the trend was declining
# - Need to improve the cleaning of the corpus to match the L1/L2 lexicon
# - NB model unable to predict the `bearish` messages.
# - The poor data processing leads to a wrong model prediction both for the sentiment analysis and Granger causality. 
# - No coefficients in the time series are coefficient, although the returns look stationary 

# # Appendix: Details steps & analytics
# 

# ### TF-IDF:  Our approach
# 
# As explained in the previous post, the tf-idf vectorization of a corpus of text documents assigns each word in a document a number that is proportional to its frequency in the document and inversely proportional to the number of documents in which it occurs
# 
# TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
# 
# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# 
# IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:
# 
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# Compute the IDFs

# Compute the TFIDF score
# 
# The higher the TF*IDF score (weight), the rarer the term and vice versa.

# ## VADER technique

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[ ]:


sid = SentimentIntensityAnalyzer()


# In[ ]:


query ={
    "sentiment":{ "$ne": "Neutral" }
}
text = pd.DataFrame(list(db.messages.find(query)))
text.head()


# In[ ]:


def exampleVADER(df,x =0):
    """
    message from Xtrain and return
    VADER sentiment scrore
    """
    print("- The message is the following \n {} \n- VADER classification \n {}".format(
        df.iloc[x],
        sid.polarity_scores(df.iloc[x])))


# In[ ]:


exampleVADER(df = text['body'],x = 0)


# In[ ]:


exampleVADER(df = X_train,x = 0)


# In[ ]:


for i in range(0, 5):
    exampleVADER(df = text['body'], x = i)


# In[ ]:


for i in range(0, 5):
    exampleVADER(df = X_train, x = i)


# In[ ]:


summary = {"positive": 0, "neutral": 0, "negative": 0}
for x in text['body']:
    ss = sid.polarity_scores(x)
    if ss["compound"] == 0.0:
        summary["neutral"] += 1
    elif ss["compound"] > 0.0:
        summary["positive"] += 1
    else:
        summary["negative"] += 1


# In[ ]:


summary


# In[ ]:


summary = {"positive": 0, "neutral": 0, "negative": 0}
for x in X_train:
    ss = sid.polarity_scores(x)
    if ss["compound"] == 0.0:
        summary["neutral"] += 1
    elif ss["compound"] > 0.0:
        summary["positive"] += 1
    else:
        summary["negative"] += 1


# In[ ]:


summary

