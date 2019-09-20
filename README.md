# PythonFinanceClass

Lecture from: [Thomas Renault](http://thomas-renault.com/)

## Notebook:

Online Code editor from the lecture: [Repl](https://repl.it/@trenault/StockTwits101)

The notebook is available here for read only: [nbviewer](https://nbviewer.jupyter.org/github/thomaspernet/PythonFinanceClass/blob/master/FinancialProject.ipynb)

To open the notebook with Binder (i.e. Online Jupyter notebook): [Online Notebook](https://mybinder.org/v2/gh/thomaspernet/PythonFinanceClass/master?filepath=FinancialProject.ipynb) Or with [colab](https://drive.google.com/file/d/1qbbXN9nD5R7urt4Eu1kzZcORwK7mFS2y/view?usp=sharing)

## WHAT YOU WILL LEARN ?

- How to extract data from social media (Stocktwits) using an Application
Programming Interface
- How to store data using MongoDB and how to query a database in Python
- How to use a lexicon-based approach to convert qualitative messages into
quantitative sentiment indicators
- How to analyze a text using Natural Language Processing (NLP)
- How to use Scikit-learn for sentiment analysis
- How to use a lexicon-based approach to convert qualitative messages into
quantitative sentiment indicators
- How to analyze a text using Natural Language Processing (NLP)
- How to use Scikit-learn for sentiment analysis

## STOCKTWITS & SENTIMENT ANALYSIS

Stocktwits : Social media platform designed for sharing ideas between investors, traders, and entrepreneurs – http://www.stocktwits.com

Founded in 2008. Approximately 150 millions messages on the platform.


Main question: What is the relation between investor sentiment on social media and stock market returns ? Any predictive power ?

Step by step:



1 - Extract data from StockTwits: https://api.stocktwits.com/developers/docs



2 – Store the data (Json, CSV, MongoDB, SQL…)



3 – Convert each message into a quantitative sentiment variable : +1 (positive), 0 (neutral), -1 (negative)



4 – Create an aggregate sentiment variable (daily) for a given stock



5 – Test the Granger Causality between sentiment on social media and stock returns

## LITERATURE REVIEW & OBJECTIVES

### NARDO ET AL. (2016)

Financial prices and transactions are not predictable according to the efficient market hypothesis.



“A  blindfolded  chimpanzee  throwing  darts  at  The  Wall  Street  Journal  could  select  a portfolio that would do as well as the (stock market) experts. However, what if this chimpanzee  could  browse  the  Internet  before  throwing  any  darts?”



Researchers  in  this  field  are  mainly  concerned  with  two  questions:  whether  the  mood  is  correctly identified by the machine, that is, accuracy, and to what extent the tonality is related to financial movements

### OLIVEIRA ET AL. (2016)

Propose an automated and fast approach to create stock market lexicons for microblogging messages.



250,000 messages on the training dataset



Holdout split method: 75% StockTwits classified messages are

used to create lexicons (training set) and the remaining (most recent)

25% posts are used for evaluation purposes (test set).

Substitute all cashtags by a unique term, thus avoiding cashtags to gain a sentiment value related with a particular time period



Replace numbers by a single tag, since the whole set of distinct numbers is too vast



For privacy reasons, all mentions and URL addresses were normalized to “@user” and “URL”, respectively



Exclude messages composed only by cashtags, url links, mentions or punctuation



Stanford CoreNLP tool to execute common natural language processing operations, such as tokenization, part of speech (POS) tagging and lemmatization.

### RENAULT (2017)

750,000 messages published on StockTwits to create a lexicon, following Oliveira et al. (2016)



Examine the relation between online investor sentiment and intraday stock returns using an extensive dataset of nearly 60 million messages published by online investors over a five-year period, from January 2012 to December 2016

INSERT IMAGE

### MAHMOUDI ET AL. (2018)

458,067 bullish and 108,659 bearish messages published on StockTwits



Compare different machine learning methods (Naïve Bayes, SVM, Neural Network).

INSERT IMAGE

Results: Including emojis improves the accuracy. Maximum Entropy gives better results than SVM and Naïve Bayes when considering bigrams.

