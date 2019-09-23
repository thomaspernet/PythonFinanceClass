#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2019

@author: Eric Vansteenberghe
Broker-dealer Leverage and the Stock Market
# data from https://www.federalreserve.gov/releases/z1/20160310/data.htm
# codes of the data are in https://www.federalreserve.gov/releases/z1/20160310/Coded/coded.pdf
# we are interested in table L.129 Security Brokers and Dealers
"""

import pandas as pd
import numpy as np
import os
import statsmodels.tsa.stattools
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scipy.stats
#We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Google Drive/empirical_finance')

ploton = 0

#%% Starting dates
# we can chose the starting dates for the filter of our time series
startdate = '1965-12-01'

#%% Borker dealer data
# import the data set
df = pd.read_csv('R_data/ltab129d.prn',delimiter='\"\s\"',engine = 'python')
# keep only the line which have NA on a give column
df = df.loc[~(df['FL664090005.Q']==' NA'),:]

# we want to expand the cell into a full line seperated by a space character
for i in range(0,len(df)):
    df.iloc[i,:] = df.iloc[i,0].split()
del i

# remove the character "quotes"
df = df.replace({'"':''},regex=True)
# transform the dates into dates recognized by python
df.index = [pd.to_datetime(x[:4]) + pd.offsets.QuarterBegin(int(x[5:])) for x in df.iloc[:,0]]


# Broker-dealer total financial assets: FL664090005 
# Broker-dealer total liabilities: FL664190005 
# keep on ly the columns with the total financial assets and the total liabilities
df = df.loc[:,['FL664090005.Q','FL664190005.Q']]
# keep only the dates after mid 1966
df = df.loc[df.index > startdate]
# rename the columns with simple acronyms
df.columns = ['TFA','TL']
# convert the data to numeric, type float
df = df.astype(float)
# compute the leverage in log as in the paper
df['log broker-dealer leverage'] = np.log(df['TFA'] /  (df['TFA'] - df['TL']))
# compute the leverage and the leverage growth rate
df['bdleverage'] = df['TFA'] /  (df['TFA'] - df['TL'])
df['broker-dealer leverage growth rate'] = (df.bdleverage - df.bdleverage.shift(1)) / df.bdleverage.shift(1)

if ploton == 1:
    # as in the paper, plot the log leverage and the leverage growth rate
    ax = df.loc[:,['log broker-dealer leverage','broker-dealer leverage growth rate']].plot(secondary_y='log broker-dealer leverage')
    fig = ax.get_figure()
    fig.savefig('brokerdealerleverage.pdf')

#%% Stock price data

# import the S&P 500 data
SP500i = pd.read_csv('R_data/SandPlong.csv',index_col=0)
# convert the dates into python date format
SP500i.index = pd.to_datetime(SP500i.index,format='%d/%m/%Y')

"""
#We tried to divide by the US GDP to see if the results are robust
code = 'INDPRO'

# r to disable escape
base_url = r'https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}'
starti = datetime(1967, 1, 1)
endi = datetime(2016,1,1)
url = base_url.format(code=code)
fred = pd.read_csv(url)
fred.DATE = pd.to_datetime(fred.DATE,format='%Y-%m-%d')
fred.index = fred.DATE
fred = fred.drop('DATE',axis=1)


del url, base_url, starti, endi, code

fred = fred.apply(pd.to_numeric,errors='corece')
fred.plot()

SP500i = pd.concat([SP500i,fred],axis = 1)
"""

# we want quarterly data, we resample
SP500i = SP500i.resample('Q').mean()

# we bound the data set in the time period of study
SP500i = SP500i.loc[SP500i.index > startdate]
SP500i= SP500i.loc[SP500i.index < '2016-01-01']

"""
# change the column name
SP500i.columns = ['SP500','GDP']

# divide by the GDP
SP500i.SP500 = SP500i.SP500 * SP500i.GDP.iloc[0] / SP500i.GDP
del SP500i['GDP']
"""
SP500i.columns = ['SP500']

# compute the quarterly growth rate
SP500i['SP500 growth rate'] = (SP500i.SP500 - SP500i.SP500.shift(1)) / SP500i.SP500.shift(1)

if ploton == 1:
    # plot the index and its growth rate
    ax = SP500i.loc[:,['SP500','SP500 growth rate']].plot(secondary_y='SP500 growth rate')
    fig = ax.get_figure()
    fig.savefig('sandpbrokerdealer.pdf')

#%% Data frame under study

# concatenate the series of interest
df = pd.concat([df['bdleverage'] ,SP500i['SP500']],axis=1)
del SP500i
# the date do not exactly correspond, we resample as a trick to have the values of a quarter on a same date
df = df.resample('Q').mean()
# the the not available data
df = df.dropna(how='any')

# compute the quarterly returns
dfx = (df - df.shift(1)) / df.shift(1)
# create the error correction model data frame
dfD = df - df.shift(1)
# keep only data from 1967
dfx = dfx.loc[dfx.index > '1966-12-31']
df = df.loc[df.index > '1966-12-31']

dfx.columns = ['broker-dealer leverage growth rate','SP500 growth rate']
if ploton == 1:
    ax = dfx.plot()
    fig = ax.get_figure()
    fig.savefig('brokerdealerSP500gr.pdf')

# rename the columns with simple names
df.columns = ['bdl','s']
dfx.columns = ['bdlr','sr']
# export the data to csv
df.loc[:,['bdl','s']].to_csv('dfbrokerdealerleverage.csv')
dfx.to_csv('dfxbrokerdealerleverage.csv')

#%% Unit root and stationarity tests

# unit root test, taking the trend into account
statsmodels.tsa.stattools.adfuller(df.bdl,regression='ct')
statsmodels.tsa.stattools.adfuller(dfx.bdlr,regression='c')
statsmodels.tsa.stattools.adfuller(df.s,regression='ct')
statsmodels.tsa.stattools.adfuller(dfx.sr,regression='c')
# the broker dealer leverage seems I(0), the stock exchange seems I(1)

#%% Correlation is not causation

dfx.corr()

#%% cointegration test
statsmodels.tsa.stattools.coint(dfx.bdlr,dfx.sr)
# The Null hypothesis is that there is no cointegration, we reject

#%% cointegration regression
# we want the ordinary least squares regression residuals from the cointegrating regression

cointreg = smf.ols('s ~ bdl',data = df).fit()

# show that the residuals are stationary for this regression
# ADF H0: the time series has a unit root
statsmodels.tsa.stattools.adfuller(cointreg.resid) # we could reject H0 at the 10% level but not the 5% level

# QQ plot of the normalized residuals
#scipy.stats.probplot(cointreg.resid_pearson, dist="norm", plot=plt)
# there seems to be one strong oultier in our residuals (one exercise would be to check if there is the need to remove this observation)



#%% Manual Granger causality test

dfD.columns = ['bdl','s']

# create the lag of the cointegration regression residuals
dfD['resid1'] = cointreg.resid.shift(1)

# create lag of return variables
dfD['bdl1'] = dfD['bdl'].shift(1)
dfD['bdl2'] = dfD['bdl'].shift(2)
dfD['bdl3'] = dfD['bdl'].shift(3)
dfD['s1'] = dfD['s'].shift(1)
dfD['s2'] = dfD['s'].shift(2)
dfD['s3'] = dfD['s'].shift(3)

# keep only data from 1967
dfD = dfD.loc[dfD.index > '1966-12-31']


#%% GRANGER TEST for s

# lag order selection with AIC
print('AIC (1,1)' , smf.ols('s ~ s1 +  bdl1 + resid1',data = dfD).fit().aic)
print('AIC (2,1)' , smf.ols('s ~ s1 + s2 +  bdl1 + resid1',data = dfD).fit().aic)
print('AIC (3,1)' , smf.ols('s ~ s1 + s2 + s3 + bdl1 + resid1',data = dfD).fit().aic)
print('AIC (1,2)' , smf.ols('s ~ s1 +  bdl1 + bdl2 + resid1',data = dfD).fit().aic)
print('AIC (1,3)' , smf.ols('s ~ s1 +  bdl1 + bdl2 + bdl3 + resid1',data = dfD).fit().aic)
# we do not find the same ordering as in the paper, that could come from the transformation with the CRB BLS spot index that we did not introduce
print('AIC (2,2)' , smf.ols('s ~ s1 + s2 +  bdl1 + bdl2 + resid1',data = dfD).fit().aic)

# do the broker dealer leverage Granger cause the stock prices?
# restricted model
modelrestricted = smf.ols('s ~ s1',data = dfD).fit()
modelrestricted.summary()

# unrestricted model
modelunrestricted = smf.ols('s ~ s1 +  bdl1 + bdl2 + bdl3 + resid1',data = dfD).fit()
modelunrestricted.summary()
dir(modelrestricted)
# F statistic
# n = 4
# m = 1 (don't forget the constant term)
Fstat1 = ( (modelunrestricted.rsquared - modelrestricted.rsquared) / 4) / ( (1 - modelunrestricted.rsquared) / (len(dfD) - 4 - 1 - 1) )
# or equivalently
Fstat1 = ( (modelrestricted.ssr - modelunrestricted.ssr) / modelunrestricted.ssr ) * ((len(dfD) - 4 - 1 - 1) / 4)
# critical value for F(4,190) at the 5% level: around 2.4
# we are above the critical value, we reject H0
# Granger causality?


# unrestricted model bis, as in the paper
modelunrestricted = smf.ols('s ~ s1 + bdl1 + resid1',data = dfD).fit()
modelunrestricted.summary()

# F statistic bis
Fstat1bis = ( (modelrestricted.ssr - modelunrestricted.ssr) / modelunrestricted.ssr ) * (len(dfD) - 2 - 1 - 1) / 2
# critical value for F(1,192) at the 5% level: around 3.9
# we are above the critical value, we reject H0
# Granger causality

#%% GRANGER TEST for bdl

# do the stock prices Granger cause the broker dealer leverage?
# lag order selection with AIC
print('AIC (1,1)' , smf.ols('bdl ~ s1 +  bdl1 + resid1',data = dfD).fit().bic)
print('AIC (2,1)' , smf.ols('bdl ~ s1 + s2 +  bdl1 + resid1',data = dfD).fit().bic)
print('AIC (3,1)' , smf.ols('bdl ~ s1 + s2 + s3 + bdl1 + resid1',data = dfD).fit().bic)
# we find the same ordering as in the paper
print('AIC (1,2)' , smf.ols('bdl ~ s1 +  bdl1 + bdl2 + resid1',data = dfD).fit().bic)
print('AIC (1,3)' , smf.ols('bdl ~ s1 +  bdl1 + bdl2 + bdl3 + resid1',data = dfD).fit().bic)
print('AIC (2,2)' , smf.ols('bdl ~ s1 + s2 +  bdl1 + bdl2 + resid1',data = dfD).fit().bic)

# restricted model
modelrestricted = smf.ols('bdl ~ bdl1',data = dfD).fit()
modelrestricted.summary()

# unrestricted model
modelunrestricted = smf.ols('bdl ~ s1 + s2 + s3 + bdl1 + resid1',data = dfD).fit()
modelunrestricted.summary()

# F statistic
Fstat2 = ( (modelrestricted .ssr - modelunrestricted.ssr) / modelunrestricted.ssr ) * ((len(dfD) - 4 - 1 - 1) / 4)
# critical value for F(4,190) at the 5% level: around 2.4
# we are abovethe critical value, we reject H0
# Granger Causality