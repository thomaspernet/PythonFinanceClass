#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:06:16 2019

@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - Monte Carlo
2019
"""

import pandas as pd
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools
import statsmodels.formula.api as smf

# to plot, set ploton to ploton to 1
ploton = 0

# change the working directory
os.chdir('//Users/skimeur/Google Drive/empirical_finance/')

#%% Import the data on French population again as in part 1
df = pd.read_csv('R_data/Valeurs.csv',sep = ';',encoding = 'latin1',skiprows = [0,1,2],header=None,index_col=False)
df = df.iloc[::-1]
df.columns = ['Year','Month','Population']
dates2 = pd.date_range('1994-01', '2016-10', freq='M') 
df.index = dates2
df = df.drop(['Year','Month'],1)
df = df.replace({' ': ''}, regex=True) 
df = df.astype(float)
df = df/1000

#%% GDP data

gdp = pd.read_csv('R_data/GDP.csv',sep=';',encoding='latin1',skiprows = [0,1])
gdp = gdp.iloc[::-1]
gdp.columns = ['Quarter','GDP']
dates3 = pd.date_range('1949-01', '2016-09', freq='Q') 
gdp.index = dates3
gdp = gdp.drop('Quarter',1)
gdp = gdp.replace({' ': ''}, regex=True) 
gdp = gdp.astype(float)

#%% Concatenate data and compute growth rates

df = pd.concat([df, gdp],axis=1)

del gdp

# drop rows with missing values
df = df.dropna(axis=0)

df = (df - df.shift(1)) / df.shift(1)
df = df.dropna()



#%% OLS coefficient t-test

# STEP 1
# original regression
OLSfull = smf.ols('Population ~ GDP',data = df).fit()
OLSfull.summary()
# extract estimates:
alphahat = OLSfull.params[0]
betahat = OLSfull.params[1]
#meanerr = OLSfull.resid.mean() # we assume E(residuals) = 0
stderr = OLSfull.resid.std()
mugdp = df.GDP.mean()
stdgdp = df.GDP.std()

# STEP 2
# we generate MCrun of time series ytilde
MCrun = 10000
tvaluesintercept = []
tvaluesbeta = []
lenseries = len(df)
#lenseries = 600
constants = [alphahat] * lenseries
for i in range(0,MCrun):
    # generate residuals randomly
    epsilontilde = np.random.normal(0,stderr,lenseries)
    # generate x randomly
    xtilde = betahat * np.random.normal(mugdp,stdgdp,lenseries)
    # STEP 3
    # ytilde is the sum of the constant and residuals generated
    # we are under the null hypothesis
    #ytilde = [sum(x) for x in zip(constants, xtilde, epsilontilde)]
    ytilde = [sum(x) for x in zip(constants, epsilontilde)]
    # STEP 4
    # we perform the regression of ytilde on xtilde
    dfolsMC = pd.DataFrame([ytilde, xtilde]).T
    dfolsMC.columns = ['Population','GDP']
    OLSfullMC = smf.ols('Population ~ GDP',data = dfolsMC).fit()
    # we are interested in the t-values of the intercept and of beta
    tvaluesintercept.append(OLSfullMC.tvalues[0])
    tvaluesbeta.append(OLSfullMC.tvalues[1])


# we can plot a ytilde against xtilde
# and compate this to the y and x original
if ploton == 1:
    dfolsMC.plot(secondary_y='Population')
    df.plot(secondary_y='Population')

# STEP 5
# the t-values we extracted are in the H0 rejection 0
# they a distributed as a t-Student
# it is a two-tailed test

# plot the histogram of the t-values from the MC simulations
if ploton == 1:
    pd.DataFrame(tvaluesbeta).hist(bins=50)
    
# we can compute the critical thresholds
print('1 percent critical threshold', np.percentile(tvaluesbeta, 99.5) )
print('5 percent critical threshold', np.percentile(tvaluesbeta, 97.5) )
# Critical threshold parametric, cf VaR section
criticaltparametric5percent = np.mean(tvaluesbeta) + scipy.stats.t.ppf(1-0.025, lenseries-2) * np.std(tvaluesbeta) * np.sqrt((lenseries-2)/(lenseries))   
print('5 percent critical threshold, parametric', criticaltparametric5percent )
print('10 percent critical threshold', np.percentile(tvaluesbeta, 95) )

# compute the p-value from the original regression
# parametric, times 2 as it is two-sided
pvaluetdistbetas = 2 * scipy.stats.t.cdf(-OLSfull.tvalues[1], df = lenseries - 2)

# from our Monte Carlo simulation
# in tvaluesbetanum we store all the t-values from our MC simulations that were above the t-value from the original OLS
tvaluesbetanum = [i for i in tvaluesbeta if i >= OLSfull.tvalues[1]]
# the p-value is the ratio of t-values form the MC below the OLS t-statistics
# don't forget the factor 2 as this is a two sided test
pvaluebetaMC = 2 * (len(tvaluesbetanum) / len(tvaluesbeta))

# compare our Monte Carlo approach with the p-value usually given by the test
print('beta coef p-value MC 5 percents',pvaluebetaMC,'OLS p-value 5 percents',OLSfull.pvalues[1])


# to comfort ourself with our method, check the 5% p-value
tvaluesbetanum = [i for i in tvaluesbeta if i <= np.percentile(tvaluesbeta, 2.5)]
pvalue5percent = 2 * len(tvaluesbetanum) / len(tvaluesbeta)


#%% Visual of the histogram and critical values

if ploton == 1:
    # we want to plot the normal distribution
    dx = 0.001  # resolution
    x = np.arange(-4, 4, dx)
    # normal distribution
    pdf = scipy.stats.t.pdf(x, loc=np.mean(tvaluesbeta), scale=np.std(tvaluesbeta), df=lenseries-2)
    alpha = 0.025  # confidence level
    LeftThres = scipy.stats.t.ppf(alpha, loc=np.mean(tvaluesbeta), scale=np.std(tvaluesbeta), df=lenseries-2)
    RightThres = scipy.stats.t.ppf(1-alpha, loc=np.mean(tvaluesbeta), scale=np.std(tvaluesbeta), df=lenseries-2)
    plt.figure(num=1, figsize=(11, 6))
    plt.plot(x, pdf, 'b', label="t-Student distributed t-values")
    #plt.hold(True)
    plt.axis("tight")
    # Vertical lines
    plt.plot([LeftThres, LeftThres], [0, 0.054], c='r')
    plt.plot([RightThres, RightThres], [0, 0.054], c='r')
    plt.xlim([-4, 4])
    plt.ylim([0, 0.45])
    plt.legend(loc="best")
    plt.xlabel("t-values")
    plt.ylabel("Probability of occurence")
    plt.title("t-values critical threshold for 5 percents p-values")
    #plt.show()
    plt.savefig('tvalues_dist.png')



#%% Tyring a Monte Carlo approach for the augmented Dickey-Fuller test

# Critical Values for Cointegration Tests
# James G. MacKinnon, 2010
# further elements: https://stats.stackexchange.com/questions/213551/how-is-the-augmented-dickey-fuller-test-adf-table-of-critical-values-calculate

## regression of the full model for the 'nc' version
# detrend
df.GDP = df.GDP - df.GDP.mean()
df['GDP1'] = df.GDP.shift(1)
df['dGDP'] = df.GDP.diff()
df.mean()
#df = df.dropna()

aDFfull = smf.ols('dGDP ~ GDP1 - 1',data = df).fit()
aDFfull.summary()


#deflate = np.sqrt((len(df)  - 2)/(len(df) -1))
#aDFfull.tvalues[0] * deflate

MCrun = 100000
# burn in phase
#burnin = 50
tvalues = []
lenseries = len(df) 
meanerr = aDFfull.resid.mean()
stderr = aDFfull.resid.std()
#constants = np.cumsum([alphahat] * lenseries)
for i in range(0,MCrun):
    #epsilontilde = np.random.normal(0,stderr,lenseries).cumsum()
    #ytilde = [sum(x) for x in zip(constants, epsilontilde)]
    ytilde = np.random.normal(0,stderr,lenseries).cumsum()
    dfadfMC = pd.DataFrame(ytilde)
    dfadfMC.columns = ['GDP']
    dfadfMC['GDP1'] = dfadfMC.GDP.shift(1)
    dfadfMC['dGDP'] = dfadfMC.GDP.diff()
    aDFfullMC = smf.ols('dGDP ~ GDP1 - 1',data = dfadfMC).fit()
    tvalues.append(aDFfullMC.tvalues[0])


# we can plot a ytilde against xtilde
# and compate this to the y and x original
dfcompare = pd.DataFrame([dfadfMC.GDP.values,df.GDP.values]).T
dfcompare.columns = ['GDPMC','GDP']
if ploton == 1:
    dfcompare.plot()

# plot the histogram of the t-values from the MC simulations
if ploton == 1:
    tvaluedf = pd.DataFrame(tvalues)
    tvaluedf.columns = ['t-values']
    ax = tvaluedf['t-values'].hist(bins=200) # when T = 10,000, not well behaved
    fig = ax.get_figure()
    fig.savefig('tvaluesadf.pdf')
    

# we can compute the critical thresholds
MacKinnon2010_5percent = -1.941  - 0.2686 /len(df) - 3.365 /len(df)**2 + 31.223 / len(df)**3 

print('1 percent critical threshold', np.percentile(tvalues, 1) )
print('5 percent critical threshold', np.percentile(tvalues, 5) )
print('MacKinnon 2010 threshold at 5%', MacKinnon2010_5percent)
print('10 percent critical threshold', np.percentile(tvalues, 10) )


# from our Monte Carlo simulation
# in tvaluesbetanum we store all the t-values from our MC simulations that were above the t-value from the original OLS
tvaluesbelow = [i for i in tvalues if i <= aDFfull.tvalues[0] ]
# the p-value is the ratio of t-values form the MC below the OLS t-statistics
# this test is non-symmetrical, hence no factor 2
pvalueMC = len(tvaluesbelow) / len(tvalues)


# compare with the p-value given following MacKinnon table:
adftest = statsmodels.tsa.stattools.adfuller(df.GDP, regression='nc', regresults=True)
print(adftest)
# we can see what regression they used (same as ours)
adftest[3].resols.summary()
# compare our Monte Carlo approach with the p-value usually given by the test
print('beta coef p-value MC 5 percents', pvalueMC, 'aDF test p-value', adftest[1], 'OLS p-value 5 percents', aDFfull.pvalues[0])

