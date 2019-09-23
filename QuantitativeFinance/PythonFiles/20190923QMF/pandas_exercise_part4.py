#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 4
2019
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools
import statsmodels.formula.api as smf
from scipy.stats import norm, ttest_ind
import scipy.stats


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
if ploton == 1:
    gdp.plot()

# concatenate both time series into one data frame
both = pd.concat([df,gdp],axis=1)

# drop rows with missing values
both = both.dropna(axis=0)

if ploton == 1:
    ax = both.plot(secondary_y=['Population'])
    fig = ax.get_figure()
    #fig.savefig('GDP_pop.png')

# compute quarterly changes
both_change = both/both.shift(1)-1

if ploton == 1:
    ax2 = both_change.plot()
    fig2 = ax2.get_figure()
    #fig2.savefig('GDP_pop_change.png')

#%% Before the OLS: visulization of the data set

if ploton == 1:
    axall = both_change.plot.scatter(x='GDP', y='Population')
    figall = axall.get_figure()
    #figall.savefig('gdppopscatterplot.pdf')
    plt.close()

#%% Correlation between population and GDP growth rates

both_change.corr()

del both, both_change, dates3


#%% Unit Root test - Dickey-Fuller test
# on gdp data
dickeyfullerdf = gdp.copy(deep=True)
dickeyfullerdf['Deltagdp'] = (gdp - gdp.shift(1))
dickeyfullerdf['GDP1'] = dickeyfullerdf['GDP'].shift(1)
# OLS of the delta GDP on the GDP
dickeyfuller_reg = smf.ols('Deltagdp ~ GDP1 -1',data = dickeyfullerdf).fit()
dickeyfuller_reg.summary()

# we should compare the t statistics of the GDP coefficient against
# the critical value of the t-distribution at 95%
# the degree of freedom are we estimate one coefficient):
degreeoffreedom = len(dickeyfullerdf.dropna()) - 2
proba = 0.95
scipy.stats.t.ppf(proba, degreeoffreedom)

# 10.41 is above 1.65
# low risk to wrongly reject H0 but
# for simplicity here we decide that rho - 1 is not equal to 0
# we decide that there is no unit root int his time series (be careful as the lenght of this series is limited)

del degreeoffreedom, dickeyfullerdf, proba

#%% Unit Root test - Augmented Dickey-Fuller test
#Before we start regressing variables:
#Test integration order:

#we compute the changes
df_change = df/df.shift(1)-1
df_change = df_change.dropna()
gdp_change = gdp/gdp.shift(1)-1
gdp_change = gdp_change.dropna()

#Population
statsmodels.tsa.stattools.adfuller(df.Population, regression='ct')
statsmodels.tsa.stattools.adfuller(df_change.Population, regression='c')
#GDP
statsmodels.tsa.stattools.adfuller(gdp.GDP, regression='ct')
statsmodels.tsa.stattools.adfuller(gdp_change.GDP, regression='c')

# imposing 'nc' to regression mean that we assume a random walk
# imposing 'c' means you assume a random walk with a drift
# imposing 'ct' would have ment that both series could have been trend stationary, in which cas the trend t should have been added in the regression




del df_change, gdp_change

#%% Concatenate all variables in one data frame
dfall = pd.concat([df,gdp],axis=1)
dfall = dfall.dropna()
dfallx = dfall / dfall.shift(1) - 1


#%% cointegration test
# H0: no cointegration
statsmodels.tsa.stattools.coint(dfall.GDP,dfall.Population)

#%% Cointegation: building two cointegrated time series

xt = [0]
yt = [0]
# define the beta of the system
beta_coint = 0.3
beta_y = 0.9
# length of our data
lent = 1000
for i in range(1,lent):
    xt.append((xt[i-1] + np.random.normal(0,1,1))[0])
    yt.append((beta_y * yt[i-1] + beta_coint * xt[i] + np.random.normal(0,1,1))[0])

dfcoint = pd.DataFrame(np.matrix([yt,xt]).transpose())
dfcoint.columns = ['yt','xt']

if ploton == 1:
    ax = dfcoint.loc[:,['yt','xt']].plot(title='Two cointegrated time series, with both a unit root')
    fig = ax.get_figure()
    #fig.savefig('cointillustration.pdf')

# Unit root test, H0: there is a unit root
statsmodels.tsa.stattools.adfuller(dfcoint['yt'], regression='c')
statsmodels.tsa.stattools.adfuller(dfcoint['xt'], regression='c')
if ploton == 1:
    dfcoint.diff().plot()
statsmodels.tsa.stattools.adfuller(dfcoint['yt'].diff().dropna(), regression='nc')
statsmodels.tsa.stattools.adfuller(dfcoint['xt'].diff().dropna(), regression='nc')
# both series are I(1)

# Cointegration test
# H0: no cointegration
statsmodels.tsa.stattools.coint(dfcoint['yt'],dfcoint['xt']) # cointegration

dfcoint.mean()
# as series mean are different than 0 we add a constant in the cointegration regression

# regress one series on the other for the cointegration regression
model_coint = smf.ols('yt ~ xt',data=dfcoint).fit()
model_coint.summary()
dir(model_coint) # we can use "resid" from the model to work on the residuals of this regression
# we check that the cointegrating residuals are I(0)
# Unit root test, H0: there is a unit root
# in fact, as demonstrated in Phillips and Ouliaris (1990), one cannot use the ADF test because of the spurious nature of the regression
#statsmodels.tsa.stattools.adfuller(model_coint.resid) # cointegrating residuals are stationary, hence I(0)

dfcoint['xt_modif'] = model_coint.params[0] + model_coint.params[1] * dfcoint['xt']

if ploton == 1:
    ax = dfcoint.loc[:,['yt','xt_modif']].plot(title='Forces de rappel')
    fig = ax.get_figure()
    #fig.savefig('cointforcesrappel.pdf')


dfcoint['Dxt'] = dfcoint['xt'].diff()
dfcoint['Dyt'] = dfcoint['yt'].diff()
dfcoint['Dxt1'] = dfcoint['Dxt'].shift(1)
dfcoint['Dyt1'] = dfcoint['Dyt'].shift(1)
dfcoint['cointerr'] = model_coint.resid

ecm1 = smf.ols('Dyt ~ cointerr + Dyt1 + Dxt1',data=dfcoint).fit()
ecm2 = smf.ols('Dxt ~ cointerr + Dyt1 + Dxt1',data=dfcoint).fit()

ecm1.summary()
ecm2.summary()

#%% Cointegrated data: an example

#%% Cointegrated variables, exercise: Euribor 1 year and 3 months
# Data sources:
# http://sdw.ecb.europa.eu/quickview.do?SERIES_KEY=143.FM.M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA
# Euribor 1 year
eu1 = pd.read_csv('R_data/FM.M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA.csv',skiprows=4)
eu1 = eu1.iloc[::-1]
eu1.columns = ['date','Euribor1year']
eu1.index = pd.to_datetime(eu1['date'],format = '%Y%b')
del eu1['date']

# Euribor 3 months
eu3 = pd.read_csv('R_data/FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA.csv',skiprows=4)
eu3 = eu3.iloc[::-1]
eu3.columns = ['date','Euribor3months']
eu3.index = pd.to_datetime(eu3['date'],format = '%Y%b')
del eu3['date']

dfeuribor = pd.concat([eu1,eu3],axis=1)
dfeuribor['spread'] = dfeuribor['Euribor1year'] - dfeuribor['Euribor3months']
if ploton == 1:
    dfeuribor.plot()


dfeuriborx = dfeuribor/dfeuribor.shift(1)-1
dfeuriborx = dfeuriborx.dropna()

# Unit root test, H0: there is a unit root
statsmodels.tsa.stattools.adfuller(dfeuribor['Euribor1year'], regression='c')
statsmodels.tsa.stattools.adfuller(dfeuriborx['Euribor1year'], regression='c')
statsmodels.tsa.stattools.adfuller(dfeuribor['Euribor3months'], regression='c')
statsmodels.tsa.stattools.adfuller(dfeuriborx['Euribor3months'], regression='c')
# both series are I(1)

# Cointegration test
# H0: no cointegration
statsmodels.tsa.stattools.coint(dfeuribor['Euribor1year'],dfeuribor['Euribor3months']) # cointegration at the 5% threshold


# show that the residuals are stationary for this regression
modeleuribor = smf.ols('Euribor1year ~ Euribor3months',data=dfeuribor).fit()
dir(modeleuribor)
# in fact, as demonstrated in Phillips and Ouliaris (1990), one cannot use the ADF test because of the spurious nature of the regression
# in our case, both series have drifts, then unit root test statistics follow the DF distributions adjusted for a constant and trend
statsmodels.tsa.stattools.adfuller(modeleuribor.resid,regression="ct")

# build an ECM with the Euribor rates
dfeuribord = dfeuribor - dfeuribor.shift(1)
dfeuribord['errors'] = modeleuribor.resid.shift(1)
dfeuribord = dfeuribord.dropna()

ecmeuribor = smf.ols('Euribor1year ~ Euribor3months + errors',data=dfeuribord).fit()
ecmeuribor.summary()

# idea: could introduce half-life of shocks

dfeuribord['LE1'] = dfeuribord.Euribor1year.shift(1)
dfeuribord['LE3'] = dfeuribord.Euribor3months.shift(1)


ecmeuribor2 = smf.ols('Euribor1year ~ Euribor3months + errors + LE1 + LE3',data=dfeuribord).fit()
ecmeuribor2.summary()

del dfeuribor, dfeuriborx, eu1, eu3, dfeuribord

#%% Find cointegrated variables
# Import the data set
pwt = pd.read_csv('R_data/pwt90.csv',encoding='latin1')

# keep only the data for Italy
pwt = pwt.loc[pwt.country=='Italy']
# keep only the column with the year, cgdpe and cda data
pwt = pwt.loc[:,['year','cgdpe','cda']]
# this can be done in one line:
#pwt = pwt.loc[pwt.country=='Italy',['year','cgdpe','cda']]

# set the year as the index
pwt.index = pwt['year']
del pwt['year']

# compute yearly changes and drop the first line which has only NA
pwtx = (pwt - pwt.shift(1)) / pwt.shift(1)
pwtx = pwtx.dropna()

if ploton == 1:
    pwt.plot()

#pwtx.plot() # there seem to be a drift in the returns

# Unit root test, H0: there is a unit root
statsmodels.tsa.stattools.adfuller(pwt['cda'], regression='ct')
statsmodels.tsa.stattools.adfuller(pwt['cgdpe'], regression='ct')
statsmodels.tsa.stattools.adfuller(pwtx['cda'], regression='c')
statsmodels.tsa.stattools.adfuller(pwtx['cgdpe'], regression='c')
statsmodels.tsa.stattools.adfuller(pwtx['cda'].diff().dropna(), regression='c')
statsmodels.tsa.stattools.adfuller(pwtx['cgdpe'].diff().dropna(), regression='c')
# both series are I(2), an ECM would not be very well suited

# Plot the Italy's values
if ploton == 1:
    ax = pwt.plot(title = "Italy's Expenditure-side real GDP and Real domestic absorption")
    fig = ax.get_figure()
    #fig.savefig('Italy_cointegrated.pdf')

# Do the test for the United Kingdom and the same variables

del pwt, pwtx





#%% OLS
# we can do a linear regression on stationary series, the returns
results = smf.ols('Population ~ GDP',data = dfallx).fit()
dir(results)
results.summary()
results.rsquared - dfallx.loc[:,['Population','GDP']].corr().iloc[0,1]**2


# are the mean of our series 0, if yes one can apply a regression with no constant term
rnull = dfallx['GDP'] * 0
# T-test: is the mean return statistically different from 0?
# H0: the 2 independent samples have identical average (expected) values
ttest_ind(dfallx['GDP'].dropna(),rnull.dropna())
ttest_ind(dfallx['Population'].dropna(),rnull.dropna())
# both series have a mean different than 0, hence we need to have a constant in the regression

# we cannot do a linear regression with no constant term
results_no_constant = smf.ols('Population ~ GDP - 1',data=dfallx).fit()
results_no_constant.summary()
# compare the R-square for the models, with no constant, it artificially seems high

# manual computation of the beta for the model with no constant
multiplies = dfallx.loc[:,'GDP'] * dfallx.loc[:,'Population']
squareds = dfallx.loc[:,'GDP']**2

betacalcule = multiplies.sum() / squareds.sum()

betacalcule == results_no_constant.params[0]

# manual computation of the beta for the model with a constant
multiplies = dfallx.loc[:,'GDP'] * ( dfallx.loc[:,'Population'] - dfallx.loc[:,'Population'].mean() )
squareds = dfallx.loc[:,'GDP'] * (dfallx.loc[:,'GDP'] - dfallx.loc[:,'GDP'].mean() )

beta1calcule = multiplies.sum() / squareds.sum()

beta1calcule - results.params[1]

beta0calcule = dfallx.loc[:,'Population'].mean() - beta1calcule * dfallx.loc[:,'GDP'].mean()

beta0calcule - results.params[0]

del betacalcule, df, gdp, multiplies, squareds, beta1calcule, beta0calcule, rnull

#%% Visual of the regression

# we tae the constant and the estimated beta
alpha = results.params[0]
beta = results.params[1]
beta2 = results_no_constant.params[0]


# plot the data
stepsize = 0.001
x = np.arange(1.1*dfallx['GDP'].min(),1.1*dfallx['GDP'].max(),stepsize)
if ploton == 1:
    axes = plt.gca()
    axes = plt.scatter(dfallx['GDP'],dfallx['Population'])
    axes = plt.xlabel('French GDP changes')
    axes = plt.ylabel('French Population changes')
    axes = plt.plot(x, alpha+beta*x ,'-')
    axes = plt.plot(x, beta2 * x ,'-')
    #figaxes = axes.get_figure()
    #figaxes.savefig('GDP_Pop_scatter.png')
    del axes

#df=df.resample('Q').mean()

del alpha, beta, beta2, x, stepsize

#%% Outlier detection and regression coefficients robustness
# we detect a first outlier with the minimum value of GDP growth rate
outlier1 = pd.DataFrame(dfallx.sort_values(by='GDP',ascending=True).iloc[0,:])
# second outlier with the maximum growth rate of population
outlier2 = pd.DataFrame(dfallx.sort_values(by='Population',ascending=False).iloc[0,:])

# create dummy variable for both outlier
dfallx['dummy1'] = 0
dfallx.loc[outlier1.columns,'dummy1'] = 1
dfallx['dummy2'] = 0
dfallx.loc[outlier2.columns,'dummy2'] = 1

# outlier 1 test
results_outlier1 = smf.ols('Population ~ GDP + dummy1',data = dfallx).fit()
results_outlier1.summary()
# outlier 2 test
results_outlier2 = smf.ols('Population ~ GDP + dummy2',data = dfallx).fit()
results_outlier2.summary()
# the second outlier seems to be influential
# use the Bonferroni adjustment for the critical threshold
tstatBonferroni = scipy.stats.t.ppf(1-0.05/(2*len(dfallx)),len(dfallx)-2-1)
# we are still above the critical threshold and we reject H0, our dummy2 coefficient is statistically significantly different from 0

# we take the second outlier out of the data set and regress again
results = smf.ols('Population ~ GDP',data = dfallx).fit()

results_outlier2_out = smf.ols('Population ~ GDP',data = dfallx.loc[~ (dfallx.index == outlier2.columns[0]),:]).fit()

results.summary()
results_outlier2_out.summary()

del outlier1, outlier2, tstatBonferroni


#%% QQ plots

# number of generated points
Ni = 10000
# generate points from a standard normal law
x = np.random.normal(0,1,Ni)
x.sort()
#plt.hist(x,bins=50)
#dfQQ.columns = ['sample']
#dfQQ = dfQQ.sort_values(by='sample',ascending=True)
#dfQQ['normalquantile'] = 0
y = []
for i in range(0,Ni):
    #dfQQ.iloc[i,1] = norm.ppf((i+1)/Ni)
    y.append(norm.ppf((i+1)/Ni))
 
if ploton == 1:
    plt.plot(x,y,'r.') # x vs y
    plt.plot(np.arange(-3,3,6/Ni),np.arange(-3,3,6/Ni),'k-') # identity line
    plt.xlabel('empirical quantiles')
    plt.ylabel('theoretical quantiles')
    plt.show()    

del x, y, Ni, i

dir(results)
# we use the normalized residuals
if ploton == 1:
    scipy.stats.probplot(results.resid_pearson, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.show()

if ploton == 1:
    scipy.stats.probplot(results_outlier2_out.resid_pearson, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot, outlier 2 taken out")
    plt.show()

