#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 5
2019
"""

import pandas as pd
import os
import numpy as np
import statsmodels.tsa.stattools
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.linalg as la # for LDL decomposition

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


#%% Unemployment data in France
u = pd.read_csv('R_data/unemployment_france.csv',sep=';',encoding='latin1',skiprows = [0,1])
u = u.iloc[::-1]
u.columns = ['Quarter','u']
dates4 = pd.date_range('1996-01', '2016-09', freq='Q') 
u.index = dates4
u = u.drop('Quarter',1)
u = u.replace({' ': ''}, regex=True) 
u = u.replace({',':'.'}, regex=True) 
u = u.astype(float)

del dates4

#%% Concatenate all variables in one data frame
dfall = pd.concat([u,df,gdp],axis=1)
dfall = dfall.dropna()
dfallx = dfall / dfall.shift(1) - 1

#%% VAR model

# plot the return data
if ploton == 1:
    dfallx.plot(secondary_y = 'u')

# drop rows with missign values (NA)
dfallx = dfallx.dropna()

# VAR with population and GDP growth rates
dfallx2 = dfallx.loc[:,['Population','GDP']]

# create a VAR
varmodel = VAR(dfallx2,freq='Q')

# select the order of the VAR
print(varmodel.select_order(maxlags = 15,trend = 'c'))
# you might want to check with no constant term
# print(varmodel.select_order(maxlags = 15,trend = 'nc'))
# select lag order 1
result_var = varmodel.fit(1)
# with no constant term
#result_var = varmodel.fit(1,trend = 'nc')
result_var.summary()

# check what information you can get from the VAR
dir(result_var)

# get the matrix
matrixVAR = result_var.params.iloc[1:,]

# compute the eigenvalues
eigenvaluesVAR = np.linalg.eig(matrixVAR)[0]
sum(eigenvaluesVAR >= 1)
# no eigenvalue greater or equal to 1, our VAR is stable
# then the reduced form VAR presented in equation can be consistently 
# estimated by OLS equation by equation.

# other way to check for the VAR stability
# our VAR is stable if the roots lies outside the unit circle
sum(result_var.roots <= 1) 
# notroots are greater or equal to one, our VAR is stable

# add lagged variables
dfallx2['GDP1'] = dfallx2['GDP'].shift(1)
dfallx2['Population1'] = dfallx2['Population'].shift(1)

# estimation by OLS of the reduced form VAR
resultpop = smf.ols('Population ~ Population1 + GDP1',data = dfallx2).fit()
resultGDP = smf.ols('GDP ~ Population1 + GDP1',data = dfallx2).fit()

resultpop.summary()
resultGDP.summary()
# indeed, as no eigenvalue greater or equal to 1, the coefficients are the same


# compute the matrix D from the schock terms
resultpop = smf.ols('Population ~ GDP + Population1 + GDP1',data = dfallx2).fit()
resultGDP = smf.ols('GDP ~ Population + Population1 + GDP1',data = dfallx2).fit()
Dmverif = np.matrix([[resultpop.resid.std()**2,0],[0,resultGDP.resid.std()**2]])

# compute the Omega matrix from the forecast error terms
Omegam = np.cov(result_var.resid.T)
# we do a Cholesky decomposition, a LDL decomposition
Binv, Dm, p = la.ldl(Omegam)

# check our decomposition
Omegam - Binv.dot(Dm).dot(Binv.T)

# We found the matrix B
Bm = np.linalg.inv(Binv)

# In practice, this decomposition is done for you
# so you can use the irf command

# plot impulse response functions
if ploton == 1:
    result_var.irf().plot()

# Granger causality
print(result_var.test_causality('Population','GDP'))

#%% Add unemployment to the VAR
    # create a VAR
varmodel = VAR(dfallx,freq='Q')

# select the order of the VAR
print(varmodel.select_order(maxlags = 15,trend = 'c'))
# you might want to check with no constant term
#print(varmodel.select_order(maxlags = 15,trend = 'nc'))
# select lag order 1
result_var = varmodel.fit(1)

# add the unemployment and test Granger causality between GDP and unemployment
print(result_var.test_causality('u','GDP'))


del dfall, dfallx, matrixVAR, eigenvaluesVAR


#%% From Regis Bourbonnais book "Econometrie" Dunod

# import the data set, drop missing values, set the dates as index
bour = pd.read_excel('R_data/C10EX2.XLS')
bour = bour.dropna(axis=0)
bour.index = pd.date_range('2001-01', '2019-01', freq='Q')
bour = bour.drop('Date',axis=1)

#bour.plot()

# ADF test
statsmodels.tsa.stattools.adfuller(bour['Y1'], regression='nc')
statsmodels.tsa.stattools.adfuller(bour['Y2'], regression='nc')
# both series are stationary

# we define a function to print if the series has unit root or not
def hasUR(df,threshold):
    if statsmodels.tsa.stattools.adfuller(df, regression='nc')[1] > threshold:
        print("your series has a unit root")
    else:
        print("Your series doesn't seem to have unit root")
    
# we set a threshold at 5% for the p-value of or ADF test
thresholdi = 0.05
hasUR(bour['Y1'],thresholdi)
hasUR(bour['Y2'],thresholdi)

# apply the VAR model
varmodelb = VAR(bour)

#Lag selection
print(varmodelb.select_order(15)) # we select order 1
# fit a VAR(1)
result_varb = varmodelb.fit(1)
result_varb.summary()

#  check for the VAR stability
# our VAR is stable if the roots lies in the unit circle
sum(result_varb.roots <= 1)
sum(result_varb.roots < 1) # one root is greater than one, our VAR is not stable, no possibility of cointegration


# is Y2 causing Y1, our test says yes
print(result_varb.test_causality('Y1','Y2'))
#print(result_varb.test_causality('Y2','Y1'))

# impulse response funciton plot
if ploton == 1:
    result_varb.irf().plot()

#Let's define a moving OLS ourselves:
def movOLS(df,window):
    Betas = []
    df.columns = ['X','Y']
    for i in range(0,(len(df)-window)):
        resultat = sm.OLS(df[i:(i+window)].X,df[i:(i+window)].Y).fit()
        Betas.append(resultat.params[0])
    return Betas;
    
window = 10
betas = movOLS(bour[['Y2','Y1']],window)
if ploton == 1:
    pd.DataFrame(betas).plot(title = 'evolution of the beta over time')

debut = pd.DataFrame(np.zeros(window))

Betas = debut.append(betas)
Betas.index = bour.index

bour['Y2_hatbis'] = Betas.multiply(bour['Y1'],axis=0)
if ploton == 1:
    bour[['Y2_hatbis','Y2']].plot()

#diff=bour['Y2_hat']-bour['Y2_hatbis']
#diff.plot()

del Betas, betas, bour, debut, thresholdi, window


#%% Wheat production and price:
wheatprice = pd.read_csv('R_data/146908e8-7a8a-40ea-b670-b39daab67a15.csv')
wheatprod = pd.read_csv('R_data/b24e9b0e-4b97-4acc-90f8-599cc178d434.csv')

wheatprice2 = wheatprice.groupby('Year')['Value'].mean()
wheatprod2 = wheatprod.groupby('Year')['Value'].sum()

wheat = pd.concat([wheatprice2,wheatprod2],axis=1).dropna(axis=0)
wheat.columns = ['price','supply']
if ploton == 1:
    wheat.plot(secondary_y='price',title='Wheat price and supply, yearly')

## equivalence between diff() and using shift(1)
#wheat1stdiff = wheatprod2.diff()
#wheat1stdiffcheck = wheatprod2 - wheatprod2.shift(1)
#
#dfcompare = pd.concat([wheat1stdiff,wheat1stdiffcheck], axis=1)
#dfcompare.plot()

# Correlation
wheat.corr()


#ADF test
statsmodels.tsa.stattools.adfuller(wheat['price'])
statsmodels.tsa.stattools.adfuller(wheat['supply'])
# both time series have a unit root

# we compute the growth rate
wheatx = wheat/wheat.shift(1) - 1
wheatx = wheatx.dropna(axis=0)
#ADF test
statsmodels.tsa.stattools.adfuller(wheatx['price'])
statsmodels.tsa.stattools.adfuller(wheatx['supply'])
# both time series are now stationary

#%% Cointegration test
statsmodels.tsa.stattools.coint(wheat.price,wheat.supply)
# The probability to wrongly reject H0 is too high, we accept it, there is no cointegration
# No need for a VECM


#%% compare supply and price

# is the supply at the next period influenced by the price today? and vice-versa?
wheatx.loc[:,'supply1'] = wheatx.loc[:,'supply'].shift(1)
wheatx.loc[:,'price1'] = wheatx.loc[:,'price'].shift(1)

wheatx.loc[:,['price','supply1']].corr()
wheatx.loc[:,['price1','supply']].corr()

if ploton == 1:
    ax = wheatx.plot(title = '1: means lag at t-1')
    fig = ax.get_figure()
    #fig.savefig('wheatsupply1.PNG')

wheat['price1'] = wheat['price'].shift(1)

if ploton == 1:
    axw = wheat.loc[:,['price','supply']].plot(secondary_y = 'price',title='wheat price and supply in levels')
    figw = axw.get_figure()
    #figw.savefig('wheatsupplylevel.pdf')

if ploton == 1:
    axw = wheat.loc[:,['price1','supply']].plot(secondary_y = 'price1',title='wheat price and supply a time t + 1 in levels')
    figw = axw.get_figure()
    #figw.savefig('wheatsupplylevel1.pdf')



# EXERCISE: apply a VAR and test for Granger causality, then do impulse response functions


#%% Rice production and prices:
riceprod = pd.read_csv('R_data/5413f61a-51fe-423f-a942-64990af7d5e1.csv')
riceprice = pd.read_csv('R_data/cc87ba8c-13ed-4ba1-ba6e-f3584f8395bc.csv')

riceprice2 = riceprice.groupby('Year')['Value'].mean()
riceprod2 = riceprod.groupby('Year')['Value'].sum()

rice = pd.concat([riceprice2,riceprod2],axis=1).dropna(axis=0)
rice.columns = ['price','supply']
if ploton == 1:
    rice.plot(secondary_y='price')

rice.corr()

# EXERCISE: apply a VAR and test for Granger causality, then do impulse response functions
