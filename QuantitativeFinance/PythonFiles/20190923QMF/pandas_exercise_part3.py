#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eric Vansteenberghe
Quantitative Methods in Finance
Beginner exercise with pandas DataFrames - part 3
2018
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats
from scipy.optimize import minimize
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

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
df = df / 1000


#%% Presenting the concept of unit root

# we define a time series following a process with a unit root
yt = [1000]
# we define a shock happening at mid period
shocky = 50
# length of our data
lent = 1000
for i in range(1,lent):
    if i == int(lent/2):
        yt.append(( yt[i-1] +  shocky + np.random.normal(0,1,1))[0])
    else: 
        yt.append(( yt[i-1] + np.random.normal(0,1,1))[0])

dfy = pd.DataFrame(np.matrix([yt]).transpose()) 
dfy.columns = ['yt']

# now compute and plot the rate of change of y and show that it was not impacted by the change except on the date of the impact
dfy['r_yt'] = (dfy.yt - dfy.yt.shift(1)) / dfy.yt.shift(1)

if ploton == 1:
    ax = dfy.yt.plot(title='time series with a unit root and a shock at mid period')
    fig=ax.get_figure()
    fig.savefig('unitrootjump.pdf')
    plt.close()
    ax = dfy['r_yt'].plot(title='returns of the time series')
    fig=ax.get_figure()
    fig.savefig('rateunitrootjump.pdf')
    
# Augmented Dickey-Fuller test: is there a unit root?
# H0: there is a unit root
adfuller(dfy.yt, regression='nc')
# imposing 'nc' to regression mean that we assume a random walk
# imposing 'c' means you assume a random walk with a drift
# p-value high in any case, we have a high probability to wrongly reject H0 we do notreject H0
# we assume thate there is a unit root
adfuller(dfy['r_yt'].dropna(), regression='nc')

del dfy, i, lent, shocky, yt

#%% Before regressing, a spurious regression

# we create two independent variables
# we loop over several model generation and compute the mean R square

# number of loops
looplen = 20

# to store the adjusted Rsquares
Rsquarelist = []

for loopi in range(0,looplen):
    yt = [0]
    xt = [0]
    # length of our data
    lent = 1000
    for i in range(1,lent):
        yt.append((yt[i-1] + np.random.normal(0,1,1))[0])
        xt.append((xt[i-1] + np.random.normal(0,1,1))[0])
    
    dfspur = pd.DataFrame(np.matrix([yt,xt]).transpose())
    dfspur.columns = ['yt','xt']
    
    # regress one series on the other
    results_spur = smf.ols('yt ~ xt',data=dfspur).fit()
    
    #results_spur.summary()
    Rsquarelist.append(results_spur.rsquared_adj)

# we expect the R^2 to be close to zero, but we find that this is not the case
np.mean(Rsquarelist)

# Example with the last two draws:
results_spur.summary()
dfspur.corr()
if ploton == 1:
    ax = dfspur.plot(title='Two independent time series, with both a unit root')
    fig = ax.get_figure()
    #fig.savefig('spuriousillustration.pdf')

del looplen, Rsquarelist, yt, xt, lent, dfspur, loopi, i
        

#%% AR(2), non stationary process with a unit root

# we define an AR(2) process
# process length AR2length
AR2length = 100
ARp2 = []
alphaAR2 = 0
beta1 = 1.6
beta2 = -0.6
# check that beta1 + beta2 == 1
beta1 + beta2 == 1
# if you want to convince yourself of the non-stationarity, you can input a small shock and the effect doesn't vanish
epsilon0 = 10**(-10)
epsilon1 = 0
ARp2.append(alphaAR2 + epsilon0)
ARp2.append(alphaAR2 + beta1 * ARp2[0] + epsilon1)
for ti in range(2,AR2length):
    ARp2.append(alphaAR2 + beta1 * ARp2[ti-1] +  beta2 * ARp2[ti-2])

# plot our AR(2) process
if ploton == 1:
    plt.plot(ARp2)

# compute the matrix of the AR(2) process in a VAR form
ARp2matrix = pd.DataFrame([[beta1, beta2],[1,0]])

# compute the eigenvalues of the process
np.abs(np.linalg.eig(ARp2matrix)[0]) # there is a unit root
if ploton ==1:
    plt.plot(pd.DataFrame(ARp2).diff())
    
# Augmented Dickey-Fuller test: is there a unit root?
# H0: there is a unit root
adfuller(ARp2[20:], regression='nc')
# imposing 'nc' to regression mean that we assume a random walk
adfuller(ARp2[20:], regression='c')
# imposing 'c' means you assume a random walk with a drift
# NB: here we remove the first 20 observations as there are just noize before the signal stabilize
# this is close to the concept of burn in

# as an illustration, if we do not remove the initial 20 first observations, our conclusion differs:
adfuller(ARp2, regression='nc')
adfuller(ARp2, regression='c')

del ARp2, ARp2matrix, alphaAR2, beta1, beta2, epsilon0, epsilon1, ti

# exercise: change the values of beta1 and beta2 to have a stationary process

#%% Back to building an AR(p) of population

# demeaned population growth rate
ytild = ((df - df.shift(1)) / df.shift(1)) - ((df - df.shift(1)) / df.shift(1)).mean()
ytild = ytild.dropna()

if ploton == 1:
    ytild.plot()

adfuller(ytild['Population'], regression='nc')

# prepare for the OLS
dftild = pd.concat([ytild, ytild.shift(1), ytild.shift(2), ytild.shift(3)], axis=1).dropna(how='any')
dftild.columns = ['yt','ytminus1','ytminus2','ytminus3']

# OLS
modeltild = smf.ols('yt ~ ytminus1 - 1',data=dftild).fit()
# show the result of our OLS
modeltild.summary()

# the coefficient of our AR(1) would be
thetatild = modeltild.params[0]

# ex: show with Monte Carlo method that the estimation of an AR(1) wit OLS regression leads to biased estimate

# are the residuals serially-correlated: cf. Ljung-Box test
# in case of residuals serial-correlation we would need to increse the order p of the AR

# ex: using thetatild, residuals plot projections and confidence interval

#AR(3)
# OLS
modeltild3 = smf.ols('yt ~ ytminus1 + ytminus2 + ytminus3 - 1',data=dftild).fit()
# show the result of our OLS
modeltild3.summary()

#plot_pacf(ytild,lags=10)

# AR(1) log-likelihood
def loglikeAR1(thetai,seriei):
    # we compute the standard deviation of the innovations
    voli = (seriei - thetai * seriei.shift(1)).std()[0]**2
    fun = len(seriei) * np.log(2*np.pi) + np.log( voli / (1-thetai**2 )) + seriei.iloc[0,0]**2 / (voli / (1-thetai**2 )) + (len(seriei)-1) * np.log(voli)
    fun += (( 1 / voli) *  (seriei-thetai*seriei.shift(1))**2).sum()[0]
    return fun 

# we maximize the log-likelihood (or minimize -log-lik)
#value to start seraching for theta
thetastart = 0.8
MLAR1 = minimize(loglikeAR1, [thetastart], args=(ytild), method='Nelder-Mead')
print('estimated theta',MLAR1.x)
print('success of ML estimation',MLAR1.success)

# Fit an ARMA with the procedure and test with AIC or BIC
ARMAfit = sm.tsa.arma_order_select_ic(ytild, ic=['aic', 'bic'], trend='nc', max_ma=0)
ARMAfit.aic_min_order
ARMAfit.bic_min_order


# fit the arma
model = ARIMA(ytild, order=(3,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
dir(model_fit)
if ploton == 1:
    ax = model_fit.resid.plot()
    fig = ax.get_figure()
    fig.savefig('AR3_resid.pdf')


#%% Ljung-Box test
# function to compute serial correlation
def autocorr(x, t):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0,1]

# compute the Ljung-Box Q-statistic:
def qstatfun(x,t):
    sumpart = 0
    for i in range(1,t+1):
        sumpart += sumpart + (autocorr(x,i)**2) / (len(x)-i)        
    return (len(x) * (len(x)+2)) * sumpart

# Ljung-Box test, H0: no serial correlation at lag max
# out: critical value at 5% level, Qstat and p-value
def LjungBox(x,t,ARorder):
    Qstat = qstatfun(x,t)
    threshold = scipy.stats.chi2.ppf(q = 0.95, df = t-ARorder)
    pvalue = 1 - scipy.stats.chi2.cdf(x=Qstat,  df = t-ARorder)
    return threshold, Qstat, pvalue

# Ljung-Box test
maxlag = 10

dfLB = pd.DataFrame(index=['Q-statistic','p-value'],columns=['AR3'])
test = LjungBox(model_fit.resid,maxlag,3)
dfLB.loc['Q-statistic','AR3'] = test[1]
dfLB.loc['p-value','AR3'] = test[2]
