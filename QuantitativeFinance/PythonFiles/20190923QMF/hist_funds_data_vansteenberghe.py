#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2019

@author: Eric Vansteenberghe
Quantitative Methods in Finance
Asset returns risk modelisation and selection, VaR and ES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import t
from scipy.stats import levy_stable
import os
#We set the working directory (useful to chose the folder where to export output files)
os.chdir('/Users/skimeur/Google Drive/empirical_finance')
from pylevy import fit_levy

# if you want to plot, set ploton to 1
ploton = 0
# if you want to compute the loops (time consuming)
loopcomp = 0


#%% Data Import
df = pd.read_csv('R_data/fonds_merged_2018_08_28.csv',index_col=0)
df.index = pd.to_datetime(df.index,format="%Y-%m-%d")


#%% clean the data set: find anomalies

# filter the series by business days
#create an index of just the date portion of your index (this is the slow step)
ts_days = pd.to_datetime(df.index.date)

#create a range of business days over that period
bdays = pd.bdate_range(start=df.index[0].date(), end=df.index[-1].date())

#Filter the series to just those days contained in the business day range.
df = df[ts_days.isin(bdays)]

# when we have prices at 0, this could be default of the company or lack of data
dfzeros = df.min().sort_values()

# list of stock with prices equal or below 0
neglist = list(dfzeros.loc[dfzeros<=0].index)
if ploton == 1:
    for negi in neglist[:4]:
        df.loc[:,negi].plot(title=negi)
        plt.pause(1)
        plt.close()

zerolist = list(dfzeros.loc[dfzeros==0].index)
if ploton == 1:
    for negi in zerolist:
        df.loc[:,negi].plot(title=negi)
        plt.pause(1)
        plt.close()
        print(df.loc[:,negi].mean())

# remove which mean are 0
#df = df.loc[:,df.mean()>0]

# we remove those three stocks stuck at zero
stocklist = list(set(list(df.columns)) - set(zerolist))

df = df.loc[:,stocklist]

df.to_csv('R_data/fonds_merged_2018_11_07.csv')

#%% fill NA

# forward fill
df = df.fillna(method='ffill')
# then backward fill
df = df.fillna(method='bfill')
df.head()

df.to_csv('R_data/fonds_merged_2018_11_07.csv')


#%% Plotting some data

if ploton == 1:
    # create a data frame with only the assets for which we have price information every day
    dffull = df.dropna(axis =1,how='any')
    ax = df.loc[:,['BNPP.PA','CNAT.PA','LP60038754']].plot(secondary_y='LP60038754')
    fig = ax.get_figure()
    fig.savefig('plot_3_assets.pdf')
    # in order to plot 20 randomly selected assets we divide the series by their means to have comparable scales
    dffull = dffull / dffull.mean()
    dffull.sample(20,axis = 1).plot(title = "Plot a random sample of assets prices, divided by their mean over the period")

#%% compute the dailty returns
dfx = (df - df.shift(1)) / df.shift(1)
del df

dfx = dfx.replace([np.inf, -np.inf], np.nan)

# remove returns with length less than one year of data 252 days
dayslimit = 252
dfx = dfx.loc[:,dfx.count() > dayslimit]
del dayslimit


# winsorize the returns
#dfxwinsorized = dfx.copy(deep = True)
#for asseti in dfx.columns:
#    dfxwinsorized.loc[:,asseti] = list(scipy.stats.mstats.winsorize(dfx[asseti], limits = 0.001))


# cap the returns by -200% and 200%
dfxwinsorized = dfx.clip(-2,2)

# count the number of days with positive returns
dfx[dfx > 0].count().sum()
# count the number of days with negative returns
dfx[dfx < 0].count().sum()

# ratio of postivie daily returns over total (note that many daily returns are null)
dfx[dfx > 0].count().sum() / ( dfx[dfx > 0].count().sum() + dfx[dfx < 0].count().sum())



#%% Assuming normal distribution and computing the VaR and ES
    

#dfx.sample(1,axis=1).plot()

# do it for the fist asset, BNP
muBNP = dfx['BNPP.PA'].mean()
sigmaBNP = dfx['BNPP.PA'].std()

# compute the quantile, one day out of 252 trading days
q = scipy.stats.norm.ppf(1 - 1 / 252)

# parametric VaR at 99.6%
VaR99_6BNP_param = muBNP - q * sigmaBNP

# historical VaR at 99.6%
rBNPsorted = dfx['BNPP.PA'].sort_values()
VaR99_6BNP_hist = rBNPsorted.iloc[round(len(rBNPsorted)*1 / 252)]

# parametric ES at 99.6%
ES99_6BNP_param = muBNP - 252 * scipy.stats.norm.pdf(scipy.stats.norm.ppf(1/252)) * sigmaBNP

# historical ES at 99.6%
ES99_6BNP_hist = rBNPsorted.iloc[0:round(len(rBNPsorted)*1 / 252)].mean()

#%% Are the returns normally distributed?

# we generate random returns from the normal law using the mean and standard deviation computed
#We generate random numbers
rBNPnormal = np.random.normal(muBNP, sigmaBNP, len(dfx['BNPP.PA'].dropna(axis=0)))
#We want to plot both histogram one above the other, are the return following a normal law from what you can observe?
if ploton==1:
    bins = np.linspace(dfx['BNPP.PA'].min(), dfx['BNPP.PA'].max(), 200)
    plt.hist(dfx['BNPP.PA'].dropna(axis=0), bins, alpha=0.5, label='r')
    plt.hist(rBNPnormal, bins, alpha=0.5, label='r if normal')
    plt.legend(loc='upper right')
    plt.savefig('normal_BNP.pdf')
    plt.show()
    plt.close()
    
#Kolmogorov-Smirnov test: low p-value we reject normal distribution of BNP's returns
pvalueBNPnormal = scipy.stats.ks_2samp(dfx['BNPP.PA'].dropna(axis=0),rBNPnormal)[1]

#%% Student t distribution
# standardize returns (zero mean and one standard deviation)
dfxsd = (dfx - dfx.mean())/dfx.std()

# fit a t Student distribution to the standardized returns
rBNPstudentdf = t.fit(dfxsd['BNPP.PA'].dropna(axis=0),floc=0,fscale=1)
# generate returns from this distribution
rBNPstudent = t.rvs(rBNPstudentdf[0] , size=len(dfx['BNPP.PA'].dropna(axis=0)))
#We want to plot both histogram one above the other, are the return following a normal law from what you can observe?
if ploton==1:
    bins = np.linspace(dfxsd['BNPP.PA'].min(), dfxsd['BNPP.PA'].max(), 200)
    plt.hist(dfxsd['BNPP.PA'].dropna(axis=0), bins, alpha=0.5, label='r')
    plt.hist(rBNPstudent, bins, alpha=0.5, label='r if student')
    plt.legend(loc='upper right')
    plt.savefig('student_BNP.pdf')
    plt.show()
    plt.close()
    
#Kolmogorov-Smirnov test: low p-value we reject Student's-t distribution of BNP's returns
pvalueBNPstudent = scipy.stats.ks_2samp(dfxsd['BNPP.PA'].dropna(axis=0),rBNPstudent)[1]
 
#%% Assuming Student t distribution and computing the VaR and ES

# parametric VaR at 99.6%
VaR99_6BNP_param_stud = muBNP -((rBNPstudentdf[0]-2)/rBNPstudentdf[0])**(0.5) * t.ppf(1-1/252, rBNPstudentdf[0]) * sigmaBNP

# parametric ES at 99.6%
ES99_6BNP_param_stud = muBNP + (252) * (1-rBNPstudentdf[0])**(-1) * (rBNPstudentdf[0] - 2 + t.ppf(1/252, rBNPstudentdf[0])**2) * t.pdf(t.ppf(1/252, rBNPstudentdf[0]), rBNPstudentdf[0]) * sigmaBNP

# generate returns from the Student's t distribution
samplesize = 10**6
rBNPstudentdf = t.fit(dfx['BNPP.PA'].dropna(axis=0),floc=dfx['BNPP.PA'].mean(),fscale=dfx['BNPP.PA'].std())
rBNP_student_MC = pd.DataFrame(t.rvs(rBNPstudentdf[0] , size= samplesize))
rBNP_student_MC.columns = ['returns']
rBNP_student_MC = dfx['BNPP.PA'].mean() + dfx['BNPP.PA'].std() *  rBNP_student_MC * ((rBNPstudentdf[0]-2)/rBNPstudentdf[0])**(0.5)
VaR99_6BNP_param_MC_stud = rBNP_student_MC.dropna(axis=0).sort_values(by='returns').iloc[round(len(rBNP_student_MC.dropna(axis=0)) * 1 / 252)].iloc[0]
ES99_6BNP_param_MC_stud = rBNP_student_MC.dropna(axis=0).sort_values(by='returns').iloc[0:round(len(rBNP_student_MC.dropna(axis=0)) * 1 / 252)].mean().iloc[0]

#%% Stable law
# for the levy stable law, we import the PyLevy package from https://github.com/josemiotto/pylevy/tree/master/levy
# fit a stable law and find the parameters
rBNPlevydf = fit_levy(dfxsd['BNPP.PA'].dropna(axis=0))
print('alpha={:.2f}, beta={:.2f}, mu_0={:.2f}, sigma={:.2f}, neglog={:.2f}'.format(*rBNPlevydf))
# generate returns from the stable law
rBNPlevy = levy_stable.rvs(rBNPlevydf[0],rBNPlevydf[1],loc=rBNPlevydf[2],scale=rBNPlevydf[3],random_state=None,size=len(dfx['BNPP.PA'].dropna(axis=0)))
#We want to plot both histogram one above the other, are the return following a normal law from what you can observe?
if ploton==1:
    bins = np.linspace(dfxsd['BNPP.PA'].min(), dfxsd['BNPP.PA'].max(), 200)
    plt.hist(dfxsd['BNPP.PA'].dropna(axis=0), bins, alpha=0.5, label='r')
    plt.hist(rBNPlevy, bins, alpha=0.5, label='r if Levy stable')
    plt.legend(loc='upper right')
    plt.savefig('levy_BNP.pdf')
    plt.show()
    plt.close()
    
#Kolmogorov-Smirnov test: p-value near 5%, we just reject normal distribution of BNP's returns
pvalueBNPlevy = scipy.stats.ks_2samp(dfxsd['BNPP.PA'].dropna(axis=0),rBNPlevy)[1]
 
#%% Assuming stable law and computing the VaR and ES 
# generate returns from the stable law
samplesize = 10**6
rBNPlevydf = fit_levy(dfx['BNPP.PA'].dropna(axis=0))
rBNPlevy_MC = pd.DataFrame(levy_stable.rvs(rBNPlevydf[0],rBNPlevydf[1],loc=rBNPlevydf[2],scale=rBNPlevydf[3],random_state=None,size=samplesize))
rBNPlevy_MC.columns = ['returns']

VaR99_6BNP_MC_levy = rBNPlevy_MC.sort_values(by='returns').iloc[round(len(rBNPlevy_MC.dropna(axis=0))*1 / 252)].iloc[0]
ES99_6BNP_MC_levy = rBNPlevy_MC.sort_values(by='returns').iloc[0:round(len(rBNPlevy_MC.dropna(axis=0))*1 / 252)].mean().iloc[0]

#%% parametric and historical VaR and ES of the sample

# compute the parametric and historical VaR  and ES for each asset
dVaR = pd.DataFrame(columns=['historical VaR','parametric VaR normal','parametric VaR student','MC VaR levy','historical ES','parametric ES normal','parametric ES student','MC ES levy'],index=dfx.columns)
# compute the quantile, one day out of 252 trading days
q = scipy.stats.norm.ppf(1 - 1 / 252)
samplesize = 10**5
if loopcomp == 1:
    for asseti in dfx.columns:
        try:
        # fit the Student's-t distribution
            rstudentdf = t.fit(dfx[asseti].dropna(axis=0),floc=dfx[asseti].mean(),fscale=dfx[asseti].std())
            dVaR.loc[asseti,'parametric VaR student'] = dfx[asseti].mean() -((rstudentdf[0]-2)/rstudentdf[0])**0.5 * t.ppf(1-1/252, rstudentdf[0]) * dfx[asseti].std()
            dVaR.loc[asseti,'parametric ES student'] = dfx[asseti].mean() + 252 * (1-rstudentdf[0])**(-1) * (rstudentdf[0] - 2 + t.ppf(1/252, rstudentdf[0])**2) * t.pdf(t.ppf(1/252, rstudentdf[0]), rstudentdf[0]) * dfx[asseti].std()
            dVaR.loc[asseti,'parametric VaR normal'] = dfx[asseti].mean() - q * dfx[asseti].std()
            dVaR.loc[asseti,'historical VaR'] = dfx[asseti].dropna(axis=0).sort_values().iloc[round(len(dfx[asseti].dropna(axis=0))*1 / 252)]
            dVaR.loc[asseti,'parametric ES normal'] = dfx[asseti].mean() - 252 * scipy.stats.norm.pdf(scipy.stats.norm.ppf(1/252)) * dfx[asseti].std()
            dVaR.loc[asseti,'historical ES'] = dfx[asseti].dropna(axis=0).sort_values().iloc[0:round(len(dfx[asseti].dropna(axis=0))*1 / 252)].mean()   
            rlevydf = fit_levy(dfx[asseti].dropna(axis=0))
            rlevy_MC = pd.DataFrame(levy_stable.rvs(rlevydf[0],rlevydf[1],loc=rlevydf[2],scale=rlevydf[3],random_state=None,size=samplesize))
            rlevy_MC.columns = ['returns']
            dVaR.loc[asseti,'MC VaR levy'] = rlevy_MC.sort_values(by='returns').iloc[round(len(rlevy_MC.dropna(axis=0))*1 / 252)].iloc[0]
            dVaR.loc[asseti,'MC ES levy'] = rlevy_MC.sort_values(by='returns').iloc[0:round(len(rlevy_MC.dropna(axis=0))*1 / 252)].mean().iloc[0]
        except:
            print('we could not fit distributions')
    dVaR.to_csv('R_data/dVaR.csv')
else:
    dVaR = pd.read_csv('R_data/dVaR.csv',index_col=0)
# we observe that our parameteric risk measures are less conservative than the historical observations
dVaR['diff VaR normal'] = dVaR['parametric VaR normal'] - dVaR['historical VaR']
dVaR['diff ES normal'] = dVaR['parametric ES normal'] - dVaR['historical ES']
dVaR['diff VaR student'] = dVaR['parametric VaR student'] - dVaR['historical VaR']
dVaR['diff ES student'] = dVaR['parametric ES student'] - dVaR['historical ES']

outVaRlatex = dVaR.median().round(3).to_latex()



#%% Test for all the assets in our data frame

# for each distribution type and for each asset, we store the p-value of the Kolmogorov-Smirnov test in a dataframe
if loopcomp == 1:
    dtest = pd.DataFrame(columns=['KS p-value normal','KS p-value student','KS p-value stable','t-Student nu','levy alpha','levy beta','levy mu','levy sigma','levy neglog'],index=dfxsd.columns)
    for asseti in dfxsd.columns:
        try:
            dtest.loc[asseti,'KS p-value normal'] = scipy.stats.ks_2samp(dfxsd[asseti].dropna(axis=0),np.random.normal(dfxsd[asseti].mean(), dfxsd[asseti].std(), len(dfxsd[asseti].dropna(axis=0))))[1]
            rstudentdf = t.fit(dfxsd[asseti].dropna(axis=0),floc=0,fscale=1)
            dtest.loc[asseti,'t-Student nu'] = rstudentdf[0]
            dtest.loc[asseti,'KS p-value student'] = scipy.stats.ks_2samp(dfxsd[asseti].dropna(axis=0),t.rvs(rstudentdf[0] , size=len(dfxsd[asseti].dropna(axis=0))))[1]
            rlevydf = fit_levy(dfxsd[asseti].dropna(axis=0))
            dtest.loc[asseti,'levy alpha'] = rlevydf[0]
            dtest.loc[asseti,'levy beta'] = rlevydf[1]
            dtest.loc[asseti,'levy mu'] = rlevydf[2]
            dtest.loc[asseti,'levy sigma'] = rlevydf[3]
            dtest.loc[asseti,'levy neglog'] = rlevydf[4]
            dtest.loc[asseti,'KS p-value stable'] = scipy.stats.ks_2samp(dfxsd[asseti].dropna(axis=0),levy_stable.rvs(rlevydf[0],rlevydf[1],loc=rlevydf[2],scale=rlevydf[3],random_state=None,size=len(dfxsd[asseti].dropna(axis=0))))[1]
        except:
            print('we could not fit distributions')
    dtest.to_csv('R_data/dtest.csv')
else:
    dtest = pd.read_csv('R_data/dtest.csv',index_col=0)


# NB: the test are not stable and several random draws should be done before concluding!

normal = dtest.loc[dtest['KS p-value normal']>0.05]
# we find 82 candidates for the normal distribution
studentl = dtest.loc[dtest['KS p-value student']>0.05]
# we find 71 candidates for the Student t distribution
stablel = dtest.loc[dtest['KS p-value stable']>0.05]
# we find 419 candidates for the Levy stable law, which is less than 10% of the initial sample

liststable = list(stablel.index)

dVaRstable = dVaR.loc[liststable,'MC ES levy']
dVaRstable.columns = ['ES']
dVaRstable = dVaRstable.astype(float)
if ploton == 1:
    ax = dVaRstable.hist(bins=200)
    fig = ax.get_figure()
    fig.savefig('ESstablehist.pdf')
    plt.close(fig)




#%% Working with the noraml and Student's t distributions
if ploton == 1:
    # we want to plot the normal distribution N(0,1)
    dx = 0.0001  # resolution
    x = np.arange(-5, 5, dx)
    # normal distribution N(0,1)
    pdf = scipy.stats.norm.pdf(x, 0, 1)
    # t Student distribution, nu = 5
    nu = 5
    pdf2 = t.pdf(x, nu, 0, 1)
    
    alpha = 0.05  # significance level
    StudenthVaR = -((nu-2)/nu)**0.5 * t.ppf(1-alpha, nu)
    NormalhVaR = -scipy.stats.norm.ppf(1-alpha)
    
    plt.figure(num=1, figsize=(11, 6))
    plt.plot(x, pdf, 'b', label="Normal PDF")
    plt.axis("tight")
    plt.plot(x, pdf2, 'g', label="Student t PDF, nu = 5")
    # Student VaR line
    plt.plot([StudenthVaR, StudenthVaR], [0, 1], c='g')
    # Normal VaR line
    plt.plot([NormalhVaR, NormalhVaR], [0, 1], c='b')
    plt.text(NormalhVaR - 1, 0.37, "Norm VaR", color='b')
    plt.text(StudenthVaR + 0.1 , 0.05, "Student t VaR", color='g')
    plt.xlim([-5, 5])
    plt.ylim([0, 0.4])
    plt.legend(loc="best")
    plt.xlabel("Return value")
    plt.ylabel("Probability of occurence")
    plt.show()
    plt.close()




#%% create an index equally weighed


# find the VaR at 20%
quantilei = 0.2
VaR20 = dfxwinsorized['BNPP.PA'].dropna(axis=0).sort_values().iloc[round(len(dfxwinsorized['BNPP.PA'].dropna(axis=0)) * quantilei)]
if ploton == 1:
    ax = dfxwinsorized['BNPP.PA'].hist(cumulative = True,bins = 100,normed = True)
    ax.axvline(x=VaR20, color='r', linestyle='dashed', linewidth=2)
    fig = ax.get_figure()
    fig.savefig('BNPcdf.pdf')
    plt.close(fig)

# find for each asset the dates at which it crosses the likelihood
if loopcomp == 1:
    quantilei = 0.01
    scoredates = dfxwinsorized.copy(deep=True)
    scoredates = scoredates * 0
    for asseti in scoredates.columns:
        VaRi = dfxwinsorized[asseti].dropna(axis=0).sort_values().iloc[round(len(dfxwinsorized[asseti].dropna(axis=0)) * quantilei)]
        scoredates.loc[:,asseti] = dfxwinsorized[asseti] < VaRi
    scoredates.to_csv('scoredates.csv')
else:
    scoredates = pd.read_csv('scoredates.csv',index_col=0)

scoredatessum = scoredates.sum(axis = 1)
del scoredates

if ploton == 1:
    ax = scoredatessum.plot(title = 'Number of asset returns crossing their VaR per day')
    fig = ax.get_figure()
    fig.savefig('VaRcrossing.pdf')
    plt.close(fig)

# compute the average correlation over the full sample
#avcorrfull = ( dfxwinsorized.corr().sum().sum() - np.nansum(np.diag(dfxwinsorized.corr())) ) / dfxwinsorized.corr().count().sum()
avcorrfull = 0.16072186528126772

# we want to compute a rolling correlation, of excess correlation over the average correlation
if loopcomp == 1:
    window = 20 # one week window
    rollingcorrelation = pd.DataFrame(0,index = dfxwinsorized.index,columns=['correlation'])
    for i in range(1, len(rollingcorrelation)-window):
        corrmat = dfxwinsorized.iloc[i:(i+window),:].corr()
        rollingcorrelation.iloc[i+window] = (corrmat.sum().sum() - np.nansum(np.diag(corrmat))) / corrmat.count().sum()
    rollingcorrelation.to_csv('R_data/rollingcorrelation.csv')
else:
    rollingcorrelation = pd.read_csv('R_Data/rollingcorrelation.csv',index_col=0)

if ploton == 1:
    ax = rollingcorrelation.plot(title = 'Rolling asset returns correlation')
    fig = ax.get_figure()
    fig.savefig('rollingcorrelationexcess.pdf')
    plt.close(fig)

# concatenate both measures
df_VaR_corr = pd.concat([scoredatessum,rollingcorrelation],axis = 1)
df_VaR_corr.columns = ['VaRexceed','corr']
if ploton == 1:
    ax = df_VaR_corr.plot.scatter(x='VaRexceed',y='corr')
    fig = ax.get_figure()
    fig.savefig('VaR_corr_scatter.pdf')
    plt.close(fig)

datespb = list(df_VaR_corr.loc[(df_VaR_corr.VaRexceed > 600)&(df_VaR_corr['corr'] > avcorrfull)].index)

# constitue an index
indexassets = dfxwinsorized.mean(axis = 1)

if ploton == 1:
    ax = indexassets.plot()
    for i in range(0,len(datespb)):
        ax.axvline(x=datespb[i], color='r', linestyle='dashed', linewidth=2)
    fig = ax.get_figure()
    fig.savefig('indexassets.pdf')
    plt.close(fig)

# if you invested one euro on the 2nd of January 1998, equaly spread, and reinvested it evry day, how much would you have on the 6th August 2018?
from functools import reduce
reduce(lambda x, y: x*y, list(indexassets.dropna() + 1))-1

