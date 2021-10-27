# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:49:27 2021

@author: User
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch, random, numpy as np, pandas as pd

def generateData(nObs, size0, size1, sigma1):
    # time series of correlated variables
    # 1) generating uncorrelated data
    np.random.seed(seed=12345); random.seed(12345)
    x = np.random.normal(0,1,size=(nObs, size0)) # each row is a variable
    # 2) creating correlation between the variables
    cols = [random.randint(0, size0-1) for i in range(size1)]
    y = x[:,cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns = list(range(1, x.shape[1]+1)))
    return x, cols

def getIVP(cov, **kwargs):
    # compute the inverse-variance portfolio
    ivp = 1./np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov, cItems):
    # compute variance per cluster
    cov_ = cov.loc[cItems, cItems] # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0,0]
    return cVar

def getQuasiDiag(link):
    # sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1,0], link[-1,1]])
    numItems = link[-1,3] # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = list(range(0, sortIx.shape[0]*2,2)) # make space
        df0 = sortIx[sortIx >= numItems] # find clusters
        i = df0.index ; j = df0.values - numItems
        sortIx[i] = link[j, 0] # item 0 
        df0 = pd.Series(link[j, 1], index= i + 1)
        sortIx = sortIx.append(df0) # item 2
        sortIx = sortIx.sort_index() # re-sort
        sortIx.index = list(range(sortIx.shape[0])) # re-index
    return sortIx.tolist()

def getRecBipart(cov, sortIx):
    # compute hrp alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]
    while len(cItems) > 0:
        cItems = [i[int(j):int(k)] for i in cItems for j,k in ((0, len(i)/2), \
            (len(i)/2, len(i))) if len(i)>1] # bi-section
        for i in list(range(0, len(cItems), 2)): # parse in pairs
            cItems0 = cItems[i] # cluster 1
            cItems1 = cItems[i+1] # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cvar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cvar1)
            w[cItems0] *= alpha # weight 1
            w[cItems1] *= 1 - alpha
    return w
        
def correlDist(corr):
    # a distance matrix based on correlation, where 0 <= d[i,j] <= 1
    # this is a proper distance metric
    dist = ((1 - corr)/2.)**.5 # distance matrix
    return dist

def plotCorrMatrix(path, corr, labels=None):
    # heatmap of correlation matrix
    if labels is None: labels=[]
    mpl.figure(figsize=(22,14))
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5, corr.shape[0]+.5), labels)
    mpl.xticks(np.arange(.5, corr.shape[0]+.5), labels, rotation=45)
    mpl.savefig(path)
    mpl.clf();mpl.close() # reset pylab
    return

# South African equity shares for IG CFD broker and corresponding Yahoo finance codes
counters = pd.read_csv(r'IG Shares.csv')
counter_dict = (counters.set_index('YF Counter')).to_dict()['Company name']
sa_40_counters = ' '.join((counters['YF Counter'].dropna()).to_list())

# price and volume data from yahoo finance
data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = sa_40_counters,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "max",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1d",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

data_closes = ((((data.stack().loc[(slice(None), 'Close'), :].reset_index()).drop('level_1', axis=1)).set_index('Date')).dropna(how='all')).fillna(method='ffill')

data_volumes = ((((data.stack().loc[(slice(None), 'Volume'), :].reset_index()).drop('level_1', axis=1)).set_index('Date')).dropna(how='all')).fillna(method='ffill')

data_turnover = data_closes * data_volumes

# top 100 shares by value exchanged
top_100_value_traded_counters = data_turnover.mean().sort_values().iloc[-100:].index.to_list()

# returns
rets = data_closes.pct_change()
rets = rets[top_100_value_traded_counters]

# selecting shares with more than 3500 days of samples
long_history = (rets.isnull().sum()<3500)
long_history = long_history[long_history!=False]

rets = rets[long_history.index.to_list()]
data_closes = data_closes[long_history.index.to_list()]
data_volumes = data_volumes[long_history.index.to_list()]

# writing subselection shares prices and volumes to file
data_closes.to_csv(r'top40_close_prices.csv')
data_volumes.to_csv(r'top40_volumes.csv')

# checking cumulative returns for errors/outliers
cumrets = (rets+1).cumprod()
cumrets.plot(legend=False, figsize=(12,7))

# compute covariance and correlation matrices
cov, corr = rets.cov(), rets.corr()

# compute and plot corr matrix
plotCorrMatrix(r'HRP3_corr0.png', corr, labels=corr.columns)
# clusters
dist = correlDist(corr)
counter_names = (pd.DataFrame(dist.index.to_list())).to_dict()[0]

# compute hierarchical linkage between distance matrix entries
link = sch.linkage(dist, 'ward')
sortIx = getQuasiDiag(link)
sortIx = corr.index[sortIx].to_list() # recover labels
df0 = corr.loc[sortIx, sortIx] # reorder
plotCorrMatrix(r'HRP3_corr1.png', df0, labels=df0.columns)

# create and edit dendogram
fig = mpl.figure(figsize=(12,6))
sch.dendrogram(link)
ax = mpl.gca()
labels = [counter_dict[counter_names[int(item.get_text())]] for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)
mpl.savefig(r'cluster_dendogram.png', dpi=100, bbox_inches='tight')

# compute hrp weights (inverse inter/intra cluster variance)
hrp = getRecBipart(cov, sortIx).reset_index()
hrp['index'] = [counter_dict[e] for e in hrp['index']]
hrp = hrp.set_index('index')
hrp.columns = ['weights']
hrp = hrp.sort_values(by='weights', ascending=False)
# subelect weights that are more than 0.5% and rebalance
hrp = hrp[hrp>0.005].dropna()
hrp = hrp/hrp.sum()

# write hrp portfolio weights to file
hrp.to_csv(r'hrp_weights.csv')