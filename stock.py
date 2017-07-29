# coding: utf-8
import bs4 as bs
import requests
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from xgboost import XGBRegressor

def sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].find('a')['href']
        
        tickers.append(ticker.rsplit('/', 1)[-1].replace("XNYS", "NYSE"))
    print(tickers)
    return tickers


tickers = sp500_tickers()
start = datetime.datetime(2005, 1, 15)
end = datetime.datetime(2017, 5, 31)

def get_px(stock, start, end):
    return pdr.DataReader(stock, 'google', start, end)['Close']

def predict_price(stock):
    
    px = pd.DataFrame(get_px(stock, start, end)).reset_index()
    dates = px['Date'].values.astype(dtype='datetime64[D]').reshape(-1, 1)
    prices = px['Close'].values

    rfr = RandomForestRegressor(n_estimators=3, max_depth=10)
    reg = linear_model.LinearRegression()
    xgb = XGBRegressor()
    
    rfr.fit(dates, prices)
    reg.fit(dates.astype('float64'), prices)
    xgb.fit(dates.astype('float64'), prices)

    plt.scatter(dates, prices, color='Black', label='Data')
    
    plt.plot(dates, rfr.predict(dates), color='red', label='RFC model')
    plt.plot(dates, reg.predict(dates.astype('float64')), color='blue', label='BREG model')
    plt.plot(dates, xgb.predict(dates.astype('float64')), color='green', label='XGB')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression/RFR for %s' % stock)
    plt.legend()
    plt.show()

for n in tickers:
    predict_price(n)
