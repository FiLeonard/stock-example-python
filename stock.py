# coding: utf-8

import pandas as pd
import pandas_datareader.data as pdr
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from xgboost import XGBRegressor
get_ipython().magic('matplotlib inline')


start = datetime.datetime(1995, 1, 15)
end = datetime.datetime(2017, 5, 31)
def get_px(stock, start, end):
    return pdr.DataReader(stock, 'google', start, end)
px = pd.DataFrame(get_px('AAPL', start, end))

print("modify data")
dates = np.squeeze(px.reset_index().as_matrix(columns=['Date'])).astype(dtype='datetime64[D]').reshape(-1, 1)
prices = np.squeeze(px.as_matrix(columns=['Close']))
print("modified data")

def predict_price(dates, prices):
    
    #svr_lin = SVR(kernel= 'linear')#, C=1e3)
    #svr_poly = SVR(kernel= 'poly')#, C=1e3, degree = 2)
    #svr_rbf = SVR(kernel= 'rbf')#, C=1e3, gamma=0.1)
    rfr = RandomForestRegressor(n_estimators=3, max_depth=10)
    reg = linear_model.LinearRegression()
    xgb = XGBRegressor()
    
    print("fitting model")
    #svr_lin.fit(dates, prices)
    #svr_poly.fit(dates, prices)
    #svr_rbf.fit(dates, prices)
    rfr.fit(dates, prices)
    reg.fit(dates.astype('float64'), prices)
    xgb.fit(dates.astype('float64'), prices)
    print("models fitted")
    
    plt.scatter(dates, prices, color='Black', label='Data')
    #plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    #plt.plot(dates, svr_poly.predict(dates), color='green', label='Poly model')
    #plt.plot(dates, svr_lin.predict(dates), color='blue', label='Linear model')
    plt.plot(dates, rfr.predict(dates), color='red', label='RFC model')
    plt.plot(dates, reg.predict(dates.astype('float64')), color='blue', label='BREG model')
    plt.plot(dates, xgb.predict(dates.astype('float64')), color='green', label='XGB')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression/RFR')
    plt.legend()
    plt.show()


predict_price(dates, prices)
