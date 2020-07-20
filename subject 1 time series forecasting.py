# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:23:42 2019

@author: Al Rahrooh
"""

#AutoRegression

from statsmodels.tsa.ar_model import AR
from random import random
data =  X_train, y_train
model = AR(data)
model_fit = model.fit()
yhat = model_fit.predict(len(data), len(data))
print(yhat)



#Moving Average

from statsmodels.tsa.arima_model import ARMA
from random import random
data = [x + random() for x in range(1, 100)]
model = ARMA(data, order=(0, 1))
model_fit = model.fit(disp=False)
yhat = model_fit.predict(len(data), len(data))
print(yhat)


#Autoregressive Moving Average
from statsmodels.tsa.arima_model import ARMA
from random import random
data = [random() for x in range(1, 100)]
model = ARMA(data, order=(2, 1))
model_fit = model.fit(disp=False)
yhat = model_fit.predict(len(data), len(data))
print(yhat)



# Autoregressive integrated moving average
from statsmodels.tsa.arima_model import ARIMA
from random import random
data = [x + random() for x in range(1, 100)]
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit(disp=False)
yhat = model_fit.predict(len(data), len(data), typ='levels')
print(yhat)



#Vector Autoregression
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
data = list()
for i in range(100):
v1 = i + random()
v2 = v1 + random()
row = [v1, v2]
data.append(row)
model = VAR(data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)