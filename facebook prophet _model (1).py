#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import fbprophet
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


#did groupby on device col dropped it ,datetime is resampled in min 
df = pd.read_excel(r'C:\Users\admin\data_device1.xlsx',parse_dates=["Datetime"])
df


# In[99]:


df_co=pd.DataFrame(df, columns=['Datetime','co'])
df_humidity=pd.DataFrame(df, columns=['Datetime','humidity'])
df_temp=pd.DataFrame(df, columns=['Datetime','temp'])


# In[100]:


df_co.head(2)


# In[101]:


df_temp.head(2)


# In[85]:


#renaming datetime-ds and temp-y as per prophet requirements
df_temp.columns = ['ds','y']
df_temp.head()


# In[86]:


df_temp['ds'] = pd.to_datetime(df_temp['ds'])


# In[87]:


df_temp


# In[88]:


#splitting the data set for training and validation
train=df_temp[(df_temp['ds'] >= '2020-07-12 00:01:00') & (df_temp['ds'] <= '2020-07-16 00:01:00')]
test=df_temp[(df_temp['ds'] > '2020-07-16 00:01:00')]


# In[94]:


train.shape


# In[95]:


test.shape


# In[39]:


from fbprophet import Prophet


# In[70]:


#trying to fit the model with confidence interval of 95% on test temperature data
m_temp = Prophet(interval_width=0.95,daily_seasonality=True)
model = m_temp.fit(test)


# In[72]:


#making future predictions for test dates starting from 2020-07-16 to 2020-07-20
future = m_temp.make_future_dataframe(periods=0)
forecasttest = m_temp.predict(future)


# In[73]:


forecasttest[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)


# In[92]:


import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df_temp['ds'] , y=df_temp["y"],
                    mode='lines+markers',
                     name='actual test temp '))
fig.add_trace(go.Scatter(x=forecasttest['ds']  , y= forecasttest["yhat"],
                     mode='lines+markers',
                     name='Forecasted test temp'))



fig.update_layout(
    title="Actual Vs Forecasted test temp",
    xaxis_title="Days ",
    yaxis_title= "forecasted temp",
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="RebeccaPurple"
    )
)

fig.show()


# In[44]:


#future forecast for dates 21,22,23 of temp data
mf = Prophet(interval_width=0.95, daily_seasonality=True)
model = mf.fit(df_temp)


# In[47]:


#making predictions from 2020-07-20(00:04:00) to 2020-07-23(00:03:00) based on periods value 4320\60\24=3 days as freq is min
future = mf.make_future_dataframe(periods=4320,freq='Min')
forecast = mf.predict(future)
forecast.tail()


# In[50]:


#Prophet includes functionality for time series cross validation to measure forecast error using historical data
from fbprophet.diagnostics import cross_validation
cv_results2 = cross_validation( model = mf, initial = pd.to_timedelta(1152,unit='Min'), horizon =pd.to_timedelta(438,unit='Min'))


# In[51]:


from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(cv_results2)
df_p


# In[90]:


import plotly.graph_objects as go


# In[57]:


import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df_temp['ds'] , y=df_temp["y"],
                    mode='lines+markers',
                     name='Actual Temperature '))
fig.add_trace(go.Scatter(x=forecast['ds']  , y= forecast["yhat"],
                     mode='lines+markers',
                     name='Forecasted Temperature'))



fig.update_layout(
    title="Actual Vs Forecasted Temperature ",
    xaxis_title="Days ",
    yaxis_title= "Temperature",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()


# In[56]:


import plotly.graph_objects as go


# In[103]:


df_humidity.columns = ['ds','y']
df_humidity.head()


# In[104]:


#future forecast for dates 21,22,23 for humidity data
mhumidity= Prophet(interval_width=0.95, daily_seasonality=True)
model = mhumidity.fit(df_humidity)


# In[105]:


#making predictions from 2020-07-20(00:04:00) to 2020-07-23(00:03:00) based on periods value 4320\60\24=3 days as freq is min
future = mhumidity.make_future_dataframe(periods=4320,freq='Min')
forecasthumidity = mhumidity.predict(future)
forecasthumidity.tail()


# In[61]:


from fbprophet.diagnostics import cross_validation
cv_resultsh = cross_validation( model = mhumidity, initial = pd.to_timedelta(1152,unit='Min'), horizon =pd.to_timedelta(438,unit='Min'))


# In[62]:


from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(cv_resultsh)
df_p


# In[107]:


import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df_humidity['ds'] , y=df_humidity["y"],
                    mode='lines+markers',
                     name='Actual humidity '))
fig.add_trace(go.Scatter(x=forecasthumidity['ds']  , y= forecasthumidity["yhat"],
                     mode='lines+markers',
                     name='Forecasted humidity'))



fig.update_layout(
    title="Actual Vs Forecasted humidity",
    xaxis_title="Days ",
    yaxis_title= "humidity",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="RebeccaPurple"
    )
)

fig.show()


# In[63]:


df_co.columns = ['ds','y']
df_co.head()


# In[64]:


#future forecast for dates 21,22,23 of co data
mco= Prophet(interval_width=0.95, daily_seasonality=True)
model = mco.fit(df_co)


# In[65]:


#making predictions from 2020-07-20(00:04:00) to 2020-07-23(00:03:00) based on periods value 4320\60\24=3 days as freq is min
future = mco.make_future_dataframe(periods=4320,freq='Min')
forecastco = mco.predict(future)
forecastco.tail()


# In[81]:


import plotly.graph_objects as go

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=df_co['ds'] , y=df_co["y"],
                    mode='lines+markers',
                     name='Actual co '))
fig.add_trace(go.Scatter(x=forecastco['ds']  , y= forecastco["yhat"],
                     mode='lines+markers',
                     name='Forecasted co'))



fig.update_layout(
    title="Actual Vs Forecasted co",
    xaxis_title="Days ",
    yaxis_title= "co",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="RebeccaPurple"
    )
)

fig.show()


# In[ ]:


conclussion:
    predicting the future forecasts is easy with fb prophet but the predicted values are not fitting properly on the top and bottom notches of the actual graph
    as perfectly as arima
    


# In[ ]:


References: https://facebook.github.io/prophet/docs/diagnostics.html
https://github.com/krishnaik06/FbProphet
https://github.com/nicknochnack/TimeSeriesForecastingProphet/blob/main/Facebook%20Prophet.ipynb
https://github.com/srivatsan88/End-to-End-Time-Series/blob/master/Time_Series_using_Prophet.ipynb    
    

