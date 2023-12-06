#!/usr/bin/env python
# coding: utf-8

# In[14]:


import yfinance as yf


# In[15]:


sp500 = yf.Ticker("^GSPC")


# In[16]:


sp500 = sp500.history(period="MAX")


# In[17]:


sp500


# In[18]:


sp500.index


# In[19]:


sp500.plot.line(y="Close",use_index = True)


# In[20]:


del sp500["Dividends"]
del sp500["Stock Splits"]


# In[21]:


sp500["Tomorrow"] = sp500["Close"].shift(-1)


# In[22]:


sp500


# In[23]:


sp500 ["Target"] = (sp500["Tomorrow"]>sp500["Close"]).astype(int)


# In[24]:


sp500


# In[25]:


sp500 = sp500.loc["1990-1-1":].copy()


# In[26]:


sp500


# In[28]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Open", "High", "Low", "Close", "Volume"]
model.fit(train[predictors], train["Target"])


# In[29]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])


# In[30]:


preds


# In[31]:


import pandas as pd


# In[32]:


preds = pd.Series(preds, index=test.index)


# In[33]:


precision_score(test["Target"], preds)


# In[34]:


combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()


# In[35]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[36]:


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)


# In[37]:


predictions = backtest(sp500, model, predictors)


# In[38]:


predictions["Predictions"].value_counts()


# In[39]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[40]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[41]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]
    
    


# In[42]:


sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])


# In[43]:


sp500


# In[44]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[45]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[46]:


predictions = backtest(sp500, model, new_predictors)


# In[47]:


predictions["Predictions"].value_counts()


# In[48]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[ ]:





# In[ ]:




