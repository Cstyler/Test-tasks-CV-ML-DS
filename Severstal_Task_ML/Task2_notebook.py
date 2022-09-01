#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
from task2 import prepare_data_array
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb
seed = 42


# Читаем датасет, анализируем даты

# In[55]:


file_name = 'Test2/X_data.csv'
df_x = pd.read_csv(file_name, sep=';')


# In[56]:


df_x.rename(columns={'Unnamed: 0': 'date'}, inplace=True)


# In[6]:


file_name = 'Test2/Y_submit.csv'
df_submit = pd.read_csv(file_name, sep=';')


# In[7]:


file_name = 'Test2/Y_train.csv'
df_labels = pd.read_csv(file_name, sep=';')


# In[9]:


values = df_submit.date.values
values.min(), values.max()


# In[10]:


values = df_labels.date.values
values.min(), values.max()


# Исходя из дат, которые указаны в Y_submit, Y_train вырезаем нужные строки из датасета

# In[57]:


df_submit_x = df_x.loc[(df_x.date >= '2018-05-03 23:06:00') & (df_x.date <= '2018-12-31 23:05:00')]
df_x = df_x.loc[(df_x.date >= '2015-01-03 23:06:00') & (df_x.date <= '2018-05-03 23:05:00')]


# In[11]:


values = df_x.date.values
values.min(), values.max()


# Сконкатенируем фичи за каждую минуту в один вектор размера 60*17=1020

# In[12]:


x_data = prepare_data_array(df_x)


# In[13]:


y_data = df_labels.quality.values


# Чтобы не переобучиться делим датасет на train/val

# In[14]:


test_size = 0.05
X_train, X_val, y_train, y_val = train_test_split(x_data,  y_data, test_size=test_size, random_state=seed)


# Нормализуем фичи, т.к. они у нас в разных диапазонах

# In[16]:


scaler = StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.transform(X_val);


# Обучим линейную модель с l2 регуляризацией. Коэффиент подберём через GridSearch

# In[44]:


model = Ridge(random_state=seed)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)
params = {'alpha': [.1, .5, 1, 10, 20, 30, 40, 50]}
ridge_model = GridSearchCV(model, params, n_jobs=10, cv=10, scoring=scorer)
ridge_model.fit(X_train, y_train)


# In[46]:


ridge_model.best_params_


# In[45]:


y_pred = ridge_model.predict(X_val)
mean_absolute_error(y_val, y_pred)


# Обучим XGBoost модель. Параметры подберём так же через GridSearch

# In[91]:


xgb_model = xgb.XGBRegressor(n_jobs=10,  learning_rate=1)
params = {'n_estimators': [2, 3, 4, 5], 'max_depth': [2, 5, 6, 7, 8, 9], 'reg_lambda': [1, 2, 3], 'reg_alpha': [0, 1, 2]}
cv = ShuffleSplit(n_splits=1, test_size=0.05, random_state=seed)
xgb_model = GridSearchCV(xgb_model, params, n_jobs=20, cv=cv, scoring=scorer)
xgb_model.fit(X_train, y_train)


# Метрика улучшилась. Выбираем эту модель

# In[100]:


y_pred = xgb_model.predict(X_val)
mean_absolute_error(y_val, y_pred)


# In[95]:


xgb_model.best_params_


# Делаем аналогичный препроцессинг для submit, делаем предсказание и записываем результат в Y_submit

# In[96]:


submit_data = prepare_data_array(df_submit_x)
scaler.transform(submit_data);


# In[97]:


submit_pred = ridge_model.predict(submit_data).round(3)


# In[103]:


df_submit.quality = submit_pred


# In[106]:


df_submit.to_csv('Test2/Y_submit.csv', index=None)


# In[ ]:




