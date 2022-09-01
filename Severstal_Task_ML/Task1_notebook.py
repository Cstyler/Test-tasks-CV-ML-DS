#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[7]:


file_name = 'statistics_2018.csv'
df = pd.read_csv(file_name)


# Задание а. Более 3 бракованных листов на партию выходит 53 раза для стали марки А, для стали марки B 44 раза

# In[8]:


steel_grade_a = df.loc[df.steel_grade == 'A']
steel_grade_b = df.loc[df.steel_grade == 'B']
len(steel_grade_a[steel_grade_a.defective_num > 3]), len(steel_grade_b[steel_grade_b.defective_num > 3])


# Задание б. При скоростях прокатки более 4 м/с свыше 3 бракованных листов стали на партию выходит 74 раза, при меньших скоростях прокатки 23 раза

# In[9]:


fast_rolling = df.loc[df.rolling_speed > 4]
slow_rolling = df.loc[df.rolling_speed <= 4]
len(fast_rolling.loc[fast_rolling.defective_num > 3]), len(slow_rolling.loc[slow_rolling.defective_num > 3])


# In[ ]:




