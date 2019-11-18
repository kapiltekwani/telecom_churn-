
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import io

churn = pd.read_csv('telecom_churn_data.csv')
churn.head()


# In[3]:


temp = churn


# In[4]:


#Assign churn column based on the usage done in 9th month
churn['churn'] = churn.apply(lambda x: 1 if (x['total_ic_mou_9'] == 0 and x['total_og_mou_9'] == 0 and x['vol_2g_mb_9'] == 0 and x['vol_3g_mb_9'] == 0) else 0, axis=1)
churn.head()


# In[5]:


#drop last date columns as it is not giving any important data
churn.drop(columns=['last_date_of_month_6','last_date_of_month_7','last_date_of_month_8', 'last_date_of_month_9'], inplace=True)
churn.head()


# In[6]:


churn.nunique()


# In[7]:


#Deleting following columns as they have only 1 unique value, so not getting any information from these columns
churn.drop(columns=['circle_id', 'loc_og_t2o_mou', 'std_og_t2o_mou','loc_ic_t2o_mou'], inplace=True)
churn.head()


# In[8]:


#Delete the columns which ends with _9, as we dont need to analyze the last month data 
churn.drop(columns= [col for col in churn if col.endswith('_9')], inplace=True)
churn.head()


# In[9]:


#Impute the data 

churn['total_rech_data_6'].fillna(value = 0, inplace=True)
churn['total_rech_data_7'].fillna(value = 0, inplace=True)
churn['total_rech_data_8'].fillna(value = 0, inplace=True)

churn['av_rech_amt_data_6'].fillna(value = 0, inplace=True)
churn['av_rech_amt_data_7'].fillna(value = 0, inplace=True)
churn['av_rech_amt_data_8'].fillna(value = 0, inplace=True)


churn['max_rech_data_6'].fillna(value = 0, inplace=True)
churn['max_rech_data_7'].fillna(value = 0, inplace=True)
churn['max_rech_data_8'].fillna(value = 0, inplace=True)

#Impute the data 
churn['night_pck_user_6'].fillna(value = -1, inplace=True)
churn['night_pck_user_7'].fillna(value = -1, inplace=True)
churn['night_pck_user_8'].fillna(value = -1, inplace=True)

churn['fb_user_6'].fillna(value = -1, inplace=True)
churn['fb_user_7'].fillna(value = -1, inplace=True)
churn['fb_user_8'].fillna(value = -1, inplace=True)


# In[10]:


churn['total_rech_data_amt_6'] = churn['total_rech_data_6'] * churn['av_rech_amt_data_6']
churn['total_rech_data_amt_7'] = churn['total_rech_data_7'] * churn['av_rech_amt_data_7']
churn['total_rech_data_amt_8'] = churn['total_rech_data_8'] * churn['av_rech_amt_data_8']

churn['avg_total_rech_amt'] = (churn['total_rech_amt_6'] + churn['total_rech_amt_7'] + churn['total_rech_data_amt_6'] + churn['total_rech_data_amt_7'])/2


# In[11]:


#threshold value for the revenue is 70 percentile
threshold_val = churn.avg_total_rech_amt.quantile(0.7)
threshold_val


# In[27]:


#filter data based on the threshold value
churn = churn.loc[churn.avg_total_rech_amt >= threshold_val]
churn.shape


# In[13]:


#This will help us to identify, how many values are missing in each columns
# also lets define the threshold 50%, if values are less than this
# we need to decide, if is it good feature to keep ?

temp = pd.DataFrame((churn.isnull().sum(axis=0)/churn.shape[0])*100)
temp = temp.reset_index()
temp.rename(columns={0:"count", "index":"colName"}, inplace=True)
temp1 = temp.loc[temp['count'] > 10]
temp1


# In[14]:


colList = temp1.colName.tolist()
churn.drop(columns=colList, inplace=True)
churn.head()


# In[15]:


churn.nunique()


# In[16]:


churn.monthly_3g_6.unique()


# In[17]:


churn.columns.values


# In[32]:


churn.drop(columns = ['last_day_rch_amt_6', 'last_day_rch_amt_7', 'last_day_rch_amt_8'], inplace=True)


# In[33]:


churn.shape


# In[34]:


churn.nunique()


# In[ ]:


######create dummy variable for fb_user column ##########

## If all are zero, then it means that its convertible
fb_user_dummies = pd.get_dummies(churn['fb_user_6'], drop_first=True)
churn = pd.concat([churn, fb_user_dummies],axis=1)
churn.drop(['fb_user_6'],axis=1,inplace=True)

