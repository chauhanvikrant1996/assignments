#!/usr/bin/env python
# coding: utf-8

# # PART 2 PRODUCT MATCHING

# In[1]:


# import necessary library
import pandas as pd 


# In[2]:


# Import flipkart data in python environment. 
# Note if you run this file on your system. please save file with same location and name 
df=pd.read_csv("e://flipkart.csv")


# In[3]:


df.head(1)


# In[4]:


# import amazon data in python environment.
df1=pd.read_csv("e://amazon.csv",encoding="ISO-8859-1")


# In[5]:


df1.head(1)


# In[6]:


# merge data with flipkart and amazon with one file and save this in data name using merge  in python
data=pd.merge(df,df1,how="outer")


# In[7]:


data.head(2)


# In[8]:


df.dtypes
# checking data types of columns in flipkart data 


# In[9]:


df1.dtypes  
# checking data types of column in amazon data and overview of data 


# In[10]:


# merge data in one file flipkart and amazon and using common column name "uniq_id" and save this in df3 and this is final for our next process 
df3=pd.merge(df, df1, on ="uniq_id")


# In[11]:


# all data in one file with df3 name 
df3.head(2)


# In[12]:


# checking logic for one product name for random and save in df4 with query option for pandas and pass for product name in format method 
pname="Alisha Solid Women's Cycling Shorts"
df4=df3.query('product_name_x=="{0}"'.format(pname))
clm=[ 'product_name_x',
       'retail_price_x',
       'discounted_price_x', 
       'product_name_y',  'retail_price_y',
       'discounted_price_y']
dffinal=df4.loc[:,clm]
dffinal.columns=['Product name in Flipkart','Retail Price in Flipkart','Discounted Price in Flipkart','Product name in Amazon',
                             'Retail Price in Amazon' ,'Discounted Price in Amazon']
dffinal


# In[13]:


# checking columns name for our requirement in output 
df3.columns


# In[14]:


# these are our final columns name in output 
['uniq_id',  'product_name_x',
       'retail_price_x',
       'discounted_price_x', 
       'product_name_y',  'retail_price_y',
       'discounted_price_y']


# In[15]:


# Make function productname and bind above logic in this function for output return our table with details and columns name 
def productname(pname):
 #       pname="Alisha Solid Women's Cycling Shorts"
        df4=df3.query('product_name_x=="{0}"'.format(pname))
        clm=[ 'product_name_x',
       'retail_price_x',
       'discounted_price_x', 
       'product_name_y',  'retail_price_y',
       'discounted_price_y']
        dffinal=df4.loc[:,clm]
        dffinal.columns=['Product name in Flipkart','Retail Price in Flipkart','Discounted Price in Flipkart','Product name in Amazon',
                             'Retail Price in Amazon' ,'Discounted Price in Amazon']
        return dffinal


# In[16]:


# Checking function performance for random product name 
productname("Alisha Solid Women's Cycling Shorts")


# In[17]:


# import interact library for output and taken output from user 
from ipywidgets import interact


# In[18]:


interact(productname,pname="Alisha Solid Women's Cycling Shorts")


# In[20]:


pname=input("enter product name to match ")
productname(pname)


# In[ ]:




