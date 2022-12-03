#!/usr/bin/env python
# coding: utf-8

# 
# # HOUSE PRICE PREDICTION 

# In[1]:


# IMPORT NECESSITY LIRARY 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 


# In[2]:


# IMPORT DATA IN PYTHON ENVIRONMENT
#  IF YOU RUN THIS FILE IN YOUR COMPUTER PLEASE SAVE THIS FILE IN SAME NAME AND SAME LOCATION
df=pd.read_excel("E:/Shacklabs/DS - Assignment Part 1 data set.xlsx")


# In[3]:


df.head(1)


# In[4]:


df.describe()
# we clearly see that our target value is continous so we can use REGRESSION TECHNIQUES 


# In[5]:


df.info()
# HERE WE HAVE NO MISSING VALUE AND OUR ALL COLUMNS IS CONTINOUS SO WE CAN BUILD OUR MODEL NOT DO EDA EXPLOTARY DATA ANALYSIS


# FIRST TECHNIQUES IS USED LINEAR REGRESSION 

# In[6]:


# here we take 4 features of house to predict the score and these 4 features have a data varies.
#                    so this is why we include these 4 features of house 
x=df[['Number of bedrooms','House Age',
      'House size (sqft)','Number of convenience stores']] 
y=df['House price of unit area']


# In[7]:


#import library sklearn for model and import linear regression techniques for prediction because data is quantitative. 
from sklearn.linear_model import LinearRegression


# WE TAKE LINEARREGRESSION IS 1ST MODEL WITH LIMITED FEATURES 

# In[8]:


lr=LinearRegression()  # initialize the object for linear regression techniques
lr.fit(x,y)


# In[9]:


# here we check score for our 1st model with 4 features include 
lr.score(x,y)


# In[10]:


#0.3838169804617867    our score is low so we can change our features for better score 


# In[11]:


lr.coef_


# In[12]:


lr.intercept_


# In[13]:


# so we include all features of house for better prediction 
x=df.iloc[:,:-1]
y=df['House price of unit area']


# AGAIN WE TAKE LINEAR REGRESSION WITH ALL FEATURES OF HOUSE 

# In[14]:


lr=LinearRegression()
lr.fit(x,y)


# In[15]:


# here we check our model score with new features include 
lr.score(x,y)


# In[16]:


#0.5826048002262252 this score is also very low but we check our predicted price first 


# In[17]:


df['linearpredictedprice']=lr.predict(x)


# In[18]:


df.head(5)


# In[19]:


'''clearly see the predicted price of house is very high and this techniques OR MODEL  is not predicted  the price approximately.
so we can change our technique for better result '''


# TECHNIQUE 2 :=  RANDOM FOREST

# In[20]:


df.head(1)


# In[21]:


X=df.iloc[:,:-2]
Y=df.iloc[:,-2].values


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# here we split our data for training and testing and for training and testing our model


# In[23]:


#  NOW WE CREATE OUR MODEL FOR RANDOM FOREST TECHNIQUE 


# In[24]:


from sklearn.ensemble import RandomForestRegressor


# In[25]:


model=RandomForestRegressor()


# In[26]:


# train  a model
model.fit(X_train,y_train)


# In[27]:


# model prediction 
RandomForestPredict1=model.predict(X_test)


# In[28]:


# check our score 
model.score(X_test,y_test)


# In[29]:


# model acuracy
from sklearn.metrics import r2_score

r2_score(y_test,RandomForestPredict1)


# In[30]:


# 0.7226427336430679
# here our score is low with all feature we reduce some features for better score 


# LIMITED FEATURES OF HOUSE WE TAKE 

# In[31]:


x=df[['House Age','Number of convenience stores','Number of bedrooms','House size (sqft)','Distance from nearest Metro station (km)']]
y=df['House price of unit area']


# In[32]:


from sklearn.ensemble import RandomForestRegressor


# In[33]:


model_4=RandomForestRegressor()
model_4.fit(x,y)


# In[34]:


model_4.score(x,y)*100


# In[35]:


# 94.87118555029427
# here we see a good score so we can accept this random forest model for selection 


# In[36]:


df['RandomForestmodel']=model_4.predict(x)


# In[37]:


df.head(1)


# In[38]:


#check for random input the price prediction is nearly approx and this is our final model for random input uses by end user from input 

input =np.array([8.1,104.81010,5,1,597])
input=input.reshape(1,-1)
print("The predicted price is ",model_4.predict(input))


# In[39]:


df.tail(1)


# # Part 2: Product matching

# In[40]:


# import necessary library
import pandas as pd 


# In[41]:


# Import flipkart data in python environment. 
# Note if you run this file on your system. please save file with same location and name 
df=pd.read_csv("e://flipkart.csv")


# In[42]:


df.head(1)


# In[43]:


# import amazon data in python environment.
df1=pd.read_csv("e://amazon.csv",encoding="ISO-8859-1")


# In[44]:


df1.head(1)


# In[45]:


# merge data with flipkart and amazon with one file and save this in data name using merge  in python
data=pd.merge(df,df1,how="outer")


# In[46]:


data.head(2)


# In[47]:


df.dtypes
# checking data types of columns in flipkart data 


# In[48]:


df1.dtypes  
# checking data types of column in amazon data and overview of data 


# In[49]:


# merge data in one file flipkart and amazon and using common column name "uniq_id" and save this in df3 and this is final for our next process 
df3=pd.merge(df, df1, on ="uniq_id")


# In[50]:


# all data in one file with df3 name 
df3.head(2)


# In[51]:


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


# In[52]:


# checking columns name for our requirement in output 
df3.columns


# In[53]:


# these are our final columns name in output 
['uniq_id',  'product_name_x',
       'retail_price_x',
       'discounted_price_x', 
       'product_name_y',  'retail_price_y',
       'discounted_price_y']


# In[54]:


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


# In[55]:


# Checking function performance for random product name 
productname("Alisha Solid Women's Cycling Shorts")


# In[56]:


# import interact library for output and taken output from user 
from ipywidgets import interact


# In[57]:


interact(productname,pname="Alisha Solid Women's Cycling Shorts")


# In[ ]:




