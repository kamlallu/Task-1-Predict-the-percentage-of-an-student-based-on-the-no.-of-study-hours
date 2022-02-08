#!/usr/bin/env python
# coding: utf-8

# # By-Priyanka Kamlallu

# # TASK 1: Prediction using Supervised ML

# # Step1: Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Step2: Import the dataset

# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("DataSet imported successfully!")


# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


df.head()


# Step:3 Visualizing the dataset by finding relationship between data points with graphical respresentation.
# 

# In[6]:


df.plot(x= "Hours", y="Scores",style= "o" ,c="g")
plt.xlabel("Hours Studied")
plt.ylabel("Score in percentage")
plt.title("Hours VS Scores")
plt.show()


# Relationship:Therefore a positive Linear Regression is observed.

# In[ ]:





# Step4: Let's Prepare the data by dividing the data into 'attributes' and 'label'

# In[8]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# Step:5 Splitting the dataset into training and testing models.

# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 0 )


# In[13]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


# In[14]:


print("Training has been successfully completed!")


# # Step:6 Let's plot the regression line.

# In[15]:


print(lr.intercept_)
print(lr.coef_)


# In[16]:


line = lr.coef_* x + lr.intercept_  
plt.scatter(x,y)
plt.plot(x ,line, c="g")
plt.xlabel("Hours Studied")
plt.ylabel("Score in percentage")
plt.show()


# # Step:7 Now let's make predictions.

# In[17]:


print(x_test) 
y_pred = lr.predict(x_test)


# In[18]:


df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1


# # Step:8 Evaluating mean absolute error

# In[19]:


from sklearn import metrics
print('Mean Aboslute Error: ', metrics.mean_absolute_error(y_test, y_pred))


# # Step:9 Predicting the score if a student studies for 9.25 hours a day.

# In[20]:


hours =[[9.25]]
pred_value = lr.predict(hours)
print('Number of total hours : {}'.format(hours))
print('Predicted Score : {}'.format(pred_value[0]))


# # Therefore, the predicted score of a student who studies for 9.25 hours/day is 93.691
#  

# In[ ]:




