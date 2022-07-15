#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with Python Scikit Learn
# 

# ## Simple Linear Regression
# ###### In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[1]:


# import libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load dataseat
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

df.head(5)


# In[3]:


df.info


# In[4]:


df.describe()


# In[5]:


df.shape


# ###### And finally, let’s plot our data points on a 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data using the below script :

# In[6]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Scores')  
plt.xlabel('Hours')  
plt.ylabel('Scores')  
plt.show()


# ###### Our next step is to divide the data into “attributes” and “labels”.
# 
# Attributes are the independent variables while labels are dependent variables whose values are to be predicted. In our dataset, we only have two columns. We want to predict the scores depending upon the Houers recorded. Therefore our attribute set will consist of the “Houers” column which is stored in the X variable, and the label will be the “Scores” column which is stored in y variable.

# In[10]:



X = df.iloc[:, :1].values  
y = df.iloc[:, 1:].values  


# In[11]:


X


# ###### Next, we split 80% of the data to the training set while 20% of the data to test set using below code.
# 
# The test_size variable is where we actually specify the proportion of the test set.

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ###### After splitting the data into training and testing sets, finally, the time is to train our algorithm. For that, we need to import LinearRegression class, instantiate it, and call the fit() method along with our training data.

# In[13]:


model = LinearRegression()  
model.fit(X_train, y_train) #training the algorithm
print("Training complete")


# ###### the linear regression model basically finds the best value for the intercept and slope, which results in a line that best fits the data. To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.

# In[14]:


#To retrieve the intercept:
print(model.intercept_)

#For retrieving the slope:
print(model.coef_)


# ##### fitting linear regrassion

# In[15]:


df['intercept']=1


# In[16]:


Lm = sm.OLS(df['Scores'],df['Hours'])


# In[17]:


results = Lm.fit()
results.summary()


# In[18]:


## Plotting the regrassin
plt.scatter (df['Scores'],df['Hours'])
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ###### Now that we have trained our algorithm, it’s time to make some predictions. To do so, we will use our test data and see how accurately our algorithm predicts the percentage score. To make predictions on the test data, execute the following script:

# In[24]:


y_prediction=model.predict(X_test)
y_prediction


# ###### Now compare the actual output values for X_test with the predicted values, execute the following script:

# In[25]:


data= pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
data


# ###### We can also visualize comparison result as a bar graph using the below script :

# In[21]:


df1 = df.head()
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# ###### Though our model is not very precise, the predicted percentages are close to the actual ones.

# ###### Let's plot our straight line with the test data :

# In[22]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# ###### Prediction score if a student study  9.25 hrs/day

# In[29]:


predict_score=model.predict([[9.25]])
print("Prediction score if a student study 9.25 hrs/day :",float(predict_score))


# ### Evaluation The Model

# In[30]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

