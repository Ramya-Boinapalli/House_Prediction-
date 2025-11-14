import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

st.title("House Price Prediction")

# Importing DataSet 
dataset = pd.read_csv("C:\\Users\\Ramya\\VS Code Project\\Streamlit\\DS10AM\\ml\\House_data.csv")
space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# Splitting the data into Train and Test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting simple linear regression to the Training Set
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predicting the prices
pred = regressor.predict(xtest)

# Visualizing the Training Set Results
fig1, ax1 = plt.subplots()
ax1.scatter(xtrain, ytrain, color='red')
ax1.plot(xtrain, regressor.predict(xtrain), color='blue')
ax1.set_title("Visuals for Training Dataset")
ax1.set_xlabel("Space")
ax1.set_ylabel("Price")
st.pyplot(fig1)

# Visualizing the Test Set Results
fig2, ax2 = plt.subplots()
ax2.scatter(xtest, ytest, color='red')
ax2.plot(xtrain, regressor.predict(xtrain), color='blue')
ax2.set_title("Visuals for Test Dataset")
ax2.set_xlabel("Space")
ax2.set_ylabel("Price")
st.pyplot(fig2)
