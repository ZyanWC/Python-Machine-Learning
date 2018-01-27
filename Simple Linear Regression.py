#Simple Linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting the dataset into the Trainning set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StanderScaler()
x_train = sc_x.fit_transformation(x_train)
x_test = sc_x.transform(x_test)
sc_y = StanderScaler()
y_train = sc_y.fit_transformation(y_train)
y_test = sc_=y.transform(y_test)"""

#Fitting Simple Lineaar Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test Set
y_hat = regressor.predict(x_test)

#Visualing the Training Data
plt.scatter(x_train, y_train, color = "blue")
plt.plot(x_train, regressor.predict(x_train), color = "red")
plt.title("Salary vs Experience (Training)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualing the Test Data
plt.scatter(x_test, y_test, color = "blue")
plt.plot(x_train, regressor.predict(x_train), color = "red")
plt.title("Salary vs Experience (Test)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()