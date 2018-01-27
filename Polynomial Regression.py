#Polynomial Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting the dataset into the Trainning set and Test set
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StanderScaler()
x_train = sc_x.fit_transformation(x_train)
x_test = sc_x.transform(x_test)
sc_y = StanderScaler()
y_train = sc_y.fit_transformation(y_train)
y_test = sc_=y.transform(y_test)"""

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#Visualisingthe Linear Regression results
plt.scatter(x, y, color = "blue")
plt.plot(x, lin_reg.predict(x), color = "red")
plt.title("Truth or Bluff? (Linear)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualizing the Polynomial Regression results
x_grid = np.arange(min(x), max(x), 0.1) #To make the line for accurate by incrementing in smaller steps
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "blue")
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = "red")
plt.title("Truth or Bluff? (Polynomial)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Predicting a new user input value (Linear)
lin_reg.predict(6.5)


#Predicting a new user input value (Polynomial)
lin_reg2.predict(poly_reg.fit_transform(6.5))