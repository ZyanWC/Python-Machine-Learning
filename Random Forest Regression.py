#Random Forest Regression

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
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StanderScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_=y.transform(y_test)"""

#Fitting Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, 
                                   random_state = 0)
regressor.fit(x, y)

#Predicting a new user input value
y_hat = regressor.predict(6.5)

#Visualizing the Random Forest Regression results (for smoother curves and better quality)
x_grid = np.arange(min(x), max(x), 0.01) 
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "blue")
plt.plot(x_grid, regressor.predict(x_grid), color = "red")
plt.title("Truth or Bluff? (Random Forest Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()