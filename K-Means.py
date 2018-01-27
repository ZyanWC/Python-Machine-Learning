#K Means gerneral use

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

#Using the Elbow Method
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, 
                    init = "k-means++", 
                    max_iter = 300, 
                    n_init = 10, 
                    random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

#Applying K Means to dataset
kmeans = KMeans(n_clusters = 5, 
                    init = "k-means++", 
                    max_iter = 300, 
                    n_init = 10, 
                    random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualizing the Clusters (Only for 2D)
plt.scatter(x[y_kmeans ==0, 0], x[y_kmeans == 0, 1], s = 100, c = "red", label = "Careful")
plt.scatter(x[y_kmeans ==1, 0], x[y_kmeans == 1, 1], s = 100, c = "blue", label = "Standard")
plt.scatter(x[y_kmeans ==2, 0], x[y_kmeans == 2, 1], s = 100, c = "green", label = "Target")
plt.scatter(x[y_kmeans ==3, 0], x[y_kmeans == 3, 1], s = 100, c = "orange", label = "Careless")
plt.scatter(x[y_kmeans ==4, 0], x[y_kmeans == 4, 1], s = 100, c = "purple", label = "Sensible")
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = "Centroids")
plt.title("Cluster of clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
