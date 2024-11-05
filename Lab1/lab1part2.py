import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import AgglomerativeClustering

# 200x5 matrix
# Contains CustomerID, Genre, Age, Annual Income, & Spending Score
customer_data = pd.read_csv('shopping_data.csv')

data = customer_data.iloc[:, 3:5].values 

plt.figure(figsize=(10, 7)) # Size
plt.subplots_adjust(bottom=0.1)
plt.scatter(data[:, 0], data[:, 1], label="True Position") # plots the datapoints

plt.show() 

#-------------------------------------------------------------------------------
# linked = linkage(data, method="average")
# # methods: single, ward, average 
# plt.figure(figsize=(10, 7))
# dendrogram(
#     linked,
#     orientation="top", 
#     distance_sort="descending", 
#     show_leaf_counts=True,
# )

# plt.show()

# cluster = AgglomerativeClustering(n_clusters=7, metric='euclidean', linkage='ward')
# cluster.fit_predict(data)
# plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
# plt.show()