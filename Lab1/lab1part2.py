import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 

customer_data = pd.read_csv('shopping_data.csv')
data = customer_data.iloc[:, 3:5].values
print(customer_data.head())


linked = linkage(data, 'single')
labelList = range(1, 201)
plt.figure(figsize=(10, 7))
dendrogram(linked,
    orientation='top',
    labels=labelList,
    distance_sort='descending',
    show_leaf_counts=True)
plt.show()
