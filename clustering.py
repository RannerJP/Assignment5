#-------------------------------------------------------------------------
# AUTHOR: Jose
# FILENAME: clustering.py
# SPECIFICATION: Tests different sizes of K-means clusters 
# FOR: CS 4210- Assignment #5
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] 
k_shadows = []
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
for k in range(2, 21):
     kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
     kmeans.fit(X_training)          

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     k_shadows.append(silhouette_score(X_training, kmeans.labels_))
kmeans = KMeans(n_clusters=(k_shadows.index(max(k_shadows)) + 2), random_state=0, n_init=10)
kmeans.fit(X_training) 
#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.scatter(range(2,21), k_shadows)
plt.xticks(range(2, 21), range(2, 21))
plt.show()
#reading the test data (clusters) by using Pandas library
df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
labels = np.array(df.values).reshape(1,len(df))[0]

#Calculate and print the Homogeneity of this kmeans clustering
#print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
