import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('mavoix_ml_sample_dataset.csv')
X = dataset.iloc[:,2:9].values

from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)

kmeans = KMeans(n_clusters = 7,init = 'k-means++',n_init = 10,max_iter = 300,random_state =0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100,c = 'red',label = 'cluster1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100,c = 'blue',label = 'cluster2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100,c = 'green',label = 'cluster3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s=100,c = 'cyan',label = 'cluster4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s=100,c = 'magenta',label = 'cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',s=300,label = 'centroids')