#!/usr/bin/env python
# coding: utf-8

# In[1]:


# K-Means Clusterin


# In[ ]:


# Importing the libraries


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


#Importing the mall dataset with pandas


# In[4]:


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values


# In[5]:


# Using the elbow method to find the optimal number of clusters


# In[6]:


from sklearn.cluster import KMeans
wcss =[]
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[7]:


# Plot the graph to visualize the Elbow Method to find the optimal number of cluster


# In[8]:


plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[9]:


# Applying KMeans to the dataset with the optimal number of cluster


# In[10]:


kmeans=KMeans(n_clusters= 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# In[11]:


# Visualising the clusters


# In[ ]:


plt.scatter(X[Y_Kmeans == 0, 0], X[Y_Kmeans == 0,1],s = 100, c='red', label = 'Cluster 1')


# In[ ]:


plt.scatter(X[Y_Kmeans == 1, 0], X[Y_Kmeans == 1,1],s = 100, c='blue', label = 'Cluster 2')


# In[ ]:


plt.scatter(X[Y_Kmeans == 2, 0], X[Y_Kmeans == 2,1],s = 100, c='green', label = 'Cluster 3')


# In[ ]:


plt.scatter(X[Y_Kmeans == 3, 0], X[Y_Kmeans == 3,1],s = 100, c='cyan', label = 'Cluster 4')


# In[ ]:


plt.scatter(X[Y_Kmeans == 4, 0], X[Y_Kmeans == 4,1],s = 100, c='magenta', label = 'Cluster 5')


# In[ ]:


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
    


# In[ ]:


plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()

