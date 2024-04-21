#!/usr/bin/env python
# coding: utf-8

# In[169]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[170]:


data = pd.read_csv("C:/Users/mahle/Downloads/archive (17)/shopping_trends.csv")


# In[171]:


data.head()


# In[172]:


data.shape


# In[173]:


print("Missing values count:")
print(data.isnull().sum())


# In[174]:


data.describe()


# In[175]:


plt.figure(figsize=(12, 8))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[176]:


plt.figure(figsize=(12, 8))
sns.histplot(data['Purchase Amount (USD)'], bins=20, kde=True)
plt.title('Distribution of Purchase Amount (USD)')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Count')
plt.show()


# In[177]:


selected_features = ['Age', 'Purchase Amount (USD)', 'Review Rating']


# In[178]:


from sklearn.preprocessing import Normalizer

scaler = Normalizer()
data_scaled = scaler.fit_transform(data[selected_features])


# In[179]:


kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)


# In[180]:


from sklearn.metrics import silhouette_score
silhouette_score(data_scaled, data['Cluster'])


# In[181]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='Age', y='Purchase Amount (USD)', hue='Cluster', data=data, palette='viridis', legend='full')
plt.title('Customer Segments based on Age and Purchase Amount')
plt.xlabel('Age')
plt.ylabel('Purchase Amount (USD)')
plt.show()


# In[182]:


# Analyze customer characteristics within each cluster
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nCluster {cluster} Characteristics:")
    print(cluster_data.describe())


# In[183]:


numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
cluster_means = data.groupby('Cluster')[numeric_columns].mean()
print("Cluster Characteristics:")
print(cluster_means)


# In[184]:


for cluster in range(3):
    print(f"\nMarket Strategy for Cluster {cluster}:")
    cluster_data = data[data['Cluster'] == cluster]
    
    # Analyze customer behavior and preferences in this cluster
    # Example: Identify top-selling products, popular categories, preferred payment methods, etc.
    top_products = cluster_data['Item Purchased'].value_counts().head(3)
    popular_categories = cluster_data['Category'].value_counts().head(3)
    preferred_payment = cluster_data['Payment Method'].value_counts().idxmax()
    
    print(f"Top Products: {top_products.index.tolist()}")
    print(f"Popular Categories: {popular_categories.index.tolist()}")
    print(f"Preferred Payment Method: {preferred_payment}")


# In[185]:


categorical_columns = ['Gender', 'Subscription Status', 'Payment Method']

for column in categorical_columns:
    plt.figure(figsize=(12, 8))
    sns.countplot(x=column, hue='Cluster', data=data)
    plt.title(f'Distribution of {column} by Cluster')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.show()


# In[186]:


cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
cluster_centers['Cluster'] = range(3)

plt.figure(figsize=(12, 8))
sns.barplot(data=cluster_centers.melt(id_vars='Cluster'), x='variable', y='value', hue='Cluster')
plt.title('Cluster Centroids of Numeric Features')
plt.xlabel('Feature')
plt.ylabel('Mean Value')
plt.legend(title='Cluster')
plt.show()


# In[187]:


for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    plt.figure(figsize=(12, 8))
    sns.pairplot(cluster_data[numeric_columns], diag_kind='kde')
    plt.suptitle(f'Pairwise Relationships within Cluster {cluster}', y=1.02)
    plt.show()

