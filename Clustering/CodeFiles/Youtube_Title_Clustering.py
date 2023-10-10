
# IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# KMeans 
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------

# Reading the data file
data = pd.read_csv('data/Cleaned/Cleaned_videoData.csv')
#data.head()
# -----------------------------------------------------------------------

# USING CHANNEL DESCRIPTION AS CLUSTERING COl

df = data[['cleaned_title_ns']].copy()
sampled_data = df[['cleaned_title_ns']].sample(frac=0.4, random_state=42)
sampled_data.dropna(inplace = True, axis = 0)
# sampled_data.shape

df_list = sampled_data['cleaned_title_ns'].to_list()
len(df_list)

# Word frequency
from sklearn.feature_extraction.text import CountVectorizer   


MyCountV=CountVectorizer(
        input="content", 
        lowercase=True, 
        stop_words = "english"
        )
 
MyDTM = MyCountV.fit_transform(df_list)  # create a sparse matrix

#vocab is a vocabulary list
vocab = MyCountV.get_feature_names_out()  # change to a list
print(list(vocab)[10:20])


MyDTM = MyDTM.toarray()  # convert to a regular array
ColumnNames=MyCountV.get_feature_names_out()
MyDTM_DF=pd.DataFrame(MyDTM,columns=ColumnNames)

# -----------------------------------------------------------------------

# CLUSTERING TITLE DATA
sampled_data = MyDTM_DF.sample(frac=0.07, random_state=42)


kmeans = KMeans(n_clusters=2).fit(sampled_data)
centroids = kmeans.cluster_centers_

# ELBOW METHOD

#initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

#create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(sampled_data)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()



# SILHOUETTE METHOD

# Silhouette method
from sklearn.metrics import silhouette_score
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(sampled_data)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg.append(silhouette_score(sampled_data, cluster_labels))


plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('‘Values of K’')
plt.ylabel('‘Silhouette score’')
plt.title('‘Silhouette analysis For Optimal k’')
plt.show()

# -----------------------------------------------------------------------

# K MEANS
from sklearn.decomposition import PCA

kmeans = KMeans(n_clusters=3, random_state = 1).fit(sampled_data)
centroids = kmeans.cluster_centers_

labels = kmeans.predict(sampled_data)
pca = PCA(n_components = 2)
df = pca.fit_transform(sampled_data)

plt.scatter(df[:, 0], df[:, 1], c = labels)
plt.title('Clustering Youtube News')
plt.xlabel('Com 1')
plt.ylabel('Com 2')
plt.legend()
plt.show()

# -----------------------------------------------------------------------
