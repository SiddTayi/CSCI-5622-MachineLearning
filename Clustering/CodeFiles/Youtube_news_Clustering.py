import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# -------------------------------------------------------------------------------------#
data = pd.read_csv('youtube_tokenized.csv')
data.head()

data.drop('Unnamed: 0', inplace = True, axis = 1)
# -------------------------------------------------------------------------------------#



# -------------------------------------------------------------------------------------#
# Optimal value of K
kmeans = KMeans(n_clusters=2).fit(data)
centroids = kmeans.cluster_centers_

# Elbow Method 1

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
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# -------------------------------------------------------------------------------------#
# Silhouette method
from sklearn.metrics import silhouette_score
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg.append(silhouette_score(data, cluster_labels))


plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('‘Values of K’')
plt.ylabel('‘Silhouette score’')
plt.title('‘Silhouette analysis For Optimal k’')
plt.show()

# -------------------------------------------------------------------------------------#

# K Means
kmeans = KMeans(n_clusters=3, random_state = 42).fit(data)
centroids = kmeans.cluster_centers_
labels = kmeans.predict(data)

pca = PCA(n_components = 2)
df = pca.fit_transform(data)

plt.scatter(df[:, 0], df[:, 1], c = labels)
plt.title('Clustering Youtube News')
plt.xlabel('Com 1')
plt.ylabel('Com 1')
plt.show()

# -------------------------------------------------------------------------------------#
