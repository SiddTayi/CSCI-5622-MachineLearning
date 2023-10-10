import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------------------------------------
data = pd.read_csv('data/Cleaned/Cleaned_channelData.csv')
data.head()

data.drop('Unnamed: 0', axis = 1, inplace = True)
data.columns

# -------------------------------------------------------------------------------------------------------

# Selecting quantitative columns
cols = ['Subscribers', 'Viewers', 'Videos_made', 'start_day']
sns.histplot(data['start_year'], bins = 50)
# -------------------------------------------------------------------------------------------------------

# KMeans
kmeans = KMeans(n_clusters=2).fit(df)
centroids = kmeans.cluster_centers_
print(centroids[:, 0], centroids[:, 1])

# -------------------------------------------------------------------------------------------------------

# Add a 'Cluster' column to your DataFrame to indicate cluster membership
df['Cluster'] = kmeans.labels_

# Create a scatter plot to visualize the data points and centroids
plt.figure(figsize=(10, 6))

# Scatter plot for each cluster
for cluster in range(3):
    plt.scatter(df[df['Cluster'] == cluster][cols[0]],
                df[df['Cluster'] == cluster][cols[1]],
                label=f'Cluster {cluster}')

# Plot cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='black', label='Centroids')

# Label axes
plt.xlabel(cols[0])
plt.ylabel(cols[1])

# Add a legend
plt.legend()

# Show the plot
plt.show()


# -------------------------------------------------------------------------------------------------------

scaled_df = StandardScaler().fit_transform(df)

# Elbow method

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
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
# -------------------------------------------------------------------------------------------------------

# Silhouette Method
from sklearn.metrics import silhouette_score
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(scaled_df)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg.append(silhouette_score(scaled_df, cluster_labels))


plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('‘Values of K’')
plt.ylabel('‘Silhouette score’')
plt.title('‘Silhouette analysis For Optimal k’')
plt.show()
# -------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------
