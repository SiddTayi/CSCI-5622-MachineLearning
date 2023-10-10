
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

# -----------------------------------------------------------------------

# Selecting only the quantitative variables
cols = ['view_count', 'likes', 'dislikes', 'comment_count', 'start_day']
df = data[cols]
df.dropna(inplace = True, axis = 0)
# df.head()
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------

# Kmeans for numerical columns

# Add a 'Cluster' column to your DataFrame to indicate cluster membership
df['Cluster'] = kmeans.labels_

# Create a scatter plot to visualize the data points and centroids
plt.figure(figsize=(10, 6))

# Scatter plot for each cluster
for cluster in range(3):
    plt.scatter(df[df['Cluster'] == cluster][cols[0]],
                df[df['Cluster'] == cluster][cols[3]],
                label=f'Cluster {cluster}')

# Plot cluster centroids
plt.scatter(centroids[:, 0], centroids[:, 3], marker='x', s=100, color='black', label='Centroids')

# Label axes
plt.xlabel(cols[1])
plt.ylabel(cols[2])

# Add a legend
plt.legend()

# Show the plot
plt.show()
# -----------------------------------------------------------------------

#ELBOW METHOD

from sklearn.preprocessing import StandardScaler
scaled_df = StandardScaler().fit_transform(df)

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

# Plot 2
Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
 kmeans = KMeans(n_clusters=num_clusters)
 kmeans.fit(scaled_df)
 Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()
# -----------------------------------------------------------------------