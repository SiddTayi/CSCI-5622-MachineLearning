# Loading package
library(dplyr)
library(proxy)
library(cluster)  # For calculating cosine distance
library(gplots)    # For plotting the dendrogram


setwd("D:/Masters-2023/Machine Learning/")
df <- read.csv("ML_Project/data/Cleaned/Cleaned_channelData.csv")
head(df)
dim(df)

set.seed(123)  # Set a seed for reproducibility
sampled_df <- sample_n(df, 100)
dim(sampled_df)
head(sampled_df)

cols <- c('Subscribers', 'Viewers', 'Videos_made')

# Create a new dataframe containing the selected columns from the existing dataframe
sampled_df <- sampled_df[cols]
head(sampled_df)


# Calculate cosine similarity matrix
cosine_similarity_matrix <- proxy::simil(as.matrix(sampled_df), method = "cosine")

# Ensure diagonals (self-similarity) are set to 0
diag(cosine_similarity_matrix) <- 0

# Perform hierarchical clustering using cosine similarity
hc <- hclust(as.dist(1 - cosine_similarity_matrix), method = "complete")

# Plot the dendrogram
plot(hc, labels = rownames(sampled_df), main = "Hierarchical Clustering Dendrogram (Cosine Similarity)", xlab = "Data Points", ylab = "Distance")


# Dissimilarity matrix
dis_mat <- dist(sampled_df, method = "cosine") # Computes the dissimilarity (distance) matrix using the cosine distance metric for the data

# Hierarchical clustering using Complete Linkage
ComLink <- hclust(dis_mat, method = "ward.D2") # Complete linkage calculates the maximum distance between clusters.

# Plot the obtained dendrogram
# cex = ddjust the size of the labels in the plot (character expansion)
plot(ComLink, cex = 0.3, hang = -1, main = "Hierarchical Clustering Dendrogram (Cosine Similarity)")
rect.hclust(ComLink, k = 4)


# Calculate cosine similarity matrix
cosine_similarity_matrix <- proxy::simil(as.matrix(sampled_df), method = "cosine")

# Ensure diagonals (self-similarity) are set to 0
diag(cosine_similarity_matrix) <- 0

# Perform hierarchical clustering using cosine similarity
hc <- hclust(as.dist(1 - cosine_similarity_matrix), method = "ward.D2")

# Plot the dendrogram
plot(hc, cex = 0.4, hang = -500, main = "Hierarchical Clustering Dendrogram (Cosine Similarity)", xlab = "Data Points")
rect.hclust(hc, k = 4, border = 2:5)


# Calculate cosine similarity matrix
cosine_similarity_matrix <- proxy::simil(as.matrix(t(sampled_df)), method = "cosine")

# Ensure diagonals (self-similarity) are set to 0
diag(cosine_similarity_matrix) <- 0

# Perform hierarchical clustering using cosine similarity
hc2 <- hclust(as.dist(1 - cosine_similarity_matrix), method = "ward.D2")

# Plot the dendrogram
plot(hc2, cex = 0.4, hang = -500, main = "Hierarchical Clustering Dendrogram (Cosine Similarity)", xlab = "Data Vectorized")
rect.hclust(hc2, k = 2, border = 2:5)


# Cut tree into 4 groups
sub_grp <- cutree(hc, k = 4)

# Number of members in each cluster
table(sub_grp)

library(factoextra)
fviz_cluster(list(data = sampled_df, cluster = sub_grp))
