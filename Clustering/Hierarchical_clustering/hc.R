# Hierarchical Clustering

# Importing the dataset
dataset <- read.csv("Mall_Customers.csv")
X <- dataset[, 4:5]

# Using dendogram to determine the optimal number of clusters
dendogram = hclust(dist(X, method = "euclidean"), method = "ward.D")
plot(dendogram,
     main = paste("Dendogram"),
     xlab = "Customers",
     ylab = "Euclidean distances")

# Fitting hierarchial clustering to the mall dataset
hc = hclust(dist(X, method = "euclidean"), method = "ward.D")
y_hc = cutree(hc, k = 5)

# Visualizing the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score'
)
