import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset (you should change to your own csv file directory)
df = pd.read_csv("C:/Users/user/Desktop/Term_project/new_ds_remove0.csv")

# Fill missing values with 0
df = df.fillna(0)

# Define the features for clustering
features = ["coef_subscribers_viewmean", "view_ratio"]

# Scale the features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Choose the number of clusters
k = 3

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0).fit(df[features])

# Add cluster labels to the dataframe
df["cluster"] = kmeans.labels_

# Print cluster sizes
print("Cluster sizes:")
print(df["cluster"].value_counts())

# Calculate and print the mean values of the features for each cluster
print("\nCluster centers:")
for i in range(k):
    print(f"Cluster {i}:")
    print(df.loc[df["cluster"] == i, features].mean())

# Plot the clusters
plt.scatter(df["coef_subscribers_viewmean"], df["view_ratio"], c=df["cluster"])
plt.xlabel("coef_subscribers_viewmean")
plt.ylabel("view_ratio")
plt.title("K-Means Clustering")
plt.show()
