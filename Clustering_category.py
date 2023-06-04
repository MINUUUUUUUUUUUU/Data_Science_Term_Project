import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the regression results(you should change to your own csv file directory)
df = pd.read_csv("C:/Users/user/Desktop/Term_project/regression_results.csv")
df = df.fillna({"Correlation_Video_Count": 0, "Correlation_Average_View_Count": 0})


#Feature explanation-> Correlation_Video_Count: Regression of Subscribers & Total Video amount
#                      Correlation_Average_View_Count: Regression of Subscribers & Recent 10 Videos' average View
features = ["Correlation_Video_Count", "Correlation_Average_View_Count"]

# Choose the number of clusters
k = 4

# Do clustering
# Used Clustering method -> K-means

kmeans = KMeans(n_clusters=k, random_state=0).fit(df[features])


# add the cluster labels to our dataframe
df["Cluster"] = kmeans.labels_

# Print some statistics to help analyze the results
print("Cluster sizes:")
print(df["Cluster"].value_counts())

# Calculate and print the mean values of the features for each cluster
print("\nCluster centers (in the scaled feature space):")
for i in range(k):
    print(f"Cluster {i}:")
    print(df.loc[df["Cluster"] == i, features].mean())

# Plot the clusters
plt.scatter(df["Correlation_Average_View_Count"], df["Correlation_Video_Count"], c=df["Cluster"])
plt.xlabel("Correlation_Average_View_Count")
plt.ylabel("Correlation_Video_Count")
plt.show()
