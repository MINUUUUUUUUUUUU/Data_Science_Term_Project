import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Load the dataset(you should change to your own csv file directory)
df = pd.read_csv("ds_remove0.csv")
# Calculate view_mean as view_count / video_count
df["view_mean"] = df["view_count"] / df["video_count"]
# Create a new dataframe to store the results
results = df.copy()


# Calculate view_ratio as view_count / subscribers
results["view_ratio"] = df["view_count"] / df["subscribers"]

# Regression: -> LinearRegression for every category: subscribers vs view_mean(view_count/video_count)
model = LinearRegression()

# Initialize a dictionary to store the coefficients
coef_dict = {}

# Perform linear regression for each category
for category in df["category"].unique():
    df_category = df[df["category"] == category]
    if len(df_category) >= 5:
        X = df_category[["subscribers"]]
        y = df_category["view_mean"]
        model.fit(X, y)
        coef_dict[category] = model.coef_[0]

        # Perform K-Fold Cross Validation ( K = 5)
        scores = cross_val_score(model, X, y, cv=5)

        # Print Cross Validation scores for each category
        print(
            f"Average cross-validation score of {category}: {np.mean(scores)}")

    else:
        # use 0 for categories with too few data points
        coef_dict[category] = 0

    # Scatter plot of actual data points
    plt.scatter(X, y, label="Actual")

    # Regression line plot
    plt.plot(X, model.predict(X), color="red", label="Regression Line")

    plt.xlabel("Subscribers")
    plt.ylabel("View Mean")
    plt.title(f"Linear Regression - Category: {category}")
    plt.legend()
    plt.show()

# Map the coefficients back to the corresponding categories in the original dataframe
results["coef_subscribers_viewmean"] = df["category"].map(coef_dict)


# Save the results to a CSV file(you should change to your own directory)
results.to_csv("new_ds_remove0.csv", index=False)
