import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def run_kmeans_clustering(qb_df_clean):
    """
    Applies K-means clustering on QB data based on performance metrics.

    :param qb_df_clean: Spark DataFrame containing QB data
    :return: Pandas DataFrame with cluster assignments
    """
    # Select relevant columns and convert to Pandas DataFrame
    features_df = qb_df_clean.select(["Name", "Passing Yards", "TD Passes", "Ints", "Passer Rating", "Rushing Yards"])
    features_df = features_df.fillna(0)  # Fill any null values
    features_pd = features_df.toPandas()  # Convert to Pandas DataFrame

    # Convert non-numeric values like '--' to NaN and handle them
    features_pd.replace('--', np.nan, inplace=True)

    # Fill NaN values with 0 (you can also use mean/median if preferred)
    features_pd.fillna(0, inplace=True)

    # Convert all columns to numeric types (just in case)
    features_pd = features_pd.apply(pd.to_numeric, errors='coerce')

    # Extract player names before scaling
    player_names = features_pd.pop("Name")

    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_pd)

    # Use the elbow method to find the optimal K (optional)
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow method to determine K
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bo-')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Fit K-means with the optimal number of clusters (e.g., K=3)
    kmeans = KMeans(n_clusters=3, random_state=0)
    qb_clusters = kmeans.fit_predict(scaled_features)

    # Add cluster assignments to the dataframe
    features_pd["Cluster"] = qb_clusters
    features_pd["Name"] = player_names

    # Return the dataframe with clusters
    return features_pd


def analyze_clusters(clustered_df):
    """
    Analyzes the clusters and displays information on quarterbacks in each cluster.

    :param clustered_df: Pandas DataFrame containing QB data with cluster assignments
    """
    # Group by cluster and view average performance metrics
    cluster_analysis = clustered_df.groupby("Cluster").mean()
    print("Cluster Performance Averages:")
    print(cluster_analysis)

    for cluster_num in clustered_df["Cluster"].unique():
        print(f"Quarterbacks in Cluster {cluster_num}:")
        print(clustered_df[clustered_df["Cluster"] == cluster_num]["Name"].values)
        print("\n")
