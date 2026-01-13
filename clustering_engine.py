import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Global variables
kmeans = None
rfm_data = None

def find_optimal_clusters(rfm_scaled_df):
    """Determine optimal number of clusters"""
    
    wcss = []
    silhouette_scores = []
    cluster_range = list(range(2, 11))
    
    for k in cluster_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_model.fit(rfm_scaled_df)
        wcss.append(kmeans_model.inertia_)
        
        score = silhouette_score(rfm_scaled_df, kmeans_model.labels_)
        silhouette_scores.append(score)
    
    best_k = cluster_range[np.argmax(silhouette_scores)]
    
    return {
        "cluster_range": cluster_range,
        "wcss": wcss,
        "silhouette_scores": silhouette_scores,
        "optimal_k": best_k,
        "best_silhouette": max(silhouette_scores)
    }



def apply_clustering(rfm_scaled_df, rfm_data, optimal_k=None):
    """Apply K-means clustering cleanly without global"""

    if optimal_k is None:
        results = find_optimal_clusters(rfm_scaled_df)
        optimal_k = results["optimal_k"]

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm_data = rfm_data.copy()
    rfm_data["Cluster"] = kmeans.fit_predict(rfm_scaled_df)

    cluster_analysis = rfm_data.groupby("Cluster").agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'R_Score': 'mean',
        'F_Score': 'mean',
        'M_Score': 'mean',
        'CustomerID': 'count',
        'CustAccountBalance': 'mean',
        'custGender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(2)

    cluster_analysis.columns = [
        'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary',
        'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score',
        'Count', 'Avg_Account_Balance', 'Most_Common_Gender'
    ]

    return cluster_analysis, rfm_data, optimal_k



def assign_cluster_name(stats):
    """Assign a descriptive name to a customer cluster based on RFM averages"""
    recency = stats['Avg_Recency']
    frequency = stats['Avg_Frequency']
    monetary = stats['Avg_Monetary']

    if recency > 365:
        if monetary > 20000:
            return "High-Value Dormant Customers"
        else:
            return "Dormant Low-Value Customers"

    elif recency > 180:
        if monetary > 20000:
            return "High-Value Inactive Customers"
        else:
            return "Inactive Low-Value Customers"

    if frequency > 10 and monetary > 30000:
        return "Active VIP Customers"
    elif frequency > 8 and monetary < 10000:
        return "Active Low-Value Customers"
    else:
        return "Regular Customers"

def assign_cluster_names(rfm_df, cluster_analysis):
    """Map cluster IDs to descriptive names using RFM statistics"""
    rfm_df = rfm_df.copy()
    rfm_df['Cluster_Name'] = rfm_df['Cluster'].map(
        lambda x: assign_cluster_name(cluster_analysis.loc[x])
    )
    return rfm_df
