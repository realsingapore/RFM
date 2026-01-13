import streamlit as st
import pandas as pd

from data_processor import load_and_clean_data
from rfm_analyser import compute_rfm
from clustering_engine import run_clustering
from visualization import (
    plot_rfm_histograms,
    plot_clusters,
    plot_recency_vs_monetary
)

st.set_page_config(page_title="RFM Dashboard", layout="wide")

st.title("📊 Customer RFM Analysis Dashboard")

# Sidebar
st.sidebar.header("Settings")

data_option = st.sidebar.selectbox(
    "Choose data source",
    ["Use default dataset", "Upload your own CSV"]
)

# Load data
if data_option == "Use default dataset":
    df = load_and_clean_data("C:/Users/elsingy/Documents/AMDARI DS/Internship/RFM/Data/Bank_Trust_Dataset.csv")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# Compute RFM
rfm_df = compute_rfm(df)

st.subheader("RFM Table")
st.dataframe(rfm_df.head())

# Clustering
n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)
clustered_df = run_clustering(rfm_df, n_clusters=n_clusters)

st.subheader("Clustered RFM Table")
st.dataframe(clustered_df.head())

# Visualizations
st.subheader("RFM Distributions")
plot_rfm_histograms(rfm_df)

st.subheader("Cluster Scatter Plot")
plot_clusters(clustered_df)

st.subheader("Recency vs Monetary")
plot_recency_vs_monetary(clustered_df)

st.success("Dashboard loaded successfully.")