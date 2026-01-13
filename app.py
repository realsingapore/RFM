import streamlit as st
import pandas as pd
from data_processor import fetch_data, preprocess_data, calculate_rfm_metrics
from rfm_analyzer import calculate_rfm_scores, prepare_for_clustering
from clustering_engine import apply_clustering, assign_cluster_names
from visualization import (
    generate_cluster_profiles,
    plot_segmentation_distribution,
    plot_rfm_comparison,
    plot_segment_sizes,
    plot_segment_revenue_percentage, plot_behavior_changes
)
import openai
from dotenv import load_dotenv
load_dotenv(override=True)
import os

def main():
    st.set_page_config(
        page_title="Bank Trust RFM Analysis",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💳 Bank Trust Customer RFM Analysis")
    st.markdown("""
    This dashboard automatically performs RFM analysis, clustering, and visualization
    for Bank Trust customers based on their transaction data.
    """)

    # Run the analysis pipeline
    raw_data = fetch_data("C:/Users/elsingy/Documents/AMDARI DS/Internship/RFM/Data/Bank_Trust_Dataset.csv")
    processed_data = preprocess_data(raw_data)

    rfm_data = calculate_rfm_metrics(processed_data)

    # Calculate RFM scores
    rfm_data = calculate_rfm_scores(rfm_data)

    # FIX: send only numeric columns (exactly like Colab)
    rfm_scaled_df = prepare_for_clustering(rfm_data)

    # Apply Clustering
    cluster_analysis, rfm_data, _ = apply_clustering(rfm_scaled_df, rfm_data)
    clestered_data = assign_cluster_names(rfm_data, cluster_analysis)

    cluster_profiles = generate_cluster_profiles(clustered_data)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 RFM Visualization", "📊 Customer category", "Recommendation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_segmentation_distribution(clustered_data))
            st.pyplot(plot_rfm_comparison(cluster_profiles))
        with col2:
            st.pyplot(plot_segment_sizes(cluster_profiles))
            st.pyplot(plot_segment_revenue_percentage(clustered_data))

        behavior_fig = plot_behavior_changes(clustered_data)
        if behavior_fig:
            st.pyplot(behavior_fig)


    with tab2:
        def get_segmented_customers(rfm_df):
            base_columns = [
                'CustomerID', 'Recency', 'Frequency', 'Monetary',
                'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
                'Cluster', 'Cluster_Name'
            ]
            available_columns = [col for col in base_columns if col in rfm_df.columns]
            segmented_customers = rfm_df[available_columns].copy()
            segmented_customers.sort_values(by=['Cluster_Name', 'Monetary'], ascending=[True, False], inplace=True)
            segmented_customers.reset_index(drop=True, inplace=True)
            return segmented_customers

    st.subheader("📋 Segmented Customer Data")
    segmented_df = get_segmented_customers(clustered_data)
    st.dataframe(segmented_df)

    csv_data = segmented_df.to_csv(index=False)
    st.download_button(
        label="Download Segmented Customer Data as CSV",
        data=segmented_df.to_csv(index=False).encode('utf-8'),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )


    with tab3:
        st.subheader("💡 Recommendations")

        prompt = f"""
        Based on this RFM cluster data:
        {cluster_profiles.to_dict()}

        Give recommendations to improve revenue, customer retention, and engagement.
    """

        client = openai.OpenAI(
            api_key=os.getenv("api_key"),
            base_url="https://api.groq.com/openai/v1"
        )

        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=prompt,
        )

        st.markdown(response.output_text)

if __name__ == "__main__":
    main()


"""
    cluster_analysis, rfm_data, optimal_k = apply_clustering(rfm_scaled_df, rfm_data)
    rfm_data = assign_cluster_names(rfm_data, cluster_analysis)

    # Visualizations
    cluster_profiles = generate_cluster_profiles(rfm_data)
    st.subheader("📊 Cluster Profiles")
    st.dataframe(cluster_profiles)

    st.subheader("📈 RFM Comparison Across Segments")
    st.pyplot(plot_rfm_comparison(cluster_profiles))

    st.subheader("🧩 Segmentation Distribution")
    st.pyplot(plot_segmentation_distribution(rfm_data))

    st.subheader("💰 Revenue Contribution by Segment")
    st.pyplot(plot_segment_revenue_percentage(rfm_data))

if __name__ == "__main__":
    main()
"""