import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Initialize visual style
plt.style.use('seaborn-v8')
sns.set_palette("husl")

def generate_cluster_profiles(rfm_df):
    """Generate summarized cluster profiles from RFM data"""
    cluster_profiles = rfm_df.groupby('Cluster_Name').agg({
        'Recency': ['mean', 'std'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std'],
        'CustomerID': 'count',
        'CustAccountBalance': 'mean'
    }).round(2)

    # Flatten columns
    cluster_profiles.columns = ['_'.join(col).strip() for col in cluster_profiles.columns.values]
    cluster_profiles = cluster_profiles.rename(columns={
        'Recency_mean': 'Avg_Recency',
        'Recency_std': 'Std_Recency',
        'Frequency_mean': 'Avg_Frequency',
        'Frequency_std': 'Std_Frequency',
        'Monetary_mean': 'Avg_Monetary',
        'Monetary_std': 'Std_Monetary',
        'CustomerID_count': 'Customer_Count',
        'CustAccountBalance_mean': 'Avg_Account_Balance'
    })

    cluster_profiles['Percentage'] = (cluster_profiles['Customer_Count'] / len(rfm_df) * 100).round(2)
    return cluster_profiles


def plot_segmentation_distribution(rfm_df):
    """Generate pie chart of customer segmentation distribution"""
    segment_counts = rfm_df['Segment_Name'].value_counts()
    colors = ['#45B7D1', '#96CBE4', '#FFFEAA7', '#DDA0DD', '#98D8C8']
    explode = [0.05] * len(segment_counts)

    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(
        segment_counts.values,
        labels=segment_counts.index,
        autopct=lambda p: f'{p:.1f}%\n({int(p * sum(segment_counts.values) / 100):,})',
        startangle=90,
        colors=colors[:len(segment_counts)],
        explode=explode,
        shadow=True
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.title('RFM Customer Segmentation Distribution\n', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    return fig


def plot_rfm_comparison(cluster_profiles):
    """Generate bar chart comparing RFM metrics across segments"""
    rfm_metrics = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
    x_pos = np.arange(len(cluster_profiles))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(rfm_metrics):
        values = cluster_profiles[metric].values
        if metric == 'Avg_Monetary':
            values = values / 1000  # Convert to thousands
        ax.bar(x_pos + i * width, values, width, label=metric.replace('Avg_', ''), alpha=0.8)

    ax.set_xlabel('Customer Segments')
    ax.set_ylabel('Values (Monetary in Thousands)')
    ax.set_title('Average RFM Values by Segment')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(cluster_profiles.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig



def plot_segment_revenue_percentage(rfm_df):
    """Generate a donut chart showing revenue contribution (%) of each segment with percentage labels"""
    # Calculate revenue contribution per segment
    revenue_per_segment = rfm_df.groupby('Cluster_Name')['Monetary'].sum()
    total_revenue = revenue_per_segment.sum()
    revenue_pct = (revenue_per_segment / total_revenue * 100).round(2).sort_values(ascending=False)

    # Define colors
    colors = ['#FF6B6B', "#4ADCD4", '#4A5B7D', '#96CEB4', '#FFFAA7', '#DDAD00', '#98D8C8']

    # Create donut chart
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts = ax.pie(
        revenue_pct.values,
        labels=[f"{v}%" for v in revenue_pct.values],  # show % on slices
        colors=colors[:len(revenue_pct)],
        startangle=90,
        wedgeprops=dict(width=0.4),
    )

    # Optional: Make labels larger and bold
    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight('bold')
        text.set_rotation(90)

    # Add legend for full segment names
    ax.legend(
        wedges,
        revenue_pct.index,
        title="Customer Segments",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    # Add title and layout
    ax.set_title("Revenue Contribution by Customer Segment", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig



def plot_behavior_changes(rfm_df):
    """Generate side-by-side pie charts showing behavior changes"""
    if 'Status' not in rfm_df.columns:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Status distribution
    status_counts = rfm_df['Status'].value_counts()
    ax1.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct='%1.1f%%',
        colors=['#4ECDCA', '#FF6B6B']
    )
    ax1.set_title('Customer Status Distribution')

    # Cluster changes (for existing customers)
    if 'Existing' in rfm_df['Status'].values:
        change_data = rfm_df[rfm_df['Status'] == 'Existing']
        change_counts = change_data['cluster_change'].value_counts()
        ax2.pie(
            change_counts.values,
            labels=change_counts.index,
            autopct='%1.1f%%',
            colors=['#45B7D1', '#96CEB4']
        )
        ax2.set_title('Cluster Changes (Existing Customers)')
    else:
        ax2.text(
            0.5, 0.5,
            'No existing customers\nfor comparison',
            ha='center', va='center',
            transform=ax2.transAxes
        )
        ax2.set_title('Cluster Changes')

    plt.tight_layout()
    return fig
