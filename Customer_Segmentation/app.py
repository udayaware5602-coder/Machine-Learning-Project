import os
os.environ["OMP_NUM_THREADS"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set Streamlit page layout
st.set_page_config(page_title="Customer Segmentation App | Uday Aware", layout="wide")

# Main Dashboard Header with Author Name
st.title("🎯 Customer Segmentation Dashboard")
st.markdown("### Developed by **Uday Aware**")
st.write("Analyze customer clusters and classify new incoming users in real time.")
st.markdown("---")

# ---------------------------------------------------------
# 1. DATA SYNTHESIS & MODEL TRAINING
# ---------------------------------------------------------
@st.cache_data
def load_and_train_model():
    np.random.seed(42)
    n_customers = 500

    data = pd.DataFrame({
        'CustomerID': np.arange(1001, 1001 + n_customers),
        'Age': np.random.randint(18, 70, size=n_customers),
        'AnnualIncome_k': np.random.randint(15, 140, size=n_customers),
        'SpendingScore': np.random.randint(1, 100, size=n_customers),
        'Recency_Days': np.random.randint(1, 365, size=n_customers),
        'PurchaseFrequency': np.random.randint(1, 50, size=n_customers)
    })

    features = ['Age', 'AnnualIncome_k', 'SpendingScore', 'Recency_Days', 'PurchaseFrequency']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])

    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    data['Cluster'] = data['Cluster'].astype(str)

    return data, scaler, kmeans, features

df, scaler, kmeans_model, features = load_and_train_model()

# ---------------------------------------------------------
# 2. SIDEBAR: PREDICT NEW CUSTOMER SEGMENT & CREDITS
# ---------------------------------------------------------
st.sidebar.header("User Classification Input")

input_age = st.sidebar.slider("Age", 18, 75, 35)
input_income = st.sidebar.slider("Annual Income ($k)", 15, 150, 60)
input_spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)
input_recency = st.sidebar.slider("Recency (Days)", 1, 365, 30)
input_freq = st.sidebar.slider("Purchase Frequency", 1, 50, 15)

# Predict cluster for user input
new_user_data = np.array([[input_age, input_income, input_spending, input_recency, input_freq]])
scaled_user_data = scaler.transform(new_user_data)
predicted_cluster = str(kmeans_model.predict(scaled_user_data)[0])

st.sidebar.markdown("---")
st.sidebar.subheader("Result")
st.sidebar.success(f"Assigned Segment: **Cluster {predicted_cluster}**")

# Sidebar Author Credit
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Author:** Uday Aware")
st.sidebar.caption("Data Science & Machine Learning Portfolio Project")

# ---------------------------------------------------------
# 3. DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Segment Visualizations", "📋 Cluster Profiles", "📁 Dataset Overview"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.scatter(
            df, x='AnnualIncome_k', y='SpendingScore', color='Cluster',
            title='Income vs. Spending Score by Cluster',
            labels={'AnnualIncome_k': 'Annual Income ($k)', 'SpendingScore': 'Spending Score'},
            template='plotly_white'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df, x='Age', y='PurchaseFrequency', color='Cluster',
            title='Age vs. Purchase Frequency by Cluster',
            labels={'Age': 'Age', 'PurchaseFrequency': 'Purchase Frequency'},
            template='plotly_white'
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Mean Metrics per Cluster")
    summary_df = df.groupby('Cluster')[features].mean().reset_index()
    st.dataframe(summary_df.style.highlight_max(axis=0), use_container_width=True)
    
    st.markdown("""
    **Segment Strategies:**
    * **Cluster 0:** Loyal / Frequent Shoppers — Target with subscriber loyalty perks.
    * **Cluster 1:** High Spend / Trend Seekers — Target with premium product recommendations.
    * **Cluster 2:** Low Activity / Risk of Churn — Re-engage using limited-time discounts.
    * **Cluster 3:** High Income / Low Spend — Upsell luxury products and high-margin goods.
    """)

with tab3:
    st.subheader("Raw Customer Data")
    st.dataframe(df, use_container_width=True)