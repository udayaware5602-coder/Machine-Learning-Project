import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as px_go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# PAGE CONFIG & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Species Clustering - Uday Aware",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional card layout and subtle styling
st.markdown("""
<style>
    /* Global background and padding fixes */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card Styles */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 15px;
    }
    
    .metric-card-title {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card-value {
        font-size: 1.8rem;
        color: #212529;
        font-weight: 700;
        margin-top: 5px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 15px;
        margin-bottom: 20px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
    }
    
    /* Custom Author Tag */
    .author-badge {
        background-color: #e2e8f0;
        color: #334155;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    /* Custom Footer */
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 0.9rem;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the Iris dataset from Iris.csv or fallback to scikit-learn."""
    try:
        df = pd.read_csv('Iris.csv')
    except FileNotFoundError:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        )
        df['Species'] = [iris.target_names[i] for i in iris.target]
        df.insert(0, 'Id', range(1, len(df) + 1))
    return df

@st.cache_data
def preprocess_data(df):
    """Clean data, scale features, and run PCA for 2D visual representation."""
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = df[feature_cols].copy()
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(pca_coords, columns=['PCA1', 'PCA2'])
    
    return X, X_scaled, scaler, pca, pca_df

@st.cache_resource
def train_kmeans_model(X_scaled, n_clusters, random_state):
    """Train K-Means clustering model and return model, labels, and metrics."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_
    return kmeans, labels, sil_score, inertia

# -----------------------------------------------------------------------------
# MAIN APPLICATION LOGIC
# -----------------------------------------------------------------------------
def main():
    # Load dataset & run initial processing
    df = load_data()
    X, X_scaled, scaler, pca, pca_df = preprocess_data(df)
    feature_cols = list(X.columns)

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROL PANEL
    # -------------------------------------------------------------------------
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.markdown("**Developed by:** Uday Aware")
    st.sidebar.markdown("---")

    # Section 1: Navigation
    st.sidebar.subheader("📌 Navigation")
    app_mode = st.sidebar.radio(
        "Go to Section:",
        ["🏠 Dashboard", "📊 Data Explorer", "📈 Data Visualization", "🤖 Machine Learning", "🎯 Prediction", "📊 Model Evaluation"]
    )
    st.sidebar.markdown("---")

    # Section 2: Model Configuration
    st.sidebar.subheader("⚙️ Model Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=3, step=1)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

    # Train model based on sidebar hyperparameters
    kmeans_model, labels, sil_score, inertia = train_kmeans_model(X_scaled, n_clusters, random_state)
    
    # Add cluster labels to full dataframe for export & plotting
    clustered_df = df.copy()
    clustered_df['Cluster'] = [f"Cluster {l}" for l in labels]
    clustered_df['PCA1'] = pca_df['PCA1']
    clustered_df['PCA2'] = pca_df['PCA2']

    # -------------------------------------------------------------------------
    # APP HEADER
    # -------------------------------------------------------------------------
    st.title("🌟 Iris Species Clustering & Analytics Dashboard")
    st.markdown("<div class='author-badge'>👤 Author: Uday Aware</div>", unsafe_allow_html=True)
    st.markdown("An interactive machine learning analytics dashboard for unsupervised Iris flower pattern recognition and cluster evaluation.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # 🏠 DASHBOARD
    # -------------------------------------------------------------------------
    if app_mode == "🏠 Dashboard":
        st.markdown("<div class='section-header'>🏠 Executive Dashboard</div>", unsafe_allow_html=True)
        
        # KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Rows", f"{len(df)}")
        with col2:
            st.metric("Total Columns", f"{len(df.columns)}")
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum()}")
        with col4:
            st.metric("Active Clusters (K)", f"{n_clusters}")
        with col5:
            st.metric("Silhouette Score", f"{sil_score:.3f}")

        st.markdown("### 📋 Dataset Overview")
        st.write("""
        This project performs **K-Means Unsupervised Clustering** on the classic **Iris Dataset**.
        The algorithm categorizes flowers based on sepal and petal dimensions without using species labels during training.
        Known species labels are preserved solely for validation and comparative exploration.
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Feature Summary")
            st.dataframe(pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": [str(dtype) for dtype in df.dtypes],
                "Role": ["Identifier" if col == "Id" else ("Ground Truth Label" if col == "Species" else "Feature Variable") for col in df.columns]
            }), use_container_width=True)

        with col_b:
            st.markdown("#### K-Means Clustering Result (2D PCA)")
            fig_pca = px.scatter(
                clustered_df,
                x='PCA1', y='PCA2',
                color='Cluster',
                hover_data=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'],
                title=f"PCA Visualization of {n_clusters} Clusters",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_pca.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pca, use_container_width=True)

    # -------------------------------------------------------------------------
    # 📊 DATA EXPLORER
    # -------------------------------------------------------------------------
    elif app_mode == "📊 Data Explorer":
        st.markdown("<div class='section-header'>📊 Data Explorer</div>", unsafe_allow_html=True)
        
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            selected_species = st.multiselect("Filter by Ground-Truth Species:", options=df['Species'].unique(), default=df['Species'].unique())
        with col_filter:
            columns_to_show = st.multiselect("Columns to Display:", options=df.columns.tolist(), default=df.columns.tolist())

        filtered_df = df[df['Species'].isin(selected_species)][columns_to_show]
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
        st.dataframe(filtered_df, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["📈 Descriptive Statistics", "🔍 Data Types & Missing Values", "🏷️ Ground Truth Frequencies"])
        
        with tab1:
            st.dataframe(df.describe().T, use_container_width=True)
            
        with tab2:
            info_df = pd.DataFrame({
                "Data Type": df.dtypes,
                "Non-Null Count": df.notnull().sum(),
                "Null Count": df.isnull().sum(),
                "Unique Values": df.nunique()
            })
            st.dataframe(info_df, use_container_width=True)
            
        with tab3:
            st.dataframe(df['Species'].value_counts().reset_index().rename(columns={"index": "Species", "Species": "Count"}), use_container_width=True)

    # -------------------------------------------------------------------------
    # 📈 DATA VISUALIZATION
    # -------------------------------------------------------------------------
    elif app_mode == "📈 Data Visualization":
        st.markdown("<div class='section-header'>📈 Data Visualization</div>", unsafe_allow_html=True)

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            x_var = st.selectbox("X-Axis Feature:", options=feature_cols, index=2)
        with col_ctrl2:
            y_var = st.selectbox("Y-Axis Feature:", options=feature_cols, index=3)
        with col_ctrl3:
            color_var = st.selectbox("Color By:", options=['Cluster', 'Species'])

        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### Scatter Plot Analysis")
            fig_scatter = px.scatter(
                clustered_df, x=x_var, y=y_var, color=color_var,
                hover_data=feature_cols,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_chart2:
            st.markdown("#### Feature Distribution (Histogram)")
            fig_hist = px.histogram(
                clustered_df, x=x_var, color=color_var, barmode="overlay",
                marginal="box", color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        col_chart3, col_chart4 = st.columns(2)
        
        with col_chart3:
            st.markdown("#### Correlation Heatmap")
            corr_matrix = X.corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=True, aspect="auto",
                color_continuous_scale="Viridis",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with col_chart4:
            st.markdown("#### Boxplot Outlier Detection")
            fig_box = px.box(
                clustered_df, y=x_var, color='Species',
                points="all", color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # -------------------------------------------------------------------------
    # 🤖 MACHINE LEARNING (CLUSTERING)
    # -------------------------------------------------------------------------
    elif app_mode == "🤖 Machine Learning":
        st.markdown("<div class='section-header'>🤖 K-Means Clustering Analysis</div>", unsafe_allow_html=True)

        st.write(f"Currently training **K-Means Algorithm** with **K = {n_clusters}**.")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Silhouette Score", f"{sil_score:.4f}")
        with col_m2:
            st.metric("Inertia (WCSS)", f"{inertia:.2f}")
        with col_m3:
            st.metric("Total Sample Size", f"{len(X)}")

        st.markdown("---")
        st.markdown("#### Elbow Method & Silhouette Analysis across K")
        
        # Calculate Elbow Curve dynamically
        k_values = list(range(2, 9))
        inertias = []
        silhouettes = []
        
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km_labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, km_labels))

        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            fig_elbow = px.line(
                x=k_values, y=inertias, markers=True,
                labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
                title="Elbow Method for Optimal K"
            )
            fig_elbow.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="K=3 (Known Iris Species)")
            st.plotly_chart(fig_elbow, use_container_width=True)

        with col_e2:
            fig_sil = px.line(
                x=k_values, y=silhouettes, markers=True,
                labels={'x': 'Number of Clusters (K)', 'y': 'Silhouette Score'},
                title="Silhouette Score Trend"
            )
            fig_sil.add_vline(x=n_clusters, line_dash="dash", line_color="green", annotation_text=f"Current K={n_clusters}")
            st.plotly_chart(fig_sil, use_container_width=True)

        st.markdown("#### Cluster Center Coordinates (Original Scale)")
        cluster_centers_orig = scaler.inverse_transform(kmeans_model.cluster_centers_)
        centers_df = pd.DataFrame(cluster_centers_orig, columns=feature_cols)
        centers_df.index = [f"Cluster {i}" for i in range(n_clusters)]
        st.dataframe(centers_df.style.highlight_max(axis=0), use_container_width=True)

    # -------------------------------------------------------------------------
    # 🎯 PREDICTION
    # -------------------------------------------------------------------------
    elif app_mode == "🎯 Prediction":
        st.markdown("<div class='section-header'>🎯 Interactive Cluster Prediction</div>", unsafe_allow_html=True)
        st.write("Enter flower dimensions below to determine which cluster the new specimen belongs to.")

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            sepal_len = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
            sepal_wid = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
        with col_in2:
            petal_len = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.8, step=0.1)
            petal_wid = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1)

        if st.button("🚀 Predict Cluster", use_container_width=True):
            input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
            input_scaled = scaler.transform(input_data)
            predicted_cluster = kmeans_model.predict(input_scaled)[0]
            
            # Compute distance to cluster centers
            distances = np.linalg.norm(input_scaled - kmeans_model.cluster_centers_, axis=1)
            
            st.success(f"### Result: Assigned to Cluster {predicted_cluster}")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("#### Input Feature Vector")
                st.dataframe(pd.DataFrame(input_data, columns=feature_cols), use_container_width=True)
            with col_res2:
                st.markdown("#### Distance to Cluster Centroids")
                dist_df = pd.DataFrame({
                    "Cluster": [f"Cluster {i}" for i in range(n_clusters)],
                    "Euclidean Distance (Scaled)": distances
                })
                st.dataframe(dist_df, use_container_width=True)

    # -------------------------------------------------------------------------
    # 📊 MODEL EVALUATION
    # -------------------------------------------------------------------------
    elif app_mode == "📊 Model Evaluation":
        st.markdown("<div class='section-header'>📊 Clustering vs Ground-Truth Species</div>", unsafe_allow_html=True)
        st.write("Cross-tabulation comparing unsupervised clusters with ground-truth species labels.")

        crosstab = pd.crosstab(clustered_df['Cluster'], clustered_df['Species'])
        
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            st.markdown("#### Contingency Table")
            st.dataframe(crosstab.style.background_gradient(cmap="Blues"), use_container_width=True)

        with col_eval2:
            st.markdown("#### Cluster Distribution")
            fig_bar = px.bar(
                clustered_df, x='Cluster', color='Species',
                barmode='group', title="Species Distribution per Cluster",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        csv_data = clustered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Clustered Dataset as CSV",
            data=csv_data,
            file_name=f"iris_kmeans_k{n_clusters}_results.csv",
            mime="text/csv"
        )

    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------
    st.markdown("""
        <div class="footer">
            Developed by <strong>Uday Aware</strong> | Built using Python, Streamlit, Pandas, Scikit-learn & Plotly
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    