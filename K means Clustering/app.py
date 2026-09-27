import io
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Facebook Live K-Means Clustering",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES: List[str] = [
    "num_reactions",
    "num_comments",
    "num_shares",
    "num_likes",
    "num_loves",
    "num_wows",
    "num_hahas",
    "num_sads",
    "num_angrys",
]

DROP_COLUMNS: List[str] = [
    "status_id",
    "status_published",
    "Column1",
    "Column2",
    "Column3",
    "Column4",
]

K_VALUES = list(range(2, 11))
MODEL_FILENAME = "iris_kmeans_model.pkl"


# ============================================================
# Styling
# ============================================================

def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* ===== Premium dashboard background ===== */
        .stApp {
            background:
                linear-gradient(135deg, rgba(7,12,25,0.96), rgba(10,20,40,0.91)),
                url("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=2200&q=85")
                center / cover fixed;
        }

        .main .block-container {
            max-width: 1450px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(5,10,22,0.98), rgba(12,25,48,0.96));
            border-right: 1px solid rgba(255,255,255,0.10);
        }

        section[data-testid="stSidebar"] * {
            color: #e5eefc;
        }

        /* ===== Hero ===== */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.6rem 2.8rem;
            min-height: 260px;
            border-radius: 26px;
            margin-bottom: 1.5rem;
            background:
                linear-gradient(90deg, rgba(3,8,20,0.94) 0%, rgba(3,8,20,0.70) 55%, rgba(3,8,20,0.28) 100%),
                url("https://images.unsplash.com/photo-1551434678-e076c223a692?auto=format&fit=crop&w=1800&q=85")
                center / cover;
            border: 1px solid rgba(255,255,255,0.14);
            box-shadow: 0 25px 70px rgba(0,0,0,0.35);
        }

        .hero h1 {
            color: white;
            font-size: clamp(2rem, 4vw, 3.6rem);
            line-height: 1.05;
            margin: 0 0 0.8rem 0;
            font-weight: 800;
            letter-spacing: -1px;
        }

        .hero p {
            color: #dbeafe;
            max-width: 760px;
            font-size: 1.05rem;
            line-height: 1.65;
            margin: 0;
        }

        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.18);
            border: 1px solid rgba(147,197,253,0.30);
            color: #bfdbfe;
            font-size: 0.82rem;
            font-weight: 700;
        }

        /* ===== Glass cards ===== */
        .glass-card {
            background: rgba(8,18,35,0.72);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 20px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 16px 45px rgba(0,0,0,0.22);
            backdrop-filter: blur(12px);
            margin-bottom: 1rem;
        }

        .section-title {
            color: #f8fafc;
            font-size: 1.25rem;
            font-weight: 750;
            margin-bottom: 0.75rem;
        }

        .section-text {
            color: #cbd5e1;
            line-height: 1.7;
        }

        /* ===== Metric cards ===== */
        div[data-testid="stMetric"] {
            background: rgba(8,18,35,0.74);
            border: 1px solid rgba(255,255,255,0.12);
            padding: 1rem 1.05rem;
            border-radius: 18px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.18);
            backdrop-filter: blur(10px);
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }

        /* ===== Streamlit text ===== */
        h1, h2, h3, h4, p, label {
            color: #f8fafc;
        }

        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sub-header {
            color: #cbd5e1;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        /* ===== Dataframes / charts ===== */
        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
        }

        /* ===== Buttons ===== */
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Data loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None) -> pd.DataFrame:
    """
    Load the notebook's Live.csv dataset.

    If a file is uploaded, it is used directly. Otherwise the app
    attempts to load Live.csv from the application's working directory.
    """
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes))

    return pd.read_csv("Live.csv")


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = set(FEATURES + ["status_type"] + DROP_COLUMNS)
    missing = sorted(required.difference(df.columns))
    return len(missing) == 0, missing


# ============================================================
# Notebook-equivalent preprocessing
# ============================================================

@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame):
    """
    Reproduce the notebook preprocessing:

    1. Copy dataframe.
    2. Drop identifier/text and redundant columns.
    3. Select the nine numerical engagement features.
    4. Fill missing feature values with their medians.
    5. Apply np.log1p().
    6. Standardize with StandardScaler.
    """
    data = df.copy()

    data.drop(columns=DROP_COLUMNS, inplace=True)

    X = data[FEATURES].copy()
    missing_before = X.isnull().sum()

    X = X.fillna(X.median())

    X_log = np.log1p(X)

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_log)
    X_scaled = pd.DataFrame(X_scaled_array, columns=FEATURES, index=X.index)

    return data, X, X_log, X_scaled, scaler, missing_before


# ============================================================
# K-Means model selection and training
# ============================================================

@st.cache_resource(show_spinner=False)
def evaluate_k_values(X_scaled_values: np.ndarray) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Reproduce the notebook's Elbow and Silhouette workflows
    for K=2,...,10.
    """
    if X_scaled_values.shape[0] <= max(K_VALUES):
        raise ValueError(
            f"The dataset must contain more than {max(K_VALUES)} rows "
            f"to evaluate K values from 2 to {max(K_VALUES)}."
        )

    inertia = {}
    silhouette_scores = {}

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=42,
        )
        model.fit(X_scaled_values)
        inertia[k] = model.inertia_

        labels = model.predict(X_scaled_values)
        silhouette_scores[k] = silhouette_score(X_scaled_values, labels)

    return inertia, silhouette_scores


@st.cache_resource(show_spinner=False)
def train_final_model(X_scaled_values: np.ndarray, optimal_k: int):
    """
    Reproduce the notebook's final K-Means model.
    """
    kmeans = KMeans(
        n_clusters=optimal_k,
        init="k-means++",
        n_init=10,
        random_state=42,
    )

    cluster_labels = kmeans.fit_predict(X_scaled_values)
    final_silhouette = silhouette_score(X_scaled_values, cluster_labels)

    return kmeans, cluster_labels, final_silhouette


@st.cache_resource(show_spinner=False)
def fit_pca(X_scaled_values: np.ndarray):
    """
    Reproduce the notebook's two-component PCA visualization.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled_values)
    return pca, X_pca


# ============================================================
# Plot helpers
# ============================================================

def plotly_theme(fig):
    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="",
    )
    return fig


def make_elbow_chart(inertia: Dict[int, float]):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(inertia.keys()),
            y=list(inertia.values()),
            mode="lines+markers",
            name="Inertia",
            hovertemplate="K=%{x}<br>Inertia=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Elbow Method",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia",
        xaxis=dict(dtick=1),
    )
    return plotly_theme(fig)


def make_silhouette_chart(silhouette_scores: Dict[int, float], optimal_k: int):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(silhouette_scores.keys()),
            y=list(silhouette_scores.values()),
            mode="lines+markers",
            name="Silhouette Score",
            hovertemplate="K=%{x}<br>Score=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[optimal_k],
            y=[silhouette_scores[optimal_k]],
            mode="markers",
            marker=dict(size=12),
            name=f"Selected K={optimal_k}",
            hovertemplate=f"K={optimal_k}<br>Score={silhouette_scores[optimal_k]:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Silhouette Score by K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        xaxis=dict(dtick=1),
    )
    return plotly_theme(fig)


def make_cluster_size_chart(cluster_counts: pd.Series):
    chart_df = cluster_counts.reset_index()
    chart_df.columns = ["Cluster", "Posts"]

    fig = px.bar(
        chart_df,
        x="Cluster",
        y="Posts",
        title="Number of Posts in Each Cluster",
        labels={"Posts": "Number of Posts", "Cluster": "Cluster"},
        text="Posts",
    )
    fig.update_traces(
        hovertemplate="Cluster=%{x}<br>Posts=%{y}<extra></extra>"
    )
    return plotly_theme(fig)


def make_pca_chart(pca_df: pd.DataFrame):
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="K-Means Clusters Visualized Using PCA",
        labels={
            "PC1": "Principal Component 1",
            "PC2": "Principal Component 2",
            "Cluster": "Cluster",
        },
        hover_data=["Cluster"],
    )
    fig.update_traces(marker=dict(size=7, opacity=0.65))
    return plotly_theme(fig)


def make_status_composition_chart(cluster_status: pd.DataFrame):
    plot_df = cluster_status.reset_index().melt(
        id_vars="Cluster",
        var_name="Status Type",
        value_name="Proportion",
    )

    fig = px.bar(
        plot_df,
        x="Cluster",
        y="Proportion",
        color="Status Type",
        barmode="stack",
        title="Post-Type Composition of Each Cluster",
        labels={"Proportion": "Proportion"},
        hover_data={"Proportion": ":.3f"},
    )
    fig.update_yaxes(range=[0, 1])
    return plotly_theme(fig)


# ============================================================
# Inference
# ============================================================

def predict_cluster(
    kmeans: KMeans,
    scaler: StandardScaler,
    values: Dict[str, float],
) -> int:
    input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    if input_df.isnull().any().any():
        raise ValueError("All engagement inputs must contain valid numeric values.")

    if (input_df < 0).any().any():
        raise ValueError("Engagement counts cannot be negative.")

    input_log = np.log1p(input_df)
    input_scaled = scaler.transform(input_log)

    return int(kmeans.predict(input_scaled)[0])


def render_inference(kmeans, scaler, cluster_profile: pd.DataFrame) -> None:
    st.subheader("Model Inference & Simulation")
    st.caption(
        "Enter engagement counts using the same nine features used by the notebook. "
        "The app applies log1p transformation and the trained StandardScaler before prediction."
    )

    with st.form("inference_form"):
        cols = st.columns(3)
        values = {}

        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    format="%.2f",
                )

        submitted = st.form_submit_button(
            "Predict Cluster",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            predicted_cluster = predict_cluster(kmeans, scaler, values)

            st.success(f"Predicted cluster: {predicted_cluster}")

            if predicted_cluster in cluster_profile.index:
                st.info(
                    "The predicted cluster's profile is shown below. "
                    "Cluster numbers have no inherent meaning; interpret them using their engagement profile."
                )
                st.dataframe(
                    cluster_profile.loc[[predicted_cluster]],
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")


# ============================================================
# Export
# ============================================================

def build_model_artifact(kmeans, scaler):
    return {
        "model": kmeans,
        "scaler": scaler,
        "features": FEATURES,
    }


def build_text_report(
    df: pd.DataFrame,
    optimal_k: int,
    inertia: float,
    final_silhouette: float,
    pca,
    cluster_counts: pd.Series,
) -> str:
    return f"""Facebook Live K-Means Clustering Report

Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns
Features used:
{", ".join(FEATURES)}

Transformation:
- Missing feature values: median imputation
- log1p transformation
- StandardScaler standardization

K-Means parameters:
- n_clusters: {optimal_k}
- init: k-means++
- n_init: 10
- random_state: 42

Final metrics:
- Inertia: {inertia:.4f}
- Silhouette score: {final_silhouette:.4f}

Cluster sizes:
{cluster_counts.to_string()}

PCA explained variance:
- PC1: {pca.explained_variance_ratio_[0]:.4f}
- PC2: {pca.explained_variance_ratio_[1]:.4f}
- Total: {pca.explained_variance_ratio_.sum():.4f}

Note:
K-Means is unsupervised. The status_type field is used only for
post-clustering interpretation and is not treated as a target variable.
"""


def render_export(
    result: pd.DataFrame,
    kmeans,
    scaler,
    df: pd.DataFrame,
    optimal_k: int,
    final_silhouette: float,
    pca,
    cluster_counts: pd.Series,
) -> None:
    st.subheader("Export & Analytics")

    csv_bytes = result.to_csv(index=False).encode("utf-8")

    model_data = build_model_artifact(kmeans, scaler)
    model_buffer = io.BytesIO()
    pickle.dump(model_data, model_buffer)
    model_buffer.seek(0)

    report = build_text_report(
        df=df,
        optimal_k=optimal_k,
        inertia=kmeans.inertia_,
        final_silhouette=final_silhouette,
        pca=pca,
        cluster_counts=cluster_counts,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download Clustered CSV",
            data=csv_bytes,
            file_name="facebook_live_clustered_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "Download Model",
            data=model_buffer.getvalue(),
            file_name=MODEL_FILENAME,
            mime="application/octet-stream",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            "Download Report",
            data=report.encode("utf-8"),
            file_name="facebook_live_kmeans_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("### Clustered Output Dataset")
    st.dataframe(result, use_container_width=True, height=420)


# ============================================================
# Dashboard
# ============================================================

def render_overview(
    df: pd.DataFrame,
    result: pd.DataFrame,
    optimal_k: int,
    final_silhouette: float,
    kmeans,
    missing_before: pd.Series,
) -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">● MACHINE LEARNING • K-MEANS CLUSTERING</div>
            <h1>Facebook Live<br>Engagement Intelligence</h1>
            <p>
                Explore audience engagement patterns, discover behavioral clusters,
                visualize model performance, and simulate new cluster assignments
                through an interactive analytics dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("👥 Total Posts", f"{len(df):,}")
    metric_cols[1].metric("📋 Source Columns", f"{df.shape[1]}")
    metric_cols[2].metric("🧠 ML Features", f"{len(FEATURES)}")
    metric_cols[3].metric("🎯 Optimal K", f"{optimal_k}")
    metric_cols[4].metric("📐 Silhouette", f"{final_silhouette:.4f}")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">🚀 Project Overview</div>
                <div class="section-text">
                    This dashboard applies the notebook's K-Means clustering workflow
                    to Facebook Live post engagement. Identifier and redundant columns
                    are removed, missing engagement values are handled with median
                    imputation, then the nine engagement variables are transformed
                    with <b>log1p</b> and standardized before clustering.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        cluster_counts = result["Cluster"].value_counts().sort_index()
        largest_cluster = int(cluster_counts.idxmax())
        largest_size = int(cluster_counts.max())
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">📊 Cluster Snapshot</div>
                <div class="section-text">
                    <b>{optimal_k}</b> engagement clusters were identified.<br>
                    Largest cluster: <b>Cluster {largest_cluster}</b><br>
                    Posts in largest cluster: <b>{largest_size:,}</b><br>
                    Model silhouette: <b>{final_silhouette:.4f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📈 Dataset Intelligence")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Dataset Preview**")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Clustering Feature Statistics**")
        st.dataframe(
            result[FEATURES].describe().T.round(2),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔍 Missing Values Before Imputation")
    missing_table = missing_before.rename("Missing Values").to_frame()
    st.dataframe(missing_table, use_container_width=True)


def render_eda(
    df: pd.DataFrame,
    inertia: Dict[int, float],
    silhouette_scores: Dict[int, float],
    optimal_k: int,
    result: pd.DataFrame,
    pca_df: pd.DataFrame,
    cluster_status: pd.DataFrame,
) -> None:
    st.subheader("Data Exploratory Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["K Selection", "Cluster Sizes", "PCA", "Status Interpretation"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_elbow_chart(inertia), use_container_width=True)
        with c2:
            st.plotly_chart(
                make_silhouette_chart(silhouette_scores, optimal_k),
                use_container_width=True,
            )

        score_df = pd.DataFrame(
            {
                "K": list(silhouette_scores.keys()),
                "Inertia": [inertia[k] for k in silhouette_scores],
                "Silhouette Score": list(silhouette_scores.values()),
            }
        )
        st.dataframe(score_df, use_container_width=True)

    with tab2:
        cluster_counts = result["Cluster"].value_counts().sort_index()
        st.plotly_chart(
            make_cluster_size_chart(cluster_counts),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.write("Mean Cluster Profile")
            st.dataframe(
                result.groupby("Cluster")[FEATURES].mean().round(2),
                use_container_width=True,
            )
        with c2:
            st.write("Median Cluster Profile")
            st.dataframe(
                result.groupby("Cluster")[FEATURES].median().round(2),
                use_container_width=True,
            )

    with tab3:
        st.plotly_chart(make_pca_chart(pca_df), use_container_width=True)
        st.write(
            f"PC1 explained variance: **{pca_df.attrs['pc1_variance']:.4f}**  \n"
            f"PC2 explained variance: **{pca_df.attrs['pc2_variance']:.4f}**  \n"
            f"Total explained variance: **{pca_df.attrs['total_variance']:.4f}**"
        )

    with tab4:
        st.plotly_chart(
            make_status_composition_chart(cluster_status),
            use_container_width=True,
        )
        st.caption(
            "This is interpretation only. It does not measure clustering accuracy."
        )


def render_sidebar() -> Tuple[str, bytes | None]:
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Dashboard Overview",
            "Data Exploratory Analysis",
            "Model Inference & Simulation",
            "Export & Analytics",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Live.csv",
        type=["csv"],
        help="Upload the same Live.csv dataset used by the notebook.",
    )

    uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Pipeline: median imputation → log1p → StandardScaler → "
        "K-Means++ → silhouette-based K selection"
    )

    return page, uploaded_bytes


# ============================================================
# Main application
# ============================================================

def main() -> None:
    apply_custom_css()

    page, uploaded_bytes = render_sidebar()

    try:
        df = load_data(uploaded_bytes)
    except FileNotFoundError:
        st.error(
            "Live.csv was not found. Upload Live.csv using the sidebar, "
            "or place Live.csv in the same folder as app.py."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Dataset loading failed: {exc}")
        st.stop()

    valid, missing = validate_dataset(df)

    if not valid:
        st.error(
            "The supplied dataset does not match the notebook schema. "
            f"Missing required columns: {', '.join(missing)}"
        )
        st.stop()

    try:
        (
            data,
            X,
            X_log,
            X_scaled,
            scaler,
            missing_before,
        ) = preprocess_data(df)

        inertia, silhouette_scores = evaluate_k_values(X_scaled.values)

        optimal_k = max(silhouette_scores, key=silhouette_scores.get)

        kmeans, cluster_labels, final_silhouette = train_final_model(
            X_scaled.values,
            optimal_k,
        )

        result = data.copy()
        result["Cluster"] = cluster_labels

        cluster_profile = (
            result.groupby("Cluster")[FEATURES]
            .mean()
            .round(2)
        )

        pca, X_pca = fit_pca(X_scaled.values)

        pca_df = pd.DataFrame(
            {
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
                "Cluster": cluster_labels,
            }
        )
        pca_df.attrs["pc1_variance"] = pca.explained_variance_ratio_[0]
        pca_df.attrs["pc2_variance"] = pca.explained_variance_ratio_[1]
        pca_df.attrs["total_variance"] = pca.explained_variance_ratio_.sum()

        cluster_status = pd.crosstab(
            result["Cluster"],
            result["status_type"],
            normalize="index",
        ).round(3)

    except Exception as exc:
        st.error(f"Pipeline execution failed: {exc}")
        st.stop()

    if page == "Dashboard Overview":
        render_overview(
            df=df,
            result=result,
            optimal_k=optimal_k,
            final_silhouette=final_silhouette,
            kmeans=kmeans,
            missing_before=missing_before,
        )

    elif page == "Data Exploratory Analysis":
        render_eda(
            df=df,
            inertia=inertia,
            silhouette_scores=silhouette_scores,
            optimal_k=optimal_k,
            result=result,
            pca_df=pca_df,
            cluster_status=cluster_status,
        )

    elif page == "Model Inference & Simulation":
        render_inference(
            kmeans=kmeans,
            scaler=scaler,
            cluster_profile=cluster_profile,
        )

    elif page == "Export & Analytics":
        render_export(
            result=result,
            kmeans=kmeans,
            scaler=scaler,
            df=df,
            optimal_k=optimal_k,
            final_silhouette=final_silhouette,
            pca=pca,
            cluster_counts=result["Cluster"].value_counts().sort_index(),
        )


if __name__ == "__main__":
    main()
