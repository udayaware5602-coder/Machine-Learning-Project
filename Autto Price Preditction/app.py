import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AutoPrice AI | Uday Aware Edition",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODERN DARK GLASSMORPHISM CSS WITH BACKGROUND IMAGE
# ============================================================

BACKGROUND_IMAGE_URL = "https://images.unsplash.com/photo-1617814076367-b759c7d7e738?q=80&w=2000&auto=format&fit=crop"

st.markdown(f"""
<style>

/* App-wide background image with dark overlay */
.stApp {{
    background: linear-gradient(rgba(15, 23, 42, 0.88), rgba(15, 23, 42, 0.95)),
                url("{BACKGROUND_IMAGE_URL}") no-repeat center center fixed;
    background-size: cover;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: #f8fafc;
}}

/* Header typography */
.main-title {{
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 8px;
}}

.subtitle {{
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 35px;
    font-weight: 400;
}}

/* Glassmorphism card container */
.glass-card {{
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
}}

/* Dashboard KPI metric cards */
.metric-card {{
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    transition: transform 0.3s ease, border-color 0.3s ease;
}}

.metric-card:hover {{
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.6);
}}

.metric-icon {{
    font-size: 32px;
    margin-bottom: 8px;
}}

.metric-value {{
    font-size: 28px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 4px;
}}

.metric-label {{
    font-size: 13px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}}

/* Feature Highlight Cards */
.feature-card {{
    background: rgba(15, 23, 42, 0.6);
    border-left: 4px solid #818cf8;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}

.feature-title {{
    font-size: 16px;
    font-weight: 700;
    color: #f8fafc;
}}

.feature-desc {{
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
}}

/* Team card layout */
.team-card {{
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease;
}}

.team-card:hover {{
    transform: translateY(-5px);
}}

.team-avatar {{
    font-size: 52px;
    margin-bottom: 12px;
}}

.team-name {{
    font-size: 22px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 4px;
}}

.team-role {{
    font-size: 14px;
    color: #c084fc;
    font-weight: 600;
    margin-bottom: 12px;
}}

/* Prediction display hero box */
.prediction-card {{
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.9) 0%, rgba(79, 70, 229, 0.9) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 40px 20px;
    border-radius: 24px;
    text-align: center;
    color: white;
    box-shadow: 0 15px 35px rgba(37, 99, 235, 0.4);
    margin-top: 20px;
}}

.prediction-title {{
    font-size: 18px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #93c5fd;
    font-weight: 700;
}}

.prediction-price {{
    font-size: 56px;
    font-weight: 900;
    margin: 12px 0;
    text-shadow: 0 4px 12px rgba(0,0,0,0.3);
}}

/* Sidebar customization */
[data-testid="stSidebar"] {{
    background-color: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}}

/* Primary button glowing effects */
.stButton > button {{
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border: none;
    height: 54px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.6);
}}

/* Footer */
.footer {{
    text-align: center;
    padding: 30px;
    color: #64748b;
    font-size: 14px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 40px;
}}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & FEATURES
# ============================================================

@st.cache_resource
def load_model():
    model_path = "random_forest_model.pkl"
    feature_path = "features.pkl"

    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    return model, features

model, features = load_model()


# ============================================================
# FILE CHECK
# ============================================================

if model is None:
    st.error("❌ Model or Feature files missing!")
    st.markdown("""
    <div class="glass-card">
        <h3>Required Workspace Files</h3>
        <p>Ensure these pickle files exist in your root directory:</p>
        <ul>
            <li><code>random_forest_model.pkl</code></li>
            <li><code>features.pkl</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🏎️ **AutoPrice AI**")
    st.caption("Developed by **Uday Aware**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "💰 Price Prediction",
            "📊 Dataset Analytics",
            "👥 Our Team"
        ]
    )

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
        <small style="color: #94a3b8;">LEAD ARCHITECT</small><br>
        <strong style="color: #38bdf8;">Uday Aware</strong><br><br>
        <small style="color: #94a3b8;">ENGINE MODEL</small><br>
        <strong style="color: #38bdf8;">Random Forest Regressor</strong><br><br>
        <small style="color: #94a3b8;">TARGET METRIC</small><br>
        <strong style="color: #38bdf8;">Vehicle Price Valuation</strong>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "🏠 Dashboard":

    st.markdown('<div class="main-title">🏎️ Executive AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">System overview, model telemetry, and architecture insights</div>', unsafe_allow_html=True)

    # 1. Top KPI Metrics Grid
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value">Random Forest</div>
            <div class="metric-label">Model Engine</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">{len(features)}</div>
            <div class="metric-label">Active Features</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">Regression</div>
            <div class="metric-label">Problem Domain</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">👑</div>
            <div class="metric-value">Uday Aware</div>
            <div class="metric-label">Lead Developer</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Hero Overview & Core Capabilities
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h2 style="margin-top:0; color:#38bdf8;">🚘 Intelligent Valuation Engine</h2>
            <p style="color:#cbd5e1; font-size: 15px; line-height: 1.7;">
                <strong>AutoPrice AI</strong> delivers instant high-precision market valuations for vehicles based on structural dimensions, engine specifications, fuel efficiency, and drivetrain dynamics.
            </p>
            <p style="color:#cbd5e1; font-size: 15px; line-height: 1.7;">
                By aggregating ensemble predictions from decision trees, the model mitigates variance and produces robust pricing models tailored for executive reporting and market analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h3 style="margin-top:0; color:#818cf8; margin-bottom:16px;">✨ Engine Highlights</h3>
            <div class="feature-card">
                <div class="feature-title">⚡ Real-time Prediction</div>
                <div class="feature-desc">Calculates market price estimates instantly upon parameters input.</div>
            </div>
            <div class="feature-card">
                <div class="feature-title">🛡️ Robust Multi-Feature Pipeline</div>
                <div class="feature-desc">Encodes drivetrain, fuel type, body style, and engine dimensions.</div>
            </div>
            <div class="feature-card">
                <div class="feature-title">📥 Report Export</div>
                <div class="feature-desc">Export detailed evaluation reports directly to CSV format.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Simulated Visual Performance Section
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#38bdf8;">📈 Valuation Drivers Trend</h3>
        <p style="color:#94a3b8; font-size:14px;">Simulated impact of Engine Power (HP) vs Highway Efficiency on Valuation</p>
    </div>
    """, unsafe_allow_html=True)

    # Generate visual chart data
    chart_data = pd.DataFrame({
        "Horsepower": np.linspace(80, 450, 30),
        "Estimated Valuation (₹ Lakhs)": np.sin(np.linspace(0, 3, 30)) * 15 + np.linspace(5, 55, 30),
        "Highway MPG": np.linspace(45, 15, 30)
    })

    st.line_chart(chart_data, x="Horsepower", y=["Estimated Valuation (₹ Lakhs)", "Highway MPG"], use_container_width=True)


# ============================================================
# PRICE PREDICTION PAGE
# ============================================================

elif page == "💰 Price Prediction":

    st.markdown('<div class="main-title">💰 Predict Car Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter real vehicle details to instantly generate an estimated market valuation</div>', unsafe_allow_html=True)

    # 1. Categorical Inputs Section
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#38bdf8;">🚗 Vehicle Specifications</h3>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        brand = st.selectbox("🏎️ Make / Brand", ["Toyota", "Honda", "BMW", "Audi", "Mercedes-Benz", "Nissan", "Ford", "Chevrolet", "Hyundai", "Volkswagen"])
        body_style = st.selectbox("🚘 Body Style", ["Sedan", "SUV", "Hatchback", "Convertible", "Wagon", "Coupe"])

    with col_b:
        fuel_type = st.selectbox("⛽ Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])
        transmission = st.selectbox("⚙️ Transmission", ["Automatic", "Manual"])

    with col_c:
        drive_wheels = st.selectbox("🛞 Drive Train", ["FWD (Front)", "RWD (Rear)", "4WD/AWD"])
        engine_location = st.radio("📍 Engine Position", ["Front", "Rear"], horizontal=True)

    # 2. Numerical Inputs Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0; color:#818cf8;">📊 Engine & Dimensions</h3>
    </div>
    """, unsafe_allow_html=True)

    col_1, col_2 = st.columns(2)

    with col_1:
        horsepower = st.slider("🐎 Horsepower (HP)", min_value=50, max_value=600, value=150, step=5)
        engine_size = st.slider("🔧 Engine Size (cc / cu-in)", min_value=500, max_value=6000, value=2000, step=100)
        curb_weight = st.number_input("⚖️ Curb Weight (lbs/kg)", min_value=1000, max_value=6000, value=2800, step=50)

    with col_2:
        highway_mpg = st.slider("🛣️ Highway Fuel Economy (MPG/KMPL)", min_value=10, max_value=60, value=30, step=1)
        city_mpg = st.slider("🏙️ City Fuel Economy (MPG/KMPL)", min_value=8, max_value=50, value=24, step=1)
        wheel_base = st.number_input("📏 Wheel Base", min_value=80.0, max_value=130.0, value=98.5, step=0.5)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Predict Trigger
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🚀 ESTIMATE VEHICLE PRICE", use_container_width=True)

    # 4. Handle Model Input Mapping
    if predict_button:
        try:
            input_dict = {f: 0.0 for f in features}

            num_mappings = {
                'horsepower': horsepower,
                'engine_size': engine_size,
                'enginesize': engine_size,
                'curb_weight': curb_weight,
                'curbweight': curb_weight,
                'highway_mpg': highway_mpg,
                'city_mpg': city_mpg,
                'wheel_base': wheel_base,
                'wheelbase': wheel_base
            }

            for feature_name in features:
                f_clean = feature_name.lower().replace("-", "_")
                if f_clean in num_mappings:
                    input_dict[feature_name] = float(num_mappings[f_clean])

                elif brand.lower() in f_clean:
                    input_dict[feature_name] = 1.0
                elif body_style.lower() in f_clean:
                    input_dict[feature_name] = 1.0
                elif fuel_type.lower() in f_clean:
                    input_dict[feature_name] = 1.0
                elif transmission.lower() in f_clean:
                    input_dict[feature_name] = 1.0

            input_df = pd.DataFrame([input_dict], columns=features)

            prediction = model.predict(input_df)
            predicted_price = max(0, float(prediction[0]))

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">Estimated Market Valuation</div>
                <div class="prediction-price">₹ {predicted_price:,.2f}</div>
                <span style="background: rgba(255,255,255,0.15); padding: 6px 16px; border-radius: 20px; font-size: 13px;">
                    Lead Engineer: Uday Aware | Engine: Random Forest
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            report = input_df.copy()
            report["Predicted_Price"] = predicted_price
            csv_data = report.to_csv(index=False)

            st.download_button(
                "📥 Download Valuation Report (CSV)",
                data=csv_data,
                file_name="vehicle_valuation.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Valuation Calculation Error: {str(e)}")


# ============================================================
# DATASET ANALYTICS PAGE
# ============================================================

elif page == "📊 Dataset Analytics":

    st.markdown('<div class="main-title">📊 Dataset Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore dataset distribution and model metrics</div>', unsafe_allow_html=True)

    dataset_path = "autos_dataset.csv"

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Target Variable", "Price")
        c4.metric("Missing Values", int(df.isna().sum().sum()))

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)
    else:
        st.warning("⚠️ `autos_dataset.csv` not found in root directory.")


# ============================================================
# TEAM PAGE
# ============================================================

elif page == "👥 Our Team":

    st.markdown('<div class="main-title">👥 Development Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">The engineers and data scientists behind AutoPrice AI</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown("""
        <div class="team-card">
            <div class="team-avatar">👨‍💻</div>
            <div class="team-name">Uday Aware</div>
            <div class="team-role">Lead AI & Machine Learning Engineer</div>
            <p style="color:#cbd5e1; font-size:13px;">Architected the system, trained the Random Forest Regressor model, and built the core Streamlit interface.</p>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="team-card">
            <div class="team-avatar">🎨</div>
            <div class="team-name">Aniket</div>
            <div class="team-role">UI/UX & Frontend Specialist</div>
            <p style="color:#cbd5e1; font-size:13px;">Designed custom dark-themed glassmorphism UI components and layout styling.</p>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="team-card">
            <div class="team-avatar">📊</div>
            <div class="team-name">Data Science Team</div>
            <div class="team-role">Data Engineering & Analytics</div>
            <p style="color:#cbd5e1; font-size:13px;">Cleaned automobile metrics, handled missing values, and engineered feature sets.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    🏎️ <b>AutoPrice AI Executive Edition</b> | Designed & Developed by <b>Uday Aware</b> & <b>Aniket</b> | Built with Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)