import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Job Trend Analysis",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# DARK MODE STYLE
# =============================================================================
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
h1, h2, h3, h4 { color: white; }
div[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #00ff00;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOADERS
# =============================================================================
@st.cache_data
def load_data(file):
    """Load CSV data"""
    df = pd.read_csv(file)
    return df

@st.cache_resource
def load_model(path):
    """Load pickled ML model"""
    if not Path(path).exists():
        st.error(f"❌ Model not found: {path}")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_feature_names(path="feature_names.pkl"):
    """Load saved feature names from training"""
    if not Path(path).exists():
        st.error(f"❌ File not found: {path}")
        return None
    
    try:
        with open(path, "rb") as f:
            feature_names = pickle.load(f)
        
        # Validate it's a list
        if not isinstance(feature_names, list):
            st.error("❌ feature_names.pkl is corrupted (not a list)")
            return None
        
        if len(feature_names) == 0:
            st.error("❌ feature_names.pkl is empty")
            return None
            
        return feature_names
        
    except EOFError:
        st.error("❌ feature_names.pkl is empty or corrupted!")
        st.info("The file exists but contains no data. Please recreate it.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading feature_names.pkl: {str(e)}")
        return None

# =============================================================================
# FEATURE PREPARATION (MATCH TRAINING EXACTLY)
# =============================================================================
def prepare_features(df, feature_names):
    """
    Prepare features to match training data exactly
    No new encoding - use exact same columns
    """
    df_copy = df.copy()
    
    with st.expander("🔧 Feature Preparation Details", expanded=False):
        st.write(f"**Required features:** {len(feature_names)}")
        st.write(f"**Available columns:** {len(df_copy.columns)}")
        
        # Check for missing columns
        missing = set(feature_names) - set(df_copy.columns)
        if missing:
            st.warning(f"⚠️ Missing columns (will be filled with 0): {list(missing)}")
            for col in missing:
                df_copy[col] = 0
    
    # Select only training features in exact order
    X = df_copy[feature_names].copy()
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing values
    X = X.fillna(0)
    
    return X

# =============================================================================
# VISUALIZATIONS
# =============================================================================
def plot_top_jobs(df, n=10):
    """Plot top N trending jobs"""
    if 'job_title' not in df.columns:
        return None
    
    if len(df) == 0:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    top = df["job_title"].value_counts().head(n)
    
    if len(top) == 0:
        return None
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top)))
    
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, color='white')
    ax.set_xlabel('Count', color='white', fontsize=12)
    ax.set_title(f'Top {len(top)} Job Roles', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, color='gray')
    
    plt.tight_layout()
    return fig

def plot_country_distribution(df, n=15):
    """Plot job distribution by country"""
    if 'company_location' not in df.columns:
        return None
    
    if len(df) == 0:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    top = df['company_location'].value_counts().head(n)
    
    if len(top) == 0:
        return None
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(top)))
    
    ax.bar(range(len(top)), top.values, color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=45, ha='right', color='white')
    ax.set_ylabel('Count', color='white', fontsize=12)
    ax.set_title(f'Jobs by Country (Top {len(top)})', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, color='gray')
    
    plt.tight_layout()
    return fig

def plot_salary_distribution(df):
    """Plot salary distribution"""
    if 'salary_usd' not in df.columns:
        return None
    
    if len(df) == 0:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    salary_data = df['salary_usd'].dropna()
    if len(salary_data) == 0:
        return None
    
    # Remove outliers
    q1, q3 = salary_data.quantile([0.25, 0.75])
    iqr = q3 - q1
    filtered = salary_data[(salary_data >= q1 - 1.5*iqr) & (salary_data <= q3 + 1.5*iqr)]
    
    if len(filtered) == 0:
        return None
    
    ax.hist(filtered, bins=50, color='#00d4ff', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Salary (USD)', color='white', fontsize=12)
    ax.set_ylabel('Frequency', color='white', fontsize=12)
    ax.set_title('Salary Distribution', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, color='gray')
    
    plt.tight_layout()
    return fig

def plot_prediction_distribution(predictions):
    """Plot distribution of predictions"""
    if len(predictions) == 0:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    ax.hist(predictions, bins=50, color='#00ff00', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Predicted Value', color='white', fontsize=12)
    ax.set_ylabel('Frequency', color='white', fontsize=12)
    ax.set_title('Prediction Distribution', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, color='gray')
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("📊 AI Job Trend Analysis Dashboard")
    st.markdown("### Powered by Machine Learning")
    st.markdown("---")

    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    models = {
        "Random Forest": "../Models/job_trend_model(Random Forest Model).pkl",
        "XGBoost": "../Models/job_trend_model(XGBoost).pkl",
        "Linear Regression": "../Models/job_trend_model(Linear_regression).pkl",
        "Gradient Boosting": "../Models/job_trend_model(Gradient_boosting).pkl",
        "Extra Trees": "../Models/job_trend_model(Extra_trees).pkl",
        "LightGBM": "../Models/job_trend_model(Lightgbm).pkl",
    }

    model_name = st.sidebar.selectbox("🤖 Select Model", list(models.keys()))
    st.sidebar.info(f"**Active Model:** {model_name}")
    
    # File upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    # ==========================================================================
    # MAIN CONTENT - BEFORE DATA UPLOAD
    # ==========================================================================
    
    if uploaded_file is None:
        st.info("👈 Upload a dataset from the sidebar to begin analysis")
        
        st.markdown("""
        ### 🎯 Features:
        - **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Linear Regression, Gradient Boosting, Extra Trees
        - **Interactive Visualizations**: Top jobs, country distribution, salary analysis
        - **Smart Filtering**: Filter by country and job title
        - **AI-Powered Predictions**: Get trend predictions for filtered data
        - **CSV Export**: Download predictions as CSV
        
        ### 📋 Requirements:
        1. Upload job dataset (CSV format)
        2. Ensure `feature_names.pkl` exists in the same folder as app.py
        3. Models should be in `../Models/` directory
        
        ### 🔧 Setup feature_names.pkl:
        Run this in your training notebook:
        ```python
        import pickle
        
        # After creating X_train
        feature_names = X_train.columns.tolist()
        
        with open("feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        
        print(f"✅ Saved {len(feature_names)} features")
        ```
        
        Then download and place it in the same folder as app.py.
        """)
        return

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    
    try:
        df = load_data(uploaded_file)
        st.success(f"✅ Dataset loaded: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return

    # Display column info
    with st.expander("📋 Dataset Info"):
        st.write("**Columns:**", list(df.columns))
        st.write("**Sample Data:**")
        sample = df.head(3).copy()
        for col in sample.select_dtypes(include=['object']).columns:
            sample[col] = sample[col].astype(str)
        st.dataframe(sample, width='stretch')

    st.markdown("---")

    # ==========================================================================
    # METRICS
    # ==========================================================================
    
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", f"{len(df):,}")
    
    with col2:
        if 'company_location' in df.columns:
            st.metric("Countries", df["company_location"].nunique())
        else:
            st.metric("Countries", "N/A")
    
    with col3:
        if 'job_title' in df.columns:
            st.metric("Unique Roles", df["job_title"].nunique())
        else:
            st.metric("Unique Roles", "N/A")
    
    with col4:
        if 'salary_usd' in df.columns:
            avg_salary = df['salary_usd'].mean()
            st.metric("Avg Salary", f"${avg_salary:,.0f}")
        else:
            st.metric("Avg Salary", "N/A")

    st.markdown("---")

    # ==========================================================================
    # FILTERS
    # ==========================================================================
    
    st.subheader("🔍 Filter Data")
    
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        if 'company_location' in df.columns:
            countries = ['All'] + sorted(df['company_location'].dropna().unique().tolist())
            selected_country = st.selectbox("📍 Select Country:", countries)
        else:
            selected_country = 'All'
            st.info("⚠️ Company location column not available")
    
    with filter_col2:
        if 'job_title' in df.columns:
            job_titles = ['All'] + sorted(df['job_title'].dropna().unique().tolist())
            selected_job = st.selectbox("💼 Select Job Title:", job_titles)
        else:
            selected_job = 'All'
            st.info("⚠️ Job title column not available")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_country != 'All' and 'company_location' in df.columns:
        filtered_df = filtered_df[filtered_df['company_location'] == selected_country]
    
    if selected_job != 'All' and 'job_title' in df.columns:
        filtered_df = filtered_df[filtered_df['job_title'] == selected_job]
    
    # Show filtered count
    if selected_country != 'All' or selected_job != 'All':
        st.info(f"📊 Showing **{len(filtered_df):,}** jobs after filtering (from **{len(df):,}** total)")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
        return

    st.markdown("---")

    # ==========================================================================
    # VISUALIZATIONS (Use filtered data)
    # ==========================================================================
    
    st.subheader("📈 Market Insights")
    
    tab1, tab2, tab3 = st.tabs(["🏆 Top Jobs", "🌍 By Country", "💰 Salary"])
    
    with tab1:
        fig = plot_top_jobs(filtered_df)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ No job title data available to display")
    
    with tab2:
        fig = plot_country_distribution(filtered_df)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ No country data available to display")
    
    with tab3:
        fig = plot_salary_distribution(filtered_df)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("⚠️ No salary data available to display")

    st.markdown("---")

    # ==========================================================================
    # PREDICTIONS
    # ==========================================================================
    
    st.subheader("🤖 AI Model Predictions")
    st.write("Run predictions on the filtered dataset using the selected ML model.")

    if st.button("🚀 Run Predictions", type="primary"):
        model_path = models[model_name]

        # Load model
        with st.spinner(f"Loading {model_name}..."):
            model = load_model(model_path)
        
        if model is None:
            st.error("❌ Could not load model. Please check if the model file exists.")
            return

        st.success(f"✅ Model loaded: {model_name}")

        # Load feature names
        feature_names = load_feature_names()
        
        if feature_names is None:
            st.error("❌ `feature_names.pkl` not found or corrupted!")
            st.info("**To fix this, run in your training notebook:**")
            st.code("""
import pickle

# After training (after creating X_train)
feature_names = X_train.columns.tolist()

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
    
print(f"✅ Saved {len(feature_names)} features")
print(feature_names)
""", language="python")
            st.info("Then download `feature_names.pkl` and place it in the same folder as app.py")
            return

        st.success(f"✅ Feature names loaded: **{len(feature_names)}** features")

        # Prepare features
        with st.spinner("Preparing features..."):
            try:
                X = prepare_features(filtered_df, feature_names)
                st.success(f"✅ Features prepared: **{X.shape[0]}** rows × **{X.shape[1]}** columns")
            except Exception as e:
                st.error(f"❌ Error preparing features: {str(e)}")
                return

        # Make predictions
        with st.spinner("Making predictions..."):
            try:
                predictions = model.predict(X)
                st.success(f"✅ Predictions completed: **{len(predictions)}** predictions")
            except Exception as e:
                st.error(f"❌ Error making predictions: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        # Add to dataframe
        result_df = filtered_df.copy()
        result_df["Prediction"] = predictions

        st.balloons()

        # Display results
        st.subheader("📋 Prediction Results")
        
        # Select display columns
        display_cols = ["Prediction"]
        if 'job_title' in result_df.columns:
            display_cols.insert(0, "job_title")
        if 'company_location' in result_df.columns:
            display_cols.insert(1, "company_location")
        if 'salary_usd' in result_df.columns:
            display_cols.append("salary_usd")
        
        # Convert object columns to string to avoid Arrow errors
        display_df = result_df[display_cols].head(100).copy()
        for col in display_df.select_dtypes(include=['object']).columns:
            display_df[col] = display_df[col].astype(str)
        
        st.dataframe(display_df, width='stretch')

        # Statistics
        st.subheader("📊 Prediction Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Mean", f"{predictions.mean():.2f}")
        with stat_col2:
            st.metric("Median", f"{np.median(predictions):.2f}")
        with stat_col3:
            st.metric("Std Dev", f"{predictions.std():.2f}")
        with stat_col4:
            st.metric("Range", f"{predictions.min():.1f} - {predictions.max():.1f}")

        # Prediction distribution plot
        st.subheader("📈 Prediction Distribution")
        fig_pred = plot_prediction_distribution(predictions)
        if fig_pred:
            st.pyplot(fig_pred)
            plt.close(fig_pred)

        # Download button
        st.markdown("---")
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Predictions (CSV)",
            data=csv,
            file_name=f"job_predictions_{model_name.replace(' ', '_')}.csv",
            mime="text/csv",
            type="primary"
        )

    st.markdown("---")
    
    # ==========================================================================
    # DATA PREVIEW (show filtered data)
    # ==========================================================================
    
    st.subheader("📄 Dataset Preview (Filtered)")
    preview_df = filtered_df.head(100).copy()
    for col in preview_df.select_dtypes(include=['object']).columns:
        preview_df[col] = preview_df[col].astype(str)
    st.dataframe(preview_df, width='stretch')
    
    st.info(f"💡 Showing first 100 rows of {len(filtered_df):,} filtered results")

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()