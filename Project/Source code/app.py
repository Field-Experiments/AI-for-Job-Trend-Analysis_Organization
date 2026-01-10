import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Job Trend Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR DARK MODE AND STYLING
# ============================================================================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00ff00;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stDataFrame {
        background-color: #1e1e1e;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Load and cache the uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ **Data loaded successfully!**")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model(model_path):
    """Load and cache the selected ML model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.write(f"✅ **Model loaded:** {model_path}")
        st.write(f"**Model type:** {type(model).__name__}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info(f"Looking for model at: {Path(model_path).absolute()}")
        return None

def prepare_features(df, feature_columns=None):
    """
    Prepare features for model prediction
    Handle missing values and encode categorical variables
    """
    df_processed = df.copy()
    
    st.write("**🔧 Feature Preparation:**")
    
    # Convert problematic object columns to string
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = df_processed[col].astype(str)
    
    st.write(f"Original columns: {list(df.columns)}")
    
    # Identify numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    st.write(f"Numeric columns: {numeric_cols}")
    st.write(f"Categorical columns: {categorical_cols}")
    
    # Encode categorical columns
    for col in categorical_cols:
        try:
            df_processed[col + '_encoded'] = pd.Categorical(df_processed[col]).codes
            st.write(f"✓ Encoded: {col}")
        except:
            st.write(f"✗ Failed to encode: {col}")
    
    # Get all numeric columns including encoded ones
    all_numeric = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target-like columns
    exclude_cols = ['salary_usd', 'Unnamed: 0', 'id', 'salary', 'wage']
    feature_columns = [col for col in all_numeric if col not in exclude_cols]
    
    st.write(f"**Selected features:** {feature_columns}")
    
    if len(feature_columns) == 0:
        st.error("❌ No features available for prediction!")
        return None, None
    
    # Fill missing values
    df_processed[feature_columns] = df_processed[feature_columns].fillna(0)
    
    X = df_processed[feature_columns]
    
    st.write(f"**Feature matrix shape:** {X.shape}")
    st.write(f"**Sample features:**")
    st.dataframe(X.head(3))
    
    return X, feature_columns

def make_predictions(model, X):
    """Make predictions using the loaded model"""
    try:
        st.write("**🤖 Making predictions...**")
        predictions = model.predict(X)
        st.write(f"✅ Predictions completed! Shape: {predictions.shape}")
        st.write(f"**Sample predictions:** {predictions[:5]}")
        return predictions
    except Exception as e:
        st.error(f"❌ Error making predictions: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_top_jobs(df, n=10):
    """Plot top N job titles by frequency"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    if 'job_title' not in df.columns:
        ax.text(0.5, 0.5, 'Job title column not found', 
                ha='center', va='center', color='white', fontsize=14)
        ax.axis('off')
        return fig
    
    top_jobs = df['job_title'].value_counts().head(n)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_jobs)))
    bars = ax.barh(range(len(top_jobs)), top_jobs.values, color=colors)
    ax.set_yticks(range(len(top_jobs)))
    ax.set_yticklabels(top_jobs.index, color='white')
    ax.set_xlabel('Number of Postings', color='white', fontsize=12)
    ax.set_title(f'Top {n} Trending Job Roles', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, color='gray')
    
    plt.tight_layout()
    return fig

def plot_jobs_by_country(df, n=15):
    """Plot job distribution by country"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    if 'company_location' not in df.columns:
        ax.text(0.5, 0.5, 'Company location column not found', 
                ha='center', va='center', color='white', fontsize=14)
        ax.axis('off')
        return fig
    
    top_countries = df['company_location'].value_counts().head(n)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(top_countries)))
    bars = ax.bar(range(len(top_countries)), top_countries.values, color=colors)
    ax.set_xticks(range(len(top_countries)))
    ax.set_xticklabels(top_countries.index, rotation=45, ha='right', color='white')
    ax.set_ylabel('Number of Jobs', color='white', fontsize=12)
    ax.set_title(f'Job Distribution by Country (Top {n})', color='white', fontsize=14, fontweight='bold')
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
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    if 'salary_usd' in df.columns:
        salary_data = df['salary_usd'].dropna()
        if len(salary_data) > 0:
            # Remove outliers for better visualization
            q1 = salary_data.quantile(0.25)
            q3 = salary_data.quantile(0.75)
            iqr = q3 - q1
            filtered_salary = salary_data[(salary_data >= q1 - 1.5*iqr) & 
                                          (salary_data <= q3 + 1.5*iqr)]
            
            ax.hist(filtered_salary, bins=50, color='#00d4ff', alpha=0.7, edgecolor='white')
            ax.set_xlabel('Salary (USD)', color='white', fontsize=12)
            ax.set_ylabel('Frequency', color='white', fontsize=12)
            ax.set_title('Salary Distribution', color='white', fontsize=14, fontweight='bold')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3, color='gray')
        else:
            ax.text(0.5, 0.5, 'No salary data available', 
                    ha='center', va='center', color='white', fontsize=14)
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'Salary column not found', 
                ha='center', va='center', color='white', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_experience_level_dist(df):
    """Plot experience level distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    if 'experience_level' in df.columns:
        exp_counts = df['experience_level'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        
        wedges, texts, autotexts = ax.pie(exp_counts.values, labels=exp_counts.index, 
                                            autopct='%1.1f%%', startangle=90,
                                            colors=colors[:len(exp_counts)])
        
        for text in texts:
            text.set_color('white')
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title('Experience Level Distribution', color='white', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Experience level column not found', 
                ha='center', va='center', color='white', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.title("📊 Job Trend Analysis Dashboard")
    st.markdown("### AI-Powered Job Market Intelligence")
    st.markdown("---")
    
    # ========================================================================
    # SIDEBAR - MODEL SELECTION AND FILE UPLOAD
    # ========================================================================
    
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    st.sidebar.subheader("Select ML Model")
    
    # ⭐ CHANGE MODEL PATHS HERE ⭐
    # You can use relative or absolute paths
    # Examples:
    # "models/random_forest.pkl"  -> if models are in a 'models' folder
    # "D:/path/to/model.pkl"      -> absolute path
    # "./random_forest.pkl"       -> current directory
    
    model_options = {
        "Random Forest": "../Models/job_trend_model(Random Forest Model).pkl",
        "XGBoost": "../Models/job_trend_model(XGBoost Model).pkl",
        "Linear Regression": "../Models/job_trend_model(Linear_regression).pkl",
        "Gradient Boosting": "../Models/job_trend_model(Gradient_boosting).pkl",
        "Extra Trees": "../Models/job_trend_model(Extra_trees).pkl",
        "LightGBM": "../Models/job_trend_model(Lightgbm).pkl"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Choose a model:",
        list(model_options.keys())
    )
    
    model_path = model_options[selected_model_name]
    
    # File upload
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your job dataset in CSV format"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Selected Model:** {selected_model_name}")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Dataset loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
            
            # Show column names
            with st.expander("📋 View Dataset Columns"):
                st.write("**Available columns:**")
                st.write(list(df.columns))
                st.write("**Column types:**")
                st.write(df.dtypes)
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", f"{len(df):,}")
            with col2:
                if 'company_location' in df.columns:
                    st.metric("Countries", df['company_location'].nunique())
                else:
                    st.metric("Countries", "N/A")
            with col3:
                if 'job_title' in df.columns:
                    st.metric("Unique Roles", df['job_title'].nunique())
                else:
                    st.metric("Unique Roles", "N/A")
            with col4:
                if 'salary_usd' in df.columns:
                    avg_salary = df['salary_usd'].mean()
                    st.metric("Avg Salary", f"${avg_salary:,.0f}")
                else:
                    st.metric("Avg Salary", "N/A")
            
            st.markdown("---")
            
            # ================================================================
            # FILTERS
            # ================================================================
            
            st.subheader("🔍 Filters")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                if 'company_location' in df.columns:
                    countries = ['All'] + sorted(df['company_location'].dropna().unique().tolist())
                    selected_country = st.selectbox("Select Country:", countries)
                else:
                    selected_country = 'All'
                    st.info("Country filter not available")
            
            with filter_col2:
                if 'job_title' in df.columns:
                    jobs = ['All'] + sorted(df['job_title'].dropna().unique().tolist())
                    selected_job = st.selectbox("Select Job Title:", jobs)
                else:
                    selected_job = 'All'
                    st.info("Job title filter not available")
            
            # Apply filters
            filtered_df = df.copy()
            if selected_country != 'All' and 'company_location' in df.columns:
                filtered_df = filtered_df[filtered_df['company_location'] == selected_country]
            if selected_job != 'All' and 'job_title' in df.columns:
                filtered_df = filtered_df[filtered_df['job_title'] == selected_job]
            
            st.info(f"Showing {len(filtered_df)} jobs after filtering")
            
            st.markdown("---")
            
            # ================================================================
            # MODEL PREDICTIONS
            # ================================================================
            
            st.subheader("🤖 AI Model Predictions")
            
            if st.button("🚀 Run Predictions", type="primary"):
                st.write("### Prediction Process Log:")
                
                # Check if model file exists
                if not Path(model_path).exists():
                    st.error(f"❌ Model file '{model_path}' not found in the current directory!")
                    st.info(f"Looking for: {Path(model_path).absolute()}")
                    st.info("Please ensure the .pkl file is in the same folder as app.py")
                else:
                    # Load model
                    with st.spinner(f"Loading {selected_model_name} model..."):
                        model = load_model(model_path)
                    
                    if model is not None:
                        with st.spinner("Preparing features and making predictions..."):
                            try:
                                # Prepare features
                                X, feature_names = prepare_features(filtered_df)
                                
                                if X is not None:
                                    # Make predictions
                                    predictions = make_predictions(model, X)
                                    
                                    if predictions is not None:
                                        # Add predictions to dataframe
                                        result_df = filtered_df.copy()
                                        result_df['Predicted_Value'] = predictions
                                        
                                        st.success("✅ Predictions completed successfully!")
                                        st.balloons()
                                        
                                        # Display predictions
                                        st.subheader("📋 Prediction Results")
                                        
                                        # Select columns to display
                                        display_cols = ['Predicted_Value']
                                        if 'job_title' in result_df.columns:
                                            display_cols.insert(0, 'job_title')
                                        if 'company_location' in result_df.columns:
                                            display_cols.insert(1, 'company_location')
                                        if 'salary_usd' in result_df.columns:
                                            display_cols.append('salary_usd')
                                        
                                        st.dataframe(
                                            result_df[display_cols].head(100),
                                            use_container_width=True
                                        )
                                        
                                        # Download button
                                        csv = result_df.to_csv(index=False)
                                        st.download_button(
                                            label="📥 Download Full Predictions as CSV",
                                            data=csv,
                                            file_name="job_predictions.csv",
                                            mime="text/csv"
                                        )
                                        
                                        # Prediction statistics
                                        st.subheader("📊 Prediction Statistics")
                                        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                                        
                                        with pred_col1:
                                            st.metric("Mean", f"{predictions.mean():.2f}")
                                        with pred_col2:
                                            st.metric("Median", f"{np.median(predictions):.2f}")
                                        with pred_col3:
                                            st.metric("Std Dev", f"{predictions.std():.2f}")
                                        with pred_col4:
                                            st.metric("Min/Max", f"{predictions.min():.1f} / {predictions.max():.1f}")
                                        
                                        # Plot predictions distribution
                                        st.subheader("📈 Prediction Distribution")
                                        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
                                        fig_pred.patch.set_facecolor('#0e1117')
                                        ax_pred.set_facecolor('#1e1e1e')
                                        
                                        ax_pred.hist(predictions, bins=50, color='#00ff00', alpha=0.7, edgecolor='white')
                                        ax_pred.set_xlabel('Predicted Value', color='white', fontsize=12)
                                        ax_pred.set_ylabel('Frequency', color='white', fontsize=12)
                                        ax_pred.set_title('Distribution of Predictions', color='white', fontsize=14, fontweight='bold')
                                        ax_pred.tick_params(colors='white')
                                        ax_pred.spines['bottom'].set_color('white')
                                        ax_pred.spines['left'].set_color('white')
                                        ax_pred.spines['top'].set_visible(False)
                                        ax_pred.spines['right'].set_visible(False)
                                        ax_pred.grid(axis='y', alpha=0.3, color='gray')
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig_pred)
                                        
                                else:
                                    st.error("❌ Failed to prepare features for prediction")
                            
                            except Exception as e:
                                st.error(f"❌ Error during prediction process: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
            
            st.markdown("---")
            
            # ================================================================
            # VISUALIZATIONS
            # ================================================================
            
            st.subheader("📈 Data Visualizations")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏆 Top Jobs", 
                "🌍 By Country", 
                "💰 Salary Analysis",
                "📊 Experience Level"
            ])
            
            with tab1:
                st.markdown("#### Top 10 Trending Job Roles")
                fig1 = plot_top_jobs(filtered_df, n=10)
                st.pyplot(fig1)
            
            with tab2:
                st.markdown("#### Job Distribution by Country")
                fig2 = plot_jobs_by_country(filtered_df, n=15)
                st.pyplot(fig2)
            
            with tab3:
                st.markdown("#### Salary Distribution")
                fig3 = plot_salary_distribution(filtered_df)
                st.pyplot(fig3)
            
            with tab4:
                st.markdown("#### Experience Level Distribution")
                fig4 = plot_experience_level_dist(filtered_df)
                st.pyplot(fig4)
            
            st.markdown("---")
            
            # ================================================================
            # DATA PREVIEW
            # ================================================================
            
            st.subheader("📄 Data Preview (First 50 Rows)")
            st.dataframe(filtered_df.head(50), use_container_width=True)
            
    else:
        # Welcome screen
        st.info("👈 Please upload a CSV file from the sidebar to get started")
        
        st.markdown("""
        ### 🎯 Features:
        - 🤖 **Multiple ML Models**: Random Forest, XGBoost, LightGBM, and more
        - 📊 **Interactive Visualizations**: Charts and graphs for data exploration
        - 🔍 **Advanced Filtering**: Filter by country and job title
        - 💾 **Export Results**: Download predictions as CSV
        - 📈 **Real-time Analysis**: Instant insights from your data
        
        ### 📝 How to Use:
        1. **Select a Model** from the sidebar dropdown
        2. **Upload CSV Dataset** with your job data
        3. **Apply Filters** to narrow down results (optional)
        4. **Click "Run Predictions"** to generate AI predictions
        5. **Explore Visualizations** in different tabs
        6. **Download Results** as CSV file
        
        ### 📋 Dataset Requirements:
        Your CSV should contain columns like:
        - `job_title` (recommended)
        - `company_location` (recommended)
        - `salary_usd` (optional)
        - `experience_level` (optional)
        - Other numeric features for model prediction
        
        ### 🔧 Troubleshooting:
        - Ensure your `.pkl` model files are in the same directory as `app.py`
        - Check that your CSV has numeric columns for predictions
        - Use the debug logs when running predictions to identify issues
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()