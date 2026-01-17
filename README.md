# 🚀 AI for Job Trend Analysis

🧾 **Detailed Project Report:** [View Report](https://docs.google.com/document/d/1BtuqQrqJNHSOjdRrQX0R7A0hy3FjvijYIVen_lpUpvE/edit?usp=sharing)  
🖥️ **Streamlit Dashboard:** [Launch Dashboard](https://field-experiments-ai-for-job-trend-analysis-organiza-app-8wmlxp.streamlit.app/)  
📽️ **Demo & Presentation Slides:** [Watch Demo](https://drive.google.com/file/d/1gto5HfDJBSDyuZv03ouL0bGNcCilV-54/view?usp=sharing)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Model Performance](#-model-performance)
- [Technical Details](#-technical-details)
- [Learning Outcomes](#-learning-outcomes)
- [Limitations & Future Scope](#-limitations--future-scope)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project demonstrates how to build an end-to-end AI system for job market trend analysis. It's designed for beginners but includes advanced ML techniques that make it suitable for both learning and analytical applications.

**What it does:**

- Analyzes large-scale job posting data (15,000+ records)
- Processes features like job role, country, experience level, and employment type
- Trains multiple regression models to predict job trend scores
- Provides an interactive web dashboard for predictions and visualizations
- Achieves strong performance with R² scores up to 0.65

**Target Audience:** Students, beginners in ML/Data Science, and anyone interested in job market analytics

> ⚠️ **Note:** This project focuses on job trend analysis, not in-demand skills prediction.

---

## ✨ Features

### 🔄 Data Processing
- **Real Dataset:** 15,000+ job postings with global coverage
- **Multiple Features:** Job role, country, experience level, employment type
- **Automatic Preprocessing:** Data cleaning, encoding, and scaling
- **Feature Engineering:** Trend score calculation from historical patterns

### 🤖 Machine Learning
- **Multiple Models:** Linear Regression, Random Forest, Gradient Boosting, Extra Trees, XGBoost, LightGBM
- **Pre-trained Models:** Ready-to-use `.pkl` files for instant predictions
- **Model Comparison:** Comprehensive evaluation with RMSE and R² metrics
- **Feature Consistency:** `feature_names.pkl` prevents deployment errors

### 📊 Interactive Dashboard
- **Real-time Predictions:** Upload custom data or use pre-loaded dataset
- **Beautiful Visualizations:** Interactive charts with Plotly and Seaborn
- **Model Selection:** Choose pre-trained models or train new ones
- **Trend Insights:** Identify booming, stable, or declining job roles
- **User-friendly Interface:** Built with Streamlit

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/job-trend-analysis
cd job-trend-analysis
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Explore Data Analysis
```bash
jupyter notebook job_trend_analysis.ipynb
```
View data exploration, feature engineering, and model training process.

### 3. Launch Dashboard
```bash
streamlit run app.py
```
Opens interactive web dashboard at `http://localhost:8501`

### 4. Make Predictions
- **Option 1:** Use pre-trained models for instant predictions
- **Option 2:** Train new models on uploaded data
- **Option 3:** Explore trend visualizations by country and job role

---

## 📁 Project Structure

```
AI for Market Trend Analysis/
│
├── 📂 Documentation/
│   ├── Job Trend Analysis.doc      # Detailed project report
│   └── Job Trend Analysis.ppt      # Presentation slides
│
├── 📂 Project/
│   ├── 📂 Dataset/
│   │   └── ai_job_dataset.csv            # Job postings dataset
│   │
│   └── 📂 Pre Models/
│       ├── linear_regression_model.pkl
│       ├── random_forest_model.pkl
│       ├── gradient_boosting_model.pkl
│       ├── extra_trees_model.pkl
│       ├── xgboost_model.pkl
│       └── lightgbm_model.pkl
│
├── 📊 app.py                       # Streamlit dashboard
├── 🔧 feature_names.pkl            # Feature structure for consistency
├── 📓 job_trend_analysis.ipynb    # Analysis notebook
├── 📋 requirements.txt             # Python dependencies
└── 📚 README.md                    # This file
```

---

## 🛠 Installation

### Prerequisites
- Python 3.8+ (recommended: 3.9 or 3.10)
- Git
- Internet connection (for package installation)

### Step-by-Step Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/job-trend-analysis
cd job-trend-analysis
```

**2. Create virtual environment**
```bash
python -m venv venv
```

**3. Activate virtual environment**
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Model Performance

### Regression Metrics (Default Dataset - 15,000+ records)

| Model               | RMSE | R²   | Training Time | Notes                          |
|---------------------|------|------|---------------|--------------------------------|
| Random Forest       | 0.82 | 0.65 | ~3.5s        | **Best overall performance**   |
| XGBoost             | 0.82 | 0.65 | ~5.2s        | High accuracy, robust          |
| Extra Trees         | 0.83 | 0.64 | ~2.8s        | Fast and reliable              |
| LightGBM            | 0.86 | 0.61 | ~4.1s        | Good for large datasets        |
| Gradient Boosting   | 0.89 | 0.58 | ~6.3s        | Moderate performance           |
| Linear Regression   | 1.35 | 0.04 | ~0.5s        | Baseline model                 |

### ✅ Best Models
- **Primary:** Random Forest (stable, strong R² of 0.65)
- **Secondary:** XGBoost (equally accurate, slightly more complex)

### Key Metrics
- **RMSE Range:** 0.82 - 1.35 (lower is better)
- **R² Range:** 0.04 - 0.65 (higher is better, max 1.0)
- **Evaluation Method:** Train-test split (80-20) with cross-validation
- **Why RMSE & R²?** For regression tasks, these metrics measure prediction accuracy better than classification accuracy

### Feature Importance (Top Features)
1. **Job Role** - Most influential factor in trend prediction
2. **Country/Region** - Geographic demand variations
3. **Experience Level** - Entry vs. Senior role trends
4. **Employment Type** - Full-time, part-time, contract patterns

---

## 🔬 Technical Details

### Data Pipeline
1. **Data Collection:** Historical job postings from multiple sources
2. **Data Cleaning:** Handle missing values, duplicates, and outliers
3. **Feature Engineering:** Categorical encoding, feature scaling
4. **Model Training:** Multiple regression algorithms with hyperparameter tuning
5. **Model Evaluation:** Cross-validation and metric comparison
6. **Deployment:** Saved models with feature consistency checks

### AI Task Definition
- **Problem Type:** Regression (predicting continuous trend scores)
- **Target Variable:** Job trend score (0-10 scale)
- **Features:** Job role, country, experience level, employment type, etc.
- **Evaluation Metrics:** 
  - **RMSE:** Root Mean Squared Error (measures prediction error)
  - **R²:** R-squared (measures model fit, 0-1 scale)
- **Why not Accuracy?** Accuracy is for classification; regression uses RMSE and R²

### Machine Learning Models

**Random Forest Regressor (Best Model)**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
```

**XGBoost Regressor (High Performance)**
```python
XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
```

### Technologies Used
- **Programming:** Python 3.9+
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **ML Models:** Scikit-learn, XGBoost, LightGBM
- **Deployment:** Streamlit
- **Development:** Google Colab, Jupyter Notebook, VS Code

---

## 📚 Learning Outcomes

Through this project, you will learn:

- **Data Analysis:** How to explore and visualize large datasets
- **Feature Engineering:** Encoding categorical variables and scaling features
- **Model Selection:** Comparing multiple ML algorithms for regression
- **Evaluation Metrics:** Understanding RMSE and R² for regression tasks
- **Model Deployment:** Saving models with pickle and creating consistent prediction pipelines
- **Dashboard Development:** Building interactive web apps with Streamlit
- **Real-world Problem Solving:** Handling feature mismatch errors and deployment challenges

---

## 🚧 Limitations & Future Scope

### Current Limitations
- **Historical Data Only:** Trend scores are inferred from past data, not real-time postings
- **External Factors:** Economic conditions, policy changes not included
- **Limited Features:** Does not predict required skills or salary trends
- **Geographic Coverage:** May have data imbalance across regions

### Future Improvements
- **Time-Series Analysis:** Incorporate temporal patterns for better forecasting
- **Skills Prediction:** Add NLP models to predict in-demand skills
- **Real-time Data:** Integrate live job posting APIs
- **Salary Trends:** Predict compensation patterns by role and region
- **Job Category Clustering:** Group similar roles for better insights
- **Deep Learning Models:** Experiment with neural networks for improved accuracy

---

## 🚨 Disclaimer

**IMPORTANT:** This project is for educational and analytical purposes only.

- 📚 **Educational Tool:** Designed for learning ML and data science concepts
- ❌ **Not Career Advice:** Do not use predictions as sole basis for career decisions
- 📊 **Historical Analysis:** Past trends do not guarantee future job market conditions
- 🎯 **Analytical Purpose:** Meant for understanding patterns, not making guarantees
- 💼 **Professional Guidance:** Consult career advisors and industry experts for career planning

**Done by Y.R.Rishidhar Reddy for educational purpose only.**
