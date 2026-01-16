🚀 AI for Job Trend Analysis

An end-to-end machine learning project that analyzes global job market data and predicts job trend scores to identify roles and regions with strong growth potential. The project includes data analysis, multiple trained ML models, and an interactive Streamlit dashboard for visualization and predictions.

📌 Project Overview

The job market is dynamic and varies across countries, roles, and experience levels. This project aims to analyze historical job postings data and predict job trend strength, helping users understand which roles are growing and where.

What this project does:

Analyzes large-scale job posting data (15,000+ records)

Trains and compares multiple regression models

Uses pre-trained models or allows users to train their own

Visualizes job trends across countries and job roles

Deploys predictions via a Streamlit dashboard (local or live)

⚠️ Note: This project focuses on job trends, not in-demand skills prediction.

🧠 AI Task Definition

Problem Type: Regression

Prediction Target: Job trend score (continuous value)

Purpose: Identify booming, stable, or low-growth job roles based on historical data

Evaluation Metrics: RMSE and R² (accuracy is not applicable)

🏗️ System Workflow

User uploads a job dataset (or uses the default dataset)

User selects:

Pre-trained model or

Train a new model

Data is preprocessed using the same feature pipeline

Model predicts job trend scores

Results are visualized through charts and tables in Streamlit

Trend insights and suggestions are displayed

📦 Pre-trained Models Included

The project comes with multiple pre-trained models, trained on the main dataset and saved as .pkl files:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

Extra Trees Regressor

XGBoost Regressor

LightGBM Regressor

To ensure consistency during prediction, the feature structure used during training is stored in feature_names.pkl. This prevents feature mismatch errors during deployment.

📊 Model Performance
Model	RMSE	R²
Linear Regression	1.35	0.04
Random Forest	0.82	0.65
Gradient Boosting	0.89	0.58
Extra Trees	0.83	0.64
XGBoost	0.82	0.65
LightGBM	0.86	0.61
✅ Best Models

Primary: Random Forest (stable, strong performance)

Secondary: XGBoost (high accuracy, slightly complex)

📁 Project Structure
AI for Market Trend Analysis/
│
├── Project/
│   ├── Dataset/
│   └── Pre Models/
│
├── app.py
├── feature_names.pkl
├── job_trend_analysis.ipynb
├── requirements.txt
├── README.md
│
├── Documentation/
│   ├── Job Trend Analysis.doc
│   └── Job Trend Analysis.ppt

🛠️ Technologies Used

Programming: Python

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

ML Models: Scikit-learn, XGBoost, LightGBM

Deployment: Streamlit

Development: Google Colab, VS Code

📚 Learning Outcomes

Built and compared multiple ML regression models

Understood why RMSE and R² are better than accuracy for regression

Learned to separate training, saving, and deployment workflows

Solved real-world deployment issues like feature mismatch errors

Designed a user-friendly AI dashboard for non-technical users

🚧 Limitations & Future Scope

Trend scores are inferred from historical data, not real-time postings

External economic factors are not included

Future improvements could include:

Time-series trend modeling

Job category clustering

Live job data integration

⚠️ Disclaimer

This project is intended only for educational and analytical purposes.
Predictions should not be treated as guaranteed indicators of job market outcomes.
