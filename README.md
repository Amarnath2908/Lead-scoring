# 🎯 LeadScore AI — Machine Learning Lead Scoring System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-111111?style=flat)](https://xgboost.ai/)
[![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

An end-to-end Machine Learning web application built for **X Education** to predict lead conversion probabilities and convert them into an actionable **Lead Score (0–100)** with assigned sales priority bands.

---

## 📌 Business Objective

X Education receives thousands of sales leads daily, but only **~30%** convert into paying customers. To optimize sales team bandwidth and response times, **LeadScore AI** scores every incoming lead based on historical pre-contact attributes.

* **Target Accuracy**: >80% accuracy on held-out test data.
* **Winning Model**: Calibrated AdaBoost (**81.75% accuracy**).
* **Output**: Conversion Prediction (`Will Convert` / `Won't Convert`), **Lead Score (0–100)**, **Priority Band**, and **Recommended Sales Action**.

---

## 🏷️ Lead Score Priority Bands

| Priority Band | Score Range | Recommended Sales Action |
| :--- | :---: | :--- |
| 🔥 **Very Hot** | **90 – 100** | Call immediately |
| ♨️ **Hot** | **75 – 89** | Same-day call |
| ☀️ **Warm** | **50 – 74** | Follow-up email within 24h |
| ❄️ **Cold** | **25 – 49** | Add to nurture campaign |
| 🧊 **Very Cold** | **0 – 24** | Deprioritize |

---

## 🖥️ User Interface (Streamlit App)

The Streamlit web application consists of 4 clean pages:

1. **🏠 Home**: Overview of the platform, priority band legend, dataset statistics, and raw data preview.
2. **📊 Dashboard**: High-level conversion rate analytics, leads grouped by source/origin, and real-time prediction history stats.
3. **🎯 Predict Lead**: Interactive form to enter lead attributes and receive real-time prediction scores, gauge charts, and recommendations.
4. **🗂️ Prediction History**: Searchable and filterable table of past predictions stored in SQLite with statistical breakdowns.

---

## 🛠️ Project Structure

```
LeadScoringApp/
├── app.py                     # Streamlit entry point (landing page & navigation shell)
├── config.py                  # Central configuration (paths, thresholds, score bands)
├── requirements.txt           # Project dependencies
├── Lead Scoring.csv           # Raw dataset
├── .gitignore                 # Git ignore rules
├── README.md                  # Documentation
│
├── pipeline/                  # Offline Machine Learning Pipeline
│   ├── data_preprocessing.py  # Data cleaning, null handling, leakage audit & ColumnTransformer
│   ├── feature_engineering.py # Derived features (website_engagement_score)
│   ├── model_training.py      # Candidates evaluation (6 models) + CalibratedClassifierCV
│   └── train.py               # Orchestrator script: python pipeline/train.py
│
├── models/                    # Saved Artifacts
│   ├── model.pkl              # Winner model (Calibrated AdaBoost)
│   ├── preprocessor.pkl       # Feature engineer + ColumnTransformer pipeline
│   └── model_metadata.json    # Metrics log for all candidate models
│
├── database/
│   └── predictions.db         # SQLite database for prediction history
│
├── utils/
│   ├── prediction.py          # Artifact loading & column-aligned inference
│   ├── scoring.py             # Probability -> Lead Score -> Priority Band mapping
│   ├── database.py            # SQLite CRUD & auto-recovery logic
│   └── visualization.py       # Plotly chart templates
│
└── pages/
    ├── 1_🏠_Home.py           # Home page
    ├── 2_📊_Dashboard.py      # Analytics dashboard
    ├── 3_🎯_Predict_Lead.py   # Single lead prediction interface
    └── 4_🗂️_Prediction_History.py # SQLite history log & filters
```

---

## 🤖 Model Benchmarks

Six candidate models were evaluated on an 80/20 train/test split using 5-fold probability calibration (`CalibratedClassifierCV`):

| Model Candidate | Test Accuracy | Test F1 Score | Score Range |
| :--- | :---: | :---: | :---: |
| **AdaBoost (Calibrated) ⭐** | **81.75%** | **0.7669** | **1 – 99** |
| Gradient Boosting (Calibrated) | 81.08% | 0.7571 | 1 – 99 |
| XGBoost (Calibrated) | 80.88% | 0.7539 | 7 – 92 |
| Random Forest (Calibrated) | 80.55% | 0.7494 | 8 – 91 |
| Decision Tree (Calibrated) | 79.21% | 0.7380 | 10 – 82 |
| Logistic Regression (Calibrated) | 79.14% | 0.7169 | 0 – 100 |

*Winner:* **AdaBoost with Sigmoid Calibration** was automatically selected and saved to `models/model.pkl`.

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install requirements:
```bash
cd LeadScoringApp
pip install -r requirements.txt
```

### 2. Run the Offline Training Pipeline

To retrain the models and regenerate the pipeline artifacts:
```bash
python pipeline/train.py
```

### 3. Start the Web Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open **http://localhost:8501** (or port 8502) in your browser.

---

## 🛡️ Data Cleaning & Leakage Audit

To prevent data leakage, 7 post-contact fields that are only filled *after* sales representatives contact a lead were excluded:
* `Tags`, `Lead Quality`, `Last Notable Activity`, `Asymmetrique Activity Score`, `Asymmetrique Profile Score`, `Asymmetrique Activity Index`, `Asymmetrique Profile Index`.

---

## 📄 License

Distributed under the MIT License.
