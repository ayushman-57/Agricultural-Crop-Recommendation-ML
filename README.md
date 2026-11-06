# Agricultural Crop Recommendation System Using Machine Learning

[![R](https://img.shields.io/badge/R-4.6%2B-blue.svg)](https://www.r-project.org/)
[![Shiny](https://img.shields.io/badge/Shiny-Interactive%20Dashboard-brightgreen.svg)](https://shiny.posit.co/)
[![Machine Learning](https://img.shields.io/badge/ML-caret%20%7C%20Random%20Forest-orange.svg)](https://topepo.github.io/caret/)
[![Cross-Validation](https://img.shields.io/badge/CV%20Macro--F1-90.76%25-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

An empirical machine learning decision support system and interactive analytics dashboard that recommends optimal agricultural crops based on localized soil macronutrients ($N, P, K, \text{pH}$), meteorological drivers (rainfall, humidity, temperature), and environmental farm conditions.

---

## 📌 Project Overview

In traditional agriculture, crop selection is frequently driven by ancestral precedent or short-term market speculation, leading to soil nutrient depletion and suboptimal yields. This project applies statistical hypothesis testing, multi-algorithm supervised classification, and cross-validated hyperparameter optimization to provide precision crop suitability recommendations.

### Key Highlights
- **Empirical Dataset:** 762 field observations across 17 Indian states and 99 districts.
- **Balanced Multi-Class Target:** 15 crop varieties (Rice, Wheat, Maize, Cotton, Banana, Chickpea, Groundnut, Mango, Millet, Mustard, Onion, Potato, Soybean, Sugarcane, Tomato).
- **14 Validated Predictors:** 7 soil/climate numerical features + 7 categorical farming attributes.
- **Statistical Validation:** Parametric One-way ANOVA, non-parametric Kruskal-Wallis, and Pearson Chi-Square tests of independence.
- **7 Machine Learning Algorithms:** Random Forest, Support Vector Machine (SVM), Multinomial Logistic Regression, Linear Discriminant Analysis (LDA), Gradient Boosting, K-Nearest Neighbors (KNN), and Decision Tree.
- **Champion Model:** **Tuned Random Forest** ($\text{mtry} = 6$) achieving **90.76% Cross-Validation Macro-F1** and **90.74% CV Accuracy**.
- **Interactive UI:** Field-ready R Shiny dashboard inspired by modern farm management design.

---

## 🏗️ Machine Learning Architecture

```
Raw Agricultural Data (762 Records, 21 Features)
                      │
                      ▼
Data Cleaning & Statistical Imputation (Median continuous, Mode categorical)
                      │
                      ▼
Exploratory Data Analysis (Class balance, distributions, outlier box plots, correlation)
                      │
                      ▼
Statistical Hypothesis Testing (ANOVA, Kruskal-Wallis, Chi-Square associations)
                      │
                      ▼
Stratified Train/Test Split (80% Train [610] / 20% Test [152])
                      │
                      ▼
Supervised Machine Learning (7 Implemented Classifiers)
                      │
                      ▼
10-Fold Repeated Cross-Validation & Hyperparameter Tuning (6 Tuned Models)
                      │
                      ▼
Champion Model Selection: Tuned Random Forest (mtry = 6, CV Macro-F1: 90.76%)
                      │
                      ▼
Interactive Shiny Dashboard Deployment (Real-time probabilistic inference)
```

---

## 📊 Empirical Model Benchmark

### 1. Test-Set Benchmark (152 Independent Test Records)

| Model Algorithm | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF Kernel)** | **90.13%** | 91.08% | 90.12% | **89.77%** | 89.79% | 0.8943 |
| **Multinomial Logistic Regression** | 89.47% | 90.57% | 89.39% | 89.41% | 89.52% | 0.8872 |
| **Linear Discriminant Analysis (LDA)** | 87.50% | 89.04% | 87.45% | 87.73% | 87.82% | 0.8661 |
| **Random Forest** | 86.18% | 87.07% | 86.18% | 85.93% | 85.99% | 0.8520 |
| **Gradient Boosting** | 82.89% | 83.78% | 82.79% | 82.48% | 82.59% | 0.8167 |
| **K-Nearest Neighbors (KNN)** | 72.37% | 73.01% | 72.24% | 71.72% | 71.87% | 0.7039 |
| **Decision Tree (CART)** | 45.39% | 29.68% | 44.91% | 33.90% | 34.41% | 0.4147 |

### 2. After 10-Fold Cross-Validated Hyperparameter Tuning

| Model Algorithm | Optimal Hyperparameters | CV Macro-F1 | CV Accuracy | Test Macro-F1 Gain |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest** 🏆 | `mtry = 6` | **90.76%** | **90.74%** | +3.26% |
| **Multinomial Logistic** | `decay = 1.0` | **87.58%** | **87.71%** | +0.63% |
| **Support Vector Machine** | `sigma = 0.0039, C = 11.31` | **87.54%** | **87.63%** | -1.32% |
| **Gradient Boosting** | `n.trees = 300, depth = 5, shrinkage = 0.1` | **85.41%** | **85.66%** | +4.22% |
| **Decision Tree** | `cp = 0.005` | **74.59%** | **75.09%** | **+40.71%** |
| **K-Nearest Neighbors** | `k = 15` | **67.02%** | **67.14%** | +5.23% |

> **Model Selection Rationale:** While SVM performed slightly higher on the single 152-sample test split, Random Forest exhibited the highest cross-validated generalization stability across 10-fold CV (90.76% Macro-F1), making it the selected predictive engine.

---

## 🖥️ Interactive Shiny Dashboard

The project includes an interactive web dashboard with 6 specialized modules:
1. **Overview:** Executive KPIs, project background, and end-to-end ML pipeline stepper.
2. **Dataset Explorer:** Filterable, searchable, and sortable view of the 762 records with missingness and schema audits.
3. **EDA & Visualizations:** Plotly interactive charts for class distribution, histograms, outlier box plots, and correlation heatmaps.
4. **Statistical Analysis:** Tables and effect-size charts for One-way ANOVA ($\eta^2$), Kruskal-Wallis ($\epsilon^2$), and Chi-Square Cramér's $V$.
5. **Model Benchmark:** Side-by-side performance comparison, Before vs. After tuning chart, 15x15 confusion matrix heatmap, and per-crop diagnostics.
6. **Crop Recommendation:** Real-time inference engine with input bounds validation, one-click agronomic presets, probability distribution, and top-5 alternative rankings.

---

## 📁 Repository Structure

```
.
├── dashboard/
│   └── app.R                       # Complete Shiny Dashboard (UI + Server)
├── data/
│   ├── cleaned_crop_data.csv       # 100% complete cleaned dataset
│   ├── cleaned_crop_data.rds       # R serialized cleaned dataset
│   ├── train_data.rds              # 80% stratified training partition (610 records)
│   └── test_data.rds               # 20% independent test partition (152 records)
├── documentation/
│   ├── figures/                    # High-resolution architectural & benchmark figures
│   └── Agricultural_Crop_Recommendation_Project_Report_Updated.docx
├── models/
│   ├── final_crop_model.rds        # Saved champion Random Forest model package
│   ├── tuned/                      # Individual tuned model RDS files
│   └── all_trained_models.rds      # Initial trained model objects
├── R/
│   ├── 01_data_import.R            # Data ingestion
│   ├── 02_data_cleaning.R          # Missing value statistical imputation
│   ├── 03_eda_crop_distribution.R  # Class distribution analysis
│   ├── 04_eda_numerical_analysis.R # Numerical feature distributions
│   ├── 05_eda_categorical_analysis.R
│   ├── 06_correlation_analysis.R   # Pearson correlation
│   ├── 07_eda_crop_comparison.R    # Crop-level comparisons
│   ├── 08_statistical_analysis.R   # ANOVA, Kruskal-Wallis, Chi-Square
│   ├── 09_model_training.R         # 7 supervised algorithms training
│   ├── 10_model_evaluation.R       # Test-set multiclass evaluation
│   ├── 11_hyperparameter_tuning.R  # 10-fold CV grid search
│   ├── 12_tuned_model_evaluation.R # Post-tuning comparison
│   ├── 13_final_predictive_model.R # Champion model packaging
│   └── 14_prediction_test.R        # Inference testing
├── results/                        # Empirical evaluation tables, CSVs, and plots
├── run_dashboard.R                 # One-click launcher script
└── README.md
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- R (version 4.2 or higher)
- RStudio (recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/ayushman-57/Agricultural-Crop-Recommendation-ML.git
cd Agricultural-Crop-Recommendation-ML
```

### 2. Install Required Packages
In R or RStudio console, run:
```R
install.packages(c(
  "shiny", "bslib", "plotly", "DT", 
  "dplyr", "caret", "randomForest", 
  "ggplot2", "readr"
))
```

### 3. Launch the Interactive Dashboard
You can run the dashboard in any of three ways:

- **Option A (Launcher Script):**
  ```R
  source("run_dashboard.R")
  ```
- **Option B (Direct Shiny Call):**
  ```R
  shiny::runApp("dashboard", launch.browser = TRUE)
  ```
- **Option C (RStudio):**
  Open `dashboard/app.R` and click **"Run App"** in the top right corner.

---

## 📜 Documentation

A comprehensive, publication-grade academic report is available in the `documentation/` folder:
- **Report Document:** [`Agricultural_Crop_Recommendation_Project_Report_Updated.docx`](documentation/Agricultural_Crop_Recommendation_Project_Report_Updated.docx)
- **High-Resolution Figures:** Located in [`documentation/figures/`](documentation/figures/)

---

## 👤 Author

- **Ayushman Chaudhary**
- GitHub: [@ayushman-57](https://github.com/ayushman-57)
- Email: ayushmanchaudhary57@gmail.com
