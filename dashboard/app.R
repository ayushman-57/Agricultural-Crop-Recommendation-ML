# ==============================================================================
# AGRICULTURAL CROP RECOMMENDATION SYSTEM
# Interactive Analytics & Machine Learning Decision Support Dashboard
# ==============================================================================

library(shiny)
library(bslib)
library(plotly)
library(DT)
library(dplyr)
library(caret)
library(randomForest)

# ------------------------------------------------------------------------------
# Robust Relative File Path Resolver
# ------------------------------------------------------------------------------
resolve_path <- function(rel_path) {
  if (file.exists(rel_path)) return(rel_path)
  parent_path <- file.path("..", rel_path)
  if (file.exists(parent_path)) return(parent_path)
  rel_path
}

# ------------------------------------------------------------------------------
# Data & Model Preloading with Error Handling
# ------------------------------------------------------------------------------
crop_data <- tryCatch({
  df <- readRDS(resolve_path("data/cleaned_crop_data.rds"))
  as.data.frame(df)
}, error = function(e) {
  message("Warning: could not read RDS, attempting CSV")
  read.csv(resolve_path("data/cleaned_crop_data.csv"), stringsAsFactors = FALSE)
})

# Load final model package
final_model_pkg <- tryCatch({
  readRDS(resolve_path("models/final_crop_model.rds"))
}, error = function(e) {
  message("Warning: final_crop_model.rds not loaded: ", conditionMessage(e))
  NULL
})

rf_model <- if (!is.null(final_model_pkg)) final_model_pkg$model else NULL

# Load evaluation and statistics results
results_data <- list(
  model_comparison = tryCatch(read.csv(resolve_path("results/model_evaluation/final_model_comparison.csv")), error = function(e) NULL),
  tuning_comparison = tryCatch(read.csv(resolve_path("results/hyperparameter_tuning/tuning_comparison.csv")), error = function(e) NULL),
  before_after = tryCatch(read.csv(resolve_path("results/tuned_evaluation/before_after_comparison.csv")), error = function(e) NULL),
  macro_f1_improvement = tryCatch(read.csv(resolve_path("results/tuned_evaluation/macro_f1_improvement.csv")), error = function(e) NULL),
  tuned_models = tryCatch(read.csv(resolve_path("results/tuned_evaluation/tuned_model_results.csv")), error = function(e) NULL),
  cleaning_report = tryCatch(read.csv(resolve_path("results/data_cleaning_report.csv")), error = function(e) NULL),
  crop_dist = tryCatch(read.csv(resolve_path("results/crop_distribution.csv")), error = function(e) NULL),
  correlation = tryCatch(read.csv(resolve_path("results/correlation_matrix.csv"), row.names = 1), error = function(e) NULL),
  descriptive_stats = tryCatch(read.csv(resolve_path("results/statistical_analysis/descriptive_statistics.csv")), error = function(e) NULL),
  anova = tryCatch(read.csv(resolve_path("results/statistical_analysis/anova_results.csv")), error = function(e) NULL),
  kruskal = tryCatch(read.csv(resolve_path("results/statistical_analysis/kruskal_results.csv")), error = function(e) NULL),
  chi_square = tryCatch(read.csv(resolve_path("results/statistical_analysis/chi_square_results.csv")), error = function(e) NULL)
)

# Available Model names and their result file prefixes
model_mapping <- list(
  "Random Forest" = "random_forest",
  "SVM" = "svm",
  "Multinomial Logistic Regression" = "multinomial_logistic_regression",
  "LDA" = "lda",
  "Gradient Boosting" = "gradient_boosting",
  "KNN" = "knn",
  "Decision Tree" = "decision_tree"
)

# Best tuning hyperparameters from research
tuning_params_info <- data.frame(
  Model = c("Random Forest", "Multinomial Logistic Regression", "SVM", "Gradient Boosting", "Decision Tree", "KNN"),
  Best_Parameters = c("mtry = 6", "decay = 1", "sigma = 0.0039, C = 11.31", "n.trees = 300, depth = 5, shrinkage = 0.1", "cp = 0.005", "k = 15"),
  CV_Macro_F1 = c("90.76%", "87.58%", "87.54%", "85.41%", "74.59%", "67.02%"),
  CV_Accuracy = c("90.74%", "87.71%", "87.63%", "85.66%", "75.09%", "67.14%")
)

# Crop Icons lookup
crop_emojis <- list(
  "Rice" = "🌾", "Wheat" = "🌾", "Maize" = "🌽", "Banana" = "🍌", "Chickpea" = "🧆",
  "Cotton" = "☁️", "Groundnut" = "🥜", "Mango" = "🥭", "Millet" = "🌾", "Mustard" = "🌿",
  "Onion" = "🧅", "Potato" = "🥔", "Soybean" = "🌱", "Sugarcane" = "🎋", "Tomato" = "🍅"
)

# Distinct categorical levels
categorical_options <- list(
  Soil_Type = sort(unique(crop_data$Soil_Type)),
  State = sort(unique(crop_data$State)),
  Season = sort(unique(crop_data$Season)),
  Irrigation_Type = sort(unique(crop_data$Irrigation_Type)),
  Water_Source = sort(unique(crop_data$Water_Source)),
  Land_Type = sort(unique(crop_data$Land_Type)),
  Previous_Crop = sort(unique(crop_data$Previous_Crop))
)

# Numerical ranges for validation and defaults
num_ranges <- list(
  N = list(min = 15, max = 250, default = 90, unit = "kg/ha"),
  P = list(min = 20, max = 100, default = 50, unit = "kg/ha"),
  K = list(min = 20, max = 240, default = 50, unit = "kg/ha"),
  pH = list(min = 5.0, max = 8.5, default = 6.7, unit = ""),
  Rainfall = list(min = 250, max = 2400, default = 800, unit = "mm"),
  Humidity = list(min = 30, max = 95, default = 65, unit = "%"),
  Temperature = list(min = 10, max = 40, default = 25, unit = "°C")
)

# ------------------------------------------------------------------------------
# Custom CSS inspired by FarmVista Clean UI
# ------------------------------------------------------------------------------
custom_css <- "
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --primary: #10B981;
  --primary-dark: #059669;
  --primary-light: #ECFDF5;
  --accent-blue: #3B82F6;
  --accent-purple: #8B5CF6;
  --accent-amber: #F59E0B;
  --surface: #FFFFFF;
  --bg-app: #F8FAFC;
  --text-main: #0F172A;
  --text-muted: #64748B;
  --border-color: #E2E8F0;
  --card-radius: 16px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
  --shadow-card: 0 4px 14px 0 rgba(15, 23, 42, 0.05);
  --shadow-hover: 0 8px 24px 0 rgba(16, 185, 129, 0.12);
}

body {
  font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
  background-color: var(--bg-app) !important;
  color: var(--text-main) !important;
  margin: 0;
  padding: 0;
  overflow-x: hidden;
}

/* Sidebar Styling */
.main-sidebar {
  background-color: var(--surface) !important;
  border-right: 1px solid var(--border-color) !important;
  min-height: 100vh;
  padding: 24px 16px;
}

.brand-container {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px 24px 12px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 20px;
}

.brand-logo-icon {
  width: 42px;
  height: 42px;
  background: linear-gradient(135deg, #10B981 0%, #059669 100%);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 20px;
  box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
}

.brand-title {
  font-size: 19px;
  font-weight: 800;
  color: var(--text-main);
  letter-spacing: -0.5px;
  line-height: 1.2;
}

.brand-subtitle {
  font-size: 11px;
  font-weight: 600;
  color: var(--primary-dark);
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

/* Nav Pills Customization */
.nav-pills .nav-link {
  font-family: 'Plus Jakarta Sans', sans-serif;
  color: var(--text-muted) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  margin-bottom: 6px !important;
  display: flex !important;
  align-items: center !important;
  gap: 12px !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  border: 1px solid transparent !important;
}

.nav-pills .nav-link:hover {
  background-color: #F1F5F9 !important;
  color: var(--text-main) !important;
  transform: translateX(3px);
}

.nav-pills .nav-link.active {
  background-color: var(--primary-light) !important;
  color: var(--primary-dark) !important;
  border-color: #A7F3D0 !important;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08) !important;
  font-weight: 700 !important;
}

.nav-pills .nav-link i {
  font-size: 16px;
  width: 20px;
  text-align: center;
}

/* Top App Header Banner */
.app-header {
  background-color: var(--surface);
  border-bottom: 1px solid var(--border-color);
  padding: 18px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-title-group h2 {
  font-size: 22px;
  font-weight: 800;
  color: var(--text-main);
  margin: 0;
  letter-spacing: -0.5px;
}

.header-title-group p {
  font-size: 13px;
  color: var(--text-muted);
  margin: 2px 0 0 0;
}

.header-badges {
  display: flex;
  align-items: center;
  gap: 12px;
}

.model-pill-badge {
  background: var(--primary-light);
  border: 1px solid #A7F3D0;
  color: var(--primary-dark);
  font-size: 12px;
  font-weight: 700;
  padding: 6px 14px;
  border-radius: 9999px;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* Cards & UI Blocks */
.farm-card {
  background-color: var(--surface);
  border: 1px solid var(--border-color);
  border-radius: var(--card-radius);
  box-shadow: var(--shadow-card);
  padding: 24px;
  margin-bottom: 24px;
  transition: transform 0.2s, box-shadow 0.2s;
}

.farm-card:hover {
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
}

.farm-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid #F1F5F9;
}

.farm-card-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--text-main);
  margin: 0;
}

.farm-card-subtitle {
  font-size: 12px;
  color: var(--text-muted);
}

/* KPI Summary Cards */
.kpi-card {
  background-color: var(--surface);
  border: 1px solid var(--border-color);
  border-radius: var(--card-radius);
  padding: 20px;
  box-shadow: var(--shadow-card);
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  transition: all 0.25s ease;
  height: 100%;
}

.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-hover);
  border-color: #A7F3D0;
}

.kpi-label {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}

.kpi-value {
  font-size: 26px;
  font-weight: 800;
  color: var(--text-main);
  letter-spacing: -0.5px;
  line-height: 1.1;
  margin-bottom: 6px;
}

.kpi-subtext {
  font-size: 12px;
  font-weight: 600;
  color: var(--primary-dark);
}

.kpi-icon-box {
  width: 48px;
  height: 48px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}

.icon-green { background: #ECFDF5; color: #10B981; }
.icon-blue { background: #EFF6FF; color: #3B82F6; }
.icon-purple { background: #F5F3FF; color: #8B5CF6; }
.icon-amber { background: #FFFBEB; color: #F59E0B; }
.icon-rose { background: #FFF1F2; color: #F43F5E; }
.icon-teal { background: #F0FDFA; color: #0D9488; }

/* Interactive Pipeline Stepper */
.pipeline-container {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 16px 0;
}

.pipeline-step {
  background: #F8FAFC;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px 16px;
  flex: 1 1 180px;
  min-width: 140px;
  position: relative;
  transition: all 0.2s;
}

.pipeline-step:hover {
  background: var(--primary-light);
  border-color: #6EE7B7;
}

.pipeline-step-num {
  font-size: 10px;
  font-weight: 800;
  color: var(--primary-dark);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.pipeline-step-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-main);
  margin-top: 2px;
}

.pipeline-step-desc {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 2px;
}

/* Prediction Feature Form Styling */
.form-section-title {
  font-size: 13px;
  font-weight: 800;
  color: var(--primary-dark);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  padding-bottom: 6px;
  border-bottom: 2px solid #ECFDF5;
  margin-top: 14px;
  margin-bottom: 14px;
}

.btn-recommend {
  background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
  color: white !important;
  font-size: 16px !important;
  font-weight: 800 !important;
  padding: 14px 28px !important;
  border-radius: 14px !important;
  border: none !important;
  width: 100% !important;
  box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35) !important;
  transition: all 0.2s !important;
  letter-spacing: 0.2px;
}

.btn-recommend:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 22px rgba(16, 185, 129, 0.45) !important;
  color: white !important;
}

.btn-preset {
  background: #F1F5F9 !important;
  color: #334155 !important;
  border: 1px solid var(--border-color) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  padding: 6px 12px !important;
  margin: 3px !important;
  transition: all 0.15s !important;
}

.btn-preset:hover {
  background: #E2E8F0 !important;
  color: #0F172A !important;
}

/* Recommendation Results Card */
.result-banner {
  background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
  border-radius: var(--card-radius);
  color: white;
  padding: 30px;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: 0 10px 25px rgba(6, 78, 59, 0.25);
  margin-bottom: 24px;
}

.result-crop-icon {
  font-size: 56px;
  line-height: 1;
  margin-bottom: 10px;
}

.result-crop-name {
  font-size: 34px;
  font-weight: 800;
  letter-spacing: -0.5px;
  margin: 0;
}

.result-prob-badge {
  display: inline-block;
  background: rgba(255, 255, 255, 0.18);
  border: 1px solid rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(4px);
  padding: 6px 16px;
  border-radius: 9999px;
  font-size: 14px;
  font-weight: 700;
  margin-top: 10px;
}

.progress-bar-custom {
  background: linear-gradient(90deg, #10B981, #059669);
  border-radius: 9999px;
  height: 8px;
}

.prob-row {
  margin-bottom: 12px;
}

.prob-row-header {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
}

/* DT Table Customizations */
.dataTables_wrapper {
  font-size: 13px;
}
table.dataTable thead th {
  background-color: #F8FAFC !important;
  color: #334155 !important;
  font-weight: 700 !important;
  border-bottom: 2px solid #E2E8F0 !important;
}
table.dataTable tbody tr:hover {
  background-color: #F8FAFC !important;
}

/* Disclaimer text */
.disclaimer-box {
  background: #FFFBEB;
  border-left: 4px solid #F59E0B;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 12px;
  color: #92400E;
  margin-top: 14px;
}

/* Sidebar Footer Model Card */
.sidebar-footer-card {
  background: #F8FAFC;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 14px;
  margin-top: 28px;
}

.status-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #10B981;
  margin-right: 6px;
  box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
}
"

# ------------------------------------------------------------------------------
# UI DEFINITION
# ------------------------------------------------------------------------------
ui <- fluidPage(
  tags$head(
    tags$title("AgriCrop AI | Agricultural Crop Analytics & Recommendation"),
    tags$style(HTML(custom_css)),
    tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css")
  ),
  
  fluidRow(
    # Left Navigation Sidebar
    column(
      width = 2,
      class = "main-sidebar",
      div(
        class = "brand-container",
        div(class = "brand-logo-icon", icon("seedling")),
        div(
          div(class = "brand-title", "AgriCrop AI"),
          div(class = "brand-subtitle", "Crop Analytics")
        )
      ),
      
      navlistPanel(
        id = "main_nav",
        widths = c(12, 12),
        well = FALSE,
        tabPanel(
          title = tagList(icon("th-large"), "Overview"),
          value = "page_overview"
        ),
        tabPanel(
          title = tagList(icon("database"), "Dataset Explorer"),
          value = "page_dataset"
        ),
        tabPanel(
          title = tagList(icon("chart-pie"), "EDA & Visuals"),
          value = "page_eda"
        ),
        tabPanel(
          title = tagList(icon("calculator"), "Statistical Tests"),
          value = "page_statistics"
        ),
        tabPanel(
          title = tagList(icon("microchip"), "Model Benchmark"),
          value = "page_models"
        ),
        tabPanel(
          title = tagList(icon("leaf"), "Recommend Crop"),
          value = "page_recommend"
        )
      ),
      
      # Sidebar Footer Model Status Card
      div(
        class = "sidebar-footer-card",
        div(
          style = "font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;",
          span(class = "status-indicator"), "Active Engine"
        ),
        div(style = "font-size: 13px; font-weight: 800; color: #0F172A;", "Random Forest (Tuned)"),
        div(style = "font-size: 12px; color: #059669; font-weight: 600; margin-top: 2px;", "CV Macro-F1: 90.76%"),
        div(style = "font-size: 11px; color: #64748B; margin-top: 2px;", "CV Accuracy: 90.74%")
      )
    ),
    
    # Main Content Area
    column(
      width = 10,
      style = "padding: 0 28px 40px 28px;",
      
      # Header Bar
      div(
        class = "app-header",
        div(
          class = "header-title-group",
          h2(textOutput("page_header_title")),
          p(textOutput("page_header_subtitle"))
        ),
        div(
          class = "header-badges",
          div(
            class = "model-pill-badge",
            icon("check-circle"),
            "14 Predictors Validated"
          ),
          div(
            class = "model-pill-badge",
            icon("award"),
            "Selected: Random Forest"
          )
        )
      ),
      
      # Conditional Page Content
      conditionalPanel(
        condition = "input.main_nav == 'page_overview'",
        
        # 6 KPI Top Cards
        fluidRow(
          column(
            width = 2,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Dataset Records"),
                div(class = "kpi-value", "762"),
                div(class = "kpi-subtext", "100% Imputed & Valid")
              ),
              div(class = "kpi-icon-box icon-green", icon("database"))
            )
          ),
          column(
            width = 2,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Crop Classes"),
                div(class = "kpi-value", "15"),
                div(class = "kpi-subtext", "Balanced Multi-class")
              ),
              div(class = "kpi-icon-box icon-blue", icon("seedling"))
            )
          ),
          column(
            width = 2,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "States Covered"),
                div(class = "kpi-value", "17"),
                div(class = "kpi-subtext", "Major Agro-Zones")
              ),
              div(class = "kpi-icon-box icon-purple", icon("map-marked-alt"))
            )
          ),
          column(
            width = 2,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Districts"),
                div(class = "kpi-value", "99"),
                div(class = "kpi-subtext", "Regional Diversity")
              ),
              div(class = "kpi-icon-box icon-amber", icon("location-dot"))
            )
          ),
          column(
            width = 2,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Implemented ML"),
                div(class = "kpi-value", "7"),
                div(class = "kpi-subtext", "6 Tuned Models")
              ),
              div(class = "kpi-icon-box icon-rose", icon("microchip"))
            )
          ),
          column(
            width = 2,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Selected Model"),
                div(class = "kpi-value", style = "font-size: 19px; line-height: 1.4;", "Random Forest"),
                div(class = "kpi-subtext", "CV Macro-F1: 90.76%")
              ),
              div(class = "kpi-icon-box icon-teal", icon("award"))
            )
          )
        ),
        
        br(),
        
        # Project Objective & Pipeline
        fluidRow(
          column(
            width = 7,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("bullseye"), " Project Objectives & Background"),
                  span(class = "farm-card-subtitle", "Smart Precision Agriculture Decision Framework")
                )
              ),
              p(
                "Modern agricultural productivity hinges on matching crop choices with hyper-localized soil fertility levels, micro-climatic patterns, and regional farming practices. This academic research project applies rigorous data cleaning, statistical hypothesis testing, and comparative supervised machine learning to provide intelligent, data-driven crop recommendations."
              ),
              div(
                style = "background: #F8FAFC; border-radius: 12px; padding: 16px; border: 1px solid #E2E8F0; margin-top: 14px;",
                h5(style = "font-size: 14px; font-weight: 700; color: #0F172A;", "Key Research Goals:"),
                tags$ul(
                  style = "padding-left: 20px; font-size: 13px; color: #334155; margin-bottom: 0;",
                  tags$li(strong("Empirical Dataset Cleaning & Validation:"), " Handle missing values via statistical median (numerical) and mode (categorical) imputation across 762 college-curated field records."),
                  tags$li(strong("Statistical Association & Hypothesis Testing:"), " Validate soil-crop variance using One-way ANOVA, Kruskal-Wallis non-parametric tests, and Chi-Square contingency analysis."),
                  tags$li(strong("Model Benchmarking & Hyperparameter Tuning:"), " Train 7 diverse algorithms (Tree, Ensemble, Kernel, Distance, Linear) and fine-tune hyperparameters across 10-fold cross-validation."),
                  tags$li(strong("Field-Ready Deployment:"), " Operationalize the top-performing model (Random Forest, 90.76% CV Macro-F1) into an interactive recommendation system.")
                )
              )
            )
          ),
          column(
            width = 5,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("trophy"), " Selected Predictive Engine"),
                  span(class = "farm-card-subtitle", "Cross-Validation Validated Champion")
                )
              ),
              div(
                style = "background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); border: 1px solid #A7F3D0; border-radius: 12px; padding: 18px; margin-bottom: 16px;",
                div(style = "font-size: 13px; font-weight: 800; color: #065F46; text-transform: uppercase;", "Model Architecture"),
                div(style = "font-size: 24px; font-weight: 800; color: #064E3B;", "Tuned Random Forest"),
                div(style = "font-size: 13px; color: #047857; margin-top: 4px;", "Hyperparameter: mtry = 6 (Optimal Split Feature Count)")
              ),
              fluidRow(
                column(
                  width = 6,
                  div(
                    style = "background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px; text-align: center;",
                    div(style = "font-size: 11px; font-weight: 700; color: #64748B;", "CV MACRO-F1"),
                    div(style = "font-size: 22px; font-weight: 800; color: #059669;", "90.76%"),
                    div(style = "font-size: 11px; color: #64748B;", "+3.26% after tuning")
                  )
                ),
                column(
                  width = 6,
                  div(
                    style = "background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px; text-align: center;",
                    div(style = "font-size: 11px; font-weight: 700; color: #64748B;", "CV ACCURACY"),
                    div(style = "font-size: 22px; font-weight: 800; color: #059669;", "90.74%"),
                    div(style = "font-size: 11px; color: #64748B;", "10-Fold Repeated CV")
                  )
                )
              ),
              p(
                style = "font-size: 12px; color: #64748B; margin-top: 14px;",
                "Note: While SVM scored 90.13% on the single held-out test split, Random Forest demonstrated the highest generalization stability and class balance across cross-validation (90.76% Macro-F1), making it the selected predictive engine."
              )
            )
          )
        ),
        
        # Machine Learning Pipeline Visual
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("diagram-project"), " Machine Learning & Analytical Pipeline"),
              span(class = "farm-card-subtitle", "End-to-End Methodological Architecture")
            )
          ),
          div(
            class = "pipeline-container",
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 01"),
              div(class = "pipeline-step-title", "Dataset Ingestion"),
              div(class = "pipeline-step-desc", "762 records, 21 agricultural attributes from 17 states.")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 02"),
              div(class = "pipeline-step-title", "Data Cleaning"),
              div(class = "pipeline-step-desc", "Median/mode imputation, zero missing values retained.")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 03"),
              div(class = "pipeline-step-title", "Exploratory EDA"),
              div(class = "pipeline-step-desc", "Class balance, numerical skewness, outlier boxplots.")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 04"),
              div(class = "pipeline-step-title", "Statistical Testing"),
              div(class = "pipeline-step-desc", "ANOVA (F-test), Kruskal-Wallis, Chi-Square associations.")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 05"),
              div(class = "pipeline-step-title", "Train/Test Split"),
              div(class = "pipeline-step-desc", "80% stratified training (610), 20% test partition (152).")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 06"),
              div(class = "pipeline-step-title", "Model Training"),
              div(class = "pipeline-step-desc", "7 algorithms trained with 14 soil/weather predictors.")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 07"),
              div(class = "pipeline-step-title", "Hyperparameter Tuning"),
              div(class = "pipeline-step-desc", "Grid search over 6 models optimizing Macro-F1.")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 08"),
              div(class = "pipeline-step-title", "Evaluation & Ranking"),
              div(class = "pipeline-step-desc", "Multiclass confusion matrices, precision, recall & Kappa.")
            ),
            div(
              class = "pipeline-step",
              div(class = "pipeline-step-num", "Step 09"),
              div(class = "pipeline-step-title", "Crop Recommendation"),
              div(class = "pipeline-step-desc", "Real-time probabilistic prediction for farmers.")
            )
          )
        )
      ),
      
      # PAGE 2: DATASET EXPLORER
      conditionalPanel(
        condition = "input.main_nav == 'page_dataset'",
        
        fluidRow(
          column(
            width = 3,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Dataset Shape"),
                div(class = "kpi-value", "762 × 21"),
                div(class = "kpi-subtext", "Rows × Columns")
              ),
              div(class = "kpi-icon-box icon-green", icon("table"))
            )
          ),
          column(
            width = 3,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Missing Values (Cleaned)"),
                div(class = "kpi-value", "0"),
                div(class = "kpi-subtext", "100% Complete")
              ),
              div(class = "kpi-icon-box icon-blue", icon("shield-halved"))
            )
          ),
          column(
            width = 3,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Duplicate Records"),
                div(class = "kpi-value", "0"),
                div(class = "kpi-subtext", "Unique Observations")
              ),
              div(class = "kpi-icon-box icon-purple", icon("copy"))
            )
          ),
          column(
            width = 3,
            div(
              class = "kpi-card",
              div(
                div(class = "kpi-label", "Target Classes"),
                div(class = "kpi-value", "15 Crops"),
                div(class = "kpi-subtext", "50 to 55 records/crop")
              ),
              div(class = "kpi-icon-box icon-amber", icon("tags"))
            )
          )
        ),
        
        br(),
        
        # Interactive Table and Filter Card
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("filter"), " Interactive Dataset Explorer & Filtering"),
              span(class = "farm-card-subtitle", "Search, filter by attributes, and view full cleaned agricultural records")
            )
          ),
          fluidRow(
            column(
              width = 3,
              selectInput("filter_crop", "Filter by Crop:", choices = c("All Crops", sort(unique(crop_data$Crop_Name))))
            ),
            column(
              width = 3,
              selectInput("filter_state", "Filter by State:", choices = c("All States", sort(unique(crop_data$State))))
            ),
            column(
              width = 3,
              selectInput("filter_season", "Filter by Season:", choices = c("All Seasons", sort(unique(crop_data$Season))))
            ),
            column(
              width = 3,
              selectInput("filter_soil", "Filter by Soil Type:", choices = c("All Soil Types", sort(unique(crop_data$Soil_Type))))
            )
          ),
          hr(),
          DTOutput("table_dataset")
        ),
        
        fluidRow(
          column(
            width = 6,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("broom"), " Data Cleaning & Imputation Audit"),
                  span(class = "farm-card-subtitle", "Missing values before and after statistical cleaning")
                )
              ),
              DTOutput("table_cleaning_report")
            )
          ),
          column(
            width = 6,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("circle-info"), " Column Schema & Data Types"),
                  span(class = "farm-card-subtitle", "21 features overview & statistical types")
                )
              ),
              DTOutput("table_column_types")
            )
          )
        )
      ),
      
      # PAGE 3: EDA AND VISUALIZATIONS
      conditionalPanel(
        condition = "input.main_nav == 'page_eda'",
        
        # Section A: Crop Distribution
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("chart-column"), " Crop Class Record Distribution"),
              span(class = "farm-card-subtitle", "Visualizing balanced representation across all 15 crop varieties (50-55 observations each)")
            )
          ),
          plotlyOutput("plot_crop_distribution", height = "360px")
        ),
        
        # Section B & C: Numerical Distributions and Outlier Box Plots
        fluidRow(
          column(
            width = 6,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("chart-area"), " Numerical Feature Distributions"),
                  span(class = "farm-card-subtitle", "Histogram with distribution shape")
                ),
                selectInput("eda_num_feature", NULL, choices = c("N", "P", "K", "pH", "Rainfall", "Humidity", "Temperature", "Crop_Duration"), width = "160px")
              ),
              plotlyOutput("plot_num_distribution", height = "340px")
            )
          ),
          column(
            width = 6,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("box-archive"), " Outlier Analysis (Box Plots)"),
                  span(class = "farm-card-subtitle", "Interquartile ranges and detected outliers")
                ),
                selectInput("eda_box_feature", NULL, choices = c("N", "P", "K", "pH", "Rainfall", "Humidity", "Temperature", "Crop_Duration"), selected = "Rainfall", width = "160px")
              ),
              plotlyOutput("plot_outliers", height = "340px")
            )
          )
        ),
        
        # Section D: Pearson Correlation Heatmap
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("braille"), " Pearson Correlation Heatmap (Numerical Features)"),
              span(class = "farm-card-subtitle", "Pairwise linear relationship between soil nutrients, climate variables, and crop duration")
            )
          ),
          plotlyOutput("plot_correlation_heatmap", height = "420px")
        ),
        
        # Section E: Categorical Feature Analysis
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("chart-pie"), " Categorical Feature Frequency Breakdown"),
              span(class = "farm-card-subtitle", "Examine distribution across environmental, management, and agronomic categories")
            ),
            selectInput("eda_cat_feature", NULL,
                        choices = c("Soil_Type", "Season", "Irrigation_Type", "Water_Source", "Land_Type", "Previous_Crop"),
                        width = "200px")
          ),
          plotlyOutput("plot_cat_distribution", height = "360px")
        )
      ),
      
      # PAGE 4: STATISTICAL ANALYSIS
      conditionalPanel(
        condition = "input.main_nav == 'page_statistics'",
        
        # Educational Methodology Intro
        div(
          style = "background: #F8FAFC; border: 1px solid var(--border-color); border-radius: var(--card-radius); padding: 20px; margin-bottom: 24px;",
          h4(style = "font-size: 16px; font-weight: 800; color: #0F172A; margin-bottom: 8px;", icon("book-open"), " Rigorous Statistical Testing Framework"),
          p(
            style = "font-size: 13px; color: #475569; margin: 0;",
            "Before machine learning modeling, formal hypothesis testing was conducted to determine whether crop types exhibit statistically significant differences across numerical parameters and whether categorical farming attributes show meaningful contingency associations. All p-values are adjusted for false discovery rate (FDR) using the Benjamini-Hochberg (BH) procedure."
          )
        ),
        
        # 1. Descriptive Statistics
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("chart-simple"), " 1. Descriptive Statistics"),
              span(class = "farm-card-subtitle", "Mean, median, standard deviation, variance, minimum and maximum for 8 continuous variables")
            )
          ),
          DTOutput("table_descriptive_stats")
        ),
        
        # 2 & 3. ANOVA and Kruskal-Wallis Tests
        fluidRow(
          column(
            width = 6,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("scale-balanced"), " 2. One-way ANOVA Tests"),
                  span(class = "farm-card-subtitle", "Testing mean difference across 15 crop varieties")
                )
              ),
              p(
                style = "font-size: 12px; color: #64748B;",
                strong("Null Hypothesis (H₀):"), " The mean of the numerical feature is identical across all 15 crop classes. ",
                strong("Effect size:"), " Eta-squared (η²)."
              ),
              DTOutput("table_anova"),
              br(),
              plotlyOutput("plot_anova_eta", height = "260px")
            )
          ),
          column(
            width = 6,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("chart-line"), " 3. Kruskal-Wallis Tests"),
                  span(class = "farm-card-subtitle", "Non-parametric rank sum test for non-normal features")
                )
              ),
              p(
                style = "font-size: 12px; color: #64748B;",
                strong("Null Hypothesis (H₀):"), " The median rank distributions across crop classes are equal. ",
                strong("Effect size:"), " Epsilon-squared (ε²)."
              ),
              DTOutput("table_kruskal"),
              br(),
              plotlyOutput("plot_kruskal_epsilon", height = "260px")
            )
          )
        ),
        
        # 4. Chi-Square Test
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("project-diagram"), " 4. Chi-Square Test of Independence (Categorical Predictors)"),
              span(class = "farm-card-subtitle", "Evaluating categorical association with Crop_Name using Cramér's V effect size")
            )
          ),
          p(
            style = "font-size: 12px; color: #64748B;",
            strong("Null Hypothesis (H₀):"), " The categorical agricultural feature (e.g., Soil Type, Season) and Crop Name are completely independent. ",
            strong("Association Strength:"), " Measured via Cramér's V (0 = no association, 1 = perfect association)."
          ),
          fluidRow(
            column(width = 7, DTOutput("table_chi_square")),
            column(width = 5, plotlyOutput("plot_cramers_v", height = "300px"))
          )
        )
      ),
      
      # PAGE 5: MACHINE LEARNING MODEL COMPARISON
      conditionalPanel(
        condition = "input.main_nav == 'page_models'",
        
        # Selected Model Banner
        div(
          style = "background: linear-gradient(135deg, #065F46 0%, #047857 100%); border-radius: var(--card-radius); padding: 22px 28px; color: white; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(5, 150, 105, 0.2);",
          fluidRow(
            column(
              width = 8,
              h3(style = "font-size: 22px; font-weight: 800; margin: 0 0 6px 0;", icon("trophy"), " Selected Predictive Champion: Random Forest"),
              p(style = "font-size: 13px; color: #D1FAE5; margin: 0;", "Selected based on superior 10-fold Cross-Validation Macro-F1 (90.76%) and robust multiclass generalization. 7 models successfully implemented; 6 hyperparameter tuned.")
            ),
            column(
              width = 4,
              div(
                style = "display: flex; gap: 16px; justify-content: flex-end; align-items: center;",
                div(
                  style = "text-align: right;",
                  div(style = "font-size: 11px; text-transform: uppercase; color: #A7F3D0; font-weight: 700;", "CV Macro-F1"),
                  div(style = "font-size: 26px; font-weight: 800;", "90.76%")
                ),
                div(
                  style = "text-align: right;",
                  div(style = "font-size: 11px; text-transform: uppercase; color: #A7F3D0; font-weight: 700;", "CV Accuracy"),
                  div(style = "font-size: 26px; font-weight: 800;", "90.74%")
                )
              )
            )
          )
        ),
        
        # Initial 7 Models Comparison Table & Chart
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("table-columns"), " Test-Set Performance: All 7 Implemented Algorithms"),
              span(class = "farm-card-subtitle", "Evaluated on 152 held-out independent test observations (80/20 split)")
            )
          ),
          DTOutput("table_model_comparison"),
          br(),
          plotlyOutput("plot_model_comparison_bar", height = "360px")
        ),
        
        # Before vs After Hyperparameter Tuning
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("sliders"), " Before vs. After Hyperparameter Tuning"),
              span(class = "farm-card-subtitle", "Quantifying Macro-F1 optimization for the 6 tuned algorithms")
            )
          ),
          fluidRow(
            column(width = 7, plotlyOutput("plot_before_after_tuning", height = "380px")),
            column(
              width = 5,
              h5(style = "font-size: 14px; font-weight: 700; color: #0F172A;", "Optimal Tuning Hyperparameters"),
              DTOutput("table_tuning_params"),
              div(
                class = "disclaimer-box",
                icon("info-circle"), " Decision Tree gained +40.71 percentage points in Macro-F1 (from 33.90% to 74.61%) upon tuning complexity parameter cp = 0.005, preventing early over-pruning."
              )
            )
          )
        ),
        
        # Confusion Matrix & Per-Crop Class Metrics
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("border-all"), " Multi-Class Confusion Matrix & Per-Crop Diagnostics"),
              span(class = "farm-card-subtitle", "Inspect classification fidelity, precision, recall, and class-level confusion")
            ),
            selectInput("cm_model_select", "Select Model to Inspect:",
                        choices = names(model_mapping),
                        selected = "Random Forest",
                        width = "260px")
          ),
          fluidRow(
            column(
              width = 7,
              h5(style = "font-size: 14px; font-weight: 700; color: #0F172A; margin-bottom: 12px;", "Confusion Matrix Heatmap (Actual vs. Predicted)"),
              plotlyOutput("plot_confusion_matrix", height = "420px")
            ),
            column(
              width = 5,
              h5(style = "font-size: 14px; font-weight: 700; color: #0F172A; margin-bottom: 12px;", "Per-Crop Precision, Recall & F1-Score"),
              DTOutput("table_class_metrics")
            )
          )
        )
      ),
      
      # PAGE 6: CROP RECOMMENDATION
      conditionalPanel(
        condition = "input.main_nav == 'page_recommend'",
        
        fluidRow(
          # Left Column: Inputs Form
          column(
            width = 6,
            div(
              class = "farm-card",
              div(
                class = "farm-card-header",
                div(
                  h4(class = "farm-card-title", icon("wheat-awn"), " Agricultural Conditions Input"),
                  span(class = "farm-card-subtitle", "Provide field soil metrics, weather, and environmental conditions")
                )
              ),
              
              # Quick Sample Preset Buttons
              div(
                style = "margin-bottom: 16px; padding: 12px; background: #F8FAFC; border-radius: 12px; border: 1px solid #E2E8F0;",
                div(style = "font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;", "Quick-Load Agronomic Presets:"),
                actionButton("btn_preset_rice", "🌾 Assam Kharif Rice", class = "btn-preset"),
                actionButton("btn_preset_cotton", "☁️ Gujarat Cotton", class = "btn-preset"),
                actionButton("btn_preset_chickpea", "🧆 MP Chickpea", class = "btn-preset"),
                actionButton("btn_preset_banana", "🍌 Kerala Banana", class = "btn-preset"),
                actionButton("btn_preset_maize", "🌽 Karnataka Maize", class = "btn-preset")
              ),
              
              # Section 1: Soil Nutrients
              div(class = "form-section-title", icon("flask"), " 1. Soil Chemistry & Nutrients"),
              fluidRow(
                column(
                  width = 6,
                  numericInput("in_N", "Nitrogen (N) [kg/ha]:", value = num_ranges$N$default, min = num_ranges$N$min, max = num_ranges$N$max, step = 1),
                  numericInput("in_K", "Potassium (K) [kg/ha]:", value = num_ranges$K$default, min = num_ranges$K$min, max = num_ranges$K$max, step = 1)
                ),
                column(
                  width = 6,
                  numericInput("in_P", "Phosphorus (P) [kg/ha]:", value = num_ranges$P$default, min = num_ranges$P$min, max = num_ranges$P$max, step = 1),
                  numericInput("in_pH", "Soil pH (5.0 - 8.5):", value = num_ranges$pH$default, min = num_ranges$pH$min, max = num_ranges$pH$max, step = 0.1)
                )
              ),
              
              # Section 2: Weather Conditions
              div(class = "form-section-title", icon("cloud-sun-rain"), " 2. Weather & Climatic Factors"),
              fluidRow(
                column(
                  width = 4,
                  numericInput("in_Rainfall", "Rainfall (mm):", value = num_ranges$Rainfall$default, min = num_ranges$Rainfall$min, max = num_ranges$Rainfall$max, step = 10)
                ),
                column(
                  width = 4,
                  numericInput("in_Humidity", "Humidity (%):", value = num_ranges$Humidity$default, min = num_ranges$Humidity$min, max = num_ranges$Humidity$max, step = 1)
                ),
                column(
                  width = 4,
                  numericInput("in_Temperature", "Temperature (°C):", value = num_ranges$Temperature$default, min = num_ranges$Temperature$min, max = num_ranges$Temperature$max, step = 0.5)
                )
              ),
              
              # Section 3: Agricultural & Environmental
              div(class = "form-section-title", icon("tractor"), " 3. Farming Environment & Location"),
              fluidRow(
                column(
                  width = 6,
                  selectInput("in_Soil_Type", "Soil Type:", choices = categorical_options$Soil_Type, selected = "Alluvial"),
                  selectInput("in_State", "State:", choices = categorical_options$State, selected = "Assam"),
                  selectInput("in_Season", "Season:", choices = categorical_options$Season, selected = "Kharif"),
                  selectInput("in_Irrigation_Type", "Irrigation Type:", choices = categorical_options$Irrigation_Type, selected = "Rainfed")
                ),
                column(
                  width = 6,
                  selectInput("in_Water_Source", "Water Source:", choices = categorical_options$Water_Source, selected = "Rainwater"),
                  selectInput("in_Land_Type", "Land Type:", choices = categorical_options$Land_Type, selected = "Plain"),
                  selectInput("in_Previous_Crop", "Previous Crop:", choices = categorical_options$Previous_Crop, selected = "Fallow")
                )
              ),
              
              br(),
              actionButton("btn_predict", "🌱 Recommend Optimal Crop", class = "btn-recommend"),
              br()
            )
          ),
          
          # Right Column: Prediction Results Dashboard
          column(
            width = 6,
            uiOutput("ui_prediction_result")
          )
        )
      )
    )
  )
)

# ------------------------------------------------------------------------------
# SERVER LOGIC
# ------------------------------------------------------------------------------
server <- function(input, output, session) {
  
  # Reactive Page Headers
  output$page_header_title <- renderText({
    switch(input$main_nav,
           "page_overview" = "Project Overview & Executive Summary",
           "page_dataset" = "Agricultural Dataset Explorer",
           "page_eda" = "Exploratory Data Analysis & Visualizations",
           "page_statistics" = "Statistical Hypothesis Testing & ANOVA",
           "page_models" = "Machine Learning Benchmark & Hyperparameter Tuning",
           "page_recommend" = "AI Crop Recommendation Engine")
  })
  
  output$page_header_subtitle <- renderText({
    switch(input$main_nav,
           "page_overview" = "762 Observations • 15 Crop Varieties • 7 ML Classifiers",
           "page_dataset" = "Inspect, filter, and analyze cleaned crop records across 17 Indian states",
           "page_eda" = "Deep visual insights into crop distributions, soil nutrients, climate, and categorical patterns",
           "page_statistics" = "Parametric ANOVA, non-parametric Kruskal-Wallis, and Chi-Square independence tests",
           "page_models" = "Rigorous comparison of 7 algorithms before and after hyperparameter optimization",
           "page_recommend" = "Live predictive inference using trained Random Forest ensemble (90.76% CV Macro-F1)")
  })
  
  # ----------------------------------------------------------------------------
  # PAGE 2: DATASET EXPLORER OUTPUTS
  # ----------------------------------------------------------------------------
  filtered_dataset <- reactive({
    df <- crop_data
    if (input$filter_crop != "All Crops") {
      df <- df[df$Crop_Name == input$filter_crop, ]
    }
    if (input$filter_state != "All States") {
      df <- df[df$State == input$filter_state, ]
    }
    if (input$filter_season != "All Seasons") {
      df <- df[df$Season == input$filter_season, ]
    }
    if (input$filter_soil != "All Soil Types") {
      df <- df[df$Soil_Type == input$filter_soil, ]
    }
    df
  })
  
  output$table_dataset <- renderDT({
    datatable(
      filtered_dataset(),
      options = list(
        pageLength = 10,
        scrollX = TRUE,
        autoWidth = TRUE,
        dom = "Bfrtip"
      ),
      rownames = FALSE,
      class = "compact stripe hover"
    )
  })
  
  output$table_cleaning_report <- renderDT({
    req(results_data$cleaning_report)
    datatable(
      results_data$cleaning_report,
      options = list(pageLength = 7, dom = "tp", scrollX = TRUE),
      rownames = FALSE,
      class = "compact stripe hover"
    )
  })
  
  output$table_column_types <- renderDT({
    col_summary <- data.frame(
      Column = names(crop_data),
      Data_Type = sapply(crop_data, class),
      Sample_Value = sapply(crop_data, function(x) as.character(x[1])),
      Missing_Count = sapply(crop_data, function(x) sum(is.na(x))),
      stringsAsFactors = FALSE
    )
    datatable(
      col_summary,
      options = list(pageLength = 7, dom = "tp", scrollX = TRUE),
      rownames = FALSE,
      class = "compact stripe hover"
    )
  })
  
  # ----------------------------------------------------------------------------
  # PAGE 3: EDA & VISUALIZATIONS OUTPUTS
  # ----------------------------------------------------------------------------
  output$plot_crop_distribution <- renderPlotly({
    req(results_data$crop_dist)
    cd <- results_data$crop_dist
    cd <- cd[order(-cd$Total_Records), ]
    
    plot_ly(
      data = cd,
      x = ~reorder(Crop_Name, -Total_Records),
      y = ~Total_Records,
      type = "bar",
      text = ~paste0(Crop_Name, ": ", Total_Records, " (", Percentage, "%)"),
      hoverinfo = "text",
      marker = list(
        color = "#10B981",
        line = list(color = "#059669", width = 1.5)
      )
    ) %>%
      layout(
        xaxis = list(title = "Crop Name", tickangle = -35),
        yaxis = list(title = "Number of Field Records"),
        margin = list(b = 80, t = 20),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)"
      )
  })
  
  output$plot_num_distribution <- renderPlotly({
    feat <- input$eda_num_feature
    v <- crop_data[[feat]]
    
    plot_ly(
      x = v,
      type = "histogram",
      nbinsx = 30,
      marker = list(
        color = "#3B82F6",
        line = list(color = "#2563EB", width = 1)
      ),
      name = feat
    ) %>%
      layout(
        xaxis = list(title = feat),
        yaxis = list(title = "Frequency Count"),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(t = 20)
      )
  })
  
  output$plot_outliers <- renderPlotly({
    feat <- input$eda_box_feature
    
    plot_ly(
      data = crop_data,
      x = ~Crop_Name,
      y = as.formula(paste0("~", feat)),
      type = "box",
      color = I("#8B5CF6")
    ) %>%
      layout(
        xaxis = list(title = "Crop Name", tickangle = -35),
        yaxis = list(title = feat),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(b = 80, t = 20)
      )
  })
  
  output$plot_correlation_heatmap <- renderPlotly({
    req(results_data$correlation)
    corr_mat <- as.matrix(results_data$correlation)
    
    plot_ly(
      x = colnames(corr_mat),
      y = rownames(corr_mat),
      z = corr_mat,
      type = "heatmap",
      colorscale = list(c(0, "#EFF6FF"), c(0.5, "#F8FAFC"), c(1, "#10B981")),
      zmin = -1,
      zmax = 1
    ) %>%
      layout(
        xaxis = list(tickangle = -30),
        yaxis = list(autorange = "reversed"),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(b = 80, l = 100)
      )
  })
  
  output$plot_cat_distribution <- renderPlotly({
    cat_col <- input$eda_cat_feature
    tbl <- as.data.frame(table(crop_data[[cat_col]]))
    colnames(tbl) <- c("Category", "Count")
    tbl <- tbl[order(-tbl$Count), ]
    
    plot_ly(
      data = tbl,
      labels = ~Category,
      values = ~Count,
      type = "pie",
      hole = 0.55,
      marker = list(
        colors = c("#10B981", "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16", "#64748B")
      ),
      textinfo = "label+percent"
    ) %>%
      layout(
        showlegend = TRUE,
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(t = 20, b = 20)
      )
  })
  
  # ----------------------------------------------------------------------------
  # PAGE 4: STATISTICAL ANALYSIS OUTPUTS
  # ----------------------------------------------------------------------------
  output$table_descriptive_stats <- renderDT({
    req(results_data$descriptive_stats)
    df <- results_data$descriptive_stats
    df[, -1] <- round(df[, -1], 2)
    datatable(df, options = list(dom = "t", scrollX = TRUE), rownames = FALSE, class = "compact stripe hover")
  })
  
  output$table_anova <- renderDT({
    req(results_data$anova)
    df <- results_data$anova
    df$F_Value <- round(df$F_Value, 2)
    df$Eta_Squared <- round(df$Eta_Squared, 4)
    df$P_Value <- formatC(df$P_Value, format = "e", digits = 2)
    df$Adjusted_P_Value <- formatC(df$Adjusted_P_Value, format = "e", digits = 2)
    datatable(df, options = list(dom = "t", scrollX = TRUE), rownames = FALSE, class = "compact stripe hover")
  })
  
  output$plot_anova_eta <- renderPlotly({
    req(results_data$anova)
    df <- results_data$anova
    df <- df[order(df$Eta_Squared), ]
    
    plot_ly(
      data = df,
      x = ~Eta_Squared,
      y = ~reorder(Feature, Eta_Squared),
      type = "bar",
      orientation = "h",
      marker = list(color = "#10B981")
    ) %>%
      layout(
        title = list(text = "One-way ANOVA Effect Size (Eta-Squared η²)", font = list(size = 12)),
        xaxis = list(title = "Eta-Squared (η²)"),
        yaxis = list(title = ""),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(l = 100, t = 40)
      )
  })
  
  output$table_kruskal <- renderDT({
    req(results_data$kruskal)
    df <- results_data$kruskal
    df$H_Value <- round(df$H_Value, 2)
    df$Epsilon_Squared <- round(df$Epsilon_Squared, 4)
    df$P_Value <- formatC(df$P_Value, format = "e", digits = 2)
    df$Adjusted_P_Value <- formatC(df$Adjusted_P_Value, format = "e", digits = 2)
    datatable(df, options = list(dom = "t", scrollX = TRUE), rownames = FALSE, class = "compact stripe hover")
  })
  
  output$plot_kruskal_epsilon <- renderPlotly({
    req(results_data$kruskal)
    df <- results_data$kruskal
    df <- df[order(df$Epsilon_Squared), ]
    
    plot_ly(
      data = df,
      x = ~Epsilon_Squared,
      y = ~reorder(Feature, Epsilon_Squared),
      type = "bar",
      orientation = "h",
      marker = list(color = "#3B82F6")
    ) %>%
      layout(
        title = list(text = "Kruskal-Wallis Effect Size (Epsilon-Squared ε²)", font = list(size = 12)),
        xaxis = list(title = "Epsilon-Squared (ε²)"),
        yaxis = list(title = ""),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(l = 100, t = 40)
      )
  })
  
  output$table_chi_square <- renderDT({
    req(results_data$chi_square)
    df <- results_data$chi_square
    df$Chi_Square <- round(df$Chi_Square, 2)
    df$Cramers_V <- round(df$Cramers_V, 4)
    df$P_Value <- formatC(df$P_Value, format = "e", digits = 2)
    df$Adjusted_P_Value <- formatC(df$Adjusted_P_Value, format = "e", digits = 2)
    datatable(df, options = list(dom = "t", scrollX = TRUE), rownames = FALSE, class = "compact stripe hover")
  })
  
  output$plot_cramers_v <- renderPlotly({
    req(results_data$chi_square)
    df <- results_data$chi_square
    df <- df[order(df$Cramers_V), ]
    
    plot_ly(
      data = df,
      x = ~Cramers_V,
      y = ~reorder(Feature, Cramers_V),
      type = "bar",
      orientation = "h",
      marker = list(color = "#8B5CF6")
    ) %>%
      layout(
        title = list(text = "Categorical Association Strength (Cramér's V)", font = list(size = 12)),
        xaxis = list(title = "Cramér's V"),
        yaxis = list(title = ""),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(l = 100, t = 40)
      )
  })
  
  # ----------------------------------------------------------------------------
  # PAGE 5: MACHINE LEARNING MODEL COMPARISON OUTPUTS
  # ----------------------------------------------------------------------------
  output$table_model_comparison <- renderDT({
    req(results_data$model_comparison)
    df <- results_data$model_comparison
    # Format percentages
    metric_cols <- c("Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Weighted_F1", "Kappa")
    for (col in metric_cols) {
      if (col %in% names(df)) df[[col]] <- paste0(round(df[[col]] * 100, 2), "%")
    }
    datatable(df, options = list(dom = "t", scrollX = TRUE), rownames = FALSE, class = "compact stripe hover")
  })
  
  output$plot_model_comparison_bar <- renderPlotly({
    req(results_data$model_comparison)
    df <- results_data$model_comparison
    df <- df[order(-df$Macro_F1), ]
    
    plot_ly(data = df) %>%
      add_trace(x = ~Model, y = ~Macro_F1 * 100, name = "Macro-F1 (%)", type = "bar", marker = list(color = "#10B981")) %>%
      add_trace(x = ~Model, y = ~Accuracy * 100, name = "Accuracy (%)", type = "bar", marker = list(color = "#3B82F6")) %>%
      add_trace(x = ~Model, y = ~Kappa * 100, name = "Kappa (%)", type = "bar", marker = list(color = "#F59E0B")) %>%
      layout(
        barmode = "group",
        yaxis = list(title = "Score (%)", range = c(0, 100)),
        xaxis = list(title = "", tickangle = -15),
        legend = list(orientation = "h", y = 1.15),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(b = 60)
      )
  })
  
  output$plot_before_after_tuning <- renderPlotly({
    req(results_data$macro_f1_improvement)
    df <- results_data$macro_f1_improvement
    df <- df[order(-df$Improvement), ]
    
    plot_ly(data = df) %>%
      add_trace(x = ~Model, y = ~Before_Macro_F1 * 100, name = "Before Tuning", type = "bar", marker = list(color = "#CBD5E1")) %>%
      add_trace(x = ~Model, y = ~After_Macro_F1 * 100, name = "After Tuning", type = "bar", marker = list(color = "#10B981")) %>%
      layout(
        barmode = "group",
        title = list(text = "Macro-F1 Score Improvement Post-Tuning (%)", font = list(size = 14)),
        yaxis = list(title = "Macro-F1 (%)", range = c(0, 100)),
        xaxis = list(title = "", tickangle = -15),
        legend = list(orientation = "h", y = 1.15),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(b = 60)
      )
  })
  
  output$table_tuning_params <- renderDT({
    datatable(tuning_params_info, options = list(dom = "t", scrollX = TRUE), rownames = FALSE, class = "compact stripe hover")
  })
  
  # Reactive Confusion Matrix
  selected_cm_data <- reactive({
    m_name <- input$cm_model_select
    prefix <- model_mapping[[m_name]]
    if (is.null(prefix)) prefix <- "random_forest"
    
    cm_path <- resolve_path(paste0("results/model_evaluation/", prefix, "_confusion_matrix.csv"))
    if (file.exists(cm_path)) {
      cm_df <- read.csv(cm_path, stringsAsFactors = FALSE)
      if ("X" %in% names(cm_df)) {
        rownames(cm_df) <- cm_df$X
        cm_df$X <- NULL
      }
      return(as.matrix(cm_df))
    }
    NULL
  })
  
  output$plot_confusion_matrix <- renderPlotly({
    cm_mat <- selected_cm_data()
    req(cm_mat)
    
    plot_ly(
      x = colnames(cm_mat),
      y = rownames(cm_mat),
      z = cm_mat,
      type = "heatmap",
      colorscale = list(c(0, "#F8FAFC"), c(0.1, "#D1FAE5"), c(1, "#047857")),
      hoverinfo = "text",
      text = ~outer(rownames(cm_mat), colnames(cm_mat), FUN = function(r, c) {
        paste0("Actual: ", r, "<br>Predicted: ", c, "<br>Count: ", cm_mat[r, c])
      })
    ) %>%
      layout(
        xaxis = list(title = "Predicted Crop", tickangle = -35),
        yaxis = list(title = "Actual Reference Crop", autorange = "reversed"),
        plot_bgcolor = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin = list(b = 90, l = 90)
      )
  })
  
  output$table_class_metrics <- renderDT({
    m_name <- input$cm_model_select
    prefix <- model_mapping[[m_name]]
    if (is.null(prefix)) prefix <- "random_forest"
    
    metrics_path <- resolve_path(paste0("results/model_evaluation/", prefix, "_class_metrics.csv"))
    if (file.exists(metrics_path)) {
      df <- read.csv(metrics_path, stringsAsFactors = FALSE)
      df$Precision <- paste0(round(df$Precision * 100, 1), "%")
      df$Recall <- paste0(round(df$Recall * 100, 1), "%")
      df$F1_Score <- paste0(round(df$F1_Score * 100, 1), "%")
      return(datatable(df, options = list(pageLength = 8, dom = "tp", scrollX = TRUE), rownames = FALSE, class = "compact stripe hover"))
    }
    datatable(data.frame(Status = "Metrics file not found"), rownames = FALSE)
  })
  
  # ----------------------------------------------------------------------------
  # PAGE 6: CROP RECOMMENDATION LOGIC
  # ----------------------------------------------------------------------------
  # Presets Observers
  observeEvent(input$btn_preset_rice, {
    updateNumericInput(session, "in_N", value = 82)
    updateNumericInput(session, "in_P", value = 33)
    updateNumericInput(session, "in_K", value = 44)
    updateNumericInput(session, "in_pH", value = 5.8)
    updateNumericInput(session, "in_Rainfall", value = 1250)
    updateNumericInput(session, "in_Humidity", value = 87)
    updateNumericInput(session, "in_Temperature", value = 27.5)
    updateSelectInput(session, "in_Soil_Type", selected = "Loamy")
    updateSelectInput(session, "in_State", selected = "Assam")
    updateSelectInput(session, "in_Season", selected = "Kharif")
    updateSelectInput(session, "in_Irrigation_Type", selected = "Rainfed")
    updateSelectInput(session, "in_Water_Source", selected = "Rainwater")
    updateSelectInput(session, "in_Land_Type", selected = "Plain")
    updateSelectInput(session, "in_Previous_Crop", selected = "Fallow")
  })
  
  observeEvent(input$btn_preset_cotton, {
    updateNumericInput(session, "in_N", value = 120)
    updateNumericInput(session, "in_P", value = 60)
    updateNumericInput(session, "in_K", value = 55)
    updateNumericInput(session, "in_pH", value = 7.5)
    updateNumericInput(session, "in_Rainfall", value = 650)
    updateNumericInput(session, "in_Humidity", value = 55)
    updateNumericInput(session, "in_Temperature", value = 29.0)
    updateSelectInput(session, "in_Soil_Type", selected = "Black")
    updateSelectInput(session, "in_State", selected = "Gujarat")
    updateSelectInput(session, "in_Season", selected = "Kharif")
    updateSelectInput(session, "in_Irrigation_Type", selected = "Drip")
    updateSelectInput(session, "in_Water_Source", selected = "Groundwater")
    updateSelectInput(session, "in_Land_Type", selected = "Plain")
    updateSelectInput(session, "in_Previous_Crop", selected = "Wheat")
  })
  
  observeEvent(input$btn_preset_chickpea, {
    updateNumericInput(session, "in_N", value = 40)
    updateNumericInput(session, "in_P", value = 65)
    updateNumericInput(session, "in_K", value = 78)
    updateNumericInput(session, "in_pH", value = 7.2)
    updateNumericInput(session, "in_Rainfall", value = 450)
    updateNumericInput(session, "in_Humidity", value = 48)
    updateNumericInput(session, "in_Temperature", value = 18.5)
    updateSelectInput(session, "in_Soil_Type", selected = "Clay")
    updateSelectInput(session, "in_State", selected = "Madhya Pradesh")
    updateSelectInput(session, "in_Season", selected = "Rabi")
    updateSelectInput(session, "in_Irrigation_Type", selected = "Sprinkler")
    updateSelectInput(session, "in_Water_Source", selected = "Canal")
    updateSelectInput(session, "in_Land_Type", selected = "Plain")
    updateSelectInput(session, "in_Previous_Crop", selected = "Soybean")
  })
  
  observeEvent(input$btn_preset_banana, {
    updateNumericInput(session, "in_N", value = 105)
    updateNumericInput(session, "in_P", value = 75)
    updateNumericInput(session, "in_K", value = 150)
    updateNumericInput(session, "in_pH", value = 6.2)
    updateNumericInput(session, "in_Rainfall", value = 1800)
    updateNumericInput(session, "in_Humidity", value = 82)
    updateNumericInput(session, "in_Temperature", value = 27.0)
    updateSelectInput(session, "in_Soil_Type", selected = "Laterite")
    updateSelectInput(session, "in_State", selected = "Kerala")
    updateSelectInput(session, "in_Season", selected = "Perennial")
    updateSelectInput(session, "in_Irrigation_Type", selected = "Drip")
    updateSelectInput(session, "in_Water_Source", selected = "River")
    updateSelectInput(session, "in_Land_Type", selected = "Lowland")
    updateSelectInput(session, "in_Previous_Crop", selected = "Vegetables")
  })
  
  observeEvent(input$btn_preset_maize, {
    updateNumericInput(session, "in_N", value = 90)
    updateNumericInput(session, "in_P", value = 45)
    updateNumericInput(session, "in_K", value = 40)
    updateNumericInput(session, "in_pH", value = 6.6)
    updateNumericInput(session, "in_Rainfall", value = 750)
    updateNumericInput(session, "in_Humidity", value = 64)
    updateNumericInput(session, "in_Temperature", value = 24.5)
    updateSelectInput(session, "in_Soil_Type", selected = "Red")
    updateSelectInput(session, "in_State", selected = "Karnataka")
    updateSelectInput(session, "in_Season", selected = "Kharif")
    updateSelectInput(session, "in_Irrigation_Type", selected = "Borewell")
    updateSelectInput(session, "in_Water_Source", selected = "Groundwater")
    updateSelectInput(session, "in_Land_Type", selected = "Upland")
    updateSelectInput(session, "in_Previous_Crop", selected = "Pulses")
  })
  
  # Recommendation Execution Reactive
  prediction_result <- eventReactive(input$btn_predict, {
    # 1. Validate numerical inputs
    errors <- c()
    if (is.na(input$in_N) || input$in_N < num_ranges$N$min || input$in_N > num_ranges$N$max) {
      errors <- c(errors, paste0("Nitrogen (N) must be between ", num_ranges$N$min, " and ", num_ranges$N$max, " kg/ha."))
    }
    if (is.na(input$in_P) || input$in_P < num_ranges$P$min || input$in_P > num_ranges$P$max) {
      errors <- c(errors, paste0("Phosphorus (P) must be between ", num_ranges$P$min, " and ", num_ranges$P$max, " kg/ha."))
    }
    if (is.na(input$in_K) || input$in_K < num_ranges$K$min || input$in_K > num_ranges$K$max) {
      errors <- c(errors, paste0("Potassium (K) must be between ", num_ranges$K$min, " and ", num_ranges$K$max, " kg/ha."))
    }
    if (is.na(input$in_pH) || input$in_pH < num_ranges$pH$min || input$in_pH > num_ranges$pH$max) {
      errors <- c(errors, paste0("Soil pH must be between ", num_ranges$pH$min, " and ", num_ranges$pH$max, "."))
    }
    if (is.na(input$in_Rainfall) || input$in_Rainfall < num_ranges$Rainfall$min || input$in_Rainfall > num_ranges$Rainfall$max) {
      errors <- c(errors, paste0("Rainfall must be between ", num_ranges$Rainfall$min, " and ", num_ranges$Rainfall$max, " mm."))
    }
    if (is.na(input$in_Humidity) || input$in_Humidity < num_ranges$Humidity$min || input$in_Humidity > num_ranges$Humidity$max) {
      errors <- c(errors, paste0("Humidity must be between ", num_ranges$Humidity$min, " and ", num_ranges$Humidity$max, " %."))
    }
    if (is.na(input$in_Temperature) || input$in_Temperature < num_ranges$Temperature$min || input$in_Temperature > num_ranges$Temperature$max) {
      errors <- c(errors, paste0("Temperature must be between ", num_ranges$Temperature$min, " and ", num_ranges$Temperature$max, " °C."))
    }
    
    if (length(errors) > 0) {
      return(list(success = FALSE, message = paste(errors, collapse = "<br>")))
    }
    
    # 2. Construct input data frame exactly in required order and format
    new_data <- data.frame(
      N = as.numeric(input$in_N),
      P = as.numeric(input$in_P),
      K = as.numeric(input$in_K),
      pH = as.numeric(input$in_pH),
      Rainfall = as.numeric(input$in_Rainfall),
      Humidity = as.numeric(input$in_Humidity),
      Temperature = as.numeric(input$in_Temperature),
      Soil_Type = as.character(input$in_Soil_Type),
      State = as.character(input$in_State),
      Season = as.character(input$in_Season),
      Irrigation_Type = as.character(input$in_Irrigation_Type),
      Water_Source = as.character(input$in_Water_Source),
      Land_Type = as.character(input$in_Land_Type),
      Previous_Crop = as.character(input$in_Previous_Crop),
      stringsAsFactors = FALSE
    )
    
    # 3. Model Inference
    if (is.null(rf_model)) {
      return(list(success = FALSE, message = "Saved Random Forest model (models/final_crop_model.rds) could not be loaded."))
    }
    
    pred_crop <- tryCatch({
      as.character(predict(rf_model, newdata = new_data))
    }, error = function(e) {
      return(list(success = FALSE, message = paste0("Prediction error: ", conditionMessage(e))))
    })
    
    # Probabilities
    probs_df <- tryCatch({
      p <- predict(rf_model, newdata = new_data, type = "prob")
      p_vec <- unlist(p[1, ])
      res <- data.frame(Crop = names(p_vec), Probability = as.numeric(p_vec), stringsAsFactors = FALSE)
      res[order(-res$Probability), ]
    }, error = function(e) NULL)
    
    list(
      success = TRUE,
      crop = pred_crop,
      probabilities = probs_df,
      inputs = new_data
    )
  }, ignoreNULL = FALSE)
  
  # UI Output for Prediction Result
  output$ui_prediction_result <- renderUI({
    res <- prediction_result()
    
    if (is.null(res)) {
      return(
        div(
          class = "farm-card",
          style = "text-align: center; padding: 60px 20px;",
          div(style = "font-size: 48px; color: #CBD5E1; margin-bottom: 12px;", icon("seedling")),
          h4(style = "font-weight: 700; color: #475569;", "Ready for Recommendation"),
          p(style = "font-size: 13px; color: #94A3B8;", "Adjust the field parameters on the left and click 'Recommend Optimal Crop' or load a quick preset.")
        )
      )
    }
    
    if (!res$success) {
      return(
        div(
          class = "farm-card",
          style = "border-color: #FECACA; background: #FEF2F2;",
          div(style = "font-size: 28px; color: #EF4444; margin-bottom: 8px;", icon("triangle-exclamation")),
          h4(style = "font-weight: 700; color: #991B1B;", "Input Validation Alert"),
          div(style = "font-size: 13px; color: #B91C1C; margin-top: 6px;", HTML(res$message))
        )
      )
    }
    
    crop_name <- res$crop
    crop_icon <- if (!is.null(crop_emojis[[crop_name]])) crop_emojis[[crop_name]] else "🌱"
    top_prob <- if (!is.null(res$probabilities)) round(res$probabilities$Probability[1] * 100, 1) else NULL
    
    tagList(
      # Prominent Recommendation Showcase Banner
      div(
        class = "result-banner",
        div(class = "result-crop-icon", crop_icon),
        div(style = "font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #A7F3D0; font-weight: 700;", "Recommended Crop"),
        h2(class = "result-crop-name", crop_name),
        if (!is.null(top_prob)) {
          div(
            class = "result-prob-badge",
            icon("chart-pie"), paste0(" Model-Estimated Probability: ", top_prob, "%")
          )
        },
        div(style = "font-size: 11px; color: #A7F3D0; margin-top: 8px;", "Inference Engine: Tuned Random Forest (mtry = 6, CV Macro-F1: 90.76%)")
      ),
      
      # Alternative Probability Rankings Card
      if (!is.null(res$probabilities)) {
        div(
          class = "farm-card",
          div(
            class = "farm-card-header",
            div(
              h4(class = "farm-card-title", icon("ranking-star"), " Top Candidate Crop Probabilities"),
              span(class = "farm-card-subtitle", "Relative ensemble voting distribution across crop varieties")
            )
          ),
          lapply(1:min(5, nrow(res$probabilities)), function(i) {
            row_crop <- res$probabilities$Crop[i]
            row_prob <- round(res$probabilities$Probability[i] * 100, 1)
            row_icon <- if (!is.null(crop_emojis[[row_crop]])) crop_emojis[[row_crop]] else "🌱"
            
            div(
              class = "prob-row",
              div(
                class = "prob-row-header",
                span(paste0(row_icon, " ", row_crop)),
                span(style = "color: #059669; font-weight: 700;", paste0(row_prob, "%"))
              ),
              div(
                style = "background: #E2E8F0; border-radius: 9999px; height: 8px; width: 100%; overflow: hidden;",
                div(class = "progress-bar-custom", style = paste0("width: ", max(2, row_prob), "%;"))
              )
            )
          }),
          div(
            class = "disclaimer-box",
            icon("triangle-exclamation"),
            strong(" Important Agronomic Notice:"),
            " Probabilities represent model-estimated decision-tree consensus over historical training records and do not constitute a guaranteed harvest yield. Always consider local microclimates and pest incidence."
          )
        )
      },
      
      # Summary of Entered Conditions
      div(
        class = "farm-card",
        div(
          class = "farm-card-header",
          div(
            h4(class = "farm-card-title", icon("clipboard-check"), " Tested Field Profile Summary"),
            span(class = "farm-card-subtitle", "Parameters supplied to the predictive engine")
          )
        ),
        div(
          style = "display: flex; flex-wrap: wrap; gap: 8px;",
          span(class = "model-pill-badge", paste0("N: ", res$inputs$N, " kg/ha")),
          span(class = "model-pill-badge", paste0("P: ", res$inputs$P, " kg/ha")),
          span(class = "model-pill-badge", paste0("K: ", res$inputs$K, " kg/ha")),
          span(class = "model-pill-badge", paste0("pH: ", res$inputs$pH)),
          span(class = "model-pill-badge", paste0("Rainfall: ", res$inputs$Rainfall, " mm")),
          span(class = "model-pill-badge", paste0("Humidity: ", res$inputs$Humidity, "%")),
          span(class = "model-pill-badge", paste0("Temp: ", res$inputs$Temperature, "°C")),
          span(class = "model-pill-badge", paste0("Soil: ", res$inputs$Soil_Type)),
          span(class = "model-pill-badge", paste0("State: ", res$inputs$State)),
          span(class = "model-pill-badge", paste0("Season: ", res$inputs$Season)),
          span(class = "model-pill-badge", paste0("Irrigation: ", res$inputs$Irrigation_Type))
        )
      )
    )
  })
}

# ------------------------------------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------------------------------------
shinyApp(ui = ui, server = server)
