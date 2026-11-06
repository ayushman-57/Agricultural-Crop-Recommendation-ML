
# ============================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 08: STATISTICAL ANALYSIS
# ============================================================

# 1. Load libraries
library(dplyr)
library(tidyr)

# 2. Load cleaned dataset
crop_data <- readRDS("data/cleaned_crop_data.rds")

crop_data$Crop_Name <- as.factor(crop_data$Crop_Name)

dir.create(
  "results/statistical_analysis",
  showWarnings = FALSE,
  recursive = TRUE
)

# ============================================================
# PART A: DESCRIPTIVE STATISTICS
# ============================================================

numeric_features <- c(
  "N", "P", "K", "pH",
  "Rainfall", "Humidity",
  "Temperature", "Crop_Duration"
)

descriptive_stats <- crop_data %>%
  summarise(
    across(
      all_of(numeric_features),
      list(
        Mean = ~mean(.x),
        Median = ~median(.x),
        SD = ~sd(.x),
        Variance = ~var(.x),
        Minimum = ~min(.x),
        Maximum = ~max(.x)
      )
    )
  ) %>%
  pivot_longer(
    cols = everything(),
    names_to = c("Feature", "Statistic"),
    names_sep = "_(?=[^_]+$)",
    values_to = "Value"
  ) %>%
  pivot_wider(
    names_from = Statistic,
    values_from = Value
  )

cat("\n========== DESCRIPTIVE STATISTICS ==========\n")
print(descriptive_stats)

write.csv(
  descriptive_stats,
  "results/statistical_analysis/descriptive_statistics.csv",
  row.names = FALSE
)

# ============================================================
# PART B: ONE-WAY ANOVA AND KRUSKAL-WALLIS TEST
# Compare numerical features across crop classes
# ============================================================

anova_results <- data.frame()
kruskal_results <- data.frame()

for (feature in numeric_features) {
  
  # Prepare data
  test_data <- crop_data %>%
    select(Crop_Name, all_of(feature)) %>%
    rename(Value = all_of(feature))
  
  # One-way ANOVA
  anova_model <- aov(Value ~ Crop_Name, data = test_data)
  
  anova_table <- summary(anova_model)[[1]]
  
  anova_p <- anova_table[["Pr(>F)"]][1]
  
  # Eta squared effect size
  eta_squared <- anova_table[["Sum Sq"]][1] /
    sum(anova_table[["Sum Sq"]])
  
  anova_results <- rbind(
    anova_results,
    data.frame(
      Feature = feature,
      F_Value = anova_table[["F value"]][1],
      P_Value = anova_p,
      Eta_Squared = eta_squared
    )
  )
  
  # Kruskal-Wallis test
  kw_test <- kruskal.test(
    Value ~ Crop_Name,
    data = test_data
  )
  
  # Epsilon squared effect size
  H <- as.numeric(kw_test$statistic)
  k <- length(unique(test_data$Crop_Name))
  n <- nrow(test_data)
  
  epsilon_squared <- max(
    0,
    (H - k + 1) / (n - k)
  )
  
  kruskal_results <- rbind(
    kruskal_results,
    data.frame(
      Feature = feature,
      H_Value = H,
      P_Value = kw_test$p.value,
      Epsilon_Squared = epsilon_squared
    )
  )
}

# Adjust p-values for multiple testing
anova_results$Adjusted_P_Value <-
  p.adjust(anova_results$P_Value, method = "BH")

kruskal_results$Adjusted_P_Value <-
  p.adjust(kruskal_results$P_Value, method = "BH")

cat("\n========== ONE-WAY ANOVA RESULTS ==========\n")
print(anova_results)

cat("\n========== KRUSKAL-WALLIS RESULTS ==========\n")
print(kruskal_results)

write.csv(
  anova_results,
  "results/statistical_analysis/anova_results.csv",
  row.names = FALSE
)

write.csv(
  kruskal_results,
  "results/statistical_analysis/kruskal_results.csv",
  row.names = FALSE
)

# ============================================================
# PART C: CHI-SQUARE TEST OF ASSOCIATION
# Categorical features vs Crop_Name
# ============================================================

categorical_features <- c(
  "Soil_Type",
  "Season",
  "Irrigation_Type",
  "Water_Source",
  "Land_Type",
  "State"
)

chi_results <- data.frame()

for (feature in categorical_features) {
  
  # Create contingency table
  contingency_table <- table(
    crop_data[[feature]],
    crop_data$Crop_Name
  )
  
  # Expected frequencies
  initial_test <- suppressWarnings(
    chisq.test(contingency_table)
  )
  
  # Use simulated p-value when expected counts are small
  if (any(initial_test$expected < 5)) {
    
    chi_test <- suppressWarnings(
      chisq.test(
        contingency_table,
        simulate.p.value = TRUE,
        B = 9999
      )
    )
    
    test_method <- "Chi-square with simulated p-value"
    
  } else {
    
    chi_test <- initial_test
    
    test_method <- "Pearson Chi-square"
  }
  
  # Calculate Cramer's V
  chi_value <- as.numeric(chi_test$statistic)
  
  total_n <- sum(contingency_table)
  
  min_dimension <- min(
    nrow(contingency_table) - 1,
    ncol(contingency_table) - 1
  )
  
  cramer_v <- sqrt(
    chi_value / (total_n * min_dimension)
  )
  
  chi_results <- rbind(
    chi_results,
    data.frame(
      Feature = feature,
      Chi_Square = chi_value,
      P_Value = chi_test$p.value,
      Cramers_V = cramer_v,
      Method = test_method
    )
  )
}

# Multiple testing correction
chi_results$Adjusted_P_Value <-
  p.adjust(chi_results$P_Value, method = "BH")

cat("\n========== CHI-SQUARE TEST RESULTS ==========\n")
print(chi_results)

write.csv(
  chi_results,
  "results/statistical_analysis/chi_square_results.csv",
  row.names = FALSE
)

# ============================================================
# PART D: SUMMARY OF STATISTICAL TESTS
# ============================================================

cat("\n========== STATISTICAL ANALYSIS SUMMARY ==========\n")

cat(
  "Features with significant ANOVA results:",
  sum(anova_results$Adjusted_P_Value < 0.05),
  "out of", nrow(anova_results), "\n"
)

cat(
  "Features with significant Kruskal-Wallis results:",
  sum(kruskal_results$Adjusted_P_Value < 0.05),
  "out of", nrow(kruskal_results), "\n"
)

cat(
  "Significant categorical associations:",
  sum(chi_results$Adjusted_P_Value < 0.05),
  "out of", nrow(chi_results), "\n"
)

cat("\nSTATISTICAL ANALYSIS COMPLETED SUCCESSFULLY\n")
