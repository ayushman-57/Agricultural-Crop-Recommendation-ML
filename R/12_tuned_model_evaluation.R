
# ============================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 12: TUNED MODEL EVALUATION
# ============================================================

library(caret)
library(dplyr)
library(ggplot2)
library(tidyr)

# ============================================================
# 1. LOAD DATA AND MODELS
# ============================================================

test_data <- readRDS("data/test_data.rds")

test_data$Crop_Name <- factor(test_data$Crop_Name)

tuned_models <- readRDS(
  "models/tuned/all_tuned_models.rds"
)

original_results <- read.csv(
  "results/model_evaluation/final_model_comparison.csv"
)

dir.create(
  "results/tuned_evaluation",
  recursive = TRUE,
  showWarnings = FALSE
)

# ============================================================
# 2. METRIC CALCULATION FUNCTION
# ============================================================

calculate_metrics <- function(actual, predicted) {
  
  classes <- levels(actual)
  
  cm <- table(
    Actual = factor(actual, levels = classes),
    Predicted = factor(predicted, levels = classes)
  )
  
  TP <- diag(cm)
  FP <- colSums(cm) - TP
  FN <- rowSums(cm) - TP
  
  precision <- ifelse(
    TP + FP == 0, 0, TP / (TP + FP)
  )
  
  recall <- ifelse(
    TP + FN == 0, 0, TP / (TP + FN)
  )
  
  f1 <- ifelse(
    precision + recall == 0,
    0,
    2 * precision * recall / (precision + recall)
  )
  
  support <- rowSums(cm)
  
  accuracy <- sum(TP) / sum(cm)
  macro_precision <- mean(precision)
  macro_recall <- mean(recall)
  macro_f1 <- mean(f1)
  
  weighted_f1 <- sum(f1 * support) / sum(support)
  
  kappa <- caret::confusionMatrix(
    data = factor(predicted, levels = classes),
    reference = factor(actual, levels = classes)
  )$overall[["Kappa"]]
  
  data.frame(
    Accuracy = accuracy,
    Macro_Precision = macro_precision,
    Macro_Recall = macro_recall,
    Macro_F1 = macro_f1,
    Weighted_F1 = weighted_f1,
    Kappa = as.numeric(kappa)
  )
}

# ============================================================
# 3. EVALUATE TUNED MODELS
# ============================================================

tuned_results <- data.frame()

for (model_name in names(tuned_models)) {
  
  cat("\nEvaluating tuned model:", model_name, "\n")
  
  model <- tuned_models[[model_name]]
  
  predictions <- predict(
    model,
    newdata = test_data
  )
  
  predictions <- factor(
    predictions,
    levels = levels(test_data$Crop_Name)
  )
  
  if (any(is.na(predictions))) {
    cat("Invalid predictions for:", model_name, "\n")
    next
  }
  
  metrics <- calculate_metrics(
    test_data$Crop_Name,
    predictions
  )
  
  metrics$Model <- model_name
  metrics$Version <- "After Tuning"
  
  tuned_results <- rbind(
    tuned_results,
    metrics
  )
  
  # Save individual tuned predictions
  file_name <- gsub(
    "[^A-Za-z0-9]+",
    "_",
    tolower(model_name)
  )
  
  write.csv(
    data.frame(
      Actual = test_data$Crop_Name,
      Predicted = predictions
    ),
    paste0(
      "results/tuned_evaluation/",
      file_name,
      "_predictions.csv"
    ),
    row.names = FALSE
  )
}

# Save tuned results
write.csv(
  tuned_results,
  "results/tuned_evaluation/tuned_model_results.csv",
  row.names = FALSE
)

# ============================================================
# 4. PREPARE BEFORE-TUNING RESULTS
# ============================================================

before_results <- original_results %>%
  dplyr::mutate(
    Version = "Before Tuning"
  )

# ============================================================
# 5. COMBINE BEFORE AND AFTER RESULTS
# ============================================================

comparison <- bind_rows(
  before_results,
  tuned_results
)

comparison <- comparison %>%
  arrange(Model, Version)

cat("\n========== BEFORE VS AFTER TUNING ==========\n")

print(comparison)

write.csv(
  comparison,
  "results/tuned_evaluation/before_after_comparison.csv",
  row.names = FALSE
)

# ============================================================
# 6. CALCULATE MACRO-F1 IMPROVEMENT
# ============================================================

improvement <- before_results %>%
  dplyr::select(Model, Before_Macro_F1 = Macro_F1) %>%
  inner_join(
    tuned_results %>%
      dplyr::select(Model, After_Macro_F1 = Macro_F1),
    by = "Model"
  ) %>%
  mutate(
    Improvement = After_Macro_F1 - Before_Macro_F1,
    Improvement_Percentage_Points = Improvement * 100
  ) %>%
  arrange(desc(Improvement))

cat("\n========== MACRO-F1 IMPROVEMENT ==========\n")

print(improvement)

write.csv(
  improvement,
  "results/tuned_evaluation/macro_f1_improvement.csv",
  row.names = FALSE
)

# ============================================================
# 7. CREATE BEFORE VS AFTER CHART
# ============================================================
plot_data <- comparison %>%
  dplyr::select(Model, Version, Macro_F1)

comparison_plot <- ggplot(
  plot_data,
  aes(
    x = Model,
    y = Macro_F1 * 100,
    fill = Version
  )
) +
  geom_col(
    position = "dodge"
  ) +
  labs(
    title = "Model Performance Before and After Tuning",
    x = "Machine Learning Model",
    y = "Macro F1-Score (%)",
    fill = "Model Version"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    )
  )

print(comparison_plot)

ggsave(
  "results/tuned_evaluation/before_after_chart.png",
  comparison_plot,
  width = 11,
  height = 7,
  dpi = 300
)

# ============================================================
# 8. FINAL SUMMARY
# ============================================================

cat("\n========== EVALUATION SUMMARY ==========\n")

cat("Tuned models evaluated:",
    nrow(tuned_results), "\n")

if (nrow(tuned_results) > 0) {
  
  best_model <- tuned_results %>%
    dplyr::arrange(dplyr::desc(Macro_F1)) %>%
    dplyr::slice(1)
  
  cat("\nBest tuned model by Macro-F1:\n")
  print(best_model)
}

cat("\nTUNED MODEL EVALUATION COMPLETED\n")
