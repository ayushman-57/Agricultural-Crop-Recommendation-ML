
# ============================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 10: MODEL EVALUATION
# ============================================================

library(caret)
library(dplyr)

# ============================================================
# 1. LOAD TRAINED MODELS AND TEST DATA
# ============================================================

trained_models <- readRDS("models/all_trained_models.rds")
test_data <- readRDS("data/test_data.rds")

test_data$Crop_Name <- factor(test_data$Crop_Name)

cat("\n========== MODEL EVALUATION ==========\n")

cat("Models available:", length(trained_models), "\n")
cat("Test records:", nrow(test_data), "\n")
cat("Crop classes:", nlevels(test_data$Crop_Name), "\n")

# Create output folders
dir.create(
  "results/model_evaluation",
  recursive = TRUE,
  showWarnings = FALSE
)

# ============================================================
# 2. FUNCTION TO CALCULATE CLASSIFICATION METRICS
# ============================================================

calculate_metrics <- function(actual, predicted) {
  
  classes <- levels(actual)
  
  # Confusion matrix
  cm <- table(
    Actual = factor(actual, levels = classes),
    Predicted = factor(predicted, levels = classes)
  )
  
  # True positives, false positives and false negatives
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
  
  # Overall metrics
  accuracy <- sum(TP) / sum(cm)
  
  macro_precision <- mean(precision)
  macro_recall <- mean(recall)
  macro_f1 <- mean(f1)
  
  weighted_f1 <- sum(f1 * support) / sum(support)
  
  # Cohen's Kappa
  kappa_result <- tryCatch(
    caret::confusionMatrix(
      data = factor(predicted, levels = classes),
      reference = factor(actual, levels = classes)
    )$overall[["Kappa"]],
    error = function(e) NA_real_
  )
  
  # Per-class results
  class_metrics <- data.frame(
    Crop = classes,
    Precision = precision,
    Recall = recall,
    F1_Score = f1,
    Support = support
  )
  
  overall_metrics <- data.frame(
    Accuracy = accuracy,
    Macro_Precision = macro_precision,
    Macro_Recall = macro_recall,
    Macro_F1 = macro_f1,
    Weighted_F1 = weighted_f1,
    Kappa = as.numeric(kappa_result)
  )
  
  list(
    confusion_matrix = cm,
    class_metrics = class_metrics,
    overall_metrics = overall_metrics
  )
}

# ============================================================
# 3. EVALUATE EACH SUCCESSFUL MODEL
# ============================================================

overall_results <- data.frame()
failed_evaluations <- data.frame()

for (model_name in names(trained_models)) {
  
  cat("\n------------------------------------\n")
  cat("Evaluating:", model_name, "\n")
  cat("------------------------------------\n")
  
  tryCatch({
    
    model <- trained_models[[model_name]]
    
    # Predict using untouched test data
    predictions <- predict(
      model,
      newdata = test_data
    )
    
    # Ensure predictions use target factor levels
    predictions <- factor(
      predictions,
      levels = levels(test_data$Crop_Name)
    )
    
    # Check prediction validity
    if (any(is.na(predictions))) {
      stop("Predictions contain missing or unknown crop classes.")
    }
    
    # Calculate metrics
    evaluation <- calculate_metrics(
      test_data$Crop_Name,
      predictions
    )
    
    # Add model name
    result <- evaluation$overall_metrics
    result$Model <- model_name
    
    overall_results <- rbind(
      overall_results,
      result
    )
    
    # Save confusion matrix
    file_name <- gsub(
      "[^A-Za-z0-9]+",
      "_",
      tolower(model_name)
    )
    
    write.csv(
      as.data.frame.matrix(
        evaluation$confusion_matrix
      ),
      paste0(
        "results/model_evaluation/",
        file_name,
        "_confusion_matrix.csv"
      )
    )
    
    # Save per-class metrics
    write.csv(
      evaluation$class_metrics,
      paste0(
        "results/model_evaluation/",
        file_name,
        "_class_metrics.csv"
      ),
      row.names = FALSE
    )
    
    # Save predictions
    prediction_output <- data.frame(
      Actual = test_data$Crop_Name,
      Predicted = predictions
    )
    
    write.csv(
      prediction_output,
      paste0(
        "results/model_evaluation/",
        file_name,
        "_predictions.csv"
      ),
      row.names = FALSE
    )
    
    cat("Accuracy:",
        round(result$Accuracy, 4), "\n")
    
    cat("Macro F1:",
        round(result$Macro_F1, 4), "\n")
    
    cat("Evaluation successful\n")
    
  }, error = function(e) {
    
    cat("Evaluation failed:", conditionMessage(e), "\n")
    
    failed_evaluations <<- rbind(
      failed_evaluations,
      data.frame(
        Model = model_name,
        Error = conditionMessage(e)
      )
    )
  })
}

# ============================================================
# 4. CREATE FINAL COMPARISON TABLE
# ============================================================

if (nrow(overall_results) > 0) {
  
  overall_results <- overall_results %>%
    dplyr::select(
      Model,
      Accuracy,
      Macro_Precision,
      Macro_Recall,
      Macro_F1,
      Weighted_F1,
      Kappa
    ) %>%
    dplyr::arrange(desc(Macro_F1))
  
  cat("\n========== MODEL PERFORMANCE COMPARISON ==========\n")
  
  print(overall_results)
  
  # Save comparison table
  write.csv(
    overall_results,
    "results/model_evaluation/final_model_comparison.csv",
    row.names = FALSE
  )
}

# ============================================================
# 5. SAVE EVALUATION FAILURES
# ============================================================

write.csv(
  failed_evaluations,
  "results/model_evaluation/evaluation_failures.csv",
  row.names = FALSE
)

# ============================================================
# 6. DISPLAY BEST MODEL
# ============================================================

if (nrow(overall_results) > 0) {
  
  best_model <- overall_results[1, ]
  
  cat("\n========== BEST MODEL BY MACRO F1 ==========\n")
  print(best_model)
}

# ============================================================
# 7. COMPLETION SUMMARY
# ============================================================

cat("\n========== EVALUATION SUMMARY ==========\n")

cat("Models evaluated:",
    nrow(overall_results), "\n")

cat("Evaluation failures:",
    nrow(failed_evaluations), "\n")

cat("\nMODEL EVALUATION COMPLETED SUCCESSFULLY\n")
