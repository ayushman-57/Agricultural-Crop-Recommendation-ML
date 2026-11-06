
# ============================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 13: FINAL PREDICTIVE MODEL
# ============================================================

library(caret)
library(dplyr)

# ============================================================
# 1. LOAD TUNED MODELS AND TRAINING DATA
# ============================================================

tuned_models <- readRDS(
  "models/tuned/all_tuned_models.rds"
)

train_data <- readRDS("data/train_data.rds")

# ============================================================
# 2. IDENTIFY THE BEST TUNED MODEL
# Using cross-validation Macro-F1
# ============================================================

model_ranking <- data.frame()

for (model_name in names(tuned_models)) {
  
  model <- tuned_models[[model_name]]
  
  best_row <- model$results[
    which.max(model$results$Macro_F1),
    ,
    drop = FALSE
  ]
  
  model_ranking <- rbind(
    model_ranking,
    data.frame(
      Model = model_name,
      Cross_Validation_Macro_F1 =
        best_row$Macro_F1,
      Cross_Validation_Accuracy =
        best_row$Accuracy
    )
  )
}

model_ranking <- model_ranking %>%
  dplyr::arrange(
    dplyr::desc(Cross_Validation_Macro_F1)
  )

cat("\n========== MODEL RANKING ==========\n")
print(model_ranking)

# Save model ranking
write.csv(
  model_ranking,
  "results/final_model_ranking.csv",
  row.names = FALSE
)

# ============================================================
# 3. SELECT BEST MODEL
# ============================================================

best_model_name <- model_ranking$Model[1]

final_model <- tuned_models[[best_model_name]]

cat("\nSelected model:", best_model_name, "\n")

cat(
  "Cross-validation Macro-F1:",
  round(model_ranking$Cross_Validation_Macro_F1[1], 4),
  "\n"
)

cat("\nBest tuning parameters:\n")
print(final_model$bestTune)

# ============================================================
# 4. STORE INPUT INFORMATION
# ============================================================

predictor_columns <- setdiff(
  names(train_data),
  "Crop_Name"
)

categorical_columns <- predictor_columns[
  sapply(
    train_data[predictor_columns],
    is.factor
  )
]

# Store factor levels for dashboard inputs
factor_levels <- lapply(
  train_data[categorical_columns],
  levels
)

# ============================================================
# 5. CREATE FINAL MODEL PACKAGE
# ============================================================

final_model_package <- list(
  
  model = final_model,
  
  model_name = best_model_name,
  
  target = "Crop_Name",
  
  predictors = predictor_columns,
  
  categorical_columns = categorical_columns,
  
  factor_levels = factor_levels,
  
  crop_classes = levels(train_data$Crop_Name),
  
  best_parameters = final_model$bestTune,
  
  cross_validation_macro_f1 =
    model_ranking$Cross_Validation_Macro_F1[1]
)

# ============================================================
# 6. SAVE FINAL PREDICTIVE MODEL
# ============================================================

saveRDS(
  final_model_package,
  "models/final_crop_model.rds"
)

# Save ranking
write.csv(
  model_ranking,
  "results/final_model_ranking.csv",
  row.names = FALSE
)

# ============================================================
# 7. FINAL SUMMARY
# ============================================================

cat("\n========== FINAL MODEL SUMMARY ==========\n")

cat("Selected model:", best_model_name, "\n")

cat("Predictors:",
    length(predictor_columns), "\n")

cat("Crop classes:",
    length(final_model_package$crop_classes), "\n")

cat("\nSaved file: models/final_crop_model.rds\n")

cat("\nFINAL PREDICTIVE MODEL CREATED SUCCESSFULLY\n")
