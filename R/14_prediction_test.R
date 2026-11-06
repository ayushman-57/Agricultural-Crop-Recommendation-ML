
# ============================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 14: PREDICTION TEST
# ============================================================

library(caret)

# 1. Load final model package
final_package <- readRDS(
  "models/final_crop_model.rds"
)

final_model <- final_package$model

# 2. Load test dataset
test_data <- readRDS("data/test_data.rds")

# 3. Select one test record
sample_input <- test_data[1, final_package$predictors,
                          drop = FALSE]

actual_crop <- test_data$Crop_Name[1]

# 4. Predict crop
predicted_crop <- predict(
  final_model,
  newdata = sample_input
)

# 5. Display results
cat("\n========== CROP PREDICTION TEST ==========\n")

cat("\nInput agricultural conditions:\n")
print(sample_input)

cat("\nActual crop:", as.character(actual_crop), "\n")

cat("Predicted crop:",
    as.character(predicted_crop), "\n")

cat(
  "Prediction correct:",
  as.character(actual_crop) == as.character(predicted_crop),
  "\n"
)

cat("\nPREDICTION TEST COMPLETED\n")
