
# ============================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 09: MODEL TRAINING AND INITIAL COMPARISON
# ============================================================

library(caret)
library(dplyr)
library(rpart)
library(randomForest)
library(e1071)
library(MASS)
library(nnet)
library(klaR)
library(gbm)
library(xgboost)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

crop_data <- readRDS("data/cleaned_crop_data.rds")

crop_data$Crop_Name <- as.factor(crop_data$Crop_Name)

# Select predictors available at recommendation time
model_data <- crop_data %>%
  dplyr::select(
    N, P, K, pH,
    Rainfall, Humidity, Temperature,
    Soil_Type, State,
    Season, Irrigation_Type,
    Water_Source, Land_Type,
    Previous_Crop,
    Crop_Name
  )

# Remove unused factor levels
model_data <- droplevels(model_data)

cat("\n========== DATASET INFORMATION ==========\n")
cat("Records:", nrow(model_data), "\n")
cat("Predictors:", ncol(model_data) - 1, "\n")
cat("Crop classes:", nlevels(model_data$Crop_Name), "\n")

# ============================================================
# 2. STRATIFIED TRAIN-TEST SPLIT
# ============================================================

set.seed(123)

train_index <- createDataPartition(
  model_data$Crop_Name,
  p = 0.80,
  list = FALSE
)

train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

cat("\n========== TRAIN-TEST SPLIT ==========\n")
cat("Training records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")

cat("\nTraining crop distribution:\n")
print(table(train_data$Crop_Name))

cat("\nTesting crop distribution:\n")
print(table(test_data$Crop_Name))

# Save split datasets
saveRDS(train_data, "data/train_data.rds")
saveRDS(test_data, "data/test_data.rds")

# ============================================================
# 3. CROSS-VALIDATION SETTINGS
# ============================================================

set.seed(123)

train_control <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  allowParallel = FALSE
)

# Scaling is learned inside each resampling training fold.
# It is useful for distance-based and scale-sensitive models.
scale_preprocess <- c("center", "scale")

# ============================================================
# 4. DEFINE THE 10 MACHINE LEARNING MODELS
# ============================================================

model_methods <- list(
  
  "Decision Tree" = list(
    method = "rpart",
    tuneLength = 5,
    preprocess = NULL,
    extra = list()
  ),
  
  "Random Forest" = list(
    method = "rf",
    tuneLength = 3,
    preprocess = NULL,
    extra = list(ntree = 300)
  ),
  
  "KNN" = list(
    method = "knn",
    tuneLength = 5,
    preprocess = scale_preprocess,
    extra = list()
  ),
  
  "SVM" = list(
    method = "svmRadial",
    tuneLength = 5,
    preprocess = scale_preprocess,
    extra = list()
  ),
  
  "Naive Bayes" = list(
    method = "nb",
    tuneLength = 5,
    preprocess = NULL,
    extra = list()
  ),
  
  "LDA" = list(
    method = "lda",
    tuneLength = 1,
    preprocess = scale_preprocess,
    extra = list()
  ),
  
  "Multinomial Logistic Regression" = list(
    method = "multinom",
    tuneLength = 3,
    preprocess = scale_preprocess,
    extra = list(
      trace = FALSE,
      maxit = 500,
      MaxNWts = 20000
    )
  ),
  
  "Gradient Boosting" = list(
    method = "gbm",
    tuneLength = 3,
    preprocess = NULL,
    extra = list(
      verbose = FALSE
    )
  ),
  
  "XGBoost" = list(
    method = "xgbTree",
    tuneLength = 3,
    preprocess = NULL,
    extra = list(
      verbose = 0
    )
  ),
  
  "Neural Network" = list(
    method = "nnet",
    tuneLength = 3,
    preprocess = scale_preprocess,
    extra = list(
      trace = FALSE,
      MaxNWts = 20000,
      maxit = 500,
      MaxNWts = 20000
    )
  )
)

# ============================================================
# 5. TRAIN ALL MODELS
# ============================================================

dir.create("models", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

trained_models <- list()
model_results <- data.frame()
failed_models <- data.frame()

for (model_name in names(model_methods)) {
  
  cat("\n\n====================================\n")
  cat("TRAINING:", model_name, "\n")
  cat("====================================\n")
  
  config <- model_methods[[model_name]]
  
  set.seed(123)
  
  tryCatch({
    
    # Construct training arguments
    train_args <- list(
      form = Crop_Name ~ .,
      data = train_data,
      method = config$method,
      trControl = train_control,
      tuneLength = config$tuneLength,
      metric = "Accuracy",
      preProcess = config$preprocess
    )
    
    # Add algorithm-specific arguments
    train_args <- c(train_args, config$extra)
    
    # Train model
    model <- do.call(
      caret::train,
      train_args
    )
    
    # Save successful model
    file_name <- gsub(
      "[^A-Za-z0-9]+",
      "_",
      tolower(model_name)
    )
    
    saveRDS(
      model,
      paste0("models/", file_name, ".rds")
    )
    
    trained_models[[model_name]] <- model
    
    # Extract best cross-validation result
    best_result <- model$results
    
    for (i in seq_len(nrow(best_result))) {
      
      model_results <- rbind(
        model_results,
        data.frame(
          Model = model_name,
          Accuracy = best_result$Accuracy[i],
          Kappa = best_result$Kappa[i]
        )
      )
    }
    
    cat("\nBest tuning parameters:\n")
    print(model$bestTune)
    
    cat("\nBest cross-validation results:\n")
    print(model$results[
      which.max(model$results$Accuracy),
      ,
      drop = FALSE
    ])
    
    cat("\nMODEL TRAINED SUCCESSFULLY\n")
    
  }, error = function(e) {
    
    cat("\nMODEL FAILED:", model_name, "\n")
    cat("Reason:", conditionMessage(e), "\n")
    
    failed_models <<- rbind(
      failed_models,
      data.frame(
        Model = model_name,
        Error = conditionMessage(e)
      )
    )
  })
}

# ============================================================
# 6. DISPLAY MODEL COMPARISON
# ============================================================

cat("\n\n========== CROSS-VALIDATION COMPARISON ==========\n")

if (nrow(model_results) > 0) {
  
  best_results <- model_results %>%
    group_by(Model) %>%
    slice_max(
      order_by = Accuracy,
      n = 1,
      with_ties = FALSE
    ) %>%
    ungroup() %>%
    arrange(desc(Accuracy))
  
  print(best_results)
  
  write.csv(
    best_results,
    "results/initial_model_comparison.csv",
    row.names = FALSE
  )
}

# Save all successful models together
saveRDS(
  trained_models,
  "models/all_trained_models.rds"
)

# Save full tuning results
write.csv(
  model_results,
  "results/all_model_tuning_results.csv",
  row.names = FALSE
)

# Save failures, if any
write.csv(
  failed_models,
  "results/failed_models.csv",
  row.names = FALSE
)

# ============================================================
# 7. FINAL SUMMARY
# ============================================================

cat("\n========== TRAINING SUMMARY ==========\n")

cat("Models successfully trained:",
    length(trained_models), "\n")

cat("Models failed:",
    nrow(failed_models), "\n")

cat("\nMODEL TRAINING STAGE COMPLETED\n")
