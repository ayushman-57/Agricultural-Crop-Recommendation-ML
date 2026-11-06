
# ============================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 11: HYPERPARAMETER TUNING
# ============================================================

library(caret)
library(dplyr)
library(nnet)
library(randomForest)
library(gbm)
library(kernlab)
library(rpart)

# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

train_data <- readRDS("data/train_data.rds")

train_data$Crop_Name <- droplevels(
  factor(train_data$Crop_Name)
)

dir.create(
  "models/tuned",
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  "results/hyperparameter_tuning",
  recursive = TRUE,
  showWarnings = FALSE
)

cat("\nTraining records:", nrow(train_data), "\n")
cat("Crop classes:", nlevels(train_data$Crop_Name), "\n")

# ============================================================
# 2. DEFINE MACRO-F1 EVALUATION FUNCTION
# ============================================================

macro_f1_summary <- function(data, lev = NULL, model = NULL) {
  
  classes <- lev
  
  cm <- table(
    factor(data$obs, levels = classes),
    factor(data$pred, levels = classes)
  )
  
  tp <- diag(cm)
  fp <- colSums(cm) - tp
  fn <- rowSums(cm) - tp
  
  precision <- ifelse(
    tp + fp == 0, 0, tp / (tp + fp)
  )
  
  recall <- ifelse(
    tp + fn == 0, 0, tp / (tp + fn)
  )
  
  f1 <- ifelse(
    precision + recall == 0,
    0,
    2 * precision * recall / (precision + recall)
  )
  
  c(
    Macro_F1 = mean(f1),
    Accuracy = sum(tp) / sum(cm)
  )
}

# ============================================================
# 3. CROSS-VALIDATION SETTINGS
# ============================================================

set.seed(123)

tune_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  summaryFunction = macro_f1_summary,
  savePredictions = "final",
  allowParallel = FALSE
)

# ============================================================
# 4. DEFINE MODEL-SPECIFIC TUNING GRIDS
# ============================================================

tuning_grids <- list(
  
  "SVM" = expand.grid(
    sigma = 2^seq(-10, -2, length.out = 5),
    C = 2^seq(-1, 5, length.out = 5)
  ),
  
  "Multinomial Logistic Regression" = expand.grid(
    decay = c(0, 0.001, 0.01, 0.1, 1)
  ),
  
  "Random Forest" = expand.grid(
    mtry = 2:6
  ),
  
  "Gradient Boosting" = expand.grid(
    n.trees = c(100, 200, 300),
    interaction.depth = c(1, 3, 5),
    shrinkage = c(0.01, 0.1),
    n.minobsinnode = 10
  ),
  
  "KNN" = expand.grid(
    k = c(3, 5, 7, 9, 11, 13, 15)
  ),
  
  "Decision Tree" = expand.grid(
    cp = c(0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
  )
)

# ============================================================
# 5. DEFINE MODEL METHODS AND PREPROCESSING
# ============================================================

model_methods <- list(
  
  "SVM" = list(
    method = "svmRadial",
    preprocess = c("center", "scale"),
    extra = list()
  ),
  
  "Multinomial Logistic Regression" = list(
    method = "multinom",
    preprocess = c("center", "scale"),
    extra = list(
      trace = FALSE,
      maxit = 500,
      MaxNWts = 20000
    )
  ),
  
  "Random Forest" = list(
    method = "rf",
    preprocess = NULL,
    extra = list(ntree = 300)
  ),
  
  "Gradient Boosting" = list(
    method = "gbm",
    preprocess = NULL,
    extra = list(verbose = FALSE)
  ),
  
  "KNN" = list(
    method = "knn",
    preprocess = c("center", "scale"),
    extra = list()
  ),
  
  "Decision Tree" = list(
    method = "rpart",
    preprocess = NULL,
    extra = list()
  )
)

# ============================================================
# 6. TRAIN TUNED MODELS
# ============================================================

tuned_models <- list()
tuning_results <- data.frame()
tuning_failures <- data.frame()

for (model_name in names(tuning_grids)) {
  
  cat("\n====================================\n")
  cat("TUNING:", model_name, "\n")
  cat("====================================\n")
  
  config <- model_methods[[model_name]]
  
  tryCatch({
    
    set.seed(123)
    
    train_args <- list(
      form = Crop_Name ~ .,
      data = train_data,
      method = config$method,
      trControl = tune_control,
      tuneGrid = tuning_grids[[model_name]],
      metric = "Macro_F1",
      maximize = TRUE,
      preProcess = config$preprocess
    )
    
    train_args <- c(
      train_args,
      config$extra
    )
    
    tuned_model <- do.call(
      caret::train,
      train_args
    )
    
    tuned_models[[model_name]] <- tuned_model
    
    # Save tuned model
    file_name <- gsub(
      "[^A-Za-z0-9]+",
      "_",
      tolower(model_name)
    )
    
    saveRDS(
      tuned_model,
      paste0("models/tuned/", file_name, "_tuned.rds")
    )
    
    # Extract best parameters and scores
    best <- tuned_model$results[
      which.max(tuned_model$results$Macro_F1),
      ,
      drop = FALSE
    ]
    
    result <- data.frame(
      Model = model_name,
      Best_Macro_F1 = best$Macro_F1,
      Best_Accuracy = best$Accuracy
    )
    
    tuning_results <- rbind(
      tuning_results,
      result
    )
    
    cat("\nBest parameters:\n")
    print(tuned_model$bestTune)
    
    cat("\nBest Macro-F1:",
        round(best$Macro_F1, 4), "\n")
    
    cat("TUNING SUCCESSFUL\n")
    
  }, error = function(e) {
    
    cat("\nTUNING FAILED:", conditionMessage(e), "\n")
    
    tuning_failures <<- rbind(
      tuning_failures,
      data.frame(
        Model = model_name,
        Error = conditionMessage(e)
      )
    )
  })
}

# ============================================================
# 7. SAVE TUNING RESULTS
# ============================================================

if (nrow(tuning_results) > 0) {
  
  tuning_results <- tuning_results %>%
    arrange(desc(Best_Macro_F1))
  
  cat("\n========== TUNING COMPARISON ==========\n")
  print(tuning_results)
  
  write.csv(
    tuning_results,
    "results/hyperparameter_tuning/tuning_comparison.csv",
    row.names = FALSE
  )
}

write.csv(
  tuning_failures,
  "results/hyperparameter_tuning/tuning_failures.csv",
  row.names = FALSE
)

# Save all tuned models
saveRDS(
  tuned_models,
  "models/tuned/all_tuned_models.rds"
)

cat("\n========== TUNING SUMMARY ==========\n")

cat("Successfully tuned:",
    length(tuned_models), "\n")

cat("Tuning failures:",
    nrow(tuning_failures), "\n")

cat("\nHYPERPARAMETER TUNING COMPLETED\n")
