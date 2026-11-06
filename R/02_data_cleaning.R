
# =====================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 02: DATA CLEANING AND REPORT
# =====================================================

library(dplyr)

# 1. Load original dataset
crop_data <- readRDS("data/original_crop_data.rds")

# 2. Record missing values BEFORE cleaning
missing_before <- colSums(is.na(crop_data))

cat("\n========== BEFORE CLEANING ==========\n")
print(missing_before)

# 3. Identify numerical and categorical columns
numeric_cols <- names(crop_data)[
  sapply(crop_data, is.numeric)
]

categorical_cols <- names(crop_data)[
  sapply(crop_data, function(x)
    is.character(x) || is.factor(x))
]

# 4. Function to calculate mode
get_mode <- function(x) {
  x <- x[!is.na(x)]

  if (length(x) == 0) return(NA)

  names(which.max(table(x)))
}

# 5. Create a cleaning report
cleaning_report <- data.frame(
  Column = names(crop_data),
  Data_Type = sapply(crop_data, function(x)
    class(x)[1]),
  Missing_Before = as.integer(missing_before),
  Method = "No cleaning required",
  Values_Handled = 0
)

# 6. Handle missing numerical values using median
for (col in numeric_cols) {

  count <- sum(is.na(crop_data[[col]]))

  if (count > 0) {

    median_value <- median(
      crop_data[[col]],
      na.rm = TRUE
    )

    crop_data[[col]][is.na(crop_data[[col]])] <-
      median_value

    cleaning_report$Method[
      cleaning_report$Column == col
    ] <- "Median Imputation"

    cleaning_report$Values_Handled[
      cleaning_report$Column == col
    ] <- count
  }
}

# 7. Handle missing categorical values using mode
for (col in categorical_cols) {

  count <- sum(is.na(crop_data[[col]]))

  if (count > 0) {

    mode_value <- get_mode(crop_data[[col]])

    crop_data[[col]][is.na(crop_data[[col]])] <-
      mode_value

    cleaning_report$Method[
      cleaning_report$Column == col
    ] <- "Mode Imputation"

    cleaning_report$Values_Handled[
      cleaning_report$Column == col
    ] <- count
  }
}

# 8. Record missing values AFTER cleaning
missing_after <- colSums(is.na(crop_data))

cleaning_report$Missing_After <-
  as.integer(missing_after)

cat("\n========== CLEANING REPORT ==========\n")
print(cleaning_report, row.names = FALSE)

# 9. Display overall cleaning statistics
cat("\n========== SUMMARY ==========\n")

cat("Total missing values before:",
    sum(missing_before), "\n")

cat("Total missing values after:",
    sum(missing_after), "\n")

cat("Total values handled:",
    sum(cleaning_report$Values_Handled), "\n")

cat("Duplicate records:",
    sum(duplicated(crop_data)), "\n")

# 10. Save cleaned dataset and report
saveRDS(crop_data, "data/cleaned_crop_data.rds")

write.csv(
  crop_data,
  "data/cleaned_crop_data.csv",
  row.names = FALSE
)

write.csv(
  cleaning_report,
  "results/data_cleaning_report.csv",
  row.names = FALSE
)

cat("\nDATA CLEANING AND REPORTING COMPLETED\n")
