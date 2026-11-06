
# =====================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 01: DATA IMPORT
# =====================================================

# 1. Set project directory
getwd()

# 2. Load required package
library(readr)
library(dplyr)

# 3. Import dataset
crop_data <- read_csv(
  "raw_data/crop_dataset1.csv",
  show_col_types = FALSE
)

# 4. Display first six records
head(crop_data)

# 5. Display last six records
tail(crop_data)

# 6. Check dataset dimensions
dim(crop_data)

# 7. Check column names
colnames(crop_data)

# 8. Check dataset structure
str(crop_data)

# 9. Statistical summary
summary(crop_data)

# 10. Number of crop classes
length(unique(crop_data$Crop_Name))

# 11. Crop-wise record count
table(crop_data$Crop_Name)

# 12. Number of missing values
colSums(is.na(crop_data))

# 13. Check duplicate records
sum(duplicated(crop_data))

# 14. Save imported dataset
saveRDS(
  crop_data,
  "data/original_crop_data.rds"
)

cat("\nDATASET IMPORTED SUCCESSFULLY\n")
