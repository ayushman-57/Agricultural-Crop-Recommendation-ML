
# =====================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 05: CATEGORICAL FEATURE ANALYSIS
# =====================================================

library(ggplot2)
library(dplyr)
library(forcats)

# 1. Load cleaned dataset
crop_data <- readRDS("data/cleaned_crop_data.rds")

# 2. Select categorical features
categorical_cols <- c(
  "Soil_Type",
  "State",
  "Season",
  "Irrigation_Type",
  "Water_Source",
  "Land_Type",
  "Fertilizer_Type",
  "Previous_Crop",
  "Variety",
  "Sowing_Month",
  "Harvest_Month"
)

# 3. Create results directory for plots
dir.create(
  "results/categorical_plots",
  showWarnings = FALSE
)

# 4. Analyse each categorical feature
for (col in categorical_cols) {
  
  # Calculate frequency
  frequency_table <- crop_data %>%
    count(.data[[col]], name = "Frequency") %>%
    arrange(desc(Frequency))
  
  # Rename column
  names(frequency_table)[1] <- "Category"
  
  # Save frequency table
  write.csv(
    frequency_table,
    paste0("results/", col, "_frequency.csv"),
    row.names = FALSE
  )
  
  # Keep top 15 categories if there are many
  if (nrow(frequency_table) > 15) {
    
    plot_data <- frequency_table %>%
      slice_head(n = 15)
  } else {
    
    plot_data <- frequency_table
  }
  
  # Create bar chart
  p <- ggplot(
    plot_data,
    aes(
      x = reorder(Category, Frequency),
      y = Frequency
    )
  ) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(
      title = paste("Distribution of", col),
      x = col,
      y = "Number of Records"
    ) +
    theme_minimal()
  
  # Display chart
  print(p)
  
  # Save chart
  ggsave(
    filename = paste0(
      "results/categorical_plots/",
      col,
      "_distribution.png"
    ),
    plot = p,
    width = 9,
    height = 6,
    dpi = 300
  )
}

cat("\nCATEGORICAL ANALYSIS COMPLETED\n")
