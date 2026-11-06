
# =====================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 04: NUMERICAL FEATURE ANALYSIS
# =====================================================

# 1. Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)

# 2. Load cleaned dataset
crop_data <- readRDS("data/cleaned_crop_data.rds")

# 3. Select numerical features
numeric_data <- crop_data %>%
  select(
    N, P, K, pH,
    Rainfall, Humidity,
    Temperature, Crop_Duration
  )

# =====================================================
# 4. DESCRIPTIVE STATISTICS
# =====================================================

statistics_table <- data.frame(
  Feature = names(numeric_data),
  Mean = sapply(numeric_data, mean),
  Median = sapply(numeric_data, median),
  SD = sapply(numeric_data, sd),
  Minimum = sapply(numeric_data, min),
  Maximum = sapply(numeric_data, max)
)

cat("\n========== DESCRIPTIVE STATISTICS ==========\n")
print(statistics_table)

# 5. Save descriptive statistics
write.csv(
  statistics_table,
  "results/numerical_statistics.csv",
  row.names = FALSE
)

# =====================================================
# 6. CONVERT DATA INTO LONG FORMAT
# =====================================================

numeric_long <- numeric_data %>%
  pivot_longer(
    cols = everything(),
    names_to = "Feature",
    values_to = "Value"
  )

# =====================================================
# 7. HISTOGRAM ANALYSIS
# =====================================================

histogram_plot <- ggplot(
  numeric_long,
  aes(x = Value)
) +
  geom_histogram(
    bins = 25,
    fill = "steelblue",
    color = "white"
  ) +
  facet_wrap(
    ~Feature,
    scales = "free",
    ncol = 2
  ) +
  labs(
    title = "Distribution of Numerical Features",
    x = "Value",
    y = "Frequency"
  ) +
  theme_minimal()

# Display histogram
print(histogram_plot)

# Save histogram
ggsave(
  "results/numerical_histograms.png",
  plot = histogram_plot,
  width = 10,
  height = 10,
  dpi = 300
)

# =====================================================
# 8. BOXPLOT ANALYSIS
# =====================================================

boxplot_graph <- ggplot(
  numeric_long,
  aes(x = Feature, y = Value)
) +
  geom_boxplot(
    fill = "lightblue",
    color = "black"
  ) +
  facet_wrap(
    ~Feature,
    scales = "free",
    ncol = 2
  ) +
  labs(
    title = "Box Plot Analysis of Numerical Features",
    x = "Feature",
    y = "Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_blank()
  )

# Display boxplot
print(boxplot_graph)

# Save boxplot
ggsave(
  "results/numerical_boxplots.png",
  plot = boxplot_graph,
  width = 10,
  height = 10,
  dpi = 300
)

# =====================================================
# 9. COMPLETION MESSAGE
# =====================================================

cat("\n========== ANALYSIS COMPLETED ==========\n")

cat("Numerical features analysed:",
    ncol(numeric_data), "\n")

cat("Total records analysed:",
    nrow(numeric_data), "\n")

cat("\nFiles saved successfully in results folder.\n")
