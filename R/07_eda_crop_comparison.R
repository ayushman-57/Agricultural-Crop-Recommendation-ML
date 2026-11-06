
# =====================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 07: CROP-WISE FEATURE COMPARISON
# =====================================================

library(ggplot2)
library(dplyr)
library(tidyr)

# 1. Load cleaned dataset
crop_data <- readRDS("data/cleaned_crop_data.rds")

# 2. Select important numerical features
crop_features <- crop_data %>%
  select(
    Crop_Name,
    N, P, K, pH,
    Rainfall, Humidity, Temperature
  )

# 3. Calculate crop-wise averages
crop_summary <- crop_features %>%
  group_by(Crop_Name) %>%
  summarise(
    across(
      c(N, P, K, pH, Rainfall, Humidity, Temperature),
      ~mean(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  )

# 4. Display summary
cat("\n========== CROP-WISE FEATURE SUMMARY ==========\n")
print(crop_summary)

# 5. Save summary
write.csv(
  crop_summary,
  "results/crop_wise_feature_summary.csv",
  row.names = FALSE
)

# 6. Convert data into long format
crop_long <- crop_features %>%
  pivot_longer(
    cols = -Crop_Name,
    names_to = "Feature",
    values_to = "Value"
  )

# 7. Create comparison boxplots
comparison_plot <- ggplot(
  crop_long,
  aes(
    x = Crop_Name,
    y = Value,
    fill = Crop_Name
  )
) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(
    ~Feature,
    scales = "free",
    ncol = 2
  ) +
  labs(
    title = "Crop-wise Comparison of Agricultural Features",
    x = "Crop Name",
    y = "Feature Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      size = 7
    )
  )

# 8. Display plot
print(comparison_plot)

# 9. Save plot
ggsave(
  "results/crop_wise_comparison.png",
  comparison_plot,
  width = 12,
  height = 10,
  dpi = 300
)

cat("\nCROP-WISE COMPARISON COMPLETED SUCCESSFULLY\n")
