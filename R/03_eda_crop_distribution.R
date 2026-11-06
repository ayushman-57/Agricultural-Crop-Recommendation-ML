
# =====================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 03: CROP DISTRIBUTION ANALYSIS
# =====================================================

# 1. Load required libraries
library(ggplot2)
library(dplyr)

# 2. Load cleaned dataset
crop_data <- readRDS("data/cleaned_crop_data.rds")

# 3. Check crop classes
cat("\nNumber of crop classes:\n")
print(length(unique(crop_data$Crop_Name)))

# 4. Calculate crop-wise frequency
crop_distribution <- crop_data %>%
  group_by(Crop_Name) %>%
  summarise(
    Total_Records = n(),
    .groups = "drop"
  )

# 5. Calculate percentage
crop_distribution <- crop_distribution %>%
  mutate(
    Percentage = round(
      Total_Records / sum(Total_Records) * 100,
      2
    )
  ) %>%
  arrange(desc(Total_Records))

# 6. Display distribution table
cat("\n========== CROP DISTRIBUTION ==========\n")
print(crop_distribution)

# 7. Save distribution table
write.csv(
  crop_distribution,
  "results/crop_distribution.csv",
  row.names = FALSE
)

# 8. Create bar chart
crop_plot <- ggplot(
  crop_distribution,
  aes(
    x = reorder(Crop_Name, Total_Records),
    y = Total_Records
  )
) +
  geom_col(fill = "steelblue") +
  geom_text(
    aes(label = Total_Records),
    hjust = -0.2,
    size = 3.5
  ) +
  coord_flip() +
  labs(
    title = "Distribution of Crop Classes",
    x = "Crop Name",
    y = "Number of Records"
  ) +
  theme_minimal() +
  expand_limits(
    y = max(crop_distribution$Total_Records) * 1.15
  )

# 9. Display chart
print(crop_plot)

# 10. Save chart
ggsave(
  "results/crop_distribution.png",
  plot = crop_plot,
  width = 10,
  height = 7,
  dpi = 300
)

cat("\nCROP DISTRIBUTION ANALYSIS COMPLETED\n")
