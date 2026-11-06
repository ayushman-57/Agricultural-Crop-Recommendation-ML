
# =====================================================
# PROJECT: AGRICULTURAL CROP RECOMMENDATION
# SCRIPT 06: CORRELATION ANALYSIS
# =====================================================

# 1. Load libraries
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

# 4. Calculate Pearson correlation matrix
cor_matrix <- cor(
  numeric_data,
  method = "pearson"
)

# 5. Display correlation matrix
cat("\n========== CORRELATION MATRIX ==========\n")
print(round(cor_matrix, 3))

# 6. Save correlation matrix
write.csv(
  round(cor_matrix, 3),
  "results/correlation_matrix.csv"
)

# 7. Convert matrix into long format
cor_data <- as.data.frame(as.table(cor_matrix))

colnames(cor_data) <- c(
  "Feature1",
  "Feature2",
  "Correlation"
)

# 8. Create correlation heatmap
cor_plot <- ggplot(
  cor_data,
  aes(
    x = Feature1,
    y = Feature2,
    fill = Correlation
  )
) +
  geom_tile(color = "white") +
  geom_text(
    aes(label = round(Correlation, 2)),
    size = 3
  ) +
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Pearson Correlation Heatmap",
    x = "Numerical Features",
    y = "Numerical Features",
    fill = "Correlation"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    )
  )

# 9. Display heatmap
print(cor_plot)

# 10. Save heatmap
ggsave(
  "results/correlation_heatmap.png",
  plot = cor_plot,
  width = 10,
  height = 8,
  dpi = 300
)

cat("\nCORRELATION ANALYSIS COMPLETED\n")
