# ==============================================================================
# LAUNCH SCRIPT FOR AGRICULTURAL CROP RECOMMENDATION DASHBOARD
# Run this script in RStudio or via Rscript to launch the interactive application
# ==============================================================================

if (!requireNamespace("shiny", quietly = TRUE)) {
  install.packages("shiny")
}

cat("=================================================================\n")
cat("Starting Agricultural Crop Recommendation Interactive Dashboard...\n")
cat("=================================================================\n")

shiny::runApp("dashboard", launch.browser = TRUE)
