# =============================================================================
# main.R — runs the whole pipeline in order.
# =============================================================================

SCRIPTS <- c(
  "src/preprocessing/preprocessing.R",                  # writes data/processed/traffic_volume_processed.csv
  "src/figures/figures_exploratory_data_analysis.R",    # -> output/exploratory/
  "src/figures/figures_time_series_diagnostics.R",      # -> output/diagnostics/
  # "src/tuning/tuning_lstm.R",                         # lstm hyperparameters tuning (might take around 8 hours)
  "src/models/lstm.R",
  "src/models/sarima.R",
  "src/models/seasonal_naive.R",
  "src/models/xg_boost.R"
)

for (s in SCRIPTS) {
  if (!file.exists(s)) stop("missing script: ", s)
  cat("\n", strrep("=", 70), "\n>>> ", s, "\n", strrep("=", 70), "\n", sep = "")
  t0 <- Sys.time()
  source(s, local = new.env())
  cat(sprintf("\n--- %s finished in %.1f s ---\n", s, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}

cat("\nAll scripts completed.\n")
