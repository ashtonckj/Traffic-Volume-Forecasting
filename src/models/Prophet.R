# ============================================================
#  Prophet Model — Metro Interstate Traffic Volume
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/models/  (plots + diagnostics + forecasts)
#
#  Design notes:
#   - Same input CSV and same TRAIN_N/TEST_H cut as sarima_model.R
#     (TEST_H = 24*56 = 1344 hours), so metrics are comparable
#     across models.
#   - Prophet natively supports multiple seasonal periods
#     (daily + weekly + yearly) in one model — unlike SARIMA/
#     Holt-Winters/Seasonal Naive, it doesn't need the Fourier
#     workaround from sarima_model.R. Daily/weekly/yearly
#     seasonality are all enabled explicitly below rather than
#     left on "auto", so the config is visible and reproducible.
#   - is_holiday is added as a regressor (add_regressor), not
#     via Prophet's built-in holidays dataframe mechanism, so it
#     stays consistent with how is_holiday is used as an
#     exogenous regressor in sarima_model.R. It's calendar-
#     determined and known in advance, so no leakage from
#     including it in the future/test frame.
#   - is_imputed leakage check carried over from sarima_model.R:
#     gap-filled hours in the test window are partly synthetic
#     ground truth, so accuracy there is reported both with and
#     without them.
# ============================================================


# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("prophet", "dplyr", "lubridate", "ggplot2", "jsonlite", "tibble")
new  <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))


# ── 1. Paths ─────────────────────────────────────────────────
input_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "output/models"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)


# ── 2. Load Processed Data ───────────────────────────────────
df <- read.csv(input_path, stringsAsFactors = FALSE)

df$date_time <- as.POSIXct(df$date_time,
                           format = "%Y-%m-%d %H:%M:%S",
                           tz     = "UTC")
df <- df %>% arrange(date_time)

cat("Loaded:", nrow(df), "rows |",
    "Range:", as.character(min(df$date_time)), "to",
    as.character(max(df$date_time)), "\n")

if (!"is_imputed" %in% names(df)) {
  warning("is_imputed column not found — assuming all rows observed. ",
          "Re-run the current preprocessing.R to get imputation tracking.")
  df$is_imputed <- 0L
}
cat("Imputed hours in full series:", sum(df$is_imputed), "/", nrow(df),
    sprintf("(%.1f%%)\n", 100 * mean(df$is_imputed)))


# ── 3. Guard: verify the series is truly hourly with no gaps ─
expected_hours <- as.numeric(
  difftime(max(df$date_time), min(df$date_time), units = "hours")
) + 1

if (nrow(df) != expected_hours) {
  stop(
    "Series is not a complete hourly grid. Expected ", expected_hours,
    " rows but found ", nrow(df), ". Re-run preprocessing.R first."
  )
}
cat("Grid check passed: series is complete and hourly.\n")


# ── 4. Build Prophet-Format Data ─────────────────────────────
# Prophet requires columns named exactly 'ds' (timestamp) and 'y'
# (target). is_holiday is carried along as a plain extra column
# so it can be registered as a regressor in Section 6.
df_prophet <- df %>%
  transmute(ds = date_time, y = traffic_volume, is_holiday, is_imputed)


# ── 5. Train / Test Split ────────────────────────────────────
# Same cut as sarima_model.R: final 56 days (1344 hours) held out.
# NOTE: keep this in sync with TEST_H in sarima_model.R and
# TEST_WEEKS in preprocessing.R — all three assume 8 weeks / 1344
# hours so cross-model accuracy comparisons are on the same window.
TEST_H  <- 24 * 56
TRAIN_N <- nrow(df_prophet) - TEST_H

df_train <- df_prophet %>% slice(1:TRAIN_N)
df_test  <- df_prophet %>% slice((TRAIN_N + 1):n())

cat("Train length:", nrow(df_train), "| Test length:", nrow(df_test), "\n")

# ── 5b. Imputation-leakage check on the test window ──────────
n_imputed_in_test <- sum(df_test$is_imputed)

if (n_imputed_in_test > 0) {
  cat(sprintf(
    "\nWARNING: %d of %d test-window hours (%.1f%%) are imputed, not observed.\n",
    n_imputed_in_test, TEST_H, 100 * n_imputed_in_test / TEST_H
  ))
  cat("Accuracy on those hours is scored against gap-fill values, not real\n",
      "traffic counts. A supplementary accuracy excluding these hours is\n",
      "computed in Section 9b.\n", sep = "")
} else {
  cat("\nNo imputed hours in the test window — clean for evaluation.\n")
}


# ── 6. Model Specification & Fit ─────────────────────────────
# Daily, weekly, and yearly seasonality all explicitly enabled:
#   - daily  (24h)  captures the commute peaks
#   - weekly (168h) captures weekday/weekend differences
#   - yearly         has ~3 full cycles to learn from here, unlike
#                    SARIMA/HW, which can't represent it at all
#     within a single ts() seasonal period
# seasonality.mode = "additive" is Prophet's default and a
# reasonable start for this series; switch to "multiplicative"
# if seasonal swings visibly scale with the trend level once you
# inspect eda_01_timeseries_full.png from preprocessing.R.
cat("\nFitting Prophet model on training set ...\n")

m <- prophet(
  yearly.seasonality  = TRUE,
  weekly.seasonality  = TRUE,
  daily.seasonality   = TRUE,
  seasonality.mode    = "additive",
  interval.width      = 0.95,
  fit                 = FALSE
)
m <- add_regressor(m, "is_holiday")
# NOTE: prophet's predict.prophet()/fit.prophet() expect a plain data.frame.
# df_train came through dplyr pipes (transmute/slice), which can leave it as
# a tibble; some prophet versions silently misbehave on tibbles (predict()
# returns a bare atomic vector instead of a data.frame, which then breaks on
# the first $ access downstream). Coercing explicitly avoids that.
m <- fit.prophet(m, as.data.frame(df_train %>% select(ds, y, is_holiday)))

cat("Model fit complete.\n")


# ── 7. Forecast ───────────────────────────────────────────────
# include_history = FALSE: we only need predictions over the test
# window here; in-sample fitted values for residual diagnostics
# are generated separately in Section 8.
future <- make_future_dataframe(m, periods = TEST_H, freq = "hour",
                                include_history = FALSE)
future <- future %>% left_join(df_test %>% select(ds, is_holiday), by = "ds")
future <- as.data.frame(future)  # see note in Section 6 on tibbles vs predict.prophet

if (any(is.na(future$is_holiday))) {
  stop("future/test timestamps didn't align — check for gaps introduced ",
       "upstream of this script.")
}

forecast <- predict(m, future)

# Fail loudly here rather than downstream at a confusing $ access, if this
# ever returns something unexpected again.
if (!is.data.frame(forecast) || !"yhat_lower" %in% names(forecast)) {
  stop("predict(m, future) did not return the expected forecast data.frame ",
       "(class: ", paste(class(forecast), collapse = "/"), "). Check that ",
       "'future' has a proper Date/POSIXct 'ds' column and is a data.frame.")
}


# ── 8. In-Sample Residual Diagnostics ────────────────────────
# Mirrors the residual checks in sarima_model.R so the two models'
# error behaviour can be compared on the same terms.
fitted_train <- predict(m, as.data.frame(df_train %>% select(ds, is_holiday)))
resid_train  <- df_train$y - fitted_train$yhat

lb_48 <- Box.test(resid_train, lag = 48,  type = "Ljung-Box")
lb_168 <- Box.test(resid_train, lag = 168, type = "Ljung-Box")

cat(sprintf("\nLjung-Box (lag=48 ) on training residuals: stat = %.4f, p = %.4f  (%s)\n",
            lb_48$statistic, lb_48$p.value,
            ifelse(lb_48$p.value > 0.05, "residuals ~ white noise", "autocorrelation remains")))
cat(sprintf("Ljung-Box (lag=168) on training residuals: stat = %.4f, p = %.4f  (%s)\n",
            lb_168$statistic, lb_168$p.value,
            ifelse(lb_168$p.value > 0.05, "no weekly autocorrelation detected", "weekly autocorrelation remains")))

png(file.path(output_dir, "prophet_residual_diagnostics.png"),
    width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(resid_train, main = "Training Residuals over Time",
     ylab = "Residual", xlab = "Time (hours)", col = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)

h_info <- hist(resid_train, breaks = 60, plot = FALSE)
hist(resid_train, breaks = 60, col = "#756bb1", border = "white",
     main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean(resid_train), sd(resid_train)) *
        length(resid_train) * diff(h_info$breaks[1:2]),
      col = "red", lwd = 2, add = TRUE)

acf(resid_train,  lag.max = 200, main = "ACF of Residuals (lag up to 200)")
pacf(resid_train, lag.max = 200, main = "PACF of Residuals (lag up to 200)")

dev.off()
cat("Saved -> output/models/prophet_residual_diagnostics.png\n")

# Prophet's own component breakdown (trend / yearly / weekly / daily)
png(file.path(output_dir, "prophet_components.png"), width = 1000, height = 1000, res = 120)
print(prophet_plot_components(m, fitted_train))
dev.off()
cat("Saved -> output/models/prophet_components.png\n")


# ── 9. Accuracy Metrics ───────────────────────────────────────
actual    <- df_test$y
predicted <- forecast$yhat

rmse <- sqrt(mean((actual - predicted)^2))
mae  <- mean(abs(actual - predicted))
mape <- mean(abs((actual - predicted) / ifelse(actual == 0, NA, actual)), na.rm = TRUE) * 100
smape <- mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-8), na.rm = TRUE) * 100

# MASE against a seasonal-naive (t-168) benchmark on the training set,
# for comparability with the MASE reported by sarima_model.R (via
# forecast::accuracy(), which also scales by the in-sample naive MAE).
naive_train_errors <- abs(diff(df_train$y, lag = 168))
mase_scale <- mean(naive_train_errors, na.rm = TRUE)
mase <- mae / mase_scale

cat("\n--- Forecast Accuracy (test set, ALL hours incl. any imputed) ---\n")
cat(sprintf("RMSE  : %10.4f\n", rmse))
cat(sprintf("MAE   : %10.4f\n", mae))
cat(sprintf("MAPE  : %10.4f %%\n", mape))
cat(sprintf("sMAPE : %10.4f %%\n", smape))
cat(sprintf("MASE  : %10.4f\n", mase))

# ── 9b. Supplementary accuracy excluding imputed test hours ──
if (n_imputed_in_test > 0) {
  keep <- df_test$is_imputed == 0
  actual_obs    <- actual[keep]
  predicted_obs <- predicted[keep]
  
  rmse_obs  <- sqrt(mean((actual_obs - predicted_obs)^2))
  mae_obs   <- mean(abs(actual_obs - predicted_obs))
  mape_obs  <- mean(abs((actual_obs - predicted_obs) / ifelse(actual_obs == 0, NA, actual_obs)), na.rm = TRUE) * 100
  smape_obs <- mean(2 * abs(actual_obs - predicted_obs) / (abs(actual_obs) + abs(predicted_obs) + 1e-8), na.rm = TRUE) * 100
  
  cat(sprintf("\n--- Forecast Accuracy (test set, OBSERVED hours only, n=%d) ---\n", sum(keep)))
  cat(sprintf("RMSE  : %10.4f\n", rmse_obs))
  cat(sprintf("MAE   : %10.4f\n", mae_obs))
  cat(sprintf("MAPE  : %10.4f %%\n", mape_obs))
  cat(sprintf("sMAPE : %10.4f %%\n", smape_obs))
} else {
  rmse_obs <- mae_obs <- mape_obs <- smape_obs <- NA_real_
}


# ── 10. Forecast vs Actual Plot ──────────────────────────────
fc_df <- tibble(
  date_time  = df_test$ds,
  actual     = actual,
  forecast   = predicted,
  lo95       = forecast$yhat_lower,
  hi95       = forecast$yhat_upper,
  is_imputed = df_test$is_imputed
)

p_fc <- ggplot(fc_df, aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.5) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.6, linetype = "dashed") +
  labs(
    title    = "Prophet — 56-Day Forecast vs Actual (Test Set)",
    subtitle = sprintf("RMSE = %.1f  |  MAE = %.1f  |  MAPE = %.2f%%  |  sMAPE = %.2f%%",
                       rmse, mae, mape, smape),
    x = "Date", y = "Traffic Volume (vehicles / hr)",
    caption = "Shaded band: 95% prediction interval"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey40"))

ggsave(file.path(output_dir, "prophet_forecast_vs_actual.png"),
       p_fc, width = 14, height = 5, dpi = 150)
cat("Saved -> output/models/prophet_forecast_vs_actual.png\n")


# ── 11. First-Week Zoom Plot ──────────────────────────────────
p_zoom <- ggplot(fc_df[1:(24 * 7), ], aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.7, linetype = "dashed") +
  labs(title = "Prophet — First 7 Days of Test Window (Zoom)",
       x = "Date", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "prophet_forecast_week1_zoom.png"),
       p_zoom, width = 12, height = 4, dpi = 150)
cat("Saved -> output/models/prophet_forecast_week1_zoom.png\n")


# ── 12. Save Forecast CSV ─────────────────────────────────────
fc_out <- fc_df
fc_out$date_time <- format(fc_out$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(fc_out, file.path(output_dir, "prophet_forecast_results.csv"), row.names = FALSE)
cat("Saved -> output/models/prophet_forecast_results.csv\n")


# ── 13. Save Model Summary JSON ───────────────────────────────
summary_list <- list(
  model = list(
    yearly_seasonality = TRUE,
    weekly_seasonality = TRUE,
    daily_seasonality  = TRUE,
    seasonality_mode   = "additive",
    interval_width     = 0.95,
    regressors         = "is_holiday"
  ),
  residual_tests = list(
    Ljung_Box_lag48_p  = round(lb_48$p.value,  4),
    Ljung_Box_lag168_p = round(lb_168$p.value, 4)
  ),
  accuracy = list(
    RMSE  = round(rmse, 4),
    MAE   = round(mae, 4),
    MAPE  = round(mape, 4),
    sMAPE = round(smape, 4),
    MASE  = round(mase, 4)
  ),
  accuracy_observed_only = list(
    n_imputed_in_test = n_imputed_in_test,
    RMSE  = if (n_imputed_in_test > 0) round(rmse_obs, 4)  else NA,
    MAE   = if (n_imputed_in_test > 0) round(mae_obs, 4)   else NA,
    MAPE  = if (n_imputed_in_test > 0) round(mape_obs, 4)  else NA,
    sMAPE = if (n_imputed_in_test > 0) round(smape_obs, 4) else NA
  ),
  split = list(
    train_n = TRAIN_N,
    test_h  = TEST_H
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "prophet_model_summary.json"))
cat("Saved -> output/models/prophet_model_summary.json\n")


# ── 14. Save Model Object ─────────────────────────────────────
saveRDS(m, file.path(output_dir, "prophet_model.rds"))
cat("Saved -> output/models/prophet_model.rds\n")
cat("(Reload later with: m <- readRDS('output/models/prophet_model.rds'))\n")


# ── 15. Done ───────────────────────────────────────────────────
cat("\n=============================================\n")
cat("  Prophet modelling complete.\n")
cat("  Outputs written to:", output_dir, "\n")
cat("=============================================\n")