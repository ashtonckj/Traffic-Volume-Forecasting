# BMMS2094 - Seasonal Naive Forecasting of Traffic Volume
# Run in RStudio with Source, or from the terminal:
# Rscript seasonal_naive_analysis.R "C:/path/to/traffic_volume_processed.csv"

CSV_PATH <- "data/processed/traffic_volume_processed.csv"
output_dir <- file.path(getwd(), "output/models", "seasonal_naive")

seasonal_period_weekly <- 168  # same hour in the previous week
seasonal_period_daily <- 24    # same hour on the previous day
acf_max_lag <- 24 * 14         # two weeks

# ---- 1. Read the shared processed data ----
df <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
df <- df[order(df$date_time), ]

# The shared processed data should already have one record per timestamp.
stopifnot(
  "NA values present"           = !anyNA(df),
  "grid is not strictly hourly" = all(as.numeric(diff(df$date_time), units = "hours") == 1),
  "duplicate timestamps"        = !any(duplicated(df$date_time))
)
# ---- 1b. Shared chronological 70% / 15% / 15% split ----
# The supplied group dataset already contains the agreed split labels.
# Using them ensures that Seasonal Naive is evaluated on exactly the same
# training, validation, and test observations as the other group models.
train <- df[df$split == "train", ]
validation <- df[df$split == "validation", ]
test <- df[df$split == "test", ]
if (nrow(train) <= seasonal_period_weekly) {
  stop("Not enough training data for the weekly Seasonal Naive model.")
}
if (!all(as.numeric(diff(train$date_time), units = "hours") == 1) ||
    !all(as.numeric(diff(validation$date_time), units = "hours") == 1) ||
    !all(as.numeric(diff(test$date_time), units = "hours") == 1)) {
  stop("Validation and test periods must contain consecutive hourly timestamps.")
}

# ---- 2. Forecast functions ----
seasonal_naive_forecast <- function(train_values, horizon, seasonal_period) {
  history <- as.numeric(train_values)
  forecast <- numeric(horizon)
  for (i in seq_len(horizon)) {
    forecast[i] <- history[length(history) - seasonal_period + 1]
    history <- c(history, forecast[i])
  }
  forecast
}

naive_forecast <- function(train_values, horizon) {
  rep(tail(as.numeric(train_values), 1), horizon)
}

# MASE uses one common weekly seasonal-naive scale for every model.
# It is calculated from the training split only, so neither validation nor test
# observations influence the error scale. A value below 1 means a model is
# better than the in-sample weekly seasonal-naive reference.
seasonal_mase_scale <- function(train_values, seasonal_period) {
  train_values <- as.numeric(train_values)
  if (length(train_values) <= seasonal_period) {
    stop("Not enough training observations to calculate the MASE scale.")
  }
  scale <- mean(abs(train_values[(seasonal_period + 1):length(train_values)] - train_values[1:(length(train_values) - seasonal_period)]))
  if (!is.finite(scale) || scale == 0) {
    stop("The MASE scale must be a positive finite number.")
  }
  scale
}

forecast_accuracy <- function(actual, forecast, mase_scale) {
  error <- actual - forecast
  nonzero <- actual != 0
  c(
    n_observed = length(actual),
    MAE = mean(abs(error)),
    RMSE = sqrt(mean(error^2)),
    MAPE_percent = mean(abs(error[nonzero] / actual[nonzero])) * 100,
    sMAPE_percent = mean(2 * abs(error) / (abs(actual) + abs(forecast))) * 100,
    MASE = mean(abs(error)) / mase_scale
  )
}

# Common m=168 scale used for the validation and test comparisons.
# Keep this based on the original training split so all group models can use
# exactly the same denominator when reporting MASE.
mase_scale_weekly <- seasonal_mase_scale(train$traffic_volume, seasonal_period_weekly)

# ---- 3. Validation: choose the seasonal period without using test data ----
weekly_validation_fc <- seasonal_naive_forecast(train$traffic_volume, nrow(validation), seasonal_period_weekly)
daily_validation_fc <- seasonal_naive_forecast(train$traffic_volume, nrow(validation), seasonal_period_daily)
validation_accuracy <- rbind(
  "Weekly Seasonal Naive (m=168)" = forecast_accuracy(validation$traffic_volume, weekly_validation_fc, mase_scale_weekly),
  "Daily Seasonal Naive (m=24)" = forecast_accuracy(validation$traffic_volume, daily_validation_fc, mase_scale_weekly)
)
validation_accuracy <- as.data.frame(validation_accuracy)
validation_accuracy$model <- rownames(validation_accuracy)
rownames(validation_accuracy) <- NULL
validation_accuracy <- validation_accuracy[, c("model", "n_observed", "MAE", "RMSE", "MAPE_percent", "sMAPE_percent", "MASE")]
write.csv(round(validation_accuracy[, -1], 3), file.path(output_dir, "validation_accuracy_metrics_numeric.csv"), row.names = FALSE)
write.csv(validation_accuracy, file.path(output_dir, "validation_accuracy_metrics.csv"), row.names = FALSE)

selected_period <- if (validation_accuracy$MAPE_percent[validation_accuracy$model == "Weekly Seasonal Naive (m=168)"] <
                       validation_accuracy$MAPE_percent[validation_accuracy$model == "Daily Seasonal Naive (m=24)"])
  seasonal_period_weekly else seasonal_period_daily

# ---- 4. Test: refit using training + validation, then evaluate once ----
train_validation <- rbind(train, validation)
weekly_fc <- seasonal_naive_forecast(train_validation$traffic_volume, nrow(test), seasonal_period_weekly)
daily_fc <- seasonal_naive_forecast(train_validation$traffic_volume, nrow(test), seasonal_period_daily)
persistence_fc <- naive_forecast(train_validation$traffic_volume, nrow(test))
accuracy <- rbind(
  "Weekly Seasonal Naive (m=168)" = forecast_accuracy(test$traffic_volume, weekly_fc, mase_scale_weekly),
  "Daily Seasonal Naive (m=24)" = forecast_accuracy(test$traffic_volume, daily_fc, mase_scale_weekly),
  "Naive persistence" = forecast_accuracy(test$traffic_volume, persistence_fc, mase_scale_weekly)
)
accuracy <- as.data.frame(accuracy)
accuracy$model <- rownames(accuracy)
rownames(accuracy) <- NULL
accuracy <- accuracy[, c("model", "n_observed", "MAE", "RMSE", "MAPE_percent", "sMAPE_percent", "MASE")]
accuracy_print <- accuracy
numeric_accuracy_columns <- c("n_observed", "MAE", "RMSE", "MAPE_percent", "sMAPE_percent", "MASE")
accuracy_print[, numeric_accuracy_columns] <- round(accuracy_print[, numeric_accuracy_columns], 3)
write.csv(accuracy_print, file.path(output_dir, "accuracy_metrics.csv"), row.names = FALSE)

forecast_table <- data.frame(
  date_time = test$date_time,
  actual_traffic_volume = test$traffic_volume,
  weekly_seasonal_naive_forecast_m168 = weekly_fc,
  daily_seasonal_naive_forecast_m24 = daily_fc,
  naive_persistence_forecast = persistence_fc
)
write.csv(forecast_table, file.path(output_dir, "holdout_forecasts.csv"), row.names = FALSE)

# ---- 5. Weekly seasonality profile ----
train$hour_of_day <- as.integer(format(train$date_time, "%H"))
train$day_of_week <- factor(
  weekdays(train$date_time),
  levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
)
profile <- aggregate(traffic_volume ~ hour_of_day + day_of_week, data = train, FUN = mean)
write.csv(profile, file.path(output_dir, "weekly_seasonality_profile_data.csv"), row.names = FALSE)

day_colours <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2")
png(file.path(output_dir, "weekly_seasonality_profile.png"), width = 2000, height = 1050, res = 200)
plot(NA, xlim = c(0, 23), ylim = range(profile$traffic_volume),
     xlab = "Hour of day", ylab = "Mean traffic volume",
     main = "Hourly Traffic Profile by Day of Week (Training Data)")
for (i in seq_along(levels(train$day_of_week))) {
  day <- levels(train$day_of_week)[i]
  d <- profile[profile$day_of_week == day, ]
  lines(d$hour_of_day, d$traffic_volume, type = "b", pch = 16, cex = 0.45,
        col = day_colours[i], lwd = 2)
}
legend("bottom", legend = levels(train$day_of_week), col = day_colours,
       lty = 1, pch = 16, ncol = 2, bty = "n")
dev.off()

# ---- 6. Actual versus forecast visual ----
png(file.path(output_dir, "seasonal_naive_holdout_forecast.png"), width = 2200, height = 1050, res = 200)
plot(test$date_time, test$traffic_volume, type = "l", col = "#17202a", lwd = 1.5,
     xlab = "Date", ylab = "Traffic volume",
     main = "15% Test-Set Forecast: Weekly Seasonal Naive")
lines(test$date_time, weekly_fc, col = "#d35400", lwd = 1.8)
legend("topleft", legend = c("Observed traffic volume", "Weekly Seasonal Naive (m=168)"),
       col = c("#17202a", "#d35400"), lty = 1, lwd = c(1.5, 1.8), bty = "n")
dev.off()

# ---- 7. Seasonal-period comparison on validation data ----
comparison <- validation_accuracy[validation_accuracy$model %in% c("Weekly Seasonal Naive (m=168)", "Daily Seasonal Naive (m=24)"), ]
png(file.path(output_dir, "seasonal_period_comparison.png"), width = 2000, height = 900, res = 200)
par(mfrow = c(1, 2), mar = c(5, 5, 4, 1))
for (measure in c("MAE", "RMSE")) {
  values <- comparison[[measure]]
  mids <- barplot(values, names.arg = c("Weekly\n(m=168)", "Daily\n(m=24)"),
                  col = c("#d35400", "#5d6d7e"), ylab = "Vehicles", main = measure,
                  ylim = c(0, max(values) * 1.2))
  text(mids, values, labels = sprintf("%.1f", values), pos = 3, cex = 0.9)
}
mtext("Seasonal-Period Comparison on the 15% Validation Set", outer = TRUE, line = -1.5, cex = 1.15)
dev.off()

# ---- 8. Error by hour of day in the test set ----
absolute_error <- abs(test$traffic_volume - weekly_fc)
error_by_hour <- aggregate(absolute_error ~ hour_of_day, data = data.frame(
  hour_of_day = as.integer(format(test$date_time, "%H")), absolute_error = absolute_error
), FUN = mean)
names(error_by_hour)[2] <- "mean_absolute_error"
write.csv(error_by_hour, file.path(output_dir, "weekly_seasonal_naive_error_by_hour.csv"), row.names = FALSE)

png(file.path(output_dir, "weekly_seasonal_naive_error_by_hour.png"), width = 2000, height = 900, res = 200)
barplot(error_by_hour$mean_absolute_error, names.arg = error_by_hour$hour_of_day,
        col = "#d35400", xlab = "Hour of day", ylab = "Mean absolute error (vehicles)",
        main = "Weekly Seasonal Naive Error by Hour of Day")
dev.off()

# ---- 9. ACF diagnostic using training data only ----
acf_result <- acf(train$traffic_volume, lag.max = acf_max_lag, plot = FALSE)
acf_table <- data.frame(
  lag_hours = as.integer(acf_result$lag[-1]),
  autocorrelation = as.numeric(acf_result$acf[-1])
)
write.csv(acf_table, file.path(output_dir, "autocorrelation_lags.csv"), row.names = FALSE)
acf_24 <- acf_table$autocorrelation[acf_table$lag_hours == 24]
acf_168 <- acf_table$autocorrelation[acf_table$lag_hours == 168]

png(file.path(output_dir, "autocorrelation_by_lag.png"), width = 2000, height = 1000, res = 200)
plot(acf_table$lag_hours, acf_table$autocorrelation, type = "h", lwd = 1,
     xlab = "Lag (hours)", ylab = "Autocorrelation",
     main = "Autocorrelation Function of Training Traffic Volume", ylim = c(-0.35, 1))
abline(h = 0, lwd = 1)
abline(v = 24, col = "#5d6d7e", lty = 2, lwd = 2)
abline(v = 168, col = "#d35400", lty = 2, lwd = 2)
text(24, min(acf_24 + 0.10, 0.95), labels = paste0("24 h: ", round(acf_24, 3)), pos = 4, col = "#5d6d7e")
text(168, min(acf_168 + 0.10, 0.95), labels = paste0("168 h: ", round(acf_168, 3)), pos = 4, col = "#d35400")
dev.off()

# ---- 10. Data summary and console results ----
timestamp_gaps <- sum(as.numeric(diff(df$date_time), units = "hours") > 1)
summary_table <- data.frame(
  metric = c("source_records", "analysis_start", "analysis_end", "training_observations",
             "validation_observations", "test_observations", "validation_start", "test_start",
             "selected_seasonal_period_hours", "weekly_mase_scale", "remaining_timestamp_gaps_over_1_hour",
             "acf_lag_24", "acf_lag_168"),
  value = c(nrow(df), min(df$date_time), max(df$date_time), nrow(train),
            nrow(validation), nrow(test), min(validation$date_time), min(test$date_time),
            selected_period, round(mase_scale_weekly, 5), timestamp_gaps,
            round(acf_24, 5), round(acf_168, 5))
)
write.csv(summary_table, file.path(output_dir, "data_summary.csv"), row.names = FALSE)

cat("Analysis complete. Files saved to:", normalizePath(output_dir), "\n")
cat("Training:", min(train$date_time), "to", max(train$date_time), "\n")
cat("Validation:", min(validation$date_time), "to", max(validation$date_time), "\n")
cat("Test:", min(test$date_time), "to", max(test$date_time), "\n")
cat("Selected seasonal period from validation:", selected_period, "hours\n\n")
cat("Test-set accuracy:\n")
print(accuracy_print)
cat("\nACF at 24 hours:", round(acf_24, 5), "\n")
cat("ACF at 168 hours:", round(acf_168, 5), "\n")
