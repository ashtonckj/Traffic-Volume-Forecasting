# BMMS2094 - Seasonal Naive Forecasting of Traffic Volume
# Run in RStudio with Source, or from the terminal:
# Rscript seasonal_naive_analysis.R "C:/path/to/traffic_volume_processed.csv"

default_data <- "C:/Users/newcr/Downloads/traffic_volume_processed (1).csv"
args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else default_data
output_dir <- file.path(getwd(), "output", "seasonal_naive_r_70_15_15")

seasonal_period_weekly <- 168  # same hour in the previous week
diff_order <- 1                # group-wide first seasonal difference
diff_lag <- 168                # group-wide weekly differencing lag
acf_max_lag <- 24 * 14         # two weeks

if (!file.exists(data_path)) {
  stop("Processed dataset not found: ", data_path)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 1. Read the shared processed data ----
data <- read.csv(data_path, stringsAsFactors = FALSE)
if (!all(c("date_time", "traffic_volume") %in% names(data))) {
  stop("The dataset must contain date_time and traffic_volume columns.")
}
data$date_time <- as.POSIXct(data$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
data <- data[order(data$date_time), ]

# The shared processed data should already have one record per timestamp.
if (anyDuplicated(data$date_time)) {
  stop("Duplicate timestamps found. Use the group's processed dataset.")
}
if (anyNA(data$traffic_volume)) {
  stop("Missing traffic_volume values found. Use the group's processed dataset.")
}
# ---- 1b. Shared chronological 70% / 15% / 15% split ----
# The supplied group dataset already contains the agreed split labels.
# Using them ensures that Seasonal Naive is evaluated on exactly the same
# training, validation, and test observations as the other group models.
if ("split" %in% names(data)) {
  data$split <- tolower(trimws(as.character(data$split)))
  required_splits <- c("train", "validation", "test")
  if (!all(required_splits %in% unique(data$split))) {
    stop("The split column must contain train, validation, and test labels.")
  }
  train <- data[data$split == "train", ]
  validation <- data[data$split == "validation", ]
  test <- data[data$split == "test", ]
} else {
  # Fallback if the group provides an otherwise identical file without labels.
  # Training receives floor(70%) of rows, validation floor(15%), and the
  # remaining newest rows form the test set.
  n_total <- nrow(data)
  n_train <- floor(0.70 * n_total)
  n_validation <- floor(0.15 * n_total)
  train <- data[seq_len(n_train), ]
  validation <- data[(n_train + 1):(n_train + n_validation), ]
  test <- data[(n_train + n_validation + 1):n_total, ]
}
if (nrow(train) <= seasonal_period_weekly) {
  stop("Not enough training data for the weekly Seasonal Naive model.")
}
if (!all(as.numeric(diff(train$date_time), units = "hours") == 1) ||
    !all(as.numeric(diff(validation$date_time), units = "hours") == 1) ||
    !all(as.numeric(diff(test$date_time), units = "hours") == 1)) {
  stop("Validation and test periods must contain consecutive hourly timestamps.")
}
if (as.numeric(difftime(min(validation$date_time), max(train$date_time), units = "hours")) != 1 ||
    as.numeric(difftime(min(test$date_time), max(validation$date_time), units = "hours")) != 1) {
  stop("Training, validation, and test splits must be consecutive with no boundary gaps.")
}

# ---- 2. Forecast functions ----
naive_forecast <- function(train_values, horizon) {
  rep(tail(as.numeric(train_values), 1), horizon)
}

# Apply the group-wide D = 1 seasonal difference: z_t = y_t - y_(t-168).
seasonally_difference <- function(values, lag = diff_lag) {
  values <- as.numeric(values)
  if (length(values) <= lag) {
    stop("Not enough observations for the required seasonal difference.")
  }
  diff(values, lag = lag, differences = diff_order)
}

# Convert predicted differences back to traffic volume. The first 168 anchors
# come from observed history; later anchors are the model's earlier forecasts.
# No validation or test traffic values are used as forecast inputs.
reconstruct_traffic_volume <- function(diff_forecast, raw_history, lag = diff_lag) {
  history <- as.numeric(raw_history)
  raw_forecast <- numeric(length(diff_forecast))
  for (i in seq_along(diff_forecast)) {
    weekly_anchor <- history[length(history) - lag + 1]
    raw_forecast[i] <- diff_forecast[i] + weekly_anchor
    history <- c(history, raw_forecast[i])
  }
  raw_forecast
}

# For a D = 1 seasonal difference, Seasonal Naive assumes that the future
# change from the same hour last week is zero: z_hat_t = 0. Reintegrating this
# produces y_hat_t = y_hat_(t-168), the standard Weekly Seasonal Naive rule.
seasonal_naive_difference_forecast <- function(diff_history, horizon) {
  if (length(diff_history) == 0) {
    stop("Differenced training history is required for Seasonal Naive forecasting.")
  }
  rep(0, horizon)
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
  scale <- mean(abs(train_values[(seasonal_period + 1):length(train_values)] -
                      train_values[1:(length(train_values) - seasonal_period)]))
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

# ---- 3. Difference training data, then validate without using test data ----
diff_train <- seasonally_difference(train$traffic_volume)
weekly_validation_diff_fc <- seasonal_naive_difference_forecast(diff_train, nrow(validation))
weekly_validation_fc <- reconstruct_traffic_volume(weekly_validation_diff_fc, train$traffic_volume)
validation_accuracy <- rbind(
  "Weekly Seasonal Naive (D=1, lag=168)" = forecast_accuracy(validation$traffic_volume, weekly_validation_fc, mase_scale_weekly)
)
validation_accuracy <- as.data.frame(validation_accuracy)
validation_accuracy$model <- rownames(validation_accuracy)
rownames(validation_accuracy) <- NULL
validation_accuracy <- validation_accuracy[, c("model", "n_observed", "MAE", "RMSE", "MAPE_percent", "sMAPE_percent", "MASE")]
write.csv(round(validation_accuracy[, -1], 3), file.path(output_dir, "validation_accuracy_metrics_numeric.csv"), row.names = FALSE)
write.csv(validation_accuracy, file.path(output_dir, "validation_accuracy_metrics.csv"), row.names = FALSE)

selected_period <- seasonal_period_weekly

# ---- 4. Refit on train + validation, then test once ----
train_validation <- rbind(train, validation)
diff_train_validation <- seasonally_difference(train_validation$traffic_volume)
weekly_test_diff_fc <- seasonal_naive_difference_forecast(diff_train_validation, nrow(test))
weekly_fc <- reconstruct_traffic_volume(weekly_test_diff_fc, train_validation$traffic_volume)
persistence_fc <- naive_forecast(train_validation$traffic_volume, nrow(test))
accuracy <- rbind(
  "Weekly Seasonal Naive (D=1, lag=168)" = forecast_accuracy(test$traffic_volume, weekly_fc, mase_scale_weekly),
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
  weekly_snaive_difference_forecast_m168 = weekly_test_diff_fc,
  weekly_snaive_reconstructed_forecast_m168 = weekly_fc,
  naive_persistence_forecast = persistence_fc
)
write.csv(forecast_table, file.path(output_dir, "holdout_forecasts.csv"), row.names = FALSE)

# ---- 5. One-week future forecast beyond the available dataset ----
# This is separate from the test forecast. It uses all known observations after
# evaluation, so it cannot receive accuracy metrics until future actual values exist.
future_horizon <- seasonal_period_weekly
diff_all_data <- seasonally_difference(data$traffic_volume)
future_diff_fc <- seasonal_naive_difference_forecast(diff_all_data, future_horizon)
future_fc <- reconstruct_traffic_volume(future_diff_fc, data$traffic_volume)
future_dates <- seq(from = max(data$date_time) + 60 * 60, by = "hour", length.out = future_horizon)
future_forecast_table <- data.frame(
  date_time = future_dates,
  weekly_snaive_difference_forecast_m168 = future_diff_fc,
  weekly_snaive_reconstructed_forecast_m168 = future_fc
)
write.csv(future_forecast_table, file.path(output_dir, "one_week_future_forecast.csv"), row.names = FALSE)

# Final two observed weeks plus the one-week forecast. The dashed line is the
# boundary between observed data and the future period.
recent_data <- tail(data, 2 * seasonal_period_weekly)
png(file.path(output_dir, "one_week_future_forecast.png"), width = 2200, height = 1050, res = 200)
plot(recent_data$date_time, recent_data$traffic_volume, type = "l", col = "#17202a", lwd = 1.5,
     xlim = range(c(recent_data$date_time, future_dates)),
     ylim = range(c(recent_data$traffic_volume, future_fc)),
     xlab = "Date", ylab = "Traffic volume",
     main = "One-Week Future Forecast: Weekly Seasonal Naive (D=1, lag 168)")
lines(future_dates, future_fc, col = "#2874a6", lwd = 1.8)
abline(v = max(data$date_time), col = "#d35400", lty = 2, lwd = 2)
legend("topleft", legend = c("Observed traffic volume", "One-week future forecast", "Forecast start"),
       col = c("#17202a", "#2874a6", "#d35400"), lty = c(1, 1, 2),
       lwd = c(1.5, 1.8, 2), bty = "n")
dev.off()

# ---- 6. Weekly seasonality profile ----
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
     main = "15% Test Forecast: Weekly Seasonal Naive (D=1, lag 168)")
lines(test$date_time, weekly_fc, col = "#d35400", lwd = 1.8)
legend("topleft", legend = c("Observed traffic volume", "Weekly SNaive (D=1, lag=168)"),
       col = c("#17202a", "#d35400"), lty = 1, lwd = c(1.5, 1.8), bty = "n")
dev.off()

# ---- 7. Seasonal-difference diagnostic ----
png(file.path(output_dir, "weekly_seasonal_difference.png"), width = 2200, height = 900, res = 200)
plot(diff_train, type = "l", col = "#5d6d7e", lwd = 0.8,
     xlab = "Training observation after lag-168 differencing",
     ylab = "Traffic-volume change from same hour last week",
     main = "Weekly Seasonal Difference of Training Traffic Volume (D=1, lag=168)")
abline(h = 0, col = "#d35400", lty = 2, lwd = 2)
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
        main = "Weekly Seasonal Naive (D=1): Error by Hour of Day")
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

# ---- 11. Data summary and console results ----
timestamp_gaps <- sum(as.numeric(diff(data$date_time), units = "hours") > 1)
summary_table <- data.frame(
  metric = c("source_records", "analysis_start", "analysis_end", "training_observations",
             "validation_observations", "test_observations", "validation_start", "test_start",
             "difference_order_D", "difference_lag_hours", "selected_snaive_period_hours",
             "weekly_mase_scale", "future_forecast_horizon_hours", "future_forecast_start",
             "future_forecast_end", "remaining_timestamp_gaps_over_1_hour",
             "acf_lag_24", "acf_lag_168"),
  value = c(nrow(data), min(data$date_time), max(data$date_time), nrow(train),
            nrow(validation), nrow(test), min(validation$date_time), min(test$date_time),
            diff_order, diff_lag, selected_period, round(mase_scale_weekly, 5), future_horizon,
            min(future_dates), max(future_dates), timestamp_gaps,
            round(acf_24, 5), round(acf_168, 5))
)
write.csv(summary_table, file.path(output_dir, "data_summary.csv"), row.names = FALSE)

cat("Analysis complete. Files saved to:", normalizePath(output_dir), "\n")
cat("Training:", min(train$date_time), "to", max(train$date_time), "\n")
cat("Validation:", min(validation$date_time), "to", max(validation$date_time), "\n")
cat("Test:", min(test$date_time), "to", max(test$date_time), "\n")
cat("Group transformation: D =", diff_order, ", lag =", diff_lag, "hours\n")
cat("Selected SNaive period from validation:", selected_period, "hours\n\n")
cat("Test-set accuracy:\n")
print(accuracy_print)
cat("\nACF at 24 hours:", round(acf_24, 5), "\n")
cat("ACF at 168 hours:", round(acf_168, 5), "\n")
cat("One-week future forecast:", min(future_dates), "to", max(future_dates),
    "(", future_horizon, "hours)\n")
