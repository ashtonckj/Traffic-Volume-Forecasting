# ============================================================
# Final Tuned LSTM Forecasting of Hourly Traffic Volume
# Input:  data/processed/traffic_volume_processed.csv
#         results/lstm_tuning_results.csv (produced by tuning_lstm.R)
#
# Rebuilds the LSTM using the best hyperparameter combination identified by
# the grid search in tuning_lstm.R, trains it, and evaluates it on the
# held-out test set (2018-07-01 .. 2018-09-30) alongside a Seasonal Naive
# benchmark scored on the identical hours. All figures here describe the
# FINAL model's behaviour; the tuning process is visualised in
# tuning_lstm.R, and the 3-month forward forecast in lstm_forecast.R.
# ============================================================
source("src/models/lstm_common.R")

library(keras3)
library(tensorflow)
library(ggplot2)

set.seed(42)
tensorflow::set_random_seed(42)

tuning_results_path <- file.path(RESULTS_DIR, "lstm_tuning_results.csv")
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)


# ---- 1. Load data and engineer features ----
df <- load_processed()
df <- engineer_features(df)

cat("Loaded", nrow(df), "rows spanning", as.character(min(df$date_time)),
    "to", as.character(max(df$date_time)), "\n")

model_df <- df[, FEATURE_COLS]


# ---- 2. Chronological train / validation / test split ----
train_idx <- df$date_time <= TRAIN_END
val_idx   <- df$date_time > TRAIN_END & df$date_time <= VAL_END
test_idx  <- df$date_time > VAL_END

cat(sprintf("Train: %d (%s .. %s)\n", sum(train_idx),
            min(df$date_time[train_idx]), max(df$date_time[train_idx])))
cat(sprintf("Val:   %d (%s .. %s)\n", sum(val_idx),
            min(df$date_time[val_idx]), max(df$date_time[val_idx])))
cat(sprintf("Test:  %d (%s .. %s)\n", sum(test_idx),
            min(df$date_time[test_idx]), max(df$date_time[test_idx])))


# ---- 3. Scaling (min-max, fitted on TRAIN ONLY to avoid leakage) ----
scale_params <- fit_scaler(model_df[train_idx, ])
scaled_all   <- apply_scale(model_df, scale_params)

train_scaled <- scaled_all[train_idx, ]
val_scaled   <- scaled_all[val_idx, ]
test_scaled  <- scaled_all[test_idx, ]


# ---- 4. Load the best hyperparameter combination from the grid search ----
if (file.exists(tuning_results_path)) {
  tuning_results <- read.csv(tuning_results_path) %>% arrange(val_loss_mse)
  best_cfg <- tuning_results[1, ]
  cat("\nLoaded best configuration from tuning results:\n")
} else {
  warning("lstm_tuning_results.csv not found -- using default hyperparameters. ",
          "Run tuning_lstm.R first to select the tuned configuration.")
  best_cfg <- data.frame(look_back = 168, lstm_units = 64, dropout_rate = 0.2,
                         learning_rate = 0.001, batch_size = 64)
}
print(best_cfg)

LOOK_BACK     <- best_cfg$look_back
LSTM_UNITS    <- best_cfg$lstm_units
DROPOUT_RATE  <- best_cfg$dropout_rate
LEARNING_RATE <- best_cfg$learning_rate
BATCH_SIZE    <- best_cfg$batch_size


# ---- 5. Build sliding-window sequences ----
# Validation and test are windowed with the tail of the preceding split as
# context, so all 2,184 validation hours and all 2,208 test hours receive a
# prediction. Windowing each split in isolation would drop the first
# LOOK_BACK hours of each, leaving the LSTM scored on a shorter test period
# than the other three models in the group comparison.
train_seq <- make_sequences(train_scaled, LOOK_BACK)
val_seq   <- make_sequences(val_scaled, LOOK_BACK,
                            context = tail(train_scaled, LOOK_BACK))
test_seq  <- make_sequences(test_scaled, LOOK_BACK,
                            context = tail(val_scaled, LOOK_BACK))

cat("Sequence shapes -> train:", paste(dim(train_seq$x), collapse = "x"),
    " val:", paste(dim(val_seq$x), collapse = "x"),
    " test:", paste(dim(test_seq$x), collapse = "x"), "\n")
stopifnot(length(test_seq$y) == sum(test_idx))


# ---- 6. Build the final LSTM model (tuned hyperparameters) ----
n_features <- dim(train_seq$x)[3]

model <- keras_model_sequential() %>%
  layer_lstm(units = LSTM_UNITS, input_shape = c(LOOK_BACK, n_features),
             return_sequences = FALSE) %>%
  layer_dropout(rate = DROPOUT_RATE) %>%
  layer_dense(units = LSTM_UNITS / 2, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = LEARNING_RATE),
  loss = "mse",
  metrics = list("mae")
)

summary(model)


# ---- 7. Train ----
# Early stopping halts training once validation loss stops improving and
# restores the best-performing weights, guarding against overfitting given
# how repetitive the daily pattern is.
early_stop <- callback_early_stopping(
  monitor = "val_loss", patience = 8, restore_best_weights = TRUE
)

history <- model %>% fit(
  x = train_seq$x, y = train_seq$y,
  validation_data = list(val_seq$x, val_seq$y),
  epochs = 50,
  batch_size = BATCH_SIZE,
  callbacks = list(early_stop),
  verbose = 2
)

best_epoch <- which.min(history$metrics$val_loss)
cat("\nBest epoch:", best_epoch, "of", length(history$metrics$val_loss), "run\n")

# Figure: training/validation loss curve, showing convergence and whether
# validation loss tracks training loss (i.e. no overfitting).
p_hist <- plot(history)
ggsave(file.path(RESULTS_DIR, "lstm_final_training_history.png"), plot = p_hist,
       width = 7, height = 4.5, dpi = 300)


# ---- 8. Predict on the held-out test set ----
pred_scaled <- model %>% predict(test_seq$x, verbose = 0)
pred   <- inverse_scale_target(as.vector(pred_scaled), scale_params)
actual <- inverse_scale_target(test_seq$y, scale_params)

test_dates <- df$date_time[test_idx]
stopifnot(length(test_dates) == length(pred))


# ---- 9. Seasonal Naive benchmark on the identical hours ----
# The forecast for hour t is the observed value at t-168 (same hour, previous
# week). Reported here only so the LSTM's error can be read as a relative
# improvement rather than an unanchored number -- the group-level Seasonal
# Naive model is a separate member's contribution.
tv_all <- df[[TARGET_COL]]
snaive_pred <- tv_all[which(test_idx) - 168]

lstm_acc   <- forecast_accuracy(actual, pred)
snaive_acc <- forecast_accuracy(actual, snaive_pred)


# ---- 10. Evaluation metrics ----
results <- data.frame(
  Model         = c("LSTM (tuned)", "Seasonal Naive (benchmark)"),
  look_back     = c(LOOK_BACK, 168),
  lstm_units    = c(LSTM_UNITS, NA),
  dropout_rate  = c(DROPOUT_RATE, NA),
  learning_rate = c(LEARNING_RATE, NA),
  batch_size    = c(BATCH_SIZE, NA),
  best_epoch    = c(best_epoch, NA),
  n_test        = c(lstm_acc$n, snaive_acc$n),
  MAE           = round(c(lstm_acc$MAE, snaive_acc$MAE), 2),
  MSE           = round(c(lstm_acc$MSE, snaive_acc$MSE), 2),
  RMSE          = round(c(lstm_acc$RMSE, snaive_acc$RMSE), 2),
  MAPE          = round(c(lstm_acc$MAPE, snaive_acc$MAPE), 2)
)

cat("\n==== Test Set Performance (2018-07-01 .. 2018-09-30) ====\n")
print(results)
cat(sprintf("\nLSTM improvement over Seasonal Naive: MAE %.1f%%, RMSE %.1f%%\n",
            100 * (1 - lstm_acc$MAE / snaive_acc$MAE),
            100 * (1 - lstm_acc$RMSE / snaive_acc$RMSE)))
if (lstm_acc$n_zero_actual > 0) {
  cat("Note:", lstm_acc$n_zero_actual, "zero-traffic hour(s) excluded from MAPE.\n")
}

write.csv(results, file.path(RESULTS_DIR, "lstm_final_metrics.csv"), row.names = FALSE)


# ---- 11. Save per-hour predictions ----
# Written out so the group can pool every model's test-set predictions and
# score them on exactly the same rows when building the comparison table.
plot_df <- data.frame(
  date_time = test_dates,
  actual    = actual,
  predicted = pred,
  snaive    = snaive_pred
)
# write.csv() silently drops "00:00:00" from any row falling exactly at midnight
# while keeping the full timestamp on every other row, so a reader that infers
# the format from the first row parses every hour as midnight. Same quirk, and
# same fix, as step 13 of preprocessing.R.
write.csv(transform(plot_df, date_time = format(date_time, "%Y-%m-%d %H:%M:%S")),
          file.path(RESULTS_DIR, "lstm_test_predictions.csv"), row.names = FALSE)


# ---- 12. Plot actual vs predicted (full test period) ----
p_full <- ggplot(plot_df, aes(x = date_time)) +
  geom_line(aes(y = actual, colour = "Actual"), linewidth = 0.3) +
  geom_line(aes(y = predicted, colour = "Predicted"), linewidth = 0.3, alpha = 0.8) +
  scale_colour_manual(values = c("Actual" = "grey30", "Predicted" = "#d62728")) +
  labs(title = "Tuned LSTM: Actual vs Predicted Traffic Volume (Test Period)",
       x = "Date", y = "Traffic Volume (vehicles/h)", colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave(file.path(RESULTS_DIR, "lstm_final_actual_vs_predicted_full.png"), plot = p_full,
       width = 8, height = 3.6, dpi = 300)


# ---- 13. Plot actual vs predicted (one-week zoom) ----
# Three months of hourly points is illegible at IEEE column width; a single
# representative week makes the hour-by-hour pattern matching visible.
zoom_start <- min(test_dates) + days(30)   # a stable week away from split edges
zoom_end   <- zoom_start + days(7)
zoom_idx   <- plot_df$date_time >= zoom_start & plot_df$date_time < zoom_end

p_zoom <- ggplot(plot_df[zoom_idx, ], aes(x = date_time)) +
  geom_line(aes(y = actual, colour = "Actual"), linewidth = 0.6) +
  geom_line(aes(y = predicted, colour = "Predicted"), linewidth = 0.6, linetype = "22") +
  scale_colour_manual(values = c("Actual" = "grey30", "Predicted" = "#d62728")) +
  labs(title = "Tuned LSTM: One-Week Sample from the Test Period",
       x = NULL, y = "Traffic Volume (vehicles/h)", colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave(file.path(RESULTS_DIR, "lstm_final_actual_vs_predicted_zoom.png"), plot = p_zoom,
       width = 7, height = 3.4, dpi = 300)


# ---- 14. Residual diagnostics ----
# Where the error actually lives: if it concentrates in the commute peaks,
# that is a different story from error spread evenly across the day.
resid_df <- plot_df %>%
  mutate(residual = actual - predicted, hour = hour(date_time))

p_resid <- ggplot(resid_df, aes(x = factor(hour), y = residual)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_boxplot(outlier.size = 0.4, fill = "#1f77b4", alpha = 0.5, linewidth = 0.3) +
  labs(title = "Tuned LSTM: Residuals by Hour of Day (Test Period)",
       x = "Hour of day", y = "Actual - Predicted") +
  theme_minimal(base_size = 11)

ggsave(file.path(RESULTS_DIR, "lstm_residuals_by_hour.png"), plot = p_resid,
       width = 7, height = 3.4, dpi = 300)

cat("\nMean absolute residual by period:\n")
print(resid_df %>%
        mutate(period = case_when(
          hour %in% 0:5   ~ "Overnight (00-05)",
          hour %in% 6:9   ~ "AM peak (06-09)",
          hour %in% 10:14 ~ "Midday (10-14)",
          hour %in% 15:18 ~ "PM peak (15-18)",
          TRUE            ~ "Evening (19-23)")) %>%
        group_by(period) %>%
        summarise(mean_abs_resid = round(mean(abs(residual)), 1),
                  mean_actual = round(mean(actual)), .groups = "drop"))

cat("\nSaved final tuned model results and plots to:", RESULTS_DIR, "\n")
