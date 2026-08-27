# ============================================================
# LSTM Forecasting of Hourly Traffic Volume
# Input: traffic_volume_processed.csv (output of preprocessing.R)
# ============================================================
library(dplyr)
library(lubridate)
library(keras3)
library(tensorflow)
library(ggplot2)

set.seed(42)
tensorflow::set_random_seed(42)

# ---- 0. Paths ----
input_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "results"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)


# ---- 1. Load data ----
df <- read.csv(input_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
df <- df %>% arrange(date_time)

cat("Loaded", nrow(df), "rows spanning", as.character(min(df$date_time)),
    "to", as.character(max(df$date_time)), "\n")


# ---- 2. Feature engineering ----
# hour, month, and day_of_week are cyclical (23 -> 0 is a small step, not a
# big one), so raw integers would mislead a distance-based model like LSTM.
# Each is converted into a sin/cos pair so the "wrap-around" is represented
# correctly, instead of one-hot encoding day_of_week (which is what the
# preprocessing script left it as).
weekday_order <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
df$day_of_week_num <- match(df$day_of_week, weekday_order) - 1  # 0-6

add_cyclical <- function(df, col, period) {
  df[[paste0(col, "_sin")]] <- sin(2 * pi * df[[col]] / period)
  df[[paste0(col, "_cos")]] <- cos(2 * pi * df[[col]] / period)
  df
}
df <- add_cyclical(df, "hour", 24)
df <- add_cyclical(df, "day_of_week_num", 7)
df <- add_cyclical(df, "month", 12)

# Final feature set fed to the LSTM: weather + holiday flag + cyclical
# calendar features + the target itself (traffic_volume is included as a
# predictor of its own future values, which is standard for LSTM forecasting).
feature_cols <- c("temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday",
                  "hour_sin", "hour_cos",
                  "day_of_week_num_sin", "day_of_week_num_cos",
                  "month_sin", "month_cos",
                  "traffic_volume")
target_col <- "traffic_volume"

model_df <- df[, feature_cols]


# ---- 3. Chronological train / validation / test split ----
# Split by position, not randomly, to respect the time ordering (later data
# must never leak into training) -- consistent with the group report.
n <- nrow(model_df)
train_end <- floor(0.70 * n)
val_end   <- floor(0.85 * n)

train_raw <- model_df[1:train_end, ]
val_raw   <- model_df[(train_end + 1):val_end, ]
test_raw  <- model_df[(val_end + 1):n, ]

cat("Train:", nrow(train_raw), " Val:", nrow(val_raw), " Test:", nrow(test_raw), "\n")


# ---- 4. Scaling (min-max, fitted on TRAIN ONLY to avoid leakage) ----
scale_params <- lapply(train_raw, function(col) list(min = min(col), max = max(col)))

apply_scale <- function(data, params) {
  scaled <- data
  for (col in names(params)) {
    rng <- params[[col]]$max - params[[col]]$min
    if (rng == 0) rng <- 1  # guard against a constant column
    scaled[[col]] <- (data[[col]] - params[[col]]$min) / rng
  }
  scaled
}

train_scaled <- apply_scale(train_raw, scale_params)
val_scaled   <- apply_scale(val_raw, scale_params)
test_scaled  <- apply_scale(test_raw, scale_params)

inverse_scale_target <- function(x, params, col = "traffic_volume") {
  x * (params[[col]]$max - params[[col]]$min) + params[[col]]$min
}


# ---- 5. Build sliding-window sequences ----
# LOOK_BACK = 24 uses the past day of hourly readings to predict the next
# hour, matching the strong 24-hour seasonality found in the EDA/STL
# decomposition. Try 168 (one week) if you want to compare against the
# weekly cycle instead.
LOOK_BACK <- 24

make_sequences <- function(data, look_back, target_col) {
  x_mat <- as.matrix(data)
  n_obs <- nrow(x_mat) - look_back
  n_feat <- ncol(x_mat)

  x_arr <- array(NA, dim = c(n_obs, look_back, n_feat))
  y_vec <- numeric(n_obs)

  for (i in 1:n_obs) {
    x_arr[i, , ] <- x_mat[i:(i + look_back - 1), ]
    y_vec[i] <- x_mat[i + look_back, target_col]
  }
  list(x = x_arr, y = y_vec)
}

train_seq <- make_sequences(train_scaled, LOOK_BACK, target_col)
val_seq   <- make_sequences(val_scaled,   LOOK_BACK, target_col)
test_seq  <- make_sequences(test_scaled,  LOOK_BACK, target_col)

cat("Sequence shapes -> train:", paste(dim(train_seq$x), collapse = "x"),
    " val:", paste(dim(val_seq$x), collapse = "x"),
    " test:", paste(dim(test_seq$x), collapse = "x"), "\n")


# ---- 6. Build the LSTM model ----
n_features <- dim(train_seq$x)[3]

model <- keras_model_sequential() %>%
  layer_lstm(units = 64, input_shape = c(LOOK_BACK, n_features),
             return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse",
  metrics = list("mae")
)

summary(model)


# ---- 7. Train ----
# Early stopping halts training once validation loss stops improving, and
# restores the best-performing weights -- guards against overfitting given
# how repetitive the daily pattern is.
early_stop <- callback_early_stopping(
  monitor = "val_loss", patience = 8, restore_best_weights = TRUE
)

history <- model %>% fit(
  x = train_seq$x, y = train_seq$y,
  validation_data = list(val_seq$x, val_seq$y),
  epochs = 50,
  batch_size = 64,
  callbacks = list(early_stop),
  verbose = 2
)

plot(history)
ggsave(file.path(output_dir, "lstm_training_history.png"), width = 8, height = 5)


# ---- 8. Predict on the held-out test set ----
pred_scaled <- model %>% predict(test_seq$x)
pred <- inverse_scale_target(as.vector(pred_scaled), scale_params)
actual <- inverse_scale_target(test_seq$y, scale_params)


# ---- 9. Evaluation metrics ----
mae  <- mean(abs(actual - pred))
mse  <- mean((actual - pred)^2)
rmse <- sqrt(mse)
mape <- mean(abs((actual - pred) / actual)) * 100

results <- data.frame(
  Model = "LSTM",
  MAE   = round(mae, 2),
  MSE   = round(mse, 2),
  RMSE  = round(rmse, 2),
  MAPE  = paste0(round(mape, 2), "%")
)

cat("\n==== LSTM Test Set Performance ====\n")
print(results)
write.csv(results, file.path(output_dir, "lstm_metrics.csv"), row.names = FALSE)


# ---- 10. Plot actual vs predicted (test period) ----
test_dates <- df$date_time[(val_end + LOOK_BACK + 1):n]

plot_df <- data.frame(
  date_time = test_dates,
  actual = actual,
  predicted = pred
)

ggplot(plot_df, aes(x = date_time)) +
  geom_line(aes(y = actual, colour = "Actual"), linewidth = 0.4) +
  geom_line(aes(y = predicted, colour = "Predicted"), linewidth = 0.4, alpha = 0.8) +
  scale_colour_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = "LSTM: Actual vs Predicted Traffic Volume (Test Set)",
       x = "Date", y = "Traffic Volume", colour = NULL) +
  theme_minimal()

ggsave(file.path(output_dir, "lstm_actual_vs_predicted.png"), width = 10, height = 5)

cat("\nSaved metrics and plots to:", output_dir, "\n")