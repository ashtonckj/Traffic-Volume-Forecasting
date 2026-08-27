# ============================================================
# Hyperparameter Tuning for LSTM Forecasting of Hourly Traffic Volume
# Input: traffic_volume_processed.csv (output of preprocessing.R)
#
# Does a grid search over the LSTM's key tunable parameters, scores each
# combination on the VALIDATION set (never the test set -- that stays
# held out until the very end, same rule as lstm.R), then retrains the
# best combination and reports final performance on the test set.
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


# ---- 1. Load data (identical to lstm.R) ----
df <- read.csv(input_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
df <- df %>% arrange(date_time)

cat("Loaded", nrow(df), "rows spanning", as.character(min(df$date_time)),
    "to", as.character(max(df$date_time)), "\n")


# ---- 2. Feature engineering (identical to lstm.R) ----
weekday_order <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
df$day_of_week_num <- match(df$day_of_week, weekday_order) - 1

add_cyclical <- function(df, col, period) {
  df[[paste0(col, "_sin")]] <- sin(2 * pi * df[[col]] / period)
  df[[paste0(col, "_cos")]] <- cos(2 * pi * df[[col]] / period)
  df
}
df <- add_cyclical(df, "hour", 24)
df <- add_cyclical(df, "day_of_week_num", 7)
df <- add_cyclical(df, "month", 12)

feature_cols <- c("temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday",
                  "hour_sin", "hour_cos",
                  "day_of_week_num_sin", "day_of_week_num_cos",
                  "month_sin", "month_cos",
                  "traffic_volume")
target_col <- "traffic_volume"

model_df <- df[, feature_cols]


# ---- 3. Chronological train / validation / test split (identical to lstm.R) ----
# Date-based split: TEST is fixed to the final full calendar year so it
# covers every month/holiday exactly once, instead of an arbitrary
# percentage-based slice. See lstm.R for the full rationale.
TRAIN_END <- as.POSIXct("2017-07-31 23:00:00", tz = "UTC")
VAL_END   <- as.POSIXct("2017-09-30 23:00:00", tz = "UTC")
# Test = everything after VAL_END (2017-10-01 to 2018-09-30)

train_raw <- model_df[df$date_time <= TRAIN_END, ]
val_raw   <- model_df[df$date_time > TRAIN_END & df$date_time <= VAL_END, ]
test_raw  <- model_df[df$date_time > VAL_END, ]

cat("Train:", nrow(train_raw), " Val:", nrow(val_raw), " Test:", nrow(test_raw), "\n")


# ---- 4. Scaling (fitted on TRAIN ONLY, identical to lstm.R) ----
scale_params <- lapply(train_raw, function(col) list(min = min(col), max = max(col)))

apply_scale <- function(data, params) {
  scaled <- data
  for (col in names(params)) {
    rng <- params[[col]]$max - params[[col]]$min
    if (rng == 0) rng <- 1
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


# ---- 5. Sliding-window sequence builder (identical to lstm.R) ----
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


# ---- 6. Define the search grid ----
# Keep this modest -- each row trains a full model. Expand once you've
# confirmed the script runs end-to-end on your machine.
grid <- expand.grid(
  look_back    = c(24, 168),
  lstm_units   = c(32, 64),
  dropout_rate = c(0.2, 0.3),
  learning_rate = c(0.001),
  batch_size   = c(64),
  stringsAsFactors = FALSE
)

cat("\nSearching", nrow(grid), "hyperparameter combinations...\n")
print(grid)

# Cache: sequences depend only on look_back, so build each look_back's
# sequences once and reuse them across the other hyperparameters.
seq_cache <- list()
get_sequences <- function(look_back) {
  key <- as.character(look_back)
  if (is.null(seq_cache[[key]])) {
    seq_cache[[key]] <<- list(
      train = make_sequences(train_scaled, look_back, target_col),
      val   = make_sequences(val_scaled,   look_back, target_col)
    )
  }
  seq_cache[[key]]
}


# ---- 7. Grid search loop (scored on VALIDATION, never test) ----
build_model <- function(look_back, n_features, lstm_units, dropout_rate, learning_rate) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = lstm_units, input_shape = c(look_back, n_features),
               return_sequences = FALSE) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = lstm_units / 2, activation = "relu") %>%
    layer_dense(units = 1)

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = "mse",
    metrics = list("mae")
  )
  model
}

results_list <- list()

for (i in seq_len(nrow(grid))) {
  cfg <- grid[i, ]
  cat("\n---- Config", i, "/", nrow(grid), "----\n")
  print(cfg)

  seqs <- get_sequences(cfg$look_back)
  n_features <- dim(seqs$train$x)[3]

  set.seed(42)
  tensorflow::set_random_seed(42)

  model <- build_model(cfg$look_back, n_features, cfg$lstm_units,
                       cfg$dropout_rate, cfg$learning_rate)

  early_stop <- callback_early_stopping(
    monitor = "val_loss", patience = 8, restore_best_weights = TRUE
  )

  history <- model %>% fit(
    x = seqs$train$x, y = seqs$train$y,
    validation_data = list(seqs$val$x, seqs$val$y),
    epochs = 50,
    batch_size = cfg$batch_size,
    callbacks = list(early_stop),
    verbose = 0
  )

  val_loss <- min(history$metrics$val_loss)
  val_mae  <- history$metrics$val_mae[[which.min(history$metrics$val_loss)]]

  results_list[[i]] <- data.frame(
    look_back     = cfg$look_back,
    lstm_units    = cfg$lstm_units,
    dropout_rate  = cfg$dropout_rate,
    learning_rate = cfg$learning_rate,
    batch_size    = cfg$batch_size,
    val_loss_mse  = val_loss,
    val_mae       = val_mae,
    n_epochs_run  = length(history$metrics$val_loss)
  )

  cat("Val MSE:", round(val_loss, 5), " Val MAE:", round(val_mae, 5), "\n")

  rm(model, history)
  gc()
}

tuning_results <- do.call(rbind, results_list) %>% arrange(val_loss_mse)

cat("\n==== Tuning results (sorted by validation MSE) ====\n")
print(tuning_results)
write.csv(tuning_results, file.path(output_dir, "lstm_tuning_results.csv"), row.names = FALSE)


# ---- 8. Retrain the best config and evaluate on the TEST set ----
best_cfg <- tuning_results[1, ]
cat("\nBest config:\n")
print(best_cfg)

seqs_best <- get_sequences(best_cfg$look_back)
test_seq  <- make_sequences(test_scaled, best_cfg$look_back, target_col)
n_features <- dim(seqs_best$train$x)[3]

set.seed(42)
tensorflow::set_random_seed(42)

final_model <- build_model(best_cfg$look_back, n_features, best_cfg$lstm_units,
                           best_cfg$dropout_rate, best_cfg$learning_rate)

early_stop <- callback_early_stopping(
  monitor = "val_loss", patience = 8, restore_best_weights = TRUE
)

final_history <- final_model %>% fit(
  x = seqs_best$train$x, y = seqs_best$train$y,
  validation_data = list(seqs_best$val$x, seqs_best$val$y),
  epochs = 50,
  batch_size = best_cfg$batch_size,
  callbacks = list(early_stop),
  verbose = 2
)

plot(final_history)
ggsave(file.path(output_dir, "lstm_tuned_training_history.png"), width = 8, height = 5)

pred_scaled <- final_model %>% predict(test_seq$x)
pred <- inverse_scale_target(as.vector(pred_scaled), scale_params)
actual <- inverse_scale_target(test_seq$y, scale_params)

mae  <- mean(abs(actual - pred))
mse  <- mean((actual - pred)^2)
rmse <- sqrt(mse)
mape <- mean(abs((actual - pred) / actual)) * 100

final_results <- data.frame(
  Model = "LSTM (tuned)",
  look_back = best_cfg$look_back,
  lstm_units = best_cfg$lstm_units,
  dropout_rate = best_cfg$dropout_rate,
  learning_rate = best_cfg$learning_rate,
  batch_size = best_cfg$batch_size,
  MAE  = round(mae, 2),
  MSE  = round(mse, 2),
  RMSE = round(rmse, 2),
  MAPE = paste0(round(mape, 2), "%")
)

cat("\n==== Tuned LSTM Test Set Performance ====\n")
print(final_results)
write.csv(final_results, file.path(output_dir, "lstm_tuned_metrics.csv"), row.names = FALSE)


# ---- 9. Plot actual vs predicted (test period) for the tuned model ----
# test_raw's dates are every date_time after VAL_END; the first look_back of
# those are only lookback context (no prediction made for them), so drop
# them to line up with pred/actual.
test_dates <- df$date_time[df$date_time > VAL_END][(best_cfg$look_back + 1):nrow(test_raw)]

plot_df <- data.frame(
  date_time = test_dates,
  actual = actual,
  predicted = pred
)

ggplot(plot_df, aes(x = date_time)) +
  geom_line(aes(y = actual, colour = "Actual"), linewidth = 0.4) +
  geom_line(aes(y = predicted, colour = "Predicted"), linewidth = 0.4, alpha = 0.8) +
  scale_colour_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = "Tuned LSTM: Actual vs Predicted Traffic Volume (Test Set)",
       x = "Date", y = "Traffic Volume", colour = NULL) +
  theme_minimal()

ggsave(file.path(output_dir, "lstm_tuned_actual_vs_predicted.png"), width = 10, height = 5)

cat("\nSaved tuning results, metrics, and plots to:", output_dir, "\n")