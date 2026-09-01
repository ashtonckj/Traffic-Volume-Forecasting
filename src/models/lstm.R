suppressPackageStartupMessages(library(keras3))

# 1. THE WINNING CONFIGURATION  (from output/models/lstm/tuning_results.csv)
CSV_PATH <- "data/processed/traffic_volume_processed.csv"

cfg <- list(
  lookback      = 48L,
  units         = 64L,
  n_layers      = 1L,
  dropout       = 0.0,
  learning_rate = 0.001,
  batch_size    = 64L,
  loss          = "mse",
  target_log    = FALSE,
  horizon       = 1L,
  clipnorm      = 1.0,
  beta_2        = 0.999,
  seed          = 42L
)

# Fixed epoch count = round(mean(CV best_epochs)) for the winning config.
# Winner's best_epochs were 62;40;49;44 -> mean 48.75 -> 49.
EPOCHS <- 49L

TEST_FRAC <- 0.15
N_FOLDS   <- 4L
ASSESS    <- 2190L
SEASON_M  <- 168L

# 2. DATA
df <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
df <- df[order(df$date_time), ]
stopifnot(
  "NA values present" = !anyNA(df),
  "grid is not strictly hourly" = all(as.numeric(diff(df$date_time), units = "hours") == 1),
)

day_num <- match(df$day_of_week, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
stopifnot("unrecognised day_of_week" = !anyNA(day_num))
cyc <- function(x, p) {
  cbind(
    sin(2 * pi * x / p),
    cos(2 * pi * x / p)
  )
}

FEATURE <- cbind(
  traffic    = df$traffic_volume,
  temp       = df$temp,
  rain       = log1p(df$rain_1h),
  snow       = log1p(df$snow_1h),
  clouds     = df$clouds_all,
  is_holiday = df$is_holiday,
  cyc(df$hour, 24),
  cyc(day_num - 1, 7),
  cyc(df$month - 1, 12)
)
colnames(FEATURE) <- c("traffic", "temp", "rain", "snow", "clouds", "is_holiday",
                       "hour_sin", "hour_cos", "day_num_sin", "day_num_cos", "month_sin", "month_cos")

TARGET <- df$traffic_volume
N_ROWS <- nrow(df)
N_FEATURE <- ncol(FEATURE)
TEST_START <- floor((1 - TEST_FRAC) * N_ROWS) + 1L
POOL_END <- TEST_START - 1L

cat(sprintf("rows = %d | TRAIN pool = 1..%d (%s .. %s)\n", N_ROWS, POOL_END,
            format(df$date_time[1], "%Y-%m-%d"), format(df$date_time[POOL_END], "%Y-%m-%d")))
cat(sprintf("           TEST      = %d..%d (%s .. %s, %.1f%%)\n\n",
            TEST_START, N_ROWS, format(df$date_time[TEST_START], "%Y-%m-%d"),
            format(df$date_time[N_ROWS], "%Y-%m-%d"), 100 * (N_ROWS - TEST_START + 1) / N_ROWS))

# 3. HELPERS
make_windows <- function(rows) {
  m <- FEATURE[rows, , drop = FALSE]
  n_samples <- length(rows) - cfg$lookback - cfg$horizon + 1L
  stopifnot(n_samples > 0)
  x <- array(0, dim = c(n_samples, cfg$lookback, N_FEATURE))
  for (i in seq_len(n_samples))
    x[i, , ] <- m[i:(i + cfg$lookback - 1L), , drop = FALSE]
  list(x = x,
       y = TARGET[rows][(cfg$lookback + cfg$horizon):length(rows)],
       idx = rows[(cfg$lookback + cfg$horizon):length(rows)])
}

make_scaler <- function(train_rows) {
  mu <- colMeans(FEATURE[train_rows, , drop = FALSE])
  sd <- apply(FEATURE[train_rows, , drop = FALSE], 2, stats::sd)
  sd[sd == 0] <- 1
  yt <- if (cfg$target_log) log1p(TARGET[train_rows]) else TARGET[train_rows]
  mu_y <- mean(yt)
  sd_y <- stats::sd(yt)
  list(
    x_apply = function(a) {
      for (j in seq_len(N_FEATURE)) a[, , j] <- (a[, , j] - mu[j]) / sd[j]
      a
    },
    y_apply = function(v) ((if (cfg$target_log) log1p(v) else v) - mu_y) / sd_y,
    y_inv = function(v) {
      z <- v * sd_y + mu_y
      pmax(0, if (cfg$target_log) expm1(z) else z)
    }
  )
}

mase_scale <- function(train_rows, m = SEASON_M) {
  y <- TARGET[train_rows]
  mean(abs(y[(m + 1):length(y)] - y[1:(length(y) - m)]))
}

metrics <- function(actual, pred, scale) {
  e <- actual - pred
  nz <- actual != 0
  c(MAE   = mean(abs(e)),
    RMSE  = sqrt(mean(e^2)),
    MAPE  = if (any(nz)) mean(abs(e[nz] / actual[nz])) * 100 else NA_real_,
    sMAPE = mean(2 * abs(e) / (abs(actual) + abs(pred) + 1e-9)) * 100,
    MASE  = mean(abs(e)) / scale,
    zeros = sum(!nz))
}

# 4. TRAIN ON THE FULL POOL — fixed epochs, no validation_data, no early stopping
set_random_seed(cfg$seed)

train_rows <- 1:POOL_END
sc <- make_scaler(train_rows)
tr <- make_windows(train_rows)

cat(sprintf("Training on %d windows for exactly %d epochs (no early stopping).\n", dim(tr$x)[1], EPOCHS))
cat("The test set is NOT visible to the model during training.\n\n")

model <- keras_model_sequential(input_shape = c(cfg$lookback, N_FEATURE))
if (cfg$n_layers >= 2L)
  model <- model %>% layer_lstm(units = cfg$units, dropout = cfg$dropout, recurrent_dropout = 0, return_sequences = TRUE)
model <- model %>%
  layer_lstm(units = cfg$units, dropout = cfg$dropout, recurrent_dropout = 0) %>%
  layer_dense(units = cfg$horizon) %>%
  compile(optimizer = optimizer_adam(learning_rate = cfg$learning_rate, beta_2 = cfg$beta_2, clipnorm = cfg$clipnorm),
          loss = cfg$loss,
          metrics = "mae")

model %>% fit(
  sc$x_apply(tr$x), sc$y_apply(tr$y),
  epochs = EPOCHS, batch_size = cfg$batch_size,
  shuffle = TRUE, verbose = 2,
  # monitors TRAINING loss only — no test data involved
  callbacks = list(callback_reduce_lr_on_plateau(monitor = "loss", factor = 0.5, patience = 4, min_lr = 1e-5))
)

# 5. TEST — touched exactly once, only to predict
te <- make_windows(TEST_START:N_ROWS)
pred <- sc$y_inv(as.numeric(predict(model, sc$x_apply(te$x), verbose = 0)))
naive <- TARGET[te$idx - SEASON_M]
scale <- mase_scale(train_rows)

out <- rbind(LSTM = metrics(te$y, pred, scale), `seasonal-naive` = metrics(te$y, naive, scale))

cat("\n===================== FINAL TEST RESULTS =====================\n")
cat(sprintf("period: %s .. %s   (%d hours)\n",
            format(df$date_time[min(te$idx)], "%Y-%m-%d"),
            format(df$date_time[max(te$idx)], "%Y-%m-%d"), length(te$y)))
print(round(out, 4))
cat(sprintf("\nimprovement over seasonal-naive: MAE %.1f%%  RMSE %.1f%%\n",
            100 * (1 - out["LSTM", "MAE"] / out["seasonal-naive", "MAE"]),
            100 * (1 - out["LSTM", "RMSE"] / out["seasonal-naive", "RMSE"])))
if (out["LSTM", "zeros"] > 0)
  cat(sprintf("note: %d test rows had traffic_volume == 0, excluded from MAPE only.\n", out["LSTM", "zeros"]))
cat("==============================================================\n")

# 6. SAVE
res <- data.frame(model = rownames(out), out, row.names = NULL)
for (k in names(cfg)) res[[k]] <- cfg[[k]]
res$epochs <- EPOCHS
write.csv(res, "output/models/lstm/final_test_metrics.csv", row.names = FALSE)

write.csv(data.frame(date_time = df$date_time[te$idx],
                     actual = te$y,
                     predicted = round(pred, 2),
                     naive = naive),
          "output/models/lstm/final_test_predictions.csv", row.names = FALSE)

cat("\nSaved:\n  output/models/lstm/final_test_metrics.csv\n",
    "  output/models/lstm/final_test_predictions.csv\n", sep = "")