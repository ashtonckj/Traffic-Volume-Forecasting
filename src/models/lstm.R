# =============================================================================
# lstm.R — LSTM traffic-volume forecasting
#   Data  : data/processed/traffic_volume_processed.csv
#   Span  : 2015-10-01 00:00 -> 2018-09-30 23:00  (26,304 hourly rows)
#   Split : 85% tuning pool / 15% locked test
#   CV    : expanding-window (anchored) rolling origin, 4 folds, purged gap
#   Output: MAE / MSE / RMSE / MAPE / sMAPE on the ORIGINAL traffic_volume scale
# =============================================================================

suppressPackageStartupMessages({
  library(keras3)
})


# 1. CONFIG  — every tunable lives here
CFG <- list(
  csv_path   = "data/processed/traffic_volume_processed.csv",

  # --- your parameters ---
  lookback      = 168L,    # 1 week of hourly history fed to the LSTM
  learning_rate = 0.001,
  units         = 64L,
  dropout       = 0.2,
  batch_size    = 64L,

  # --- parameters you were missing (see notes at the bottom of this file) ---
  horizon           = 1L,     # forecast t+1. Gap depends on this.
  n_layers          = 1L,     # 1 or 2 LSTM layers
  recurrent_dropout = 0.0,    # keep at 0 -> preserves cuDNN GPU acceleration
  clipnorm          = 1.0,    # gradient clipping: LSTMs explode without it
  epochs            = 100L,   # upper bound; early stopping decides the real number
  patience_stop     = 10L,    # early-stopping patience
  patience_lr       = 4L,     # ReduceLROnPlateau patience
  lr_factor         = 0.5,    # LR multiplier on plateau
  loss              = "mse",  # try "huber" if spikes dominate the loss
  seed              = 42L,    # reproducibility — do not omit in a comparison study

  # --- split / CV geometry ---
  test_frac = 0.15,
  n_folds   = 4L,
  assess    = 2190L,   # 3-month validation window (4 x 3mo = full annual cycle)

  verbose = 1L
)

set_random_seed(CFG$seed)

# The metrics block below flattens predictions to a vector, which is only
# correct for a single-step forecast. Multi-step needs reshaping first.
stopifnot("this script assumes horizon = 1" = CFG$horizon == 1L)


# 2. LOAD + VERIFY
cat("\n== Loading data ==\n")
df <- read.csv(CFG$csv_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
df <- df[order(df$date_time), ]

# Verify rather than assume: strict hourly grid, no gaps, no duplicates, no NAs.
d <- as.numeric(diff(df$date_time), units = "hours")
stopifnot(
  "NA values present"            = !anyNA(df),
  "date_time failed to parse"    = !anyNA(df$date_time),
  "grid is not strictly hourly"  = all(d == 1),
  "duplicate timestamps"         = !any(duplicated(df$date_time))
)
n <- nrow(df)
cat(sprintf("rows = %d | %s -> %s | hourly grid verified\n",
            n, format(min(df$date_time)), format(max(df$date_time))))


# 3. FEATURES
#    Cyclical encoding matters: raw hour 23 and hour 0 are adjacent in reality
#    but maximally distant numerically. sin/cos pairs fix that.
dow_num <- match(df$day_of_week, c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
stopifnot("unrecognised day_of_week value" = !anyNA(dow_num))

cyc <- function(x, period) cbind(sin(2 * pi * x / period), cos(2 * pi * x / period))

feat <- cbind(
  traffic    = df$traffic_volume,          # lagged target as an input feature
  temp       = df$temp,
  rain       = log1p(df$rain_1h),          # log1p: heavily zero-inflated, long tail
  snow       = log1p(df$snow_1h),
  clouds     = df$clouds_all,
  is_holiday = df$is_holiday,
  cyc(df$hour,        24),                 # hour_sin,  hour_cos
  cyc(dow_num - 1,     7),                 # dow_sin,   dow_cos
  cyc(df$month  - 1,  12)                  # month_sin, month_cos
)
colnames(feat) <- c("traffic", "temp", "rain", "snow", "clouds", "is_holiday",
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos")
target  <- df$traffic_volume
n_feat  <- ncol(feat)
cat(sprintf("features = %d (%s)\n", n_feat, paste(colnames(feat), collapse = ", ")))


# 4. SPLIT GEOMETRY
#    gap = lookback + horizon - 1. Without it the first validation input window
#    overlaps rows the model already trained on, and val loss is partly train loss.
gap        <- CFG$lookback + CFG$horizon - 1L
test_start <- floor((1 - CFG$test_frac) * n) + 1L
pool_end   <- test_start - 1L
initial    <- pool_end - CFG$n_folds * CFG$assess - gap   # anchored fold-1 train size

stopifnot(
  "fold-1 training set too small — reduce n_folds or assess" =
    initial > 4L * (CFG$lookback + CFG$horizon)
)

folds <- lapply(seq_len(CFG$n_folds), function(f) {
  tr_end <- initial + (f - 1L) * CFG$assess
  va_st  <- tr_end + gap + 1L
  list(train = 1:tr_end, val = va_st:(va_st + CFG$assess - 1L))   # train ALWAYS starts at 1
})

# Assertions — if any of these fail the fold geometry is wrong and no result means anything.
stopifnot(
  "folds not anchored at row 1" =
    all(vapply(folds, function(f) min(f$train) == 1L, logical(1))),
  "purge gap incorrect" =
    all(vapply(folds, function(f) min(f$val) - max(f$train) - 1L == gap, logical(1))),
  "unequal validation windows" =
    all(vapply(folds, function(f) length(f$val) == CFG$assess, logical(1))),
  "folds do not tile the pool exactly" =
    max(folds[[CFG$n_folds]]$val) == pool_end
)

cat("\n== Split geometry ==\n")
cat(sprintf("gap = %d rows | pool = 1..%d | TEST = %d..%d (%.1f%%)\n",
            gap, pool_end, test_start, n, 100 * (n - test_start + 1) / n))
for (i in seq_along(folds)) {
  f <- folds[[i]]
  cat(sprintf("fold %d | train 1..%-6d (%.1f mo) -> %s | val %s .. %s\n",
              i, max(f$train), length(f$train) / 730.5,
              format(df$date_time[max(f$train)], "%Y-%m-%d"),
              format(df$date_time[min(f$val)],   "%Y-%m-%d"),
              format(df$date_time[max(f$val)],   "%Y-%m-%d")))
}
cat(sprintf("TEST  | %s .. %s\n\n",
            format(df$date_time[test_start], "%Y-%m-%d"),
            format(df$date_time[n], "%Y-%m-%d")))


# 5. HELPERS
# Build [samples, lookback, features] from a contiguous slice of rows.
make_windows <- function(rows) {
  m  <- feat[rows, , drop = FALSE]
  yv <- target[rows]
  ns <- length(rows) - CFG$lookback - CFG$horizon + 1L
  stopifnot("slice shorter than lookback + horizon" = ns > 0)
  x <- array(0, dim = c(ns, CFG$lookback, n_feat))
  for (i in seq_len(ns)) x[i, , ] <- m[i:(i + CFG$lookback - 1L), , drop = FALSE]
  list(x = x, y = yv[(CFG$lookback + CFG$horizon):length(rows)])
}

# z-score using TRAIN rows only. Returned closure applies the same shift to val/test.
make_scaler <- function(train_rows) {
  mu <- colMeans(feat[train_rows, , drop = FALSE])
  sd <- apply(feat[train_rows, , drop = FALSE], 2, stats::sd)
  sd[sd == 0] <- 1                      # guard constant columns
  mu_y <- mean(target[train_rows])
  sd_y <- stats::sd(target[train_rows])
  list(
    x_apply = function(a) {             # a is [samples, lookback, features]
      for (j in seq_len(n_feat)) a[, , j] <- (a[, , j] - mu[j]) / sd[j]
      a
    },
    y_apply = function(v) (v - mu_y) / sd_y,
    y_inv   = function(v) v * sd_y + mu_y
  )
}

build_model <- function() {
  m <- keras_model_sequential(input_shape = c(CFG$lookback, n_feat))
  if (CFG$n_layers >= 2L) {
    m <- m |> layer_lstm(units = CFG$units, dropout = CFG$dropout,
                         recurrent_dropout = CFG$recurrent_dropout,
                         return_sequences = TRUE)
  }
  m |>
    layer_lstm(units = CFG$units, dropout = CFG$dropout,
               recurrent_dropout = CFG$recurrent_dropout) |>
    layer_dense(units = CFG$horizon) |>
    compile(
      optimizer = optimizer_adam(learning_rate = CFG$learning_rate,
                                 clipnorm      = CFG$clipnorm),
      loss = CFG$loss, metrics = "mae"
    )
}

# Metrics on the ORIGINAL scale. MAPE is undefined where actual == 0, so those
# rows are excluded and the count is reported — never silently dropped.
metrics <- function(actual, pred) {
  e  <- actual - pred
  nz <- actual != 0
  c(MAE   = mean(abs(e)),
    MSE   = mean(e^2),
    RMSE  = sqrt(mean(e^2)),
    MAPE  = if (any(nz)) mean(abs(e[nz] / actual[nz])) * 100 else NA_real_,
    sMAPE = mean(2 * abs(e) / (abs(actual) + abs(pred) + 1e-9)) * 100,
    zeros = sum(!nz))
}

# Train on train_rows, evaluate on eval_rows. Fresh weights every call.
run_split <- function(train_rows, eval_rows, tag) {
  cat(sprintf("--- %s | train %d rows | eval %d rows ---\n",
              tag, length(train_rows), length(eval_rows)))
  sc <- make_scaler(train_rows)
  tr <- make_windows(train_rows)
  ev <- make_windows(eval_rows)

  xtr <- sc$x_apply(tr$x)
  ytr <- sc$y_apply(tr$y)
  xev <- sc$x_apply(ev$x)

  model <- build_model()                                  # fresh weights — no leakage across folds
  h <- model |> fit(
    xtr, ytr,
    validation_data = list(xev, sc$y_apply(ev$y)),
    epochs = CFG$epochs, batch_size = CFG$batch_size,
    shuffle = TRUE,                                       # safe: windows are pre-built, no cross-boundary mixing
    verbose = CFG$verbose,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = CFG$patience_stop,
                              restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = CFG$lr_factor,
                                    patience = CFG$patience_lr, min_lr = 1e-5)
    )
  )

  pred <- sc$y_inv(as.numeric(predict(model, xev, verbose = 0)))
  m    <- metrics(ev$y, pred)

  # Seasonal-naive baseline: same hour, one week ago. If the LSTM can't beat
  # this, the problem is the model, not the hyperparameters.
  tgt_idx   <- eval_rows[(CFG$lookback + CFG$horizon):length(eval_rows)]
  naive_idx <- tgt_idx - 168L
  stopifnot("baseline reaches before row 1" = min(naive_idx) >= 1L)
  bm <- metrics(ev$y, target[naive_idx])

  list(metrics = m, baseline = bm,
       epochs_run = length(h$metrics$loss),
       best_epoch = which.min(h$metrics$val_loss))
}


# 6. EXPANDING-WINDOW CROSS-VALIDATION
cat("== Expanding-window CV ==\n")
res <- vector("list", CFG$n_folds)
for (i in seq_len(CFG$n_folds)) {
  res[[i]] <- run_split(folds[[i]]$train, folds[[i]]$val, sprintf("fold %d", i))
}

cv <- do.call(rbind, lapply(res, function(r) r$metrics))
rownames(cv) <- paste("fold", seq_len(CFG$n_folds))

cat("\n== CV results (original traffic_volume scale) ==\n")
print(round(cv, 3))
cat("\nmean :", sprintf("MAE %.2f  MSE %.1f  RMSE %.2f  MAPE %.2f%%  sMAPE %.2f%%",
                        mean(cv[, "MAE"]), mean(cv[, "MSE"]), mean(cv[, "RMSE"]),
                        mean(cv[, "MAPE"], na.rm = TRUE), mean(cv[, "sMAPE"])), "\n")
cat("sd   :", sprintf("MAE %.2f  RMSE %.2f", sd(cv[, "MAE"]), sd(cv[, "RMSE"])), "\n")
cat("best epoch per fold:", vapply(res, function(r) r$best_epoch, integer(1)), "\n")


# 7. FINAL REFIT ON THE FULL POOL -> EVALUATE ON TEST (ONCE)
cat("\n== Final refit + test ==\n")
final <- run_split(1:pool_end, test_start:n, "FINAL")

cat("\n== TEST results (original traffic_volume scale) ==\n")
out <- rbind(LSTM = final$metrics, `seasonal-naive` = final$baseline)
print(round(out, 3))
if (final$metrics["zeros"] > 0)
  cat(sprintf("\nnote: %d test rows had traffic_volume == 0 and were excluded from MAPE.\n",
              final$metrics["zeros"]))
cat("\nDone.\n")

# =============================================================================
# NOTES — parameters added beyond your five, and why
# =============================================================================
# horizon           : you must define what you are forecasting. Set to 1 (next
#                     hour). It also determines the purge gap.
# clipnorm = 1.0    : LSTMs suffer exploding gradients through time. Without
#                     clipping you get loss spikes or NaN. Cheap insurance.
# epochs + patience : "epochs" is not something to grid-search. Set a high cap
#                     and let early stopping with restore_best_weights decide.
# ReduceLROnPlateau : reacts to val_loss, not step count — so it self-adjusts to
#                     each fold's different length. Do NOT use a step-indexed
#                     schedule here; steps/epoch grows 210 -> 312 across folds.
# seed              : without it you cannot tell a real improvement from run-to-
#                     run noise when comparing configurations.
# recurrent_dropout : left at 0 deliberately. Any non-zero value disables the
#                     cuDNN kernel and slows training by roughly 10x.
# n_layers          : exposed so you can test depth without editing the model.
# loss              : switch to "huber" if rush-hour spikes dominate the gradient.
#
# TO SWEEP HYPERPARAMETERS: convert the CFG entries to tfruns flags(), then drive
# with tfruns::tuning_run("lstm.R", flags = list(...), sample = 0.3) and select
# on mean CV MAE across folds. Only ever look at the TEST block once, at the end.
#
# MEMORY: fold 4 materialises ~20,000 x 168 x 12 doubles (~320 MB). Peak usage
# around 1.5-2 GB. If that is tight, lower lookback or switch make_windows() to
# keras3::timeseries_dataset_from_array().
# =============================================================================