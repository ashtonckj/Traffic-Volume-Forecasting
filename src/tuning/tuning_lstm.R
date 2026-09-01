# =============================================================================
# tune.R — SELF-CONTAINED hyperparameter tuning for the traffic-volume LSTM.
#          No source() of any other file. Just: Rscript tune.R
#
#   Data     : data/processed/traffic_volume_processed.csv
#   Protocol : 85% tuning pool / 15% locked test (test is NEVER touched here)
#   CV       : expanding-window (anchored) rolling origin, 4 folds, purged gap
#   Ranking  : mean MASE across all 4 folds
#   Output   : output/models/lstm/tuning_results.csv   (appended after EVERY config)
#
#   Restartable: a config already in the CSV is skipped. Ctrl-C is safe.
# =============================================================================

suppressPackageStartupMessages(library(keras3))

# =============================================================================
# 1. SETTINGS
# =============================================================================
CSV_PATH <- "data/processed/traffic_volume_processed.csv"
OUT_PATH <- "output/models/lstm/tuning_results.csv"

TEST_FRAC <- 0.15     # protocol — never tuned
N_FOLDS <- 4L         # protocol — never tuned
ASSESS <- 2190L       # protocol — 3-month validation window
SEASON_M <- 168L      # weekly seasonality: naive baseline + MASE denominator
PRIMARY <- "MASE"     # metric used to rank configurations

MAX_CONFIGS <- Inf    # set to e.g. 2 for a timing smoke test, then back to Inf

# --- THE SEARCH SPACE -------------------------------------------------------
GRID <- list(
  lookback      = c(48L),
  units         = c(64L, 128L),
  n_layers      = c(1L, 2L),
  dropout       = c(0.0),
  learning_rate = c(3e-4, 1e-3, 3e-3),
  batch_size    = c(64L),
  loss          = c("mse"),
  target_log    = c(FALSE)
)
TUNABLE <- names(GRID)

# --- FIXED for every configuration ------------------------------------------
FIXED <- list(
  horizon       = 1L,
  clipnorm      = 1.0,
  beta_2        = 0.999,
  epochs        = 100L,
  patience_stop = 10L,
  patience_lr   = 4L,
  lr_factor     = 0.5,
  seed          = 42L
)

# =============================================================================
# 2. DATA — loaded once, verified rather than assumed
# =============================================================================
cat("Loading data ...\n")
df <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
df <- df[order(df$date_time), ]
stopifnot(
  "NA values present"           = !anyNA(df),
  "grid is not strictly hourly" = all(as.numeric(diff(df$date_time), units = "hours") == 1),
  "duplicate timestamps"        = !any(duplicated(df$date_time))
)

# Cyclical encoding: raw hour 23 and hour 0 are adjacent in reality but
# maximally distant numerically. sin/cos pairs fix that.
dow <- match(df$day_of_week,
             c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
stopifnot("unrecognised day_of_week" = !anyNA(dow))
cyc <- function(x, p) cbind(sin(2 * pi * x / p), cos(2 * pi * x / p))

FEAT <- cbind(
  traffic    = df$traffic_volume,
  temp       = df$temp,
  rain       = log1p(df$rain_1h),      # zero-inflated with a long tail
  snow       = log1p(df$snow_1h),
  clouds     = df$clouds_all,
  is_holiday = df$is_holiday,
  cyc(df$hour,      24),
  cyc(dow - 1,       7),
  cyc(df$month - 1, 12)
)
colnames(FEAT) <- c("traffic", "temp", "rain", "snow", "clouds", "is_holiday",
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos")

TARGET     <- df$traffic_volume
N          <- nrow(df)
N_FEAT     <- ncol(FEAT)
TEST_START <- floor((1 - TEST_FRAC) * N) + 1L
POOL_END   <- TEST_START - 1L

cat(sprintf("rows = %d | %s -> %s | features = %d\n",
            N, format(min(df$date_time)), format(max(df$date_time)), N_FEAT))
cat(sprintf("tuning pool = 1..%d | TEST = %d..%d (locked, not used in this script)\n\n",
            POOL_END, TEST_START, N))

# =============================================================================
# 3. FOLDS
#
# gap = lookback + horizon - 1. Without it the first validation input window
# overlaps rows the model trained on, and val loss is partly train loss.
#
# NOTE: va_end always equals POOL_END regardless of gap, so the validation
# windows are IDENTICAL for every configuration. Only the training end shifts.
# Comparisons across the grid stay fair even when lookback varies.
# =============================================================================
make_folds <- function(cfg) {
  gap     <- cfg$lookback + cfg$horizon - 1L
  initial <- POOL_END - N_FOLDS * ASSESS - gap
  stopifnot("fold-1 training set too small" = initial > 4L * (cfg$lookback + cfg$horizon))
  f <- lapply(seq_len(N_FOLDS), function(i) {
    tr_end <- initial + (i - 1L) * ASSESS
    va_st  <- tr_end + gap + 1L
    list(train = 1:tr_end, val = va_st:(va_st + ASSESS - 1L))
  })
  stopifnot(
    "not anchored at row 1" = all(vapply(f, function(x) min(x$train) == 1L, logical(1))),
    "wrong purge gap"       = all(vapply(f, function(x) min(x$val) - max(x$train) - 1L == gap, logical(1))),
    "folds do not tile"     = max(f[[N_FOLDS]]$val) == POOL_END
  )
  f
}

# =============================================================================
# 4. WINDOWING AND SCALING
# =============================================================================
make_windows <- function(rows, cfg) {
  m  <- FEAT[rows, , drop = FALSE]
  ns <- length(rows) - cfg$lookback - cfg$horizon + 1L
  stopifnot("slice shorter than lookback + horizon" = ns > 0)
  x <- array(0, dim = c(ns, cfg$lookback, N_FEAT))
  for (i in seq_len(ns)) x[i, , ] <- m[i:(i + cfg$lookback - 1L), , drop = FALSE]
  list(x   = x,
       y   = TARGET[rows][(cfg$lookback + cfg$horizon):length(rows)],
       idx = rows[(cfg$lookback + cfg$horizon):length(rows)])
}

# z-score from TRAIN rows only — never the whole series.
make_scaler <- function(train_rows, cfg) {
  mu <- colMeans(FEAT[train_rows, , drop = FALSE])
  sd <- apply(FEAT[train_rows, , drop = FALSE], 2, stats::sd)
  sd[sd == 0] <- 1
  yt <- if (cfg$target_log) log1p(TARGET[train_rows]) else TARGET[train_rows]
  mu_y <- mean(yt)
  sd_y <- stats::sd(yt)
  list(
    x_apply = function(a) {
      for (j in seq_len(N_FEAT)) a[, , j] <- (a[, , j] - mu[j]) / sd[j]
      a
    },
    y_apply = function(v) ((if (cfg$target_log) log1p(v) else v) - mu_y) / sd_y,
    y_inv   = function(v) {
      z <- v * sd_y + mu_y
      pmax(0, if (cfg$target_log) expm1(z) else z)
    }
  )
}

# =============================================================================
# 5. MODEL
# =============================================================================
build_model <- function(cfg) {
  m <- keras_model_sequential(input_shape = c(cfg$lookback, N_FEAT))
  if (cfg$n_layers >= 2L)
    m <- m |> layer_lstm(units = cfg$units, dropout = cfg$dropout,
                         recurrent_dropout = 0, return_sequences = TRUE)
  m |>
    layer_lstm(units = cfg$units, dropout = cfg$dropout, recurrent_dropout = 0) |>
    layer_dense(units = cfg$horizon) |>
    compile(optimizer = optimizer_adam(learning_rate = cfg$learning_rate,
                                       beta_2   = cfg$beta_2,
                                       clipnorm = cfg$clipnorm),
            loss = cfg$loss, metrics = "mae")
}

# =============================================================================
# 6. METRICS — all on the ORIGINAL traffic_volume scale
#
# MASE denominator = in-sample seasonal-naive MAE on the TRAINING rows
# (Hyndman & Koehler 2006). It must not come from the evaluation set.
#   MASE < 1 -> better than "same hour last week";  = 1 -> equal;  > 1 -> worse.
# =============================================================================
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

# =============================================================================
# 7. ONE FOLD — fresh weights every call, so nothing leaks between folds
# =============================================================================
run_fold <- function(cfg, train_rows, eval_rows) {
  set_random_seed(cfg$seed)
  sc  <- make_scaler(train_rows, cfg)
  tr  <- make_windows(train_rows, cfg)
  ev  <- make_windows(eval_rows,  cfg)
  xtr <- sc$x_apply(tr$x)
  xev <- sc$x_apply(ev$x)

  model <- build_model(cfg)
  h <- model |> fit(
    xtr, sc$y_apply(tr$y),
    validation_data = list(xev, sc$y_apply(ev$y)),
    epochs = cfg$epochs, batch_size = cfg$batch_size,
    shuffle = TRUE, verbose = 2,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = cfg$patience_stop,
                              restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = cfg$lr_factor,
                                    patience = cfg$patience_lr, min_lr = 1e-5)
    )
  )

  pred <- sc$y_inv(as.numeric(predict(model, xev, verbose = 0)))
  out  <- list(metrics    = metrics(ev$y, pred, mase_scale(train_rows)),
               best_epoch = which.min(h$metrics$val_loss))
  rm(xtr, xev, tr, ev, model)
  gc(verbose = FALSE)   # free ~300 MB before the next fold
  out
}

# =============================================================================
# 8. GRID, RESUME, CSV
# =============================================================================
dir.create("results", showWarnings = FALSE)

cfg_key <- function(cfg) {
  paste(vapply(TUNABLE, function(k) as.character(cfg[[k]]), ""), collapse = "|")
}

as_cfg <- function(row) {
  cfg <- FIXED
  for (k in TUNABLE) cfg[[k]] <- row[[k]]
  cfg$lookback   <- as.integer(cfg$lookback)
  cfg$units      <- as.integer(cfg$units)
  cfg$n_layers   <- as.integer(cfg$n_layers)
  cfg$batch_size <- as.integer(cfg$batch_size)
  cfg$loss       <- as.character(cfg$loss)
  cfg$target_log <- as.logical(cfg$target_log)
  cfg
}

append_row <- function(row, path) {
  write.table(row, path, sep = ",", row.names = FALSE,
              col.names = !file.exists(path), append = file.exists(path), qmethod = "double")
}

done <- if (file.exists(OUT_PATH)) read.csv(OUT_PATH, stringsAsFactors = FALSE)$key else character(0)

TODO <- expand.grid(GRID, stringsAsFactors = FALSE)
TODO <- TODO[order(TODO$lookback, TODO$units, TODO$dropout), , drop = FALSE]  # cheapest first
if (is.finite(MAX_CONFIGS)) TODO <- TODO[seq_len(min(MAX_CONFIGS, nrow(TODO))), , drop = FALSE]

cat(sprintf("== %d configurations x %d folds = %d model fits ==\n",
            nrow(TODO), N_FOLDS, nrow(TODO) * N_FOLDS))
if (length(done)) cat(sprintf("   %d already in %s, will be skipped\n", length(done), OUT_PATH))
cat("\n")

# =============================================================================
# 9. RUN
# =============================================================================
times <- numeric(0)

for (i in seq_len(nrow(TODO))) {
  cfg <- as_cfg(TODO[i, , drop = FALSE])
  key <- cfg_key(cfg)

  if (key %in% done) {
    cat(sprintf("[%2d/%2d] %-34s skip (already done)\n", i, nrow(TODO), key))
    next
  }
  cat(sprintf("[%2d/%2d] %-34s ", i, nrow(TODO), key))
  flush.console()

  t0 <- Sys.time()
  res <- tryCatch({
    folds <- make_folds(cfg)
    lapply(seq_len(N_FOLDS), function(f) run_fold(cfg, folds[[f]]$train, folds[[f]]$val))
  }, error = function(e) {
    cat("FAILED:", conditionMessage(e), "\n")
    NULL
  })
  if (is.null(res)) next

  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  times <- c(times, secs)
  M <- do.call(rbind, lapply(res, function(r) r$metrics))

  row <- data.frame(key = key, timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                    stringsAsFactors = FALSE)
  for (k in TUNABLE) row[[k]] <- cfg[[k]]
  for (m in c("MAE", "RMSE", "MAPE", "sMAPE", "MASE")) {
    row[[paste0("mean_", m)]] <- round(mean(M[, m], na.rm = TRUE), 4)
    row[[paste0("sd_",   m)]] <- round(stats::sd(M[, m]), 4)
  }
  for (f in seq_len(N_FOLDS)) {
    row[[paste0("f", f, "_MAE")]]  <- round(M[f, "MAE"],  3)
    row[[paste0("f", f, "_MASE")]] <- round(M[f, "MASE"], 4)
  }
  row$best_epochs <- paste(vapply(res, function(r) r$best_epoch, integer(1)), collapse = ";")
  row$secs        <- round(secs, 1)

  append_row(row, OUT_PATH)   # written immediately — a crash never loses finished work

  eta_min <- mean(times) * (nrow(TODO) - i) / 60
  cat(sprintf("MASE %.4f (sd %.4f)  MAE %.1f  RMSE %.1f  [%.0fs, ETA %.0f min]\n",
              row$mean_MASE, row$sd_MASE, row$mean_MAE, row$mean_RMSE, secs, eta_min))
}

# =============================================================================
# 10. LEADERBOARD
# =============================================================================
r <- read.csv(OUT_PATH, stringsAsFactors = FALSE)
r <- r[order(r[[paste0("mean_", PRIMARY)]]), ]

cat(sprintf("\n== Leaderboard: all %d configs, ranked by mean %s ==\n", nrow(r), PRIMARY))
print(r[, c("lookback", "units", "n_layers", "dropout",
            "mean_MASE", "sd_MASE", "mean_MAE", "mean_RMSE", "mean_sMAPE", "secs")],
      row.names = FALSE, digits = 5)

best <- r[1, ]
cat(sprintf("\nBEST: lookback=%d units=%d n_layers=%d dropout=%.1f\n",
            best$lookback, best$units, best$n_layers, best$dropout))
cat(sprintf("      mean MASE %.4f (sd %.4f) | mean MAE %.2f | best epochs %s\n",
            best$mean_MASE, best$sd_MASE, best$mean_MAE, best$best_epochs))
cat(sprintf("\nFull results: %s\n", OUT_PATH))
cat("Next: put these values into lstm.R and run it once to get the TEST score.\n")