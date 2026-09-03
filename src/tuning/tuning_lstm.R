suppressPackageStartupMessages(library(keras3))

# 1. CONFIGURATION
CSV_PATH <- "data/processed/traffic_volume_processed.csv"
OUT_PATH <- "output/models/lstm/tuning_results.csv"
dir.create(dirname(OUT_PATH), recursive = TRUE, showWarnings = FALSE)

TEST_FRAC <- 0.15     # protocol — never tuned
N_FOLDS   <- 4L       # protocol — never tuned
ASSESS    <- 2190L    # protocol — 3-month validation window
SEASON_M  <- 168L     # weekly lag: differencing, seasonal-naive benchmark, MASE scale
D_ORDER   <- 1L       # seasonal differencing order, matched across all four models
PRIMARY   <- "MASE"   # metric used to rank configurations

MAX_CONFIGS <- Inf

# --- THE SEARCH SPACE -------------------------------------------------------
GRID <- list(
  lookback      = c(48L, 168L),
  units         = c(64L, 128L),
  n_layers      = c(1L, 2L),
  dropout       = c(0.0, 0.1),
  learning_rate = c(3e-4, 1e-3, 3e-3),
  batch_size    = c(64L),
  loss          = c("mse")
)
TUNABLE <- names(GRID)

FIXED <- list(
  horizon       = 1L,
  clipnorm      = 1.0,
  beta_2        = 0.999,
  epochs        = 150L,
  patience_stop = 10L,
  patience_lr   = 4L,
  lr_factor     = 0.5,
  seed          = 42L
)

# 2. DATA
cat("Loading data ...\n")
df <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
df <- df[order(df$date_time), ]
stopifnot(
  "NA values present"           = !anyNA(df),
  "grid is not strictly hourly" = all(as.numeric(diff(df$date_time), units = "hours") == 1),
  "duplicate timestamps"        = !any(duplicated(df$date_time))
)

# Seasonal differencing, D = 1 at lag 168, applied before anything reaches the
# model — the same diff() call the SARIMAX model uses, so all models receive an
# identically transformed target. The first 168 rows have no lag reference and
# are dropped. PREV keeps the lag anchor y_(t-168), which diff() discards but
# to_level() needs to turn a predicted change back into a volume.
y_full <- df$traffic_volume
TARGET <- diff(y_full, lag = SEASON_M, differences = D_ORDER)
df     <- df[-(1:(SEASON_M * D_ORDER)), , drop = FALSE]
rownames(df) <- NULL
LEVEL  <- df$traffic_volume
PREV   <- head(y_full, -(SEASON_M * D_ORDER))

# Six features, identical to lstm.R. The traffic channel carries the differenced
# series; the exogenous columns are left in levels.
FEAT <- cbind(
  traffic    = TARGET,
  temp       = df$temp,
  rain       = df$rain_1h,
  snow       = df$snow_1h,
  clouds     = df$clouds_all,
  is_holiday = df$is_holiday
)

N          <- nrow(df)
N_FEAT     <- ncol(FEAT)
TEST_START <- min(which(df$split == "test"))
POOL_END   <- TEST_START - 1L

cat(sprintf("rows = %d (%d dropped by differencing) | %s -> %s | features = %d (%s)\n",
            N, SEASON_M * D_ORDER, format(min(df$date_time)), format(max(df$date_time)),
            N_FEAT, paste(colnames(FEAT), collapse = ", ")))
cat(sprintf("differencing: D = %d at lag %d | mean %.2f | sd %.2f veh/h\n",
            D_ORDER, SEASON_M, mean(TARGET), stats::sd(TARGET)))
cat(sprintf("tuning pool = 1..%d | TEST = %d..%d (locked, not used in this script)\n\n",
            POOL_END, TEST_START, N))

# 3. FOLDS
#    gap = lookback + horizon - 1. Without it the first validation input window
#    overlaps rows the model trained on, and val loss is partly train loss.
#    va_end always equals POOL_END regardless of gap, so the validation windows
#    are IDENTICAL for every configuration and only the training end shifts —
#    comparisons stay fair even when look-back varies.
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

# 4. HELPERS
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

# Undo the differencing: y_t = y_(t-168) + predicted change.
to_level <- function(pred, idx) pmax(0, PREV[idx] + pred)

# Standardisation fitted on TRAINING rows only — never the whole series.
make_scaler <- function(train_rows, cfg) {
  mu <- colMeans(FEAT[train_rows, , drop = FALSE])
  sd <- apply(FEAT[train_rows, , drop = FALSE], 2, stats::sd)
  sd[sd == 0] <- 1
  mu_y <- mean(TARGET[train_rows])
  sd_y <- stats::sd(TARGET[train_rows])
  list(
    x_apply = function(a) {
      for (j in seq_len(N_FEAT)) a[, , j] <- (a[, , j] - mu[j]) / sd[j]
      a
    },
    y_apply = function(v) (v - mu_y) / sd_y,
    # a differenced value is legitimately negative, so it is not clipped here;
    # clipping happens once, in to_level(), after the level is reconstructed
    y_inv   = function(v) v * sd_y + mu_y
  )
}

build_model <- function(cfg) {
  m <- keras_model_sequential(input_shape = c(cfg$lookback, N_FEAT))
  if (cfg$n_layers >= 2L)
    m <- m |> layer_lstm(units = cfg$units, dropout = cfg$dropout,
                         recurrent_dropout = 0, return_sequences = TRUE)
  m |>
    layer_lstm(units = cfg$units, dropout = cfg$dropout, recurrent_dropout = 0) |>
    layer_dense(units = cfg$horizon) |>
    compile(optimizer = optimizer_adam(learning_rate = cfg$learning_rate,
                                       beta_2 = cfg$beta_2, clipnorm = cfg$clipnorm),
            loss = cfg$loss, metrics = "mae")
}

# 5. METRICS — all on the ORIGINAL traffic_volume scale, after integration.
#    MASE denominator = in-sample seasonal-naive MAE on the TRAINING rows
#    (Hyndman & Koehler 2006). It must not come from the evaluation set.
#    MASE < 1 -> better than "same hour last week"; = 1 -> equal; > 1 -> worse.
mase_scale <- function(train_rows, m = SEASON_M) {
  y <- LEVEL[train_rows]
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

# 6. ONE FOLD — fresh weights every call, so nothing leaks between folds
run_fold <- function(cfg, train_rows, eval_rows) {
  set_random_seed(cfg$seed)
  sc <- make_scaler(train_rows, cfg)
  tr <- make_windows(train_rows, cfg)
  ev <- make_windows(eval_rows, cfg)
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

  # The network predicts a change; integrate before scoring so the leaderboard
  # is in vehicles/h and comparable to the other three models.
  pred <- to_level(sc$y_inv(as.numeric(predict(model, xev, verbose = 0))), ev$idx)
  out  <- list(metrics    = metrics(LEVEL[ev$idx], pred, mase_scale(train_rows)),
               best_epoch = which.min(h$metrics$val_loss),
               n_epochs   = length(h$metrics$val_loss))
  rm(xtr, xev, tr, ev, model)
  gc(verbose = FALSE)   # free memory before the next fold
  out
}

# 7. GRID, RESUME, CSV
#    The key carries the differencing order, so a run under a different
#    differencing scheme can never be mistaken for one already completed.
cfg_key <- function(cfg) {
  paste(c(paste0("D", D_ORDER, "L", SEASON_M),
          vapply(TUNABLE, function(k) as.character(cfg[[k]]), "")), collapse = "|")
}

as_cfg <- function(row) {
  cfg <- FIXED
  for (k in TUNABLE) cfg[[k]] <- row[[k]]
  cfg$lookback   <- as.integer(cfg$lookback)
  cfg$units      <- as.integer(cfg$units)
  cfg$n_layers   <- as.integer(cfg$n_layers)
  cfg$batch_size <- as.integer(cfg$batch_size)
  cfg$loss       <- as.character(cfg$loss)
  cfg
}

append_row <- function(row, path) {
  write.table(row, path, sep = ",", row.names = FALSE,
              col.names = !file.exists(path), append = file.exists(path), qmethod = "double")
}

done <- if (file.exists(OUT_PATH)) read.csv(OUT_PATH, stringsAsFactors = FALSE)$key else character(0)

TODO <- expand.grid(GRID, stringsAsFactors = FALSE)
TODO <- TODO[order(TODO$lookback, TODO$units, TODO$n_layers, TODO$dropout), , drop = FALSE]
if (is.finite(MAX_CONFIGS)) TODO <- TODO[seq_len(min(MAX_CONFIGS, nrow(TODO))), , drop = FALSE]

cat(sprintf("== %d configurations x %d folds = %d model fits ==\n",
            nrow(TODO), N_FOLDS, nrow(TODO) * N_FOLDS))
if (length(done)) cat(sprintf("   %d already in %s, will be skipped\n", length(done), OUT_PATH))
cat("\n")

# 8. RUN
times <- numeric(0)

for (i in seq_len(nrow(TODO))) {
  cfg <- as_cfg(TODO[i, , drop = FALSE])
  key <- cfg_key(cfg)

  if (key %in% done) {
    cat(sprintf("[%2d/%2d] %-40s skip (already done)\n", i, nrow(TODO), key))
    next
  }
  cat(sprintf("[%2d/%2d] %-40s ", i, nrow(TODO), key))
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
                    D_order = D_ORDER, diff_lag = SEASON_M, stringsAsFactors = FALSE)
  for (k in TUNABLE) row[[k]] <- cfg[[k]]
  for (m in c("MAE", "RMSE", "MAPE", "sMAPE", "MASE")) {
    row[[paste0("mean_", m)]] <- round(mean(M[, m], na.rm = TRUE), 4)
    row[[paste0("sd_",   m)]] <- round(stats::sd(M[, m]), 4)
  }
  for (f in seq_len(N_FOLDS)) {
    row[[paste0("f", f, "_MAE")]]  <- round(M[f, "MAE"],  3)
    row[[paste0("f", f, "_MASE")]] <- round(M[f, "MASE"], 4)
  }
  be <- vapply(res, function(r) r$best_epoch, integer(1))
  row$best_epochs <- paste(be, collapse = ";")
  row$secs <- round(secs, 1)

  append_row(row, OUT_PATH)   # written immediately — a crash never loses finished work

  # If early stopping never fired, the epoch cap bound the run and the result
  # understates this configuration. Raise FIXED$epochs and re-run it.
  capped <- vapply(res, function(r) r$n_epochs >= cfg$epochs, logical(1))
  if (any(capped))
    cat(sprintf("        WARNING: %d/%d folds hit the %d-epoch cap — raise FIXED$epochs\n",
                sum(capped), N_FOLDS, cfg$epochs))

  eta_min <- mean(times) * (nrow(TODO) - i) / 60
  cat(sprintf("        MASE %.4f (sd %.4f)  MAE %.1f  RMSE %.1f  epochs %s  [%.0fs, ETA %.0f min]\n",
              row$mean_MASE, row$sd_MASE, row$mean_MAE, row$mean_RMSE, row$best_epochs, secs, eta_min))
}

# 9. LEADERBOARD
r <- read.csv(OUT_PATH, stringsAsFactors = FALSE)
if ("D_order" %in% names(r)) r <- r[r$D_order == D_ORDER & r$diff_lag == SEASON_M, , drop = FALSE]
r <- r[order(r[[paste0("mean_", PRIMARY)]]), ]

cat(sprintf("\n== Leaderboard: all %d configs, ranked by mean %s ==\n", nrow(r), PRIMARY))
print(r[, c("lookback", "units", "n_layers", "dropout",
            "mean_MASE", "sd_MASE", "mean_MAE", "mean_RMSE", "mean_sMAPE", "secs")],
      row.names = FALSE, digits = 5)

best <- r[1, ]
cat(sprintf("\nBEST: lookback=%d units=%d n_layers=%d dropout=%.1f lr=%g\n",
            best$lookback, best$units, best$n_layers, best$dropout, best$learning_rate))
cat(sprintf("      mean MASE %.4f (sd %.4f) | mean MAE %.2f | best epochs %s\n",
            best$mean_MASE, best$sd_MASE, best$mean_MAE, best$best_epochs))

# The standard error of a single configuration's mean. Configurations within
# one SE of the best are not distinguishable from each other; among those,
# prefer the smallest and cheapest (the parsimony rule).
se <- mean(r$sd_MASE, na.rm = TRUE) / sqrt(N_FOLDS)
tied <- sum(r$mean_MASE <= best$mean_MASE + se)
cat(sprintf("\nStandard error of a config mean: %.4f\n", se))
cat(sprintf("%d configuration(s) lie within one SE of the best — treat them as tied\n", tied))
cat("and pick the smallest/cheapest among them.\n")

cat(sprintf("\nFull results: %s\n", OUT_PATH))
cat("Next: fix the parameters that clearly won, open the next ones in GRID,\n")
cat("and run again. When the search is done, copy the winning values and\n")
cat("round(mean(best_epochs)) into lstm.R.\n")