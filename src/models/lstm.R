suppressPackageStartupMessages(library(keras3))
suppressPackageStartupMessages(library(ggplot2))

OUT <- "output/models/lstm"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# 1. CONFIGURATION  (winning configuration from tuning_results.csv)
CSV_PATH <- "data/processed/traffic_volume_processed.csv"

cfg <- list(
  lookback      = 168L,
  units         = 128L,
  n_layers      = 2L,
  dropout       = 0.0,
  learning_rate = 0.003,
  batch_size    = 64L,
  loss          = "mse",
  horizon       = 1L,
  clipnorm      = 1.0,
  beta_2        = 0.999,
  seed          = 42L
)

EPOCHS    <- 10L      # mean of the CV best_epochs for the winning config
SEASON_M  <- 168L     # weekly lag: differencing, seasonal-naive benchmark, MASE scale
D_ORDER   <- 1L       # seasonal differencing order, matched across all four models

# Forward forecast horizon: one week immediately after the last observation.
FORECAST_HOURS <- 168L

# 2. DATA
df <- read.csv(CSV_PATH, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")
df <- df[order(df$date_time), ]
stopifnot(
  "NA values present"           = !anyNA(df),
  "grid is not strictly hourly" = all(as.numeric(diff(df$date_time), units = "hours") == 1),
  "duplicate timestamps"        = !any(duplicated(df$date_time))
)

# Seasonal differencing, D = 1 at lag 168 — the same diff() call the SARIMAX
# model uses, so every model receives an identically transformed target. The
# first 168 rows have no lag reference and are dropped. PREV keeps the anchor
# y_(t-168), which diff() discards but to_level() needs.
y_full <- df$traffic_volume
split_full <- df$split
TARGET <- diff(y_full, lag = SEASON_M, differences = D_ORDER)
df     <- df[-(1:(SEASON_M * D_ORDER)), , drop = FALSE]
rownames(df) <- NULL
LEVEL  <- df$traffic_volume
PREV   <- head(y_full, -(SEASON_M * D_ORDER))

# Six features. The traffic channel carries the differenced series; the
# exogenous columns are left in levels.
FEATURE <- cbind(
  traffic    = TARGET,
  temp       = df$temp,
  rain       = df$rain_1h,
  snow       = df$snow_1h,
  clouds     = df$clouds_all,
  is_holiday = df$is_holiday
)

N_ROWS <- nrow(df)
N_FEATURE <- ncol(FEATURE)
# Take the test boundary from the shared split column so the LSTM is scored on
# exactly the rows the other models use.
TEST_START <- min(which(df$split == "test"))
POOL_END <- TEST_START - 1L

cat(sprintf("rows = %d (%d dropped by differencing) | features = %d (%s)\n",
            N_ROWS, SEASON_M * D_ORDER, N_FEATURE, paste(colnames(FEATURE), collapse = ", ")))
cat(sprintf("differencing: D = %d at lag %d | mean %.2f | sd %.2f veh/h\n",
            D_ORDER, SEASON_M, mean(TARGET), stats::sd(TARGET)))
cat(sprintf("TRAIN pool = 1..%d      (%s .. %s)\n", POOL_END,
            format(df$date_time[1], "%Y-%m-%d"), format(df$date_time[POOL_END], "%Y-%m-%d")))
cat(sprintf("TEST       = %d..%d (%s .. %s, %.1f%%)\n\n", TEST_START, N_ROWS,
            format(df$date_time[TEST_START], "%Y-%m-%d"),
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

# Undo the differencing: y_t = y_(t-168) + predicted change.
to_level <- function(pred, idx) pmax(0, PREV[idx] + pred)

# Standardisation fitted on TRAINING rows only.
make_scaler <- function(train_rows) {
  mu <- colMeans(FEATURE[train_rows, , drop = FALSE])
  sd <- apply(FEATURE[train_rows, , drop = FALSE], 2, stats::sd)
  sd[sd == 0] <- 1
  mu_y <- mean(TARGET[train_rows])
  sd_y <- stats::sd(TARGET[train_rows])
  list(
    x_apply = function(a) {
      for (j in seq_len(N_FEATURE)) a[, , j] <- (a[, , j] - mu[j]) / sd[j]
      a
    },
    y_apply = function(v) (v - mu_y) / sd_y,
    # a differenced value is legitimately negative, so it is not clipped here;
    # clipping happens once, in to_level(), after the level is reconstructed
    y_inv = function(v) v * sd_y + mu_y
  )
}

# MASE denominator: in-sample seasonal-naive MAE on TRAINING rows, original scale.
mase_scale <- function(train_rows, m = SEASON_M) {
  mean(abs(train_rows[(m + 1):length(train_rows)] - train_rows[1:(length(train_rows) - m)]))
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

# 4. TRAIN THE MODEL  — once, on the training pool only
set_random_seed(cfg$seed)

train_rows <- 1:POOL_END
sc <- make_scaler(train_rows)      # this scaler is reused everywhere below
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
  callbacks = list(callback_reduce_lr_on_plateau(monitor = "loss", factor = 0.5, patience = 4, min_lr = 1e-5))
)

# 5. TEST EVALUATION — the test set is touched exactly once, only to predict
#    The model outputs a change; it is integrated back to a volume before any
#    metric is computed, so every figure below is in vehicles/h and stays
#    comparable to the other three models.
te <- make_windows(TEST_START:N_ROWS)
pred <- to_level(sc$y_inv(as.numeric(predict(model, sc$x_apply(te$x), verbose = 0))), te$idx)
actual <- LEVEL[te$idx]
naive <- PREV[te$idx]      # seasonal naive = same hour last week = zero change
scale <- mase_scale(y_full[split_full == "train"])

out <- rbind(LSTM = metrics(actual, pred, scale), `seasonal-naive` = metrics(actual, naive, scale))

cat("\n===================== FINAL TEST RESULTS =====================\n")
cat(sprintf("period: %s .. %s   (%d hours)\n",
            format(df$date_time[min(te$idx)], "%Y-%m-%d"),
            format(df$date_time[max(te$idx)], "%Y-%m-%d"), length(actual)))
print(round(out, 4))
cat(sprintf("\nimprovement over seasonal-naive: MAE %.1f%%  RMSE %.1f%%\n",
            100 * (1 - out["LSTM", "MAE"] / out["seasonal-naive", "MAE"]),
            100 * (1 - out["LSTM", "RMSE"] / out["seasonal-naive", "RMSE"])))
if (out["LSTM", "zeros"] > 0)
  cat(sprintf("note: %d test rows had traffic_volume == 0, excluded from MAPE only.\n", out["LSTM", "zeros"]))
cat("==============================================================\n")

res <- data.frame(model = rownames(out), out, row.names = NULL)
for (k in names(cfg)) res[[k]] <- cfg[[k]]
res$epochs <- EPOCHS
res$D_order <- D_ORDER
res$diff_lag <- SEASON_M
write.csv(res, file.path(OUT, "final_test_metrics.csv"), row.names = FALSE)

test_df <- data.frame(date_time = df$date_time[te$idx],
                      actual = actual,
                      predicted = round(pred, 2),
                      naive = naive)
write.csv(transform(test_df, date_time = format(date_time, "%Y-%m-%d %H:%M:%S")),
          file.path(OUT, "final_test_predictions.csv"), row.names = FALSE)

# 6. ONE-WEEK FORWARD FORECAST  — SAME model, SAME scaler
#
# The horizon lies beyond the end of the record, so no actual values exist and
# no accuracy can be measured for it. Two inputs must be supplied:
#
#   is_holiday                          — known exactly in advance.
#   Weather (temp, rain, snow, clouds)  — unknown, so set to climatology:
#     the historical mean for that month and hour across the observed record.
#     This makes the output a forecast under average seasonal weather, which is
#     an assumption, not a weather prediction.
#
# The forecast is recursive: each predicted change is appended to the input
# window and used to predict the next, so error compounds with horizon. The
# integration anchor y_(t-168) is an OBSERVED volume for the first 168 steps,
# so within a one-week horizon no drift accumulates in the level itself.
FORECAST_START <- max(df$date_time) + 3600
future_dates <- seq(FORECAST_START, by = "hour", length.out = FORECAST_HOURS)
H <- length(future_dates)

flt <- as.POSIXlt(future_dates)
future_hour <- flt$hour
future_month <- flt$mon + 1L
future_day <- ((flt$wday + 6L) %% 7L) + 1L        # Monday = 1 ... Sunday = 7

cat(sprintf("\nForecast horizon: %d hours (%s .. %s)\n", H,
            format(min(future_dates), "%Y-%m-%d %H:%M"),
            format(max(future_dates), "%Y-%m-%d %H:%M")))

# US federal holidays; none fall in 1-7 Oct 2018, but the list keeps the code
# correct if the horizon is extended.
future_holidays <- as.Date(c("2018-10-08",   # Columbus Day
                             "2018-11-12",   # Veterans Day (observed)
                             "2018-11-22",   # Thanksgiving
                             "2018-12-25"))  # Christmas Day
future_is_holiday <- as.integer(as.Date(future_dates) %in% future_holidays)
cat(sprintf("Holiday hours in horizon: %d\n", sum(future_is_holiday)))

# Climatology by month x hour, in the same units FEATURE stores.
obs_lt <- as.POSIXlt(df$date_time)
clim <- data.frame(key = paste(obs_lt$mon + 1L, obs_lt$hour),
                   temp = df$temp, rain = df$rain_1h,
                   snow = df$snow_1h, clouds = df$clouds_all)
clim_mean <- aggregate(cbind(temp, rain, snow, clouds) ~ key, data = clim, FUN = mean)
mi <- match(paste(future_month, future_hour), clim_mean$key)
stopifnot("missing climatology for some month/hour" = !anyNA(mi))

FUTURE_FEATURE <- cbind(
  traffic    = rep(NA_real_, H),          # filled in recursively below
  temp       = clim_mean$temp[mi],
  rain       = clim_mean$rain[mi],
  snow       = clim_mean$snow[mi],
  clouds     = clim_mean$clouds[mi],
  is_holiday = future_is_holiday
)
colnames(FUTURE_FEATURE) <- colnames(FEATURE)

# Scale future rows and the seed window with the SAME scaler used for training.
future_scaled <- sc$x_apply(array(FUTURE_FEATURE, dim = c(1, H, N_FEATURE)))[1, , ]
seed_rows     <- (N_ROWS - cfg$lookback + 1L):N_ROWS      # last lookback observed hours
window        <- sc$x_apply(array(FEATURE[seed_rows, ],
                                  dim = c(1, cfg$lookback, N_FEATURE)))[1, , ]

target_col   <- which(colnames(FEATURE) == "traffic")
preds_scaled <- numeric(H)

cat(sprintf("Forecasting %d hours recursively with the trained model...\n", H))
pb <- txtProgressBar(min = 0, max = H, style = 3)
for (i in seq_len(H)) {
  p <- as.numeric(predict(model, array(window, dim = c(1, cfg$lookback, N_FEATURE)),
                          verbose = 0))
  preds_scaled[i] <- p
  nxt <- future_scaled[i, ]
  nxt[target_col] <- p                                   # prediction feeds forward
  window <- rbind(window[-1, , drop = FALSE], nxt)
  if (i %% 12 == 0 || i == H) setTxtProgressBar(pb, i)
}
close(pb)

# Integrate: each predicted change is added to the volume 168 hours earlier.
# Past 168 h the anchor would itself be a prediction; within one week it is not.
deltas <- sc$y_inv(preds_scaled)
lev <- c(LEVEL, numeric(H))
for (i in seq_len(H)) lev[N_ROWS + i] <- max(0, lev[N_ROWS + i - SEASON_M] + deltas[i])
forecast_values <- lev[(N_ROWS + 1L):(N_ROWS + H)]

forecast_df <- data.frame(
  date_time  = future_dates,
  forecast   = round(forecast_values, 1),
  day_of_week = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")[future_day],
  hour       = future_hour,
  is_holiday = future_is_holiday
)
write.csv(transform(forecast_df, date_time = format(date_time, "%Y-%m-%d %H:%M:%S")),
          file.path(OUT, "lstm_future_forecast.csv"), row.names = FALSE)

cat("\n==== One-week forward forecast summary ====\n")
cat(sprintf("mean %.0f | min %.0f | max %.0f vehicles/h\n",
            mean(forecast_values), min(forecast_values), max(forecast_values)))
cat(sprintf("peak at %s (%.0f veh/h)\n",
            format(future_dates[which.max(forecast_values)], "%a %d %b %H:%M"),
            max(forecast_values)))
cat(sprintf("mean predicted week-on-week change %+.1f veh/h (range %+.0f .. %+.0f)\n",
            mean(deltas), min(deltas), max(deltas)))
dtype <- ifelse(forecast_df$is_holiday == 1, "Holiday",
                ifelse(future_day >= 6L, "Weekend", "Weekday"))
print(data.frame(type  = names(tapply(forecast_df$forecast, dtype, mean)),
                 mean  = round(as.numeric(tapply(forecast_df$forecast, dtype, mean))),
                 hours = as.integer(table(dtype))))
cat("NOTE: no actual values exist for this period, so no accuracy is reported.\n")

# 7. FIGURES  — IEEE single column (3.5 in), 300 dpi.
COL_A <- "#1F6FB2"   # actual / LSTM     — solid
COL_B <- "#D1682A"   # predicted / naive — dashed
W <- 3.5
DPI <- 300

thm <- theme_minimal(base_size = 8, base_family = "serif") +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_line(linewidth = 0.2, colour = "grey88"),
        axis.title  = element_text(size = 8),
        axis.text   = element_text(size = 7, colour = "grey25"),
        legend.position = "top", legend.title = element_blank(),
        legend.key.width = unit(16, "pt"), legend.key.height = unit(8, "pt"),
        legend.margin = margin(0, 0, 0, 0), legend.box.margin = margin(0, 0, -6, 0),
        legend.text = element_text(size = 7),
        plot.margin = margin(2, 4, 2, 2))

test_df$err_lstm  <- abs(test_df$actual - test_df$predicted)
test_df$err_naive <- abs(test_df$actual - test_df$naive)
test_df$hr        <- as.integer(format(test_df$date_time, "%H"))
test_df$dow       <- weekdays(test_df$date_time)

## --- Fig 1: actual vs predicted, one representative test week --------------
mon <- which(test_df$dow == "Monday" & test_df$hr == 0L)
s   <- mon[which.min(abs(mon - nrow(test_df) / 2))]
wk  <- test_df[s:(s + 167L), ]
d1  <- rbind(data.frame(t = wk$date_time, v = wk$actual,    series = "Actual"),
             data.frame(t = wk$date_time, v = wk$predicted, series = "LSTM forecast"))
d1$series <- factor(d1$series, levels = c("Actual", "LSTM forecast"))

ggsave(file.path(OUT, "fig1_test_week.png"),
  ggplot(d1, aes(t, v, colour = series, linetype = series)) +
    geom_line(linewidth = 0.45) +
    scale_colour_manual(values = c("Actual" = COL_A, "LSTM forecast" = COL_B)) +
    scale_linetype_manual(values = c("Actual" = "solid", "LSTM forecast" = "22")) +
    scale_x_datetime(date_breaks = "1 day", date_labels = "%a") +
    labs(x = NULL, y = "Traffic volume (vehicles/h)") + thm,
  width = W, height = 2.1, dpi = DPI)

## --- Fig 2: cumulative distribution of absolute error ----------------------
d2 <- rbind(data.frame(e = test_df$err_lstm,  series = "LSTM"),
            data.frame(e = test_df$err_naive, series = "Seasonal naive"))
d2$series <- factor(d2$series, levels = c("LSTM", "Seasonal naive"))

ggsave(file.path(OUT, "fig2_error_ecdf.png"),
  ggplot(d2, aes(e, colour = series, linetype = series)) +
    stat_ecdf(linewidth = 0.45, pad = FALSE) +
    scale_colour_manual(values = c("LSTM" = COL_A, "Seasonal naive" = COL_B)) +
    scale_linetype_manual(values = c("LSTM" = "solid", "Seasonal naive" = "22")) +
    scale_y_continuous(labels = function(x) paste0(x * 100, "%"), breaks = seq(0, 1, .25)) +
    coord_cartesian(xlim = c(0, as.numeric(quantile(test_df$err_naive, 0.98)))) +
    labs(x = "Absolute forecast error (vehicles/h)", y = "Cumulative % of test hours") + thm,
  width = W, height = 2.0, dpi = DPI)

## --- Fig 3: mean absolute error by hour of day -----------------------------
d3 <- rbind(
  data.frame(hour = 0:23, mae = as.numeric(tapply(test_df$err_lstm,  test_df$hr, mean)), series = "LSTM"),
  data.frame(hour = 0:23, mae = as.numeric(tapply(test_df$err_naive, test_df$hr, mean)), series = "Seasonal naive"))
d3$series <- factor(d3$series, levels = c("LSTM", "Seasonal naive"))

ggsave(file.path(OUT, "fig3_error_by_hour.png"),
  ggplot(d3, aes(hour, mae, colour = series, linetype = series)) +
    geom_line(linewidth = 0.45) +
    scale_colour_manual(values = c("LSTM" = COL_A, "Seasonal naive" = COL_B)) +
    scale_linetype_manual(values = c("LSTM" = "solid", "Seasonal naive" = "22")) +
    scale_x_continuous(breaks = seq(0, 21, 3), labels = sprintf("%02d", seq(0, 21, 3))) +
    scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .08))) +
    labs(x = "Hour of day", y = "Mean absolute error (vehicles/h)") + thm,
  width = W, height = 2.0, dpi = DPI)

## --- Fig 4: observed tail + one-week forward forecast ----------------------
tail_obs <- df[df$date_time >= max(df$date_time) - 13 * 86400, ]
d4 <- rbind(data.frame(t = tail_obs$date_time, v = tail_obs$traffic_volume, series = "Observed"),
            data.frame(t = forecast_df$date_time, v = forecast_df$forecast, series = "Forecast"))
d4$series <- factor(d4$series, levels = c("Observed", "Forecast"))

ggsave(file.path(OUT, "fig4_future_forecast.png"),
  ggplot(d4, aes(t, v, colour = series, linetype = series)) +
    geom_line(linewidth = 0.4) +
    geom_vline(xintercept = as.numeric(FORECAST_START), linewidth = 0.3,
               linetype = "dotted", colour = "grey45") +
    scale_colour_manual(values = c("Observed" = COL_A, "Forecast" = COL_B)) +
    scale_linetype_manual(values = c("Observed" = "solid", "Forecast" = "22")) +
    scale_x_datetime(date_breaks = "4 days", date_labels = "%d %b") +
    labs(x = NULL, y = "Traffic volume (vehicles/h)") + thm,
  width = W, height = 2.1, dpi = DPI)

# 8. FIGURE FACTS — the numbers to quote when describing the figures.
#    Do not state a value in the report that is not printed here.
mh <- tapply(test_df$err_lstm, test_df$hr, mean)
nh <- tapply(test_df$err_naive, test_df$hr, mean)
cat("\n=============== FIGURE FACTS ===============\n")
cat(sprintf("Differencing: D = %d at lag %d\n", D_ORDER, SEASON_M))
cat(sprintf("Fig 1 week shown : %s .. %s\n",
            format(min(wk$date_time), "%d %b %Y"), format(max(wk$date_time), "%d %b %Y")))
cat(sprintf("  actual peak %.0f | actual min %.0f | MAE in this week %.1f\n",
            max(wk$actual), min(wk$actual), mean(abs(wk$actual - wk$predicted))))
cat("Fig 2 absolute-error percentiles\n")
for (q in c(.5, .9, .95, .99))
  cat(sprintf("  p%02.0f  LSTM %7.1f | naive %7.1f\n", q * 100,
              quantile(test_df$err_lstm, q), quantile(test_df$err_naive, q)))
for (thr in c(100, 200, 500))
  cat(sprintf("  %% hours error < %3d : LSTM %5.1f%% | naive %5.1f%%\n", thr,
              100 * mean(test_df$err_lstm < thr), 100 * mean(test_df$err_naive < thr)))
cat("Fig 3 error by hour\n")
cat(sprintf("  LSTM  worst %02d:00 (%.1f) | best %02d:00 (%.1f)\n",
            as.integer(names(which.max(mh))), max(mh),
            as.integer(names(which.min(mh))), min(mh)))
cat(sprintf("  LSTM beats naive in %d of 24 hours\n", sum(mh < nh)))
cat("Fig 4 forecast week\n")
cat(sprintf("  weekday mean %.0f | weekend mean %.0f\n",
            mean(forecast_df$forecast[future_day <= 5L]),
            mean(forecast_df$forecast[future_day >= 6L])))
cat(sprintf("  last observed week mean %.0f | forecast week mean %.0f\n",
            mean(tail(df$traffic_volume, 168)), mean(forecast_df$forecast)))
cat("============================================\n")

cat(sprintf("\nSaved to %s\n", OUT))