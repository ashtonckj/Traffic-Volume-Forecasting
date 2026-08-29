# ============================================================
# Shared setup for the LSTM scripts.
#
# tuning_lstm.R, lstm.R, and lstm_forecast.R all need the same feature
# engineering, the same split boundaries, the same scaler, and the same
# sequence builder. Keeping them here means the three scripts cannot
# silently drift apart -- if the split moves, it moves for all of them.
#
# Sourced from the project root:  source("src/models/lstm_common.R")
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
})

PROCESSED_PATH <- "data/processed/traffic_volume_processed.csv"
RESULTS_DIR    <- "results"

# ---- Split boundaries ----
# Chronological, matching the rest of the group's models so every method is
# scored on the identical test period:
#   TRAIN      2015-10-01 .. 2018-03-31   (21,912 h, ~30 months)
#   VALIDATION 2018-04-01 .. 2018-06-30   ( 2,184 h,  3 months)
#   TEST       2018-07-01 .. 2018-09-30   ( 2,208 h,  3 months)
#   FORECAST   2018-10-01 .. 2018-12-31   ( 2,208 h,  3 months, unobserved)
TRAIN_END     <- as.POSIXct("2018-03-31 23:00:00", tz = "UTC")
VAL_END       <- as.POSIXct("2018-06-30 23:00:00", tz = "UTC")
FORECAST_START <- as.POSIXct("2018-10-01 00:00:00", tz = "UTC")
FORECAST_END   <- as.POSIXct("2018-12-31 23:00:00", tz = "UTC")

TARGET_COL <- "traffic_volume"

FEATURE_COLS <- c("temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday",
                  "hour_sin", "hour_cos",
                  "day_of_week_num_sin", "day_of_week_num_cos",
                  "month_sin", "month_cos",
                  "traffic_volume")

WEEKDAY_ORDER <- c("Sunday", "Monday", "Tuesday", "Wednesday",
                   "Thursday", "Friday", "Saturday")


# ---- Load the processed dataset ----
load_processed <- function(path = PROCESSED_PATH) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  df <- df %>% arrange(date_time)

  # The hourly grid must be complete and gap-free: a sliding window assumes
  # row i-1 is exactly one hour before row i, and a missing hour would
  # silently shift every window that spans it.
  steps <- as.numeric(diff(df$date_time), units = "hours")
  if (any(steps != 1)) {
    stop(sum(steps != 1), " break(s) in the hourly grid. Re-run preprocessing.R.")
  }
  df
}


# ---- Cyclical encoding ----
# hour, day_of_week, and month wrap around: hour 23 is adjacent to hour 0,
# December is adjacent to January. Fed in as raw integers, the network would
# read 23 -> 0 as a jump of 23 rather than a step of 1. Each is therefore
# mapped onto a circle via a sin/cos pair, so adjacent values stay adjacent.
add_cyclical <- function(df, col, period) {
  df[[paste0(col, "_sin")]] <- sin(2 * pi * df[[col]] / period)
  df[[paste0(col, "_cos")]] <- cos(2 * pi * df[[col]] / period)
  df
}

engineer_features <- function(df) {
  df$day_of_week_num <- match(as.character(df$day_of_week), WEEKDAY_ORDER) - 1
  # wday(label = TRUE) returns localized names, so a non-English R locale
  # would produce all-NA here and poison every downstream feature silently.
  if (any(is.na(df$day_of_week_num))) {
    stop("Unrecognized day_of_week values: ",
         paste(unique(df$day_of_week[is.na(df$day_of_week_num)]), collapse = ", "),
         ". Expected English weekday names.")
  }

  df <- add_cyclical(df, "hour", 24)
  df <- add_cyclical(df, "day_of_week_num", 7)
  df <- add_cyclical(df, "month", 12)
  df
}


# ---- Min-max scaler, fitted on training rows only ----
# Fitting on the full series would leak the range of unseen future data into
# training, so the parameters come from the training rows and are then
# applied unchanged to validation, test, and forecast inputs. Values outside
# the training range map outside [0, 1], which is expected and harmless.
fit_scaler <- function(train_df) {
  lapply(train_df, function(col) list(min = min(col), max = max(col)))
}

apply_scale <- function(data, params) {
  scaled <- data
  for (col in names(params)) {
    rng <- params[[col]]$max - params[[col]]$min
    if (rng == 0) rng <- 1  # guard against a constant column
    scaled[[col]] <- (data[[col]] - params[[col]]$min) / rng
  }
  scaled
}

inverse_scale_target <- function(x, params, col = TARGET_COL) {
  x * (params[[col]]$max - params[[col]]$min) + params[[col]]$min
}


# ---- Sliding-window sequence builder ----
# `context` is the last look_back rows of the immediately preceding split.
# Without it, windowing a split in isolation consumes its first look_back
# rows as warm-up and produces no prediction for them -- with look_back =
# 168 that silently drops the first week of the test set, so the LSTM would
# be scored on fewer hours than SARIMA, Holt-Winters, and Seasonal Naive and
# the comparison table would not be like-for-like. The context rows are
# strictly past observations that would already be in hand at forecast time,
# so supplying them is not leakage.
make_sequences <- function(data, look_back, target_col = TARGET_COL, context = NULL) {
  if (!is.null(context)) {
    stopifnot(nrow(context) == look_back, identical(names(context), names(data)))
    data <- rbind(context, data)
  }

  x_mat <- as.matrix(data)
  n_obs <- nrow(x_mat) - look_back
  if (n_obs < 1) stop("Not enough rows (", nrow(x_mat), ") for look_back = ", look_back)
  n_feat <- ncol(x_mat)

  x_arr <- array(NA_real_, dim = c(n_obs, look_back, n_feat))
  y_vec <- numeric(n_obs)

  for (i in seq_len(n_obs)) {
    x_arr[i, , ] <- x_mat[i:(i + look_back - 1), ]
    y_vec[i] <- x_mat[i + look_back, target_col]
  }
  list(x = x_arr, y = y_vec)
}


# ---- Accuracy measures ----
# MAPE divides by the actual value, so any zero-traffic hour would return
# Inf and destroy the mean. Those hours are excluded and the count reported,
# rather than letting a single 3 a.m. reading decide the headline metric.
forecast_accuracy <- function(actual, pred) {
  err <- actual - pred
  nonzero <- actual != 0
  list(
    MAE           = mean(abs(err)),
    MSE           = mean(err^2),
    RMSE          = sqrt(mean(err^2)),
    MAPE          = mean(abs(err[nonzero] / actual[nonzero])) * 100,
    n             = length(actual),
    n_zero_actual = sum(!nonzero)
  )
}
