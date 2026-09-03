# ============================================================
# Three-Month Forward Forecast (2018-10-01 .. 2018-12-31)
#
# Continues on from lstm.R: run from the project root, e.g.
#   Rscript src/models/lstm_forecast.R
# `source("lstm.R")` below re-runs it, so this script inherits df, FEATURE,
# TARGET, cfg, N_FEATURE, N_ROWS, cyc(), make_scaler(), make_windows() --
# nothing is redefined or duplicated.
#
# lstm.R measures how accurate the model is on data it has never seen but
# which was actually observed. This script does the thing the model is
# ultimately for: forecasting beyond the end of the dataset, where no actual
# values exist to compare against.
#
# Two problems have to be solved to get there.
#
# 1. Which model to forecast with. lstm.R's model is trained only on the
#    train pool (1:POOL_END), because the tail was held out for an honest
#    test. Once that test has done its job, throwing away that history would
#    be wasteful, so the model is refit on ALL observed rows (1:N_ROWS)
#    using the same architecture and epoch count -- standard practice once
#    model selection is finished.
#
# 2. What to feed it for weather. temp, rain, snow, and clouds do not exist
#    for October-December 2018. Calendar features and holidays do, because
#    they are deterministic. The weather inputs are filled with climatology:
#    the historical mean for that month-and-hour across the observed record
#    (rain/snow averaged in the same log1p space FEATURE uses). This is an
#    explicit assumption of average seasonal weather, not a weather
#    forecast, and it's what makes the output a scenario projection rather
#    than a prediction.
# ============================================================
source("src/models/lstm.R")

suppressPackageStartupMessages(library(ggplot2))

FORECAST_START <- as.POSIXct("2018-10-01 00:00:00", tz = "UTC")
FORECAST_END   <- as.POSIXct("2018-12-31 23:00:00", tz = "UTC")

# 7. BUILD THE FUTURE FEATURE FRAME
# Everything here is known in advance without any forecasting: the calendar
# repeats, and US federal holidays are fixed by rule.
future_dates <- seq(FORECAST_START, FORECAST_END, by = "hour")
H <- length(future_dates)
cat(sprintf("\nForecast horizon: %d hours (%s .. %s)\n", H,
            as.character(FORECAST_START), as.character(FORECAST_END)))

future_lt        <- as.POSIXlt(future_dates)
future_hour      <- future_lt$hour
future_month     <- future_lt$mon + 1L
# wday: Sunday = 0 .. Saturday = 6 -> remap to Monday = 0 .. Sunday = 6,
# the same convention lstm.R uses for day_num - 1 in cyc().
future_day_num0  <- (future_lt$wday + 6) %% 7

# Federal holidays falling inside the horizon, following the same
# observance rule the dataset itself uses (Columbus Day = 2nd Monday of
# October, Veterans Day = Nov 11 or the Monday after when it lands on a
# Sunday, Thanksgiving = 4th Thursday of November, Christmas Day = Dec 25).
future_holidays <- as.Date(c(
  "2018-10-08",  # Columbus Day
  "2018-11-12",  # Veterans Day (observed; Nov 11 2018 is a Sunday)
  "2018-11-22",  # Thanksgiving Day
  "2018-12-25"   # Christmas Day
))
future_is_holiday <- as.integer(as.Date(future_dates) %in% future_holidays)
cat(sprintf("Holiday hours in horizon: %d (%d days)\n", sum(future_is_holiday), length(future_holidays)))

# Climatological weather: mean of each weather variable by month and hour
# across the observed record, in the same units FEATURE already stores them
# in (rain/snow are log1p'd there, so they're averaged in log1p space here
# too). Means rather than medians, because the median of rain/snow is zero
# for nearly every hour, which would encode "it never rains" instead of
# "average precipitation".
clim <- data.frame(
  key    = paste(df$month, df$hour),
  temp   = df$temp,
  rain   = log1p(df$rain_1h),
  snow   = log1p(df$snow_1h),
  clouds = df$clouds_all
)
clim_mean <- aggregate(cbind(temp, rain, snow, clouds) ~ key, data = clim, FUN = mean)
match_idx <- match(paste(future_month, future_hour), clim_mean$key)
stopifnot("missing climatology for some future month/hour combo" = !anyNA(match_idx))

FUTURE_FEATURE <- cbind(
  traffic    = rep(NA_real_, H),   # unknown -- filled in step by step below
  temp       = clim_mean$temp[match_idx],
  rain       = clim_mean$rain[match_idx],
  snow       = clim_mean$snow[match_idx],
  clouds     = clim_mean$clouds[match_idx],
  is_holiday = future_is_holiday,
  cyc(future_hour, 24),
  cyc(future_day_num0, 7),
  cyc(future_month - 1, 12)
)
colnames(FUTURE_FEATURE) <- colnames(FEATURE)

# 8. REFIT ON ALL OBSERVED DATA -- same fixed epochs, no validation, no
# early stopping, for the same reason as lstm.R section 4: the stopping
# point already came from tuning on a genuine validation set.
set_random_seed(cfg$seed)

sc_full <- make_scaler(1:N_ROWS)
tr_full <- make_windows(1:N_ROWS)

cat(sprintf("\nRefitting on all %d observed windows for %d epochs...\n", dim(tr_full$x)[1], EPOCHS))

model_full <- keras_model_sequential(input_shape = c(cfg$lookback, N_FEATURE))
if (cfg$n_layers >= 2L)
  model_full <- model_full %>% layer_lstm(units = cfg$units, dropout = cfg$dropout, recurrent_dropout = 0, return_sequences = TRUE)
model_full <- model_full %>%
  layer_lstm(units = cfg$units, dropout = cfg$dropout, recurrent_dropout = 0) %>%
  layer_dense(units = cfg$horizon) %>%
  compile(optimizer = optimizer_adam(learning_rate = cfg$learning_rate, beta_2 = cfg$beta_2, clipnorm = cfg$clipnorm),
          loss = cfg$loss,
          metrics = "mae")

model_full %>% fit(
  sc_full$x_apply(tr_full$x), sc_full$y_apply(tr_full$y),
  epochs = EPOCHS, batch_size = cfg$batch_size,
  shuffle = TRUE, verbose = 2,
  callbacks = list(callback_reduce_lr_on_plateau(monitor = "loss", factor = 0.5, patience = 4, min_lr = 1e-5))
)

# 9. RECURSIVE MULTI-STEP FORECAST
# One hour at a time: predict hour t+1 from the last `lookback` hours,
# append that prediction (with hour t+1's known calendar/holiday features
# and climatological weather) to the window, drop the oldest hour, repeat.
# Each prediction becomes an input to the next, which is why error
# accumulates with horizon and why the later weeks should be read as a
# seasonal profile rather than an hour-accurate forecast.
future_scaled <- sc_full$x_apply(array(FUTURE_FEATURE, dim = c(1, H, N_FEATURE)))[1, , ]
window <- sc_full$x_apply(array(FEATURE[(N_ROWS - cfg$lookback + 1):N_ROWS, ], dim = c(1, cfg$lookback, N_FEATURE)))[1, , ]

target_col   <- which(colnames(FEATURE) == "traffic")
preds_scaled <- numeric(H)

cat(sprintf("\nForecasting %d hours recursively...\n", H))
pb <- txtProgressBar(min = 0, max = H, style = 3)
for (i in seq_len(H)) {
  x <- array(window, dim = c(1, cfg$lookback, N_FEATURE))
  p <- as.numeric(predict(model_full, x, verbose = 0))
  preds_scaled[i] <- p

  next_row <- future_scaled[i, ]
  next_row[target_col] <- p   # the prediction feeds itself forward
  window <- rbind(window[-1, , drop = FALSE], next_row)

  if (i %% 24 == 0 || i == H) setTxtProgressBar(pb, i)
}
close(pb)

forecast_values <- sc_full$y_inv(preds_scaled)   # y_inv already clamps to >= 0

forecast_df <- data.frame(
  date_time  = future_dates,
  forecast   = round(forecast_values),
  is_holiday = future_is_holiday,
  hour       = future_hour,
  day_num0   = future_day_num0   # Monday = 0 .. Sunday = 6
)

# Formatted to a fixed string first: write.csv() drops "00:00:00" from
# midnight rows only, which makes the column unparseable as a single format
# downstream.
write.csv(transform(forecast_df, date_time = format(date_time, "%Y-%m-%d %H:%M:%S")),
          "output/models/lstm/lstm_future_forecast.csv", row.names = FALSE)

# 10. SUMMARY OF THE FORECAST
cat("\n==== Forecast summary (2018-10-01 .. 2018-12-31) ====\n")
cat(sprintf("Mean: %.0f  Min: %.0f  Max: %.0f vehicles/h\n", mean(forecast_values), min(forecast_values), max(forecast_values)))

cat("\nMonthly mean forecast vs same months in the observed record:\n")
q4 <- df$month %in% c(10, 11, 12)
obs_monthly <- tapply(df$traffic_volume[q4], df$month[q4], function(x) round(mean(x)))
fc_month    <- as.POSIXlt(forecast_df$date_time)$mon + 1L
fc_monthly  <- tapply(forecast_df$forecast, fc_month, function(x) round(mean(x)))
print(data.frame(month = as.integer(names(fc_monthly)),
                 forecast_mean = as.numeric(fc_monthly),
                 observed_mean = as.numeric(obs_monthly[names(fc_monthly)])))

cat("\nForecast mean by day type:\n")
day_type <- ifelse(forecast_df$is_holiday == 1, "Holiday",
                   ifelse(forecast_df$day_num0 %in% c(5, 6), "Weekend", "Weekday"))
day_means  <- tapply(forecast_df$forecast, day_type, function(x) round(mean(x)))
day_counts <- table(day_type)
print(data.frame(type = names(day_means),
                 mean_forecast = as.numeric(day_means),
                 hours = as.integer(day_counts[names(day_means)])))

# 11. FIGURE: OBSERVED TAIL + FORECAST
# The last two observed months are shown alongside the forecast so the join
# is visible -- a forecast that starts at the wrong level, or that decays
# to a flat line, is obvious here and nowhere else.
tail_df <- df[df$date_time >= FORECAST_START - 60 * 86400, ]
tail_df <- data.frame(date_time = tail_df$date_time, value = tail_df$traffic_volume, series = "Observed")
fc_plot <- data.frame(date_time = forecast_df$date_time, value = forecast_df$forecast, series = "Forecast")

p_fc <- ggplot(rbind(tail_df, fc_plot), aes(x = date_time, y = value, colour = series)) +
  geom_line(linewidth = 0.3) +
  geom_vline(xintercept = as.numeric(FORECAST_START), linetype = "22", colour = "grey40") +
  scale_colour_manual(values = c(Observed = "grey30", Forecast = "#d62728")) +
  labs(title = "Tuned LSTM: Three-Month Forward Forecast",
       subtitle = "Weather inputs set to monthly-hourly climatology",
       x = NULL, y = "Traffic Volume (vehicles/h)", colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave("output/models/lstm/lstm_future_forecast.png", plot = p_fc,
       width = 8, height = 3.6, dpi = 300)

# 12. FIGURE: ONE FORECAST WEEK, IN DETAIL
week_start <- FORECAST_START + 7 * 86400
week_idx <- forecast_df$date_time >= week_start & forecast_df$date_time < week_start + 7 * 86400

p_week <- ggplot(forecast_df[week_idx, ], aes(x = date_time, y = forecast)) +
  geom_line(colour = "#d62728", linewidth = 0.6) +
  labs(title = "Forecast Detail: One Week (2018-10-08 .. 2018-10-14)",
       subtitle = "Columbus Day (Mon 8 Oct) falls at the start of this week",
       x = NULL, y = "Forecast Traffic Volume (vehicles/h)") +
  theme_minimal(base_size = 11)

ggsave("output/models/lstm/lstm_future_forecast_week.png", plot = p_week, width = 7, height = 3.2, dpi = 300)

cat("\nSaved forecast and figures to: output/models/lstm\n")