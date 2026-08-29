# ============================================================
# Three-Month Forward Forecast (2018-10-01 .. 2018-12-31)
# Input:  data/processed/traffic_volume_processed.csv
#         results/lstm_tuning_results.csv (produced by tuning_lstm.R)
#
# lstm.R measures how accurate the model is on data it has never seen but
# which was actually observed. This script does the thing the model is
# ultimately for: forecasting beyond the end of the dataset, where no actual
# values exist to compare against.
#
# Two problems have to be solved to get there.
#
# 1. Which model to forecast with. The model in lstm.R is trained on data
#    ending 2018-03-31, because the last six months were reserved for
#    validation and testing. Once those held-out periods have done their job
#    (selecting hyperparameters and producing an honest accuracy estimate),
#    throwing away six months of the most recent history would be wasteful.
#    The model is therefore refit on ALL observed data using the same
#    architecture and the epoch count that was best during tuning -- standard
#    practice once model selection is finished.
#
# 2. What to feed it for weather. temp, rain_1h, snow_1h, and clouds_all do
#    not exist for October-December 2018. Calendar features and holidays do,
#    because they are deterministic. The weather inputs are therefore filled
#    with climatology: the historical mean for that month-and-hour across the
#    observed record. This is an explicit assumption of average seasonal
#    weather, not a weather forecast, and it is what makes the output a
#    scenario projection rather than a prediction.
# ============================================================
source("src/models/lstm_common.R")

library(keras3)
library(tensorflow)
library(ggplot2)

set.seed(42)
tensorflow::set_random_seed(42)

tuning_results_path <- file.path(RESULTS_DIR, "lstm_tuning_results.csv")
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)


# ---- 1. Load observed data ----
df <- load_processed()
df <- engineer_features(df)
cat("Observed series:", nrow(df), "hours,", as.character(min(df$date_time)),
    "..", as.character(max(df$date_time)), "\n")


# ---- 2. Hyperparameters from the grid search ----
if (file.exists(tuning_results_path)) {
  tuning_results <- read.csv(tuning_results_path) %>% arrange(val_loss_mse)
  best_cfg <- tuning_results[1, ]
} else {
  stop("results/lstm_tuning_results.csv not found -- run tuning_lstm.R first.")
}
print(best_cfg)

LOOK_BACK     <- best_cfg$look_back
LSTM_UNITS    <- best_cfg$lstm_units
DROPOUT_RATE  <- best_cfg$dropout_rate
LEARNING_RATE <- best_cfg$learning_rate
BATCH_SIZE    <- best_cfg$batch_size
# best_epoch was added to the tuning output; older result files may predate it.
REFIT_EPOCHS  <- if (!is.null(best_cfg$best_epoch) && !is.na(best_cfg$best_epoch)) {
  best_cfg$best_epoch
} else {
  warning("best_epoch missing from tuning results -- defaulting to 30 refit epochs.")
  30
}
cat("Refitting for", REFIT_EPOCHS, "epochs (best epoch found during tuning).\n")


# ---- 3. Build the future feature frame ----
# Everything here is known in advance without any forecasting: the calendar
# repeats, and US federal holidays are fixed by rule.
future_dates <- seq(FORECAST_START, FORECAST_END, by = "hour")
H <- length(future_dates)
cat("Forecast horizon:", H, "hours (",
    as.character(FORECAST_START), "..", as.character(FORECAST_END), ")\n")

future <- data.frame(date_time = future_dates)
future$hour        <- hour(future$date_time)
future$month       <- month(future$date_time)
future$day_of_week <- as.character(wday(future$date_time, label = TRUE, abbr = FALSE))

# Federal holidays falling inside the horizon, following the same observance
# rule the dataset itself uses (Columbus Day = 2nd Monday of October,
# Veterans Day = Nov 11 or the Monday after when it lands on a Sunday,
# Thanksgiving = 4th Thursday of November, Christmas Day = Dec 25).
future_holidays <- as.Date(c(
  "2018-10-08",  # Columbus Day
  "2018-11-12",  # Veterans Day (observed; Nov 11 2018 is a Sunday)
  "2018-11-22",  # Thanksgiving Day
  "2018-12-25"   # Christmas Day
))
future$is_holiday <- as.integer(as.Date(future$date_time) %in% future_holidays)
cat("Holiday hours in horizon:", sum(future$is_holiday),
    "(", length(future_holidays), "days )\n")

# Climatological weather: mean of each weather variable by month and hour
# across the observed record. Means rather than medians, because the median
# of rain_1h and snow_1h is zero for nearly every hour, which would encode
# "it never rains" instead of "average precipitation".
climatology <- df %>%
  group_by(month, hour) %>%
  summarise(temp       = mean(temp),
            rain_1h    = mean(rain_1h),
            snow_1h    = mean(snow_1h),
            clouds_all = mean(clouds_all),
            .groups = "drop")

future <- future %>% left_join(climatology, by = c("month", "hour"))
stopifnot(!any(is.na(future$temp)))

future <- engineer_features(future)


# ---- 4. Scale using the full observed record ----
# The refit model trains on everything observed, so the scaler is fitted on
# everything observed. Future rows are transformed with those same
# parameters; nothing about October-December 2018 informs them.
observed_df <- df[, FEATURE_COLS]
scale_params <- fit_scaler(observed_df)
observed_scaled <- apply_scale(observed_df, scale_params)

# traffic_volume is unknown for future rows -- it gets filled in step by step
# by the model itself during the recursive loop below.
future$traffic_volume <- NA_real_
future_features <- future[, FEATURE_COLS]
future_scaled <- future_features
for (col in setdiff(FEATURE_COLS, TARGET_COL)) {
  rng <- scale_params[[col]]$max - scale_params[[col]]$min
  if (rng == 0) rng <- 1
  future_scaled[[col]] <- (future_features[[col]] - scale_params[[col]]$min) / rng
}


# ---- 5. Refit on all observed data ----
train_seq <- make_sequences(observed_scaled, LOOK_BACK)
n_features <- dim(train_seq$x)[3]
cat("Refit training sequences:", paste(dim(train_seq$x), collapse = "x"), "\n")

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

# No validation split and no early stopping here: every observed hour is
# being used for training, so the stopping point comes from REFIT_EPOCHS,
# which tuning already established on a genuine validation set.
history <- model %>% fit(
  x = train_seq$x, y = train_seq$y,
  epochs = REFIT_EPOCHS,
  batch_size = BATCH_SIZE,
  verbose = 2
)


# ---- 6. Recursive multi-step forecast ----
# One hour at a time. Predict hour t+1 from the last LOOK_BACK hours, append
# that prediction (together with hour t+1's known calendar and climatological
# weather) to the window, drop the oldest hour, repeat. Each prediction
# therefore becomes an input to the next, which is why error accumulates with
# horizon and why the later weeks should be read as a seasonal profile rather
# than an hour-accurate forecast.
window <- as.matrix(tail(observed_scaled, LOOK_BACK))
preds_scaled <- numeric(H)

cat("\nForecasting", H, "hours recursively...\n")
pb <- txtProgressBar(min = 0, max = H, style = 3)
for (i in seq_len(H)) {
  x <- array(window, dim = c(1, LOOK_BACK, n_features))
  p <- as.numeric(predict(model, x, verbose = 0))

  preds_scaled[i] <- p

  next_row <- as.numeric(future_scaled[i, ])
  next_row[match(TARGET_COL, FEATURE_COLS)] <- p   # the prediction feeds itself forward
  window <- rbind(window[-1, , drop = FALSE], next_row)

  if (i %% 24 == 0 || i == H) setTxtProgressBar(pb, i)
}
close(pb)

forecast_values <- inverse_scale_target(preds_scaled, scale_params)
# Traffic volume cannot be negative; the recursive loop can drift below zero
# in the small hours late in the horizon.
forecast_values <- pmax(forecast_values, 0)

forecast_df <- data.frame(
  date_time      = future_dates,
  forecast       = round(forecast_values),
  is_holiday     = future$is_holiday,
  hour           = future$hour,
  day_of_week    = future$day_of_week
)

# Formatted to a fixed string first: write.csv() drops "00:00:00" from midnight
# rows only, which makes the column unparseable as a single format downstream
# (see step 13 of preprocessing.R).
write.csv(transform(forecast_df, date_time = format(date_time, "%Y-%m-%d %H:%M:%S")),
          file.path(RESULTS_DIR, "lstm_future_forecast.csv"), row.names = FALSE)


# ---- 7. Summary of the forecast ----
cat("\n==== Forecast summary (2018-10-01 .. 2018-12-31) ====\n")
cat(sprintf("Mean: %.0f  Min: %.0f  Max: %.0f vehicles/h\n",
            mean(forecast_values), min(forecast_values), max(forecast_values)))

cat("\nMonthly mean forecast vs same months in the observed record:\n")
obs_monthly <- df %>%
  filter(month %in% c(10, 11, 12)) %>%
  group_by(month) %>%
  summarise(observed_mean = round(mean(traffic_volume)), .groups = "drop")
fc_monthly <- forecast_df %>%
  mutate(month = month(date_time)) %>%
  group_by(month) %>%
  summarise(forecast_mean = round(mean(forecast)), .groups = "drop")
print(as.data.frame(left_join(fc_monthly, obs_monthly, by = "month")))

cat("\nForecast mean by day type:\n")
print(as.data.frame(forecast_df %>%
  mutate(type = case_when(is_holiday == 1 ~ "Holiday",
                          day_of_week %in% c("Saturday", "Sunday") ~ "Weekend",
                          TRUE ~ "Weekday")) %>%
  group_by(type) %>%
  summarise(mean_forecast = round(mean(forecast)), hours = n(), .groups = "drop")))


# ---- 8. Figure: observed tail + forecast ----
# The last two observed months are shown alongside the forecast so the join
# is visible -- a forecast that starts at the wrong level, or that decays to
# a flat line, is obvious here and nowhere else.
tail_df <- df %>%
  filter(date_time >= FORECAST_START - days(60)) %>%
  transmute(date_time, value = traffic_volume, series = "Observed")
fc_plot <- forecast_df %>%
  transmute(date_time, value = forecast, series = "Forecast")

p_fc <- ggplot(rbind(tail_df, fc_plot), aes(x = date_time, y = value, colour = series)) +
  geom_line(linewidth = 0.3) +
  geom_vline(xintercept = as.numeric(FORECAST_START), linetype = "22", colour = "grey40") +
  scale_colour_manual(values = c("Observed" = "grey30", "Forecast" = "#d62728")) +
  labs(title = "Tuned LSTM: Three-Month Forward Forecast",
       subtitle = "Weather inputs set to monthly-hourly climatology",
       x = NULL, y = "Traffic Volume (vehicles/h)", colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top")

ggsave(file.path(RESULTS_DIR, "lstm_future_forecast.png"), plot = p_fc,
       width = 8, height = 3.6, dpi = 300)


# ---- 9. Figure: one forecast week, in detail ----
week_start <- FORECAST_START + days(7)
week_idx <- forecast_df$date_time >= week_start & forecast_df$date_time < week_start + days(7)

p_week <- ggplot(forecast_df[week_idx, ], aes(x = date_time, y = forecast)) +
  geom_line(colour = "#d62728", linewidth = 0.6) +
  labs(title = "Forecast Detail: One Week (2018-10-08 .. 2018-10-14)",
       subtitle = "Columbus Day (Mon 8 Oct) falls at the start of this week",
       x = NULL, y = "Forecast Traffic Volume (vehicles/h)") +
  theme_minimal(base_size = 11)

ggsave(file.path(RESULTS_DIR, "lstm_future_forecast_week.png"), plot = p_week,
       width = 7, height = 3.2, dpi = 300)

cat("\nSaved forecast and figures to:", RESULTS_DIR, "\n")
