library(tidyverse)
library(lubridate)
library(forecast)
library(ggplot2)
library(zoo)
library(tseries)   # adf.test(), kpss.test()
library(gridExtra) # arrange ACF/PACF pairs side by side

# =============================================================================
# 0. PATHS & CONFIG
# =============================================================================

raw_path   <- "data/raw/Metro_Interstate_Traffic_Volume.csv"
output_dir <- "data/processed"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Gap-fill tiers (see Section 1h)
SHORT_MAX <- 2    # hours: linear interpolation
LONG_MAX  <- 24   # hours: t-168 lag; above this, seasonal-naive averaging

# Continuous window used to avoid the ~10-month hole in the raw data
CUTOFF_DATE <- as.POSIXct("2015-10-01 00:00:00", tz = "UTC")

# Seasonal period used for SARIMA / Holt-Winters / Seasonal Naive.
# Traffic volume has BOTH a strong daily (24h) and weekly (168h) pattern, but
# ts()-based models (Arima, HoltWinters, snaive) only support ONE seasonal
# period. We use 168 (weekly) because it's the dominant cycle your own
# gap-fill logic relies on (t-168 lag, seasonal-naive across weeks) — the
# daily pattern is still present *within* that cycle, just not modeled as
# its own seasonal term.
# If you want daily seasonality as the primary period instead, set this to
# 24 and treat weekly effects as an exogenous regressor (day_of_week dummies)
# in SARIMAX. You cannot get both periods natively out of HoltWinters/snaive;
# that would require msts() + TBATS, which is out of scope for these 4 models.
SEASONAL_PERIOD <- 168

# Train / validation / test split fractions (Section 3). Applied in
# chronological order (train earliest, test latest) so no model ever trains
# on data from after the period it's evaluated on.
TRAIN_FRAC <- 0.70
VAL_FRAC   <- 0.15
TEST_FRAC  <- 0.15   # implied remainder; kept explicit for readability

# Significance level used for both ADF and KPSS decisions (Section 3b)
ALPHA <- 0.05

# Max lags shown on ACF/PACF plots. 2 * SEASONAL_PERIOD lets you see two full
# weekly cycles worth of autocorrelation, which is what matters most here.
ACF_MAX_LAG <- 2 * SEASONAL_PERIOD


# =============================================================================
# 1. LOAD & CLEAN
# =============================================================================

df <- read.csv(raw_path, stringsAsFactors = FALSE)
names(df) <- trimws(names(df))
char_cols <- names(df)[sapply(df, is.character)]
df[char_cols] <- lapply(df[char_cols], trimws)
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

cat("=== Raw Data Overview ===\n")
cat("Raw rows read:", nrow(df), "\n")
cat("Date range   :", as.character(min(df$date_time)), "to", as.character(max(df$date_time)), "\n\n")


# ---- 1a. Remove fully duplicated rows ----
n_before <- nrow(df)
df <- distinct(df)
cat("Removed", n_before - nrow(df), "fully duplicated rows.\n")

# ---- 1b. Collapse duplicate timestamps ----
# Some hours have >1 row because OpenWeatherMap logged multiple simultaneous
# conditions for the same hour. holiday/traffic_volume are identical within
# a timestamp group (verified); weather numerics are averaged.
n_before <- nrow(df)
df <- df %>%
  group_by(date_time) %>%
  summarise(
    holiday        = first(holiday),
    temp           = round(mean(temp, na.rm = TRUE), 2),
    rain_1h        = mean(rain_1h, na.rm = TRUE),
    snow_1h        = mean(snow_1h, na.rm = TRUE),
    clouds_all     = mean(clouds_all, na.rm = TRUE),
    weather_main   = first(weather_main),
    traffic_volume = first(traffic_volume),
    .groups = "drop"
  ) %>%
  arrange(date_time)
cat("Collapsed", n_before - nrow(df), "duplicate-timestamp rows into", nrow(df), "unique hourly rows.\n")

# ---- 1c. Text normalization ----
df$holiday <- tolower(df$holiday)
df$weather_main <- tolower(df$weather_main)

# ---- 1d. Fix known mislabeled holiday date ----
# "christmas day" is mislabeled on 2016-12-26 in the raw data; belongs on 12-25.
mis_idx <- which(df$holiday == "christmas day" & as.Date(df$date_time) == as.Date("2016-12-26"))
df$holiday[mis_idx] <- "none"
fix_idx <- which(df$date_time == as.POSIXct("2016-12-25 00:00:00", tz = "UTC"))
if (length(fix_idx) > 0) df$holiday[fix_idx] <- "christmas day"

# ---- 1e. Fix "holiday labeled only on first hour" bug ----
# Propagate the holiday name to every hour of its calendar date.
df$cal_date <- as.Date(df$date_time)
holiday_lookup <- df %>%
  filter(holiday != "none") %>%
  group_by(cal_date) %>%
  summarise(holiday_name = first(holiday), .groups = "drop")

n_before_holiday <- sum(df$holiday != "none")
df <- df %>%
  left_join(holiday_lookup, by = "cal_date") %>%
  mutate(holiday = ifelse(!is.na(holiday_name), holiday_name, "none")) %>%
  select(-holiday_name)
cat("Holiday-labeled rows: ", n_before_holiday, " (before fix) -> ",
    sum(df$holiday != "none"), " (after propagating to full day)\n", sep = "")

# ---- 1f. Binary holiday feature ----
df$is_holiday <- ifelse(df$holiday != "none", 1L, 0L)
df$cal_date <- NULL
df$holiday <- NULL

# ---- 1g. Fix physically impossible values via linear interpolation ----
# temp == 0 Kelvin is impossible; rain_1h has one 9831.3 mm/hr outlier
# (next-highest valid value is 55.63, so 300 mm/hr safely isolates just it).
n_bad_temp <- sum(df$temp <= 0)
n_bad_rain <- sum(df$rain_1h > 300)
df$temp[df$temp <= 0]   <- NA
df$rain_1h[df$rain_1h > 300] <- NA
df <- df %>% arrange(date_time)
df$temp    <- na.approx(df$temp,    x = as.numeric(df$date_time), rule = 2)
df$rain_1h <- na.approx(df$rain_1h, x = as.numeric(df$date_time), rule = 2)
cat("Interpolated", n_bad_temp, "temp value(s) and", n_bad_rain, "rain_1h value(s) flagged as physically impossible.\n")

# ---- 1h. Restrict to a clean, continuous date range ----
# ~10-month gap exists between 2014-08-08 and 2015-06-11. Restrict to the
# continuous period from CUTOFF_DATE onward instead of imputing across it.
n_before <- nrow(df)
df <- df %>% filter(date_time >= CUTOFF_DATE)
cat("Removed", n_before - nrow(df), "rows before", format(CUTOFF_DATE),
    "; retained range:", as.character(min(df$date_time)), "to", as.character(max(df$date_time)),
    "(", nrow(df), "rows )\n")

# ---- 1i. Fill remaining internal hourly gaps ----
# Reindex to a complete hourly grid, then fill traffic_volume by gap length:
#   <=2h   : linear interpolation
#   3-24h  : same hour, previous week (t-168)
#   >24h   : seasonal-naive average across +/-4 weeks (same hour-of-week),
#            skipping any missing/out-of-bounds reference points
# Weather columns are linearly interpolated regardless of gap length (no
# strong weekly seasonality to preserve there). is_holiday is re-derived by
# calendar date, never interpolated.
full_grid <- data.frame(date_time = seq(min(df$date_time), max(df$date_time), by = "hour"))
df <- full_grid %>% left_join(df, by = "date_time") %>% arrange(date_time)

df$cal_date <- as.Date(df$date_time)
df$is_holiday <- ifelse(df$cal_date %in% holiday_lookup$cal_date, 1L, 0L)
df$cal_date <- NULL

run <- rle(is.na(df$traffic_volume))
gap_length <- rep(run$lengths, run$lengths)
df$gap_length <- ifelse(is.na(df$traffic_volume), gap_length, 0)

weather_cols <- c("temp", "rain_1h", "snow_1h", "clouds_all")
for (col in weather_cols) {
  df[[col]] <- round(na.approx(df[[col]], x = as.numeric(df$date_time), na.rm = FALSE), 2)
}

tv <- df$traffic_volume
medium_idx <- df$gap_length > SHORT_MAX & df$gap_length <= LONG_MAX
tv[medium_idx] <- lag(tv, 168)[medium_idx]
tv <- na.approx(tv, x = as.numeric(df$date_time), na.rm = FALSE)
tv[df$gap_length > LONG_MAX] <- NA

long_idx <- which(df$gap_length > LONG_MAX)
get_seasonal_avg <- function(i, n_weeks = 4) {
  offsets <- c(-n_weeks:-1, 1:n_weeks) * 168L
  candidate_idx <- i + offsets
  candidate_idx <- candidate_idx[candidate_idx >= 1 & candidate_idx <= length(tv)]
  vals <- tv[candidate_idx]
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0) return(NA_real_)
  mean(vals)
}
tv[long_idx] <- sapply(long_idx, get_seasonal_avg)
df$traffic_volume <- round(tv, 0)

# Keep an explicit imputation flag instead of discarding gap_length.
# This matters most for Seasonal Naive -- its forecasts ARE same-hour-lastweek
# / seasonal-average values, so if imputed hours end up in the test set you'd
# be scoring the model against numbers it effectively generated. Carry this
# flag through and exclude/flag imputed hours when evaluating (Section 3).
df$is_imputed <- ifelse(df$gap_length > 0, 1L, 0L)

n_short <- sum(df$gap_length > 0 & df$gap_length <= SHORT_MAX)
n_medium <- sum(df$gap_length > SHORT_MAX & df$gap_length <= LONG_MAX)
n_long <- length(long_idx)
n_long_unresolved <- sum(is.na(df$traffic_volume[long_idx]))
cat("Gap fill: ", n_short, " hour(s) interpolated, ", n_medium,
    " hour(s) filled via t-168, ", n_long,
    " hour(s) filled via seasonal-naive averaging (", n_long_unresolved,
    " unresolved).\n", sep = "")
cat("Total imputed hours:", sum(df$is_imputed), "/", nrow(df),
    sprintf("(%.1f%%)\n\n", 100 * mean(df$is_imputed)))

df$gap_length <- NULL

# ---- 1j. Calendar-derived features ----
df$hour        <- hour(df$date_time)
df$month       <- month(df$date_time)
df$day_of_week <- wday(df$date_time, label = TRUE, abbr = FALSE)

# ---- 1k. Final column set ----
df <- df %>%
  select(date_time, temp, rain_1h, snow_1h, clouds_all, is_holiday,
         hour, month, day_of_week, is_imputed, traffic_volume)

cat("Final cleaned dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")


# =============================================================================
# 2. EDA
# =============================================================================

# ---- IEEE plot settings ----
IEEE_W_SINGLE <- 3.5
IEEE_W_DOUBLE <- 7.16
IEEE_H        <- 2.8
IEEE_DPI      <- 300

theme_ieee <- function(base_size = 7) {
  theme_minimal(base_size = base_size) +
    theme(
      text             = element_text(family = "serif", size = base_size),
      plot.title       = element_text(size = base_size + 1, face = "bold", hjust = 0.5),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1),
      legend.text      = element_text(size = base_size - 1),
      legend.title     = element_text(size = base_size),
      strip.text       = element_text(size = base_size, face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom"
    )
}

save_ieee <- function(fname, plot, width = IEEE_W_DOUBLE, height = IEEE_H) {
  ggsave(filename = paste0(fname, ".png"), plot = plot,
         width = width, height = height, dpi = IEEE_DPI, units = "in")
  cat("Saved:", paste0(fname, ".png"), "\n")
}

# ---- 2a. Full hourly time series ----
p_ts <- ggplot(df, aes(x = date_time, y = traffic_volume)) +
  geom_line(linewidth = 0.2, colour = "steelblue") +
  theme_ieee() +
  labs(title = "Hourly Metro Interstate Traffic Volume (Oct 2015 - Sep 2018)",
       x = "Date", y = "Traffic Volume")
print(p_ts)
save_ieee("eda_01_timeseries_full", p_ts, width = IEEE_W_DOUBLE, height = 3)

# ---- 2b. Hour-of-day profile by day type ----
# Purpose: reveal the daily seasonality that a 168-period model still needs
# to "contain" implicitly, and show weekday vs weekend differ substantially.
day_profile <- df %>%
  mutate(day_type = ifelse(day_of_week %in% c("Saturday", "Sunday"), "Weekend", "Weekday")) %>%
  group_by(day_type, hour) %>%
  summarise(avg_volume = mean(traffic_volume, na.rm = TRUE), .groups = "drop")

p_profile <- ggplot(day_profile, aes(x = hour, y = avg_volume, colour = day_type)) +
  geom_line(linewidth = 0.6) +
  scale_x_continuous(breaks = seq(0, 23, 4)) +
  theme_ieee() +
  labs(title = "Average Hourly Traffic Profile: Weekday vs Weekend",
       x = "Hour of Day", y = "Average Traffic Volume", colour = NULL)
print(p_profile)
save_ieee("eda_02_hourly_profile", p_profile)

# ---- 2c. Monthly aggregate seasonal plot (annual pattern) ----
monthly_ts <- df %>%
  mutate(ym = floor_date(date_time, "month")) %>%
  group_by(ym) %>%
  summarise(avg_volume = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  pull(avg_volume) %>%
  ts(start = c(year(min(df$date_time)), month(min(df$date_time))), frequency = 12)

p_seasonal <- ggseasonplot(monthly_ts, year.labels = TRUE, year.labels.left = TRUE) +
  theme_ieee() +
  labs(title = "Monthly-Aggregated Seasonal Plot", x = "Month", y = "Average Traffic Volume")
print(p_seasonal)
save_ieee("eda_03_monthly_seasonal", p_seasonal)

# ---- 2d. Multi-seasonal STL decomposition (daily + weekly) ----
# mstl() (forecast pkg) handles multiple seasonal periods, unlike stl(), so
# this view is more honest about the data than a single-period decomposition
# would be -- useful for EDA even though the models below can only act on
# one of these periods at a time.
msts_data <- msts(df$traffic_volume, seasonal.periods = c(24, 168))
stl_fit <- mstl(msts_data)
p_stl <- autoplot(stl_fit) + theme_ieee() + labs(title = "Multi-Seasonal STL Decomposition (24h + 168h)")
print(p_stl)
save_ieee("eda_04_mstl", p_stl, height = 5)

# ---- 2e. Summary statistics ----
cat("\n=== Summary Statistics: traffic_volume ===\n")
df %>%
  summarise(Min = min(traffic_volume), Median = median(traffic_volume),
            Mean = round(mean(traffic_volume), 1), Max = max(traffic_volume),
            SD = round(sd(traffic_volume), 1)) %>%
  print()


# =============================================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# =============================================================================
# Chronological 70/15/15 split. The `split` column is the only label added
# here — all stationarity testing, ACF/PACF diagnostics, and differencing
# are handled in differencing.R which runs after this script.

n_total <- nrow(df)
train_n <- floor(TRAIN_FRAC * n_total)
val_n   <- floor(VAL_FRAC * n_total)
test_n  <- n_total - train_n - val_n

df$split <- NA_character_
df$split[1:train_n]                       <- "train"
df$split[(train_n + 1):(train_n + val_n)] <- "validation"
df$split[(train_n + val_n + 1):n_total]   <- "test"

cat("\n=== Train/Validation/Test Split ===\n")
cat(sprintf("Train      : rows   1 - %d  (%s to %s)  %.1f%%\n",
            train_n,
            as.character(df$date_time[1]),
            as.character(df$date_time[train_n]),
            100 * train_n / n_total))
cat(sprintf("Validation : rows %d - %d  (%s to %s)  %.1f%%\n",
            train_n + 1, train_n + val_n,
            as.character(df$date_time[train_n + 1]),
            as.character(df$date_time[train_n + val_n]),
            100 * val_n / n_total))
cat(sprintf("Test       : rows %d - %d  (%s to %s)  %.1f%%\n",
            train_n + val_n + 1, n_total,
            as.character(df$date_time[train_n + val_n + 1]),
            as.character(df$date_time[n_total]),
            100 * test_n / n_total))

# Imputation leakage check
for (split_name in c("validation", "test")) {
  idx <- if (split_name == "validation") {
    (train_n + 1):(train_n + val_n)
  } else {
    (train_n + val_n + 1):n_total
  }
  n_imp <- sum(df$is_imputed[idx])
  if (n_imp > 0) {
    cat("WARNING:", n_imp, "imputed hour(s) in", split_name,
        "-- exclude is_imputed==1 rows from reported metrics.\n")
  } else {
    cat("No imputed hours in", split_name, "-- clean for evaluation.\n")
  }
}


# =============================================================================
# 4. SAVE
# =============================================================================
# Saves the cleaned dataset with raw columns only + is_imputed + split.
# No differencing, scaling, or cyclical encoding -- those are model-specific
# and handled in each model's own script (or in differencing.R for the
# pre-computed differenced column).

df_out           <- df
df_out$date_time <- format(df_out$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(df_out, file.path(output_dir, "traffic_volume_processed.csv"), row.names = FALSE)
cat("\nSaved:", file.path(output_dir, "traffic_volume_processed.csv"), "\n")
cat("Columns:", paste(names(df_out), collapse = ", "), "\n")
cat("Rows   :", nrow(df_out), "\n")

cat("\n--- Preprocessing complete ---\n")
cat("Next step: run differencing.R to produce stationarity diagnostics\n")
cat("           and add traffic_volume_diff (training split only) to the CSV.\n")