library(tidyverse)
library(lubridate)
library(forecast)
library(ggplot2)
library(zoo)
library(tseries)   # adf.test(), kpss.test()

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
# Chronological 70/15/15 split, fixed across ALL models so MAE/RMSE/MAPE
# comparisons are valid. Validation is for model selection / hyperparameter
# and order tuning (e.g. picking SARIMA (p,d,q), LSTM epochs/architecture);
# test is only touched once, at the end, for the final comparison.

n_total  <- nrow(df)
train_n  <- floor(TRAIN_FRAC * n_total)
val_n    <- floor(VAL_FRAC * n_total)
test_n   <- n_total - train_n - val_n   # remainder, so all rows are used

split_data <- function(dataset, train_n, val_n) {
  list(
    full       = dataset,
    train      = dataset %>% slice(1:train_n),
    validation = dataset %>% slice((train_n + 1):(train_n + val_n)),
    test       = dataset %>% slice((train_n + val_n + 1):n())
  )
}

splits <- split_data(df, train_n, val_n)

cat("\n=== Train/Validation/Test Split Verification ===\n")
cat("Train     :", as.character(min(splits$train$date_time)), "to",
    as.character(max(splits$train$date_time)),
    sprintf("(%d hours, %.1f%%)\n", nrow(splits$train), 100 * nrow(splits$train) / n_total))
cat("Validation:", as.character(min(splits$validation$date_time)), "to",
    as.character(max(splits$validation$date_time)),
    sprintf("(%d hours, %.1f%%)\n", nrow(splits$validation), 100 * nrow(splits$validation) / n_total))
cat("Test      :", as.character(min(splits$test$date_time)), "to",
    as.character(max(splits$test$date_time)),
    sprintf("(%d hours, %.1f%%)\n", nrow(splits$test), 100 * nrow(splits$test) / n_total))

for (split_name in c("validation", "test")) {
  n_imputed <- sum(splits[[split_name]]$is_imputed)
  if (n_imputed > 0) {
    cat("WARNING:", n_imputed, "imputed hour(s) fall inside the", split_name, "set.\n",
        "Seasonal Naive scores on these hours will be artificially inflated\n",
        "since the 'ground truth' there was itself seasonal-naive-generated.\n",
        "Recommend excluding is_imputed==1 rows from reported", split_name, "metrics.\n")
  } else {
    cat("No imputed hours fall inside the", split_name, "set -- clean for cross-model comparison.\n")
  }
}


# =============================================================================
# 3b. STATIONARITY TESTING (ADF & KPSS) AND DIFFERENCING ORDER SELECTION
# =============================================================================
# Run on the TRAINING series only, so the differencing decision can't leak
# information from the test period.
#
# ADF (Augmented Dickey-Fuller): H0 = series has a unit root (non-stationary).
#   p-value < ALPHA  -> reject H0 -> evidence FOR stationarity.
# KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = series IS (trend-)stationary.
#   p-value < ALPHA  -> reject H0 -> evidence AGAINST stationarity.
# The two null hypotheses are reversed on purpose: requiring both tests to
# agree ("ADF says stationary" AND "KPSS says stationary") is a stricter,
# more defensible criterion than relying on either test alone.

train_vol <- splits$train$traffic_volume

run_stationarity_tests <- function(x, label) {
  adf_res  <- tseries::adf.test(x, alternative = "stationary")
  kpss_res <- tseries::kpss.test(x, null = "Level")
  cat("\n---", label, "---\n")
  cat(sprintf("ADF  : stat = %.4f, p-value = %s -> %s\n",
              unname(adf_res$statistic),
              format.pval(adf_res$p.value, digits = 4, eps = 0.01),
              ifelse(adf_res$p.value < ALPHA,
                     "stationary (reject H0)", "non-stationary (fail to reject H0)")))
  cat(sprintf("KPSS : stat = %.4f, p-value = %s -> %s\n",
              unname(kpss_res$statistic),
              format.pval(kpss_res$p.value, digits = 4, eps = 0.01),
              ifelse(kpss_res$p.value < ALPHA,
                     "non-stationary (reject H0)", "stationary (fail to reject H0)")))
  list(adf = adf_res, kpss = kpss_res)
}

is_stationary <- function(res) (res$adf$p.value < ALPHA) && (res$kpss$p.value >= ALPHA)

# ---- Level (raw) series ----
st_level <- run_stationarity_tests(train_vol, "Level (raw traffic_volume, train)")

# ---- Apply non-seasonal differencing until both tests agree (cap at d = 2) ----
d <- 0
diff_train <- train_vol
st_current <- st_level
while (!is_stationary(st_current) && d < 2) {
  d <- d + 1
  diff_train <- diff(train_vol, differences = d)
  st_current <- run_stationarity_tests(diff_train, paste0("Difference d = ", d))
}

if (d == 0) {
  cat("\nSeries is stationary at level -> no non-seasonal differencing applied (d = 0).\n")
} else if (is_stationary(st_current)) {
  cat("\nADF and KPSS agree the series is stationary after d =", d, "non-seasonal difference(s).\n")
} else {
  cat("\nADF/KPSS still disagree or indicate non-stationarity after d =", d,
      "(capped) -- inspect eda_01/eda_04 plots for trend/structural breaks before modeling.\n")
}

# ---- Seasonal unit-root check at the weekly period (168h) ----
# nsdiffs() (forecast pkg) runs a seasonal unit-root test (default OCSB) to
# recommend how many seasonal differences (D) the 168h cycle needs.
train_ts_weekly <- ts(train_vol, frequency = SEASONAL_PERIOD)
D_seasonal <- forecast::nsdiffs(train_ts_weekly)
cat("\nRecommended seasonal differences (D) at period =", SEASONAL_PERIOD, ":", D_seasonal, "\n")

cat("\n=== Stationarity Summary (training set) ===\n")
cat("Non-seasonal differences (d):", d, "\n")
cat("Seasonal differences (D)    :", D_seasonal, "\n")
cat("NOTE: these (d, D) orders are diagnostic evidence from the TRAIN series.\n",
    "Section 3c below applies them to the full dataset so the differenced\n",
    "column ships pre-computed in the saved CSV, instead of a model-specific\n",
    "ts object.\n", sep = "")


# =============================================================================
# 3c. APPLY DIFFERENCING TO THE FULL DATASET
# =============================================================================
# Adds a `traffic_volume_diff` column to the full dataset using the (d, D)
# orders identified above on the TRAINING split: first d non-seasonal
# (lag-1) differences, then D seasonal (lag = SEASONAL_PERIOD) differences
# on top. This is diagnostic/convenience only -- auto.arima() etc. can still
# search their own orders on the raw traffic_volume column if you prefer;
# this column just saves you from re-deriving it downstream.
#
# Differencing shortens the series, so the first (d + D * SEASONAL_PERIOD)
# rows can't have a value and are left as NA rather than dropped, keeping
# the CSV's row count and date_time index identical to the raw column.

diff_vec <- df$traffic_volume
if (d > 0) {
  diff_vec <- diff(diff_vec, differences = d)
}
if (D_seasonal > 0) {
  diff_vec <- diff(diff_vec, lag = SEASONAL_PERIOD, differences = D_seasonal)
}
n_pad <- nrow(df) - length(diff_vec)
df$traffic_volume_diff <- c(rep(NA_real_, n_pad), diff_vec)

cat("\nApplied differencing to the full dataset: d =", d, "non-seasonal (lag 1), D =",
    D_seasonal, "seasonal (lag", SEASONAL_PERIOD, ").\n")
cat("traffic_volume_diff has", n_pad, "leading NA row(s) (rows lost to differencing).\n\n")

# Re-slice train/validation/test now that the diff column exists, so
# downstream splits carry it too.
splits <- split_data(df, train_n, val_n)


# =============================================================================
# 4. MODEL-READY FEATURES (flat table, no per-model ts()/array objects)
# =============================================================================
# Everything below is added as plain columns on `df` and written out as CSV.
# Any downstream tool -- R ts(), a SARIMAX xreg matrix, a Keras windowed
# LSTM input, etc. -- can be re-derived directly from these columns, so we
# no longer build or save separate ts_data / xreg_data / lstm array objects.

# ---- 4a. Split label ----
# Same chronological boundaries as Section 3, carried as a plain column
# instead of a list of separate ts()/data-frame objects.
df$split <- NA_character_
df$split[1:train_n]                          <- "train"
df$split[(train_n + 1):(train_n + val_n)]    <- "validation"
df$split[(train_n + val_n + 1):n_total]      <- "test"

# ---- 4b. Cyclical encoding ----
# sin/cos encodings so e.g. hour 23 -> hour 0 doesn't look like a big jump
# to models that treat these as ordinary numeric features (e.g. an LSTM).
add_cyclical <- function(d) {
  d %>%
    mutate(
      dow_num   = wday(date_time, week_start = 1),  # 1 = Monday
      hour_sin  = sin(2 * pi * hour / 24),
      hour_cos  = cos(2 * pi * hour / 24),
      dow_sin   = sin(2 * pi * dow_num / 7),
      dow_cos   = cos(2 * pi * dow_num / 7),
      month_sin = sin(2 * pi * month / 12),
      month_cos = cos(2 * pi * month / 12)
    ) %>%
    select(-dow_num)
}
df <- add_cyclical(df)

# ---- 4c. Min-max scaling (fit on TRAIN rows only, to avoid leakage) ----
scale_cols <- c("temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume")
train_rows <- df$split == "train"
scale_params <- lapply(scale_cols, function(col) {
  list(min = min(df[[col]][train_rows], na.rm = TRUE),
       max = max(df[[col]][train_rows], na.rm = TRUE))
})
names(scale_params) <- scale_cols

for (col in names(scale_params)) {
  rng <- scale_params[[col]]$max - scale_params[[col]]$min
  df[[paste0(col, "_scaled")]] <- if (rng == 0) 0 else (df[[col]] - scale_params[[col]]$min) / rng
}

cat("=== Model-ready feature table ===\n")
cat("Columns:", paste(names(df), collapse = ", "), "\n")
cat("Rows   :", nrow(df), "\n\n")


# =============================================================================
# 5. SAVE (CSV only)
# =============================================================================

# 5a. Full model-ready dataset -- one flat CSV with raw + differenced +
# cyclical + scaled columns and a split label. Filter on `split` downstream
# for train/validation/test, and reconstruct whatever structure a given
# model needs (ts(), xreg matrix, windowed sequences) directly from this.
df_out <- df
df_out$date_time <- format(df_out$date_time, "%Y-%m-%d %H:%M:%S")  # avoid
# write.csv() silently dropping "00:00:00" on midnight rows otherwise
write.csv(df_out, file.path(output_dir, "traffic_volume_processed.csv"), row.names = FALSE)
cat("Saved:", file.path(output_dir, "traffic_volume_processed.csv"), "\n")

# 5b. Scaling parameters -- needed to invert the *_scaled columns back to
# raw units later (e.g. after an LSTM prediction).
scale_params_df <- data.frame(
  variable = names(scale_params),
  min      = sapply(scale_params, function(p) p$min),
  max      = sapply(scale_params, function(p) p$max),
  row.names = NULL
)
write.csv(scale_params_df, file.path(output_dir, "scale_params.csv"), row.names = FALSE)
cat("Saved:", file.path(output_dir, "scale_params.csv"), "\n")

# 5c. Stationarity test summary -- the (d, D) orders and ADF/KPSS statistics
# used to build traffic_volume_diff, as a small CSV instead of an .rds list.
stationarity_summary <- data.frame(
  alpha        = ALPHA,
  d            = d,
  D_seasonal   = D_seasonal,
  seasonal_period = SEASONAL_PERIOD,
  adf_stat_level   = unname(st_level$adf$statistic),
  adf_pvalue_level = st_level$adf$p.value,
  kpss_stat_level  = unname(st_level$kpss$statistic),
  kpss_pvalue_level = st_level$kpss$p.value
)
write.csv(stationarity_summary, file.path(output_dir, "stationarity_summary.csv"), row.names = FALSE)
cat("Saved:", file.path(output_dir, "stationarity_summary.csv"), "\n")

cat("\n--- Preprocessing complete ---\n")
cat("All models -> read traffic_volume_processed.csv and filter on the `split` column\n",
    "             (train / validation / test), then build whatever structure\n",
    "             (ts(), xreg matrix, windowed LSTM sequences) you need from\n",
    "             its plain columns.\n", sep = "")
cat("Invert scaled columns -> use scale_params.csv (min/max per variable).\n")
cat("Differencing/stationarity details -> stationarity_summary.csv\n")