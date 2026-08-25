# ============================================================
# Time Series Diagnostics: Metro Interstate Traffic Volume
# Reads the PROCESSED csv (output of preprocessing.R) and produces:
#   - ACF / PACF plots (original + differenced series)
#   - ADF and KPSS stationarity tests (original + differenced series)
#   - Recommended differencing orders (regular d, seasonal D at 24h and 168h)
#   - Ljung-Box test for residual/series autocorrelation (original + differenced)
#   - An STL decomposition plot (trend / seasonal / remainder)
# All figures are saved as PNG; all numeric results are saved as one JSON file.
#
# Intended to run AFTER preprocessing.R, on the same processed CSV used by
# visualisation.R. These diagnostics are what justify (i) whether the series
# needs differencing before SARIMA, and (ii) what seasonal period(s) and
# AR/MA orders are appropriate -- feeds directly into Section III / the
# SARIMA teammate's model-fitting step.
# ============================================================

library(dplyr)
library(ggplot2)
library(forecast)   # ggAcf, ggPacf, ndiffs, nsdiffs, mstl
library(tseries)    # adf.test, kpss.test
library(jsonlite)   # write_json

processed_path <- "data/processed/traffic_volume_processed.csv"
fig_dir  <- "output/diagnostics"
json_path <- "output/diagnostics/ts_diagnostics.json"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# ---- 0. Load processed data and build the time series object ----
df <- read.csv(processed_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S")
df <- df %>% arrange(date_time)

# Hourly series, no gaps expected (preprocessing.R already reindexed to a
# complete hourly grid and dropped the one unfillable long gap).
stopifnot(!any(is.na(df$traffic_volume)))

DAILY_PERIOD  <- 24    # hours in a day
WEEKLY_PERIOD <- 168   # hours in a week

# Base ts with daily seasonality (used for most tests/plots below)
tv_ts_daily <- ts(df$traffic_volume, frequency = DAILY_PERIOD)

cat("Loaded", length(df$traffic_volume), "hourly observations from",
    format(min(df$date_time)), "to", format(max(df$date_time)), "\n\n")

results <- list(
  n_obs = length(df$traffic_volume),
  date_range = list(start = format(min(df$date_time)), end = format(max(df$date_time)))
)

# ---- helper: run ADF + KPSS and return a tidy list ----
run_stationarity_tests <- function(x, label) {
  adf <- suppressWarnings(adf.test(x))
  kpss_level <- suppressWarnings(kpss.test(x, null = "Level"))
  kpss_trend <- suppressWarnings(kpss.test(x, null = "Trend"))

  cat("---", label, "---\n")
  cat(sprintf("ADF:  statistic = %.3f, p-value = %.4f -> %s\n",
              adf$statistic, adf$p.value,
              ifelse(adf$p.value < 0.05, "reject H0 (stationary)", "fail to reject H0 (non-stationary)")))
  cat(sprintf("KPSS (level): statistic = %.3f, p-value = %.4f -> %s\n",
              kpss_level$statistic, kpss_level$p.value,
              ifelse(kpss_level$p.value < 0.05, "reject H0 (non-stationary)", "fail to reject H0 (stationary)")))
  cat(sprintf("KPSS (trend): statistic = %.3f, p-value = %.4f -> %s\n\n",
              kpss_trend$statistic, kpss_trend$p.value,
              ifelse(kpss_trend$p.value < 0.05, "reject H0 (non-stationary around trend)", "fail to reject H0 (trend-stationary)")))

  list(
    adf = list(statistic = unname(adf$statistic), p_value = adf$p.value,
               stationary = adf$p.value < 0.05),
    kpss_level = list(statistic = unname(kpss_level$statistic), p_value = kpss_level$p.value,
                       stationary = kpss_level$p.value >= 0.05),
    kpss_trend = list(statistic = unname(kpss_trend$statistic), p_value = kpss_trend$p.value,
                       stationary = kpss_trend$p.value >= 0.05)
  )
}

# ---- helper: Ljung-Box at several lags ----
run_ljung_box <- function(x, lags, fitdf = 0, label) {
  out <- lapply(lags, function(L) {
    bt <- Box.test(x, lag = L, type = "Ljung-Box", fitdf = fitdf)
    cat(sprintf("Ljung-Box [%s] lag=%d: statistic = %.2f, p-value = %.4g -> %s\n",
                label, L, bt$statistic, bt$p.value,
                ifelse(bt$p.value < 0.05, "reject H0 (autocorrelated)", "fail to reject H0 (white noise)")))
    list(lag = L, statistic = unname(bt$statistic), p_value = bt$p.value,
         autocorrelated = bt$p.value < 0.05)
  })
  cat("\n")
  out
}

# ---- helper: save an Acf/Pacf ggplot in the same style as visualisation.R ----
save_acf_pacf <- function(x, max_lag, filename_prefix, title_suffix) {
  p_acf <- ggAcf(x, lag.max = max_lag) +
    labs(title = paste("ACF -", title_suffix), x = "Lag (hours)", y = "ACF") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  ggsave(file.path(fig_dir, paste0("acf_", filename_prefix, ".png")),
         p_acf, width = 8, height = 5, dpi = 300)

  p_pacf <- ggPacf(x, lag.max = max_lag) +
    labs(title = paste("PACF -", title_suffix), x = "Lag (hours)", y = "PACF") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  ggsave(file.path(fig_dir, paste0("pacf_", filename_prefix, ".png")),
         p_pacf, width = 8, height = 5, dpi = 300)
}


# ============================================================
# 1. Diagnostics on the ORIGINAL series
# ============================================================

# ACF/PACF out to 200 lags -> long enough to see both the daily (24h) and
# the start of the weekly (168h) seasonal spikes.
save_acf_pacf(tv_ts_daily, max_lag = 200, filename_prefix = "original",
              title_suffix = "Original traffic_volume series")

stationarity_original <- run_stationarity_tests(df$traffic_volume, "ORIGINAL series")
ljung_original <- run_ljung_box(df$traffic_volume, lags = c(24, 48, 168),
                                 fitdf = 0, label = "original")

results$original <- list(
  stationarity = stationarity_original,
  ljung_box = ljung_original
)


# ============================================================
# 2. Recommended differencing orders
# ============================================================

# Regular (non-seasonal) differencing order
d_reg <- ndiffs(tv_ts_daily, test = "adf")

# Seasonal differencing order at the DAILY period (24h)
D_daily <- nsdiffs(tv_ts_daily, test = "ocsb")

# Seasonal differencing order at the WEEKLY period (168h), tested separately
# since nsdiffs() only evaluates one seasonal period at a time.
tv_ts_weekly <- ts(df$traffic_volume, frequency = WEEKLY_PERIOD)
D_weekly <- nsdiffs(tv_ts_weekly, test = "ocsb")

cat("Recommended differencing -> non-seasonal d =", d_reg,
    ", seasonal D (period=24h) =", D_daily,
    ", seasonal D (period=168h) =", D_weekly, "\n\n")

results$recommended_differencing <- list(
  d_nonseasonal = d_reg,
  D_seasonal_24h = D_daily,
  D_seasonal_168h = D_weekly
)


# ============================================================
# 3. Apply differencing and re-check stationarity
# ============================================================
# Apply in the order: seasonal (24h) difference first if recommended, then
# regular first-difference if still recommended. (Weekly (168h) differencing
# is reported above for completeness/for the SARIMA teammate to consider as
# an alternate seasonal period, but is not chained here since applying both
# a 24h and a 168h seasonal difference on top of each other on an already
# short-relative-to-168 test window can over-difference the series.)

tv_work <- df$traffic_volume

if (D_daily > 0) {
  tv_work <- diff(tv_work, lag = DAILY_PERIOD, differences = D_daily)
  cat("Applied seasonal differencing at lag", DAILY_PERIOD,
      "(D =", D_daily, "). Remaining length:", length(tv_work), "\n")
}
if (d_reg > 0) {
  tv_work <- diff(tv_work, differences = d_reg)
  cat("Applied non-seasonal differencing (d =", d_reg,
      "). Remaining length:", length(tv_work), "\n")
}
cat("\n")

tv_diff_ts <- ts(tv_work, frequency = DAILY_PERIOD)

save_acf_pacf(tv_diff_ts, max_lag = 200, filename_prefix = "differenced",
              title_suffix = sprintf("Differenced series (d=%d, D24=%d)", d_reg, D_daily))

stationarity_differenced <- run_stationarity_tests(tv_work, "DIFFERENCED series")
ljung_differenced <- run_ljung_box(tv_work, lags = c(24, 48, 168),
                                    fitdf = 0, label = "differenced")

results$differenced <- list(
  d_applied = d_reg,
  D_applied_24h = D_daily,
  n_obs_after_differencing = length(tv_work),
  stationarity = stationarity_differenced,
  ljung_box = ljung_differenced
)


# ============================================================
# 4. STL decomposition (trend / seasonal / remainder), daily period
# ============================================================
# mstl() handles a single ts cleanly; using the daily-frequency ts here.
# This is a useful companion figure: it visually separates the trend and
# 24h-seasonal components already implied by the ACF/PACF spikes above, and
# gives a sense of how much variance is "explained" by seasonality vs. left
# in the remainder for SARIMA/LSTM to capture.
decomp <- mstl(tv_ts_daily)

p_decomp <- autoplot(decomp) +
  labs(title = "STL Decomposition of Hourly Traffic Volume (daily period = 24h)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
ggsave(file.path(fig_dir, "stl_decomposition.png"), p_decomp, width = 9, height = 7, dpi = 300)

# Rough seasonal strength measure (Hyndman & Athanasopoulos, 2021 formula)
remainder <- decomp[, "Remainder"]
seasonal  <- decomp[, "Seasonal24"]
seasonal_strength <- max(0, 1 - var(remainder) / var(remainder + seasonal))
cat("Approx. seasonal strength (24h):", round(seasonal_strength, 3), "\n\n")

results$stl_decomposition <- list(
  period_used = DAILY_PERIOD,
  approx_seasonal_strength_24h = seasonal_strength
)


# ============================================================
# 5. Save all numeric results to JSON
# ============================================================
write_json(results, json_path, auto_unbox = TRUE, pretty = TRUE, digits = 6)

cat("Saved figures to '", fig_dir, "/':\n", sep = "")
cat(" - acf_original.png / pacf_original.png\n")
cat(" - acf_differenced.png / pacf_differenced.png\n")
cat(" - stl_decomposition.png\n")
cat("Saved all test statistics/p-values to:", json_path, "\n")