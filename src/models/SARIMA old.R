# ============================================================
#  Metro Interstate Traffic Volume – Statistical Analysis
#  Preprocessing & Exploratory Analysis for SARIMA Modeling
# ============================================================
# Dataset: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
# Requires: tidyverse, lubridate, forecast, tseries, ggplot2, zoo, moments

# ── 0. Install / Load Packages ───────────────────────────────
pkgs <- c("tidyverse", "lubridate", "forecast", "tseries",
          "ggplot2", "zoo", "moments", "scales", "gridExtra")
installed <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(installed)) install.packages(installed, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

cat("✔ Packages loaded\n")


# ── 1. Load Data ─────────────────────────────────────────────
# Place the CSV in your working directory, or adjust the path below.
# Download from: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
df_raw <- read.csv("Metro_Interstate_Traffic_Volume.csv", stringsAsFactors = FALSE)

cat("✔ Raw data loaded:", nrow(df_raw), "rows ×", ncol(df_raw), "columns\n")
cat("\n── Column Names ──\n")
print(names(df_raw))
cat("\n── Head ──\n")
print(head(df_raw))


# ── 2. Preprocessing ─────────────────────────────────────────

## 2a. Parse datetime
df <- df_raw %>%
  mutate(
    date_time = as.POSIXct(date_time, format = "%Y-%m-%d %H:%M:%S", tz = "America/Chicago"),
    hour      = hour(date_time),
    day_of_week = wday(date_time, label = TRUE, abbr = TRUE),
    month     = month(date_time, label = TRUE, abbr = TRUE),
    year      = year(date_time),
    is_weekend = ifelse(wday(date_time) %in% c(1, 7), 1, 0),
    is_holiday = ifelse(holiday != "None", 1, 0)
  )

## 2b. Check for duplicates
n_dupes <- sum(duplicated(df$date_time))
cat("\n── Duplicated timestamps:", n_dupes, "──\n")
if (n_dupes > 0) {
  df <- df %>%
    arrange(date_time) %>%
    group_by(date_time) %>%
    summarise(
      traffic_volume = mean(traffic_volume, na.rm = TRUE),
      temp           = mean(temp, na.rm = TRUE),
      rain_1h        = mean(rain_1h, na.rm = TRUE),
      snow_1h        = mean(snow_1h, na.rm = TRUE),
      clouds_all     = mean(clouds_all, na.rm = TRUE),
      is_holiday     = max(is_holiday),
      is_weekend     = max(is_weekend),
      hour           = first(hour),
      day_of_week    = first(day_of_week),
      month          = first(month),
      year           = first(year),
      .groups = "drop"
    )
  cat("   → Duplicates removed by averaging. Rows now:", nrow(df), "\n")
}

## 2c. Create a complete hourly time grid & detect gaps
full_grid <- data.frame(
  date_time = seq(min(df$date_time), max(df$date_time), by = "hour")
)
df_complete <- full_grid %>%
  left_join(df, by = "date_time")

n_missing_ts <- sum(is.na(df_complete$traffic_volume))
cat("\n── Missing hourly slots after completing grid:", n_missing_ts, "──\n")

## 2d. Fill gaps with linear interpolation
df_complete <- df_complete %>%
  mutate(
    traffic_volume = na.approx(traffic_volume, na.rm = FALSE),
    temp           = na.approx(temp,           na.rm = FALSE),
    rain_1h        = na.approx(rain_1h,        na.rm = FALSE),
    snow_1h        = na.approx(snow_1h,        na.rm = FALSE),
    clouds_all     = na.approx(clouds_all,     na.rm = FALSE)
  )

## 2e. Re-derive time features on the complete grid
df_complete <- df_complete %>%
  mutate(
    hour        = hour(date_time),
    day_of_week = wday(date_time, label = TRUE, abbr = TRUE),
    month       = month(date_time, label = TRUE, abbr = TRUE),
    year        = year(date_time),
    is_weekend  = ifelse(wday(date_time) %in% c(1, 7), 1, 0)
  )

## 2f. Temperature conversion (Kelvin → Celsius) & outlier flag
df_complete <- df_complete %>%
  mutate(
    temp_C        = temp - 273.15,
    temp_outlier  = ifelse(temp_C < -50 | temp_C > 50, 1, 0)
  )
cat("── Temperature outliers flagged:", sum(df_complete$temp_outlier, na.rm = TRUE), "\n")

# Replace temperature outliers with interpolated values
if (sum(df_complete$temp_outlier, na.rm = TRUE) > 0) {
  df_complete$temp_C[df_complete$temp_outlier == 1] <- NA
  df_complete$temp_C <- na.approx(df_complete$temp_C, na.rm = FALSE)
}

cat("\n✔ Preprocessing complete. Rows:", nrow(df_complete), "\n")


# ── 3. Descriptive Statistics ─────────────────────────────────
cat("\n═══════════════════════════════════════════\n")
cat("  DESCRIPTIVE STATISTICS – traffic_volume\n")
cat("═══════════════════════════════════════════\n")

tv <- df_complete$traffic_volume
stats_tbl <- data.frame(
  Statistic = c("N", "Mean", "Median", "Std Dev", "Min", "Max",
                "Q1 (25%)", "Q3 (75%)", "IQR", "Skewness", "Kurtosis",
                "Missing (NAs)"),
  Value = c(
    length(tv),
    round(mean(tv, na.rm = TRUE), 2),
    round(median(tv, na.rm = TRUE), 2),
    round(sd(tv, na.rm = TRUE), 2),
    round(min(tv, na.rm = TRUE), 2),
    round(max(tv, na.rm = TRUE), 2),
    round(quantile(tv, 0.25, na.rm = TRUE), 2),
    round(quantile(tv, 0.75, na.rm = TRUE), 2),
    round(IQR(tv, na.rm = TRUE), 2),
    round(skewness(tv, na.rm = TRUE), 4),
    round(kurtosis(tv, na.rm = TRUE), 4),
    sum(is.na(tv))
  )
)
print(stats_tbl, row.names = FALSE)


# ── 4. Stationarity Tests ────────────────────────────────────
cat("\n═══════════════════════════════════════\n")
cat("  STATIONARITY TESTS (on hourly series)\n")
cat("═══════════════════════════════════════\n")

# Use a subset for speed (first 5,000 observations)
tv_sub <- na.omit(tv)[1:5000]

# Augmented Dickey-Fuller
adf_res <- adf.test(tv_sub, alternative = "stationary")
cat("\n[ADF Test]\n")
cat("  Statistic:", round(adf_res$statistic, 4), "\n")
cat("  p-value  :", round(adf_res$p.value, 4),
    ifelse(adf_res$p.value < 0.05, "→ STATIONARY ✔", "→ NON-STATIONARY ✗"), "\n")

# KPSS Test
kpss_res <- kpss.test(tv_sub)
cat("\n[KPSS Test]\n")
cat("  Statistic:", round(kpss_res$statistic, 4), "\n")
cat("  p-value  :", round(kpss_res$p.value, 4),
    ifelse(kpss_res$p.value > 0.05, "→ STATIONARY ✔", "→ NON-STATIONARY ✗"), "\n")

# Phillips-Perron
pp_res <- pp.test(tv_sub)
cat("\n[Phillips-Perron Test]\n")
cat("  Statistic:", round(pp_res$statistic, 4), "\n")
cat("  p-value  :", round(pp_res$p.value, 4),
    ifelse(pp_res$p.value < 0.05, "→ STATIONARY ✔", "→ NON-STATIONARY ✗"), "\n")


# ── 5. ACF / PACF Analysis ──────────────────────────────────
cat("\n[Seasonal Differencing & ndiffs / nsdiffs]\n")

# Recommended differencing orders
d_val  <- ndiffs(tv_sub)
D_val  <- nsdiffs(ts(tv_sub, frequency = 24))
cat("  Suggested d (non-seasonal):", d_val, "\n")
cat("  Suggested D (seasonal, s=24):", D_val, "\n")


# ── 6. Create ts Object ──────────────────────────────────────
# Season = 24 (daily pattern) for hourly data
# Adjust to 24*7 = 168 if you want weekly seasonality
ts_hourly <- ts(df_complete$traffic_volume, frequency = 24)

cat("\n✔ ts object created. Length:", length(ts_hourly),
    " | Frequency:", frequency(ts_hourly), "\n")


# ── 7. Visualisations ────────────────────────────────────────
cat("\n── Generating plots … ──\n")

## 7a. Full time-series overview
p1 <- df_complete %>%
  filter(!is.na(traffic_volume)) %>%
  ggplot(aes(date_time, traffic_volume)) +
  geom_line(colour = "#2c7fb8", linewidth = 0.3, alpha = 0.7) +
  labs(title = "Hourly Traffic Volume – I-94 Westbound (2012–2018)",
       x = "Date", y = "Traffic Volume (vehicles/hr)") +
  scale_x_datetime(labels = date_format("%Y"), date_breaks = "1 year") +
  theme_minimal(base_size = 11)

## 7b. Average by hour of day
p2 <- df_complete %>%
  group_by(hour) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(hour, avg_vol)) +
  geom_line(colour = "#e34a33", linewidth = 1.2) +
  geom_point(colour = "#e34a33", size = 2) +
  labs(title = "Average Traffic by Hour of Day",
       x = "Hour (0–23)", y = "Mean Volume") +
  scale_x_continuous(breaks = 0:23) +
  theme_minimal(base_size = 11)

## 7c. Average by day of week
p3 <- df_complete %>%
  group_by(day_of_week) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(day_of_week, avg_vol, fill = day_of_week)) +
  geom_col(show.legend = FALSE, alpha = 0.85) +
  labs(title = "Average Traffic by Day of Week",
       x = "Day", y = "Mean Volume") +
  theme_minimal(base_size = 11)

## 7d. Average by month
p4 <- df_complete %>%
  group_by(month) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(month, avg_vol, group = 1)) +
  geom_line(colour = "#31a354", linewidth = 1.2) +
  geom_point(colour = "#31a354", size = 2.5) +
  labs(title = "Average Traffic by Month",
       x = "Month", y = "Mean Volume") +
  theme_minimal(base_size = 11)

## 7e. Distribution / histogram
p5 <- df_complete %>%
  filter(!is.na(traffic_volume)) %>%
  ggplot(aes(traffic_volume)) +
  geom_histogram(bins = 60, fill = "#756bb1", colour = "white", alpha = 0.85) +
  labs(title = "Distribution of Hourly Traffic Volume",
       x = "Traffic Volume", y = "Count") +
  theme_minimal(base_size = 11)

## 7f. Weekday vs Weekend
p6 <- df_complete %>%
  mutate(day_type = ifelse(is_weekend == 1, "Weekend", "Weekday")) %>%
  group_by(hour, day_type) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(hour, avg_vol, colour = day_type)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 1.8) +
  labs(title = "Weekday vs Weekend Traffic Pattern",
       x = "Hour (0–23)", y = "Mean Volume", colour = NULL) +
  scale_colour_manual(values = c("Weekday" = "#2c7fb8", "Weekend" = "#e34a33")) +
  scale_x_continuous(breaks = seq(0, 23, 2)) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

## Save combined plot
grid_plot <- gridExtra::grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 2)
ggsave("traffic_eda_plots.png", grid_plot, width = 14, height = 16, dpi = 150)
cat("✔ EDA plot saved → traffic_eda_plots.png\n")

## 7g. ACF & PACF plots (for SARIMA order identification)
png("traffic_acf_pacf.png", width = 1200, height = 500, res = 120)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
acf(tv_sub,  lag.max = 72, main = "ACF  – Traffic Volume (first 5000 obs)")
pacf(tv_sub, lag.max = 72, main = "PACF – Traffic Volume (first 5000 obs)")
dev.off()
cat("✔ ACF/PACF plot saved → traffic_acf_pacf.png\n")

## 7h. STL decomposition (captures daily & weekly seasonality)
ts_week <- ts(df_complete$traffic_volume[1:(24*7*10)], frequency = 24)
stl_fit <- stl(ts_week, s.window = "periodic")

png("traffic_stl_decomposition.png", width = 1000, height = 700, res = 120)
plot(stl_fit, main = "STL Decomposition – Traffic Volume (s=24, 10 weeks)")
dev.off()
cat("✔ STL decomposition plot saved → traffic_stl_decomposition.png\n")


# ── 8. Auto SARIMA Identification (on subset for speed) ──────
cat("\n── Auto SARIMA identification (subset: 2000 obs) ──\n")
ts_sub2 <- ts(na.omit(df_complete$traffic_volume)[1:2000], frequency = 24)

auto_model <- auto.arima(
  ts_sub2,
  seasonal    = TRUE,
  stepwise    = TRUE,
  approximation = TRUE,
  trace       = TRUE
)
cat("\n✔ Best model identified:\n")
print(summary(auto_model))

# Save model summary
sink("auto_arima_summary.txt")
cat("Auto ARIMA Model Summary\n")
cat("========================\n\n")
print(summary(auto_model))
cat("\nAIC:", AIC(auto_model), "\n")
cat("BIC:", BIC(auto_model), "\n")
sink()
cat("✔ Auto ARIMA summary saved → auto_arima_summary.txt\n")


# ── 9. Residual Diagnostics ───────────────────────────────────
cat("\n── Residual Diagnostics ──\n")
residuals_model <- residuals(auto_model)

png("traffic_residual_diagnostics.png", width = 1100, height = 700, res = 120)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(residuals_model, main = "Residuals over Time",
     ylab = "Residual", xlab = "Time")
abline(h = 0, col = "red", lty = 2)
hist(residuals_model, breaks = 40, main = "Residual Distribution",
     xlab = "Residual", col = "steelblue")
acf(residuals_model,  lag.max = 48, main = "ACF of Residuals")
pacf(residuals_model, lag.max = 48, main = "PACF of Residuals")
dev.off()
cat("✔ Residual diagnostics saved → traffic_residual_diagnostics.png\n")

# Ljung-Box test on residuals
lb_test <- Box.test(residuals_model, lag = 24, type = "Ljung-Box")
cat("\n[Ljung-Box Test on Residuals (lag=24)]\n")
cat("  Statistic:", round(lb_test$statistic, 4), "\n")
cat("  p-value  :", round(lb_test$p.value, 4),
    ifelse(lb_test$p.value > 0.05,
           "→ Residuals are WHITE NOISE ✔",
           "→ Residuals show autocorrelation ✗"), "\n")


# ── 10. Forecast (next 48 hours) ─────────────────────────────
cat("\n── Generating 48-hour forecast ──\n")
fc <- forecast(auto_model, h = 48)

png("traffic_forecast_48h.png", width = 1100, height = 500, res = 120)
autoplot(fc) +
  labs(title = "48-Hour Traffic Volume Forecast",
       x = "Time (hours)", y = "Traffic Volume") +
  theme_minimal(base_size = 11)
dev.off()
cat("✔ Forecast plot saved → traffic_forecast_48h.png\n")


# ── 11. Session & Summary ─────────────────────────────────────
cat("\n═══════════════════════════════════════════════════\n")
cat("  ANALYSIS COMPLETE\n")
cat("  Output files:\n")
cat("    • traffic_eda_plots.png\n")
cat("    • traffic_acf_pacf.png\n")
cat("    • traffic_stl_decomposition.png\n")
cat("    • traffic_residual_diagnostics.png\n")
cat("    • traffic_forecast_48h.png\n")
cat("    • auto_arima_summary.txt\n")
cat("═══════════════════════════════════════════════════\n")
cat("\n── R Session Info ──\n")
print(sessionInfo())