# ============================================================
# SARIMA / SARIMAX Model — Metro Interstate Traffic Volume
# Input : data/processed/traffic_volume_processed.csv
# Output: output/models/SARIMA/ (or SARIMAX_Holiday / SARIMAX_Full)
# ============================================================

# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("forecast", "tseries", "lubridate", "dplyr",
          "ggplot2", "zoo", "jsonlite", "tibble")
new <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))


# ── 1. Paths & Model Configuration ───────────────────────────
input_path       <- "data/processed/traffic_volume_processed.csv"
MODEL_TYPE       <- "full" # Options: "pure", "holiday", "full"
REFIT_WITH_VALID <- FALSE  # Set TRUE to combine Train+Valid before test forecast
FAST_RUN         <- TRUE   # Set FALSE for exhaustive search before final submission

output_dir <- switch(MODEL_TYPE,
  "pure"    = "output/models/SARIMA",
  "holiday" = "output/models/SARIMAX_Holiday",
  "full"    = "output/models/SARIMAX_Full"
)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)


# ── 2. Load Processed Data ───────────────────────────────────
if (!file.exists(input_path)) stop("Input file not found at: ", input_path)
df <- read.csv(input_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
df <- df %>% arrange(date_time)

cat("Loaded:", nrow(df), "rows | Range:",
    as.character(min(df$date_time)), "to", as.character(max(df$date_time)), "\n")

if (!"is_imputed" %in% names(df)) {
  warning("is_imputed column not found — assuming all rows observed.")
  df$is_imputed <- 0L
}
cat(sprintf("Imputed hours: %d / %d (%.1f%%)\n",
            sum(df$is_imputed), nrow(df), 100 * mean(df$is_imputed)))


# ── 2b. Restore Factor Levels ────────────────────────────────
df$day_of_week <- weekdays(df$date_time)
day_levels <- c("Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday")
stopifnot(all(unique(df$day_of_week) %in% day_levels))
df$day_of_week <- factor(df$day_of_week, levels = day_levels)


# ── 3. Guard: Complete Hourly Grid ───────────────────────────
expected_hours <- as.numeric(difftime(max(df$date_time), min(df$date_time), units = "hours")) + 1
if (nrow(df) != expected_hours) {
  stop("Series is not a complete hourly grid. Expected ", expected_hours,
       " rows but found ", nrow(df), ". Re-run preprocessing script.")
}
cat("Grid check passed.\n")


# ── 4. Constants ─────────────────────────────────────────────
SEASON  <- 168 # Weekly seasonal period (24 hours * 7 days)
D_ORDER <- 1   # Seasonal differencing (lag=168)
d_ORDER <- 0   # Ordinal differencing


# ── 5. Train / Validation / Test Split ───────────────────────
stopifnot("split" %in% names(df))

train_idx <- which(df$split == "train")
valid_idx <- which(df$split == "validation")
test_idx  <- which(df$split == "test")

TRAIN_N <- length(train_idx)
VALID_H <- length(valid_idx)
TEST_H  <- length(test_idx)
N       <- nrow(df)

stopifnot(TRAIN_N + VALID_H + TEST_H == N)

cat("\n=== Train / Validation / Test Split ===\n")
cat(sprintf("Train : %d rows | %s to %s | %.1f%%\n",
            TRAIN_N, df$date_time[train_idx[1]], df$date_time[tail(train_idx, 1)], 100 * TRAIN_N / N))
cat(sprintf("Valid : %d rows | %s to %s | %.1f%%\n",
            VALID_H, df$date_time[valid_idx[1]], df$date_time[tail(valid_idx, 1)], 100 * VALID_H / N))
cat(sprintf("Test  : %d rows | %s to %s | %.1f%%\n",
            TEST_H, df$date_time[test_idx[1]], df$date_time[tail(test_idx, 1)], 100 * TEST_H / N))

n_imputed_in_valid <- sum(df$is_imputed[valid_idx])
n_imputed_in_test  <- sum(df$is_imputed[test_idx])
test_imputed_idx   <- df$is_imputed[test_idx]


# ── 6. Differencing (Training Isolation) ─────────────────────
train_raw <- df$traffic_volume[train_idx]
valid_raw <- df$traffic_volume[valid_idx]
test_raw  <- df$traffic_volume[test_idx]

# Training: seasonal diff
diff_train <- diff(train_raw, lag = SEASON, differences = D_ORDER)
n_pad      <- TRAIN_N - length(diff_train)

# Validation: anchor against tail of training
valid_anchor <- c(tail(train_raw, SEASON), valid_raw)
diff_valid   <- diff(valid_anchor, lag = SEASON, differences = D_ORDER)[1:VALID_H]

# Test: anchor against tail of validation
test_anchor <- c(tail(valid_raw, SEASON), test_raw)
diff_test   <- diff(test_anchor, lag = SEASON, differences = D_ORDER)[1:TEST_H]

ts_train_diff <- ts(diff_train, frequency = 1)
ts_valid_diff <- ts(diff_valid, frequency = 1)
ts_test_diff  <- ts(diff_test, frequency = 1)


# ── 6b. Stationarity Diagnostics ─────────────────────────────
cat("\n=== Stationarity Tests: RAW vs DIFFERENCED ===\n")
adf_raw  <- adf.test(train_raw, alternative = "stationary")
kpss_raw <- kpss.test(train_raw, null = "Level")

adf_diff  <- adf.test(diff_train, alternative = "stationary")
kpss_diff <- kpss.test(diff_train, null = "Level")

cat(sprintf("ADF  (raw) : stat = %.4f, p = %.4f\n", adf_raw$statistic, adf_raw$p.value))
cat(sprintf("KPSS (raw) : stat = %.4f, p = %.4f\n", kpss_raw$statistic, kpss_raw$p.value))
cat(sprintf("ADF  (diff): stat = %.4f, p = %.4f\n", adf_diff$statistic, adf_diff$p.value))
cat(sprintf("KPSS (diff): stat = %.4f, p = %.4f\n", kpss_diff$statistic, kpss_diff$p.value))

stationarity_summary <- data.frame(
  series       = c("raw_train", "diff_train"),
  adf_stat     = c(adf_raw$statistic, adf_diff$statistic),
  adf_pvalue   = c(adf_raw$p.value,  adf_diff$p.value),
  adf_verdict  = c(ifelse(adf_raw$p.value < 0.05, "stationary", "non-stationary"),
                   ifelse(adf_diff$p.value < 0.05, "stationary", "non-stationary")),
  kpss_stat    = c(kpss_raw$statistic, kpss_diff$statistic),
  kpss_pvalue  = c(kpss_raw$p.value,  kpss_diff$p.value),
  kpss_verdict = c(ifelse(kpss_raw$p.value > 0.05, "stationary", "non-stationary"),
                   ifelse(kpss_diff$p.value > 0.05, "stationary", "non-stationary"))
)
write.csv(stationarity_summary, file.path(output_dir, "stationarity_summary.csv"), row.names = FALSE)


# ── 7. Build Exogenous Matrices ──────────────────────────────
build_xreg <- function(idx, type = MODEL_TYPE) {
  if (type == "pure") return(NULL)
  d <- df[idx, ]
  if (type == "holiday") {
    mm <- model.matrix(~ is_holiday, data = d)[, -1, drop = FALSE]
  } else if (type == "full") {
    mm <- model.matrix(~ is_holiday + temp + rain_1h + snow_1h + clouds_all, data = d)[, -1, drop = FALSE]
  }
  return(mm)
}

xreg_train_full <- build_xreg(train_idx[(n_pad + 1):TRAIN_N])
xreg_valid_full <- build_xreg(valid_idx)
xreg_test_full  <- build_xreg(test_idx)


# ── 7b. Collinearity Diagnostics ─────────────────────────────
if (!is.null(xreg_train_full)) {
  qr_rank <- qr(xreg_train_full)$rank
  cat(sprintf("\nDesign matrix: %d columns, rank = %d %s\n",
              ncol(xreg_train_full), qr_rank,
              ifelse(qr_rank < ncol(xreg_train_full), "-- RANK DEFICIENT", "-- full rank")))

  weather_vars <- intersect(c("temp", "rain_1h", "snow_1h", "clouds_all"), names(df))
  if (length(weather_vars) > 1) {
    weather_cor <- cor(df[train_idx, weather_vars], use = "complete.obs")
    cat("\nWeather variable correlations (train):\n")
    print(round(weather_cor, 2))
  }
} else {
  cat("\nPure SARIMA mode selected: Exogenous design matrix is NULL.\n")
}


# ── 8. Final Model Fitting ────────────────────────────────────
if (REFIT_WITH_VALID) {
  ts_fit_final   <- ts(c(ts_train_diff, ts_valid_diff), frequency = 1)
  xreg_fit_final <- if (!is.null(xreg_train_full)) rbind(xreg_train_full, xreg_valid_full) else NULL
} else {
  ts_fit_final   <- ts_train_diff
  xreg_fit_final <- xreg_train_full
}

cat("\nFitting ARIMA model via auto.arima...\n")
fit <- auto.arima(
  ts_fit_final,
  xreg          = xreg_fit_final,
  seasonal      = FALSE,
  d             = 0,
  D             = 0,
  stepwise      = FAST_RUN,
  approximation = FAST_RUN,
  trace         = TRUE
)


# ── 9. Residual Diagnostics ───────────────────────────────────
resid <- residuals(fit)

lb_48  <- Box.test(resid, lag = 48,  type = "Ljung-Box", fitdf = sum(fit$arma[1:4]))
lb_168 <- Box.test(resid, lag = 168, type = "Ljung-Box", fitdf = sum(fit$arma[1:4]))

cat(sprintf("\nLjung-Box (lag= 48): stat = %.4f, p = %.4f (%s)\n",
            lb_48$statistic, lb_48$p.value, ifelse(lb_48$p.value > 0.05, "white noise", "autocorrelated")))
cat(sprintf("Ljung-Box (lag=168): stat = %.4f, p = %.4f (%s)\n",
            lb_168$statistic, lb_168$p.value, ifelse(lb_168$p.value > 0.05, "white noise", "autocorrelated")))

set.seed(42)
sw_sample <- as.numeric(resid)
if (length(sw_sample) > 5000) sw_sample <- sample(sw_sample, 5000)
sw <- shapiro.test(sw_sample)

png(file.path(output_dir, "sarima_residual_diagnostics.png"), width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(resid, main = "Residuals over Time", ylab = "Residual", xlab = "Time (hours)", col = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)
h_info <- hist(resid, breaks = 60, plot = FALSE)
hist(resid, breaks = 60, col = "#756bb1", border = "white", main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean(resid), sd(resid)) * length(resid) * diff(h_info$breaks[1:2]), col = "red", lwd = 2, add = TRUE)
acf(resid, lag.max = 200, main = "ACF of Residuals")
pacf(resid, lag.max = 200, main = "PACF of Residuals")
dev.off()


# ── 10. Multi-Step Test Forecasting ─────────────────────────
fc <- forecast(fit, xreg = xreg_test_full, h = TEST_H)

recon <- c(tail(valid_raw, SEASON), rep(NA_real_, TEST_H))
for (i in seq_len(TEST_H)) {
  recon[SEASON + i] <- as.numeric(fc$mean[i]) + recon[i]
}
fc_mean_raw <- recon[(SEASON + 1):(SEASON + TEST_H)]

anchor_used <- recon[1:TEST_H]
fc_lo80_raw <- as.numeric(fc$lower[, 1]) + anchor_used
fc_hi80_raw <- as.numeric(fc$upper[, 1]) + anchor_used
fc_lo95_raw <- as.numeric(fc$lower[, 2]) + anchor_used
fc_hi95_raw <- as.numeric(fc$upper[, 2]) + anchor_used

# ── 11. Forecast Evaluation Metrics ──────────────────────────
actual    <- test_raw
predicted <- fc_mean_raw

mae   <- mean(abs(actual - predicted))
rmse  <- sqrt(mean((actual - predicted)^2))
me    <- mean(actual - predicted)
mape  <- mean(abs((actual - predicted) / ifelse(actual == 0, NA, actual)), na.rm = TRUE) * 100
smape <- mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-8)) * 100

# Seasonal-naive MASE scale (lag = 168 hours)
scale <- mean(abs(
  train_raw[(SEASON + 1):length(train_raw)] - 
    train_raw[1:(length(train_raw) - SEASON)]
))

# MASE
mase <- mae / scale

cat("\n--- Forecast Accuracy (Test Set — All Hours) ---\n")
cat(sprintf("ME    : %10.4f\n", me))
cat(sprintf("RMSE  : %10.4f\n", rmse))
cat(sprintf("MAE   : %10.4f\n", mae))
cat(sprintf("MAPE  : %10.4f %%\n", mape))
cat(sprintf("sMAPE : %10.4f %%\n", smape))
cat(sprintf("MASE  : %10.4f\n", mase))

# ── 11b. Observed Hours Evaluation ──────────────────────────
if (n_imputed_in_test > 0) {
  keep          <- test_imputed_idx == 0
  actual_obs    <- actual[keep]
  predicted_obs <- predicted[keep]
  rmse_obs  <- sqrt(mean((actual_obs - predicted_obs)^2))
  mae_obs   <- mean(abs(actual_obs - predicted_obs))
  mape_obs  <- mean(abs((actual_obs - predicted_obs) / ifelse(actual_obs == 0, NA, actual_obs)), na.rm = TRUE) * 100
  smape_obs <- mean(2 * abs(actual_obs - predicted_obs) / (abs(actual_obs) + abs(predicted_obs) + 1e-8)) * 100
} else {
  rmse_obs <- rmse; mae_obs <- mae; mape_obs <- mape; smape_obs <- smape
}


# ── 12. Visualization: Forecast vs Actual ─────────────────────
fc_df <- tibble(
  date_time = df$date_time[test_idx],
  actual    = actual,
  forecast  = predicted,
  lo80      = fc_lo80_raw,
  hi80      = fc_hi80_raw,
  lo95      = fc_lo95_raw,
  hi95      = fc_hi95_raw
)

p_fc <- ggplot(fc_df, aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.5) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.6, linetype = "dashed") +
  labs(
    title    = sprintf("%s — %d-Day Test Forecast vs Actual", toupper(MODEL_TYPE), TEST_H %/% 24),
    subtitle = sprintf("RMSE = %.1f | MAE = %.1f | MAPE = %.2f%% | MASE = %.3f", rmse, mae, mape, mase),
    x        = "Date",
    y        = "Traffic Volume (vehicles / hr)",
    caption  = "Shaded bands: 80% (dark) and 95% (light) prediction intervals"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_forecast_vs_actual.png"), p_fc, width = 14, height = 5, dpi = 150)


# ── 13. Visualization: First-Week Zoom ────────────────────────
p_zoom <- ggplot(fc_df[1:min(24 * 7, nrow(fc_df)), ], aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.7, linetype = "dashed") +
  labs(title = sprintf("%s — First 7 Days of Test Window (Zoom)", toupper(MODEL_TYPE)),
       x = "Date", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_forecast_week1_zoom.png"), p_zoom, width = 12, height = 4, dpi = 150)


# ── 14. Save Forecast Output ──────────────────────────────────
fc_out <- fc_df
fc_out$date_time <- format(fc_out$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(fc_out, file.path(output_dir, "sarima_forecast_results.csv"), row.names = FALSE)


# ── 15. Export JSON Summary ───────────────────────────────────
model_order <- arimaorder(fit)

summary_list <- list(
  model = list(
    type           = MODEL_TYPE,
    order          = list(p = model_order[1], d = model_order[2], q = model_order[3]),
    seasonal_order = list(P = model_order[4], D = model_order[5], Q = model_order[6], s = SEASON),
    differencing   = list(d = d_ORDER, D = D_ORDER, lag = SEASON),
    xreg_cols      = if (!is.null(xreg_train_full)) colnames(xreg_train_full) else "None (Pure SARIMA)",
    AIC            = fit$aic,
    AICc           = fit$aicc,
    BIC            = fit$bic,
    log_likelihood = fit$loglik
  ),
  residual_tests = list(
    Ljung_Box_lag48_p  = round(lb_48$p.value, 4),
    Ljung_Box_lag168_p = round(lb_168$p.value, 4),
    Shapiro_Wilk_p     = round(sw$p.value, 4)
  ),
  accuracy = list(
    ME    = round(me, 4),
    RMSE  = round(rmse, 4),
    MAE   = round(mae, 4),
    MAPE  = round(mape, 4),
    sMAPE = round(smape, 4),
    MASE  = round(mase, 4)
  ),
  accuracy_observed_only = list(
    n_imputed_in_test = n_imputed_in_test,
    RMSE  = round(rmse_obs, 4),
    MAE   = round(mae_obs, 4),
    MAPE  = round(mape_obs, 4),
    sMAPE = round(smape_obs, 4)
  ),
  split = list(
    total_n          = N,
    train_n          = TRAIN_N,
    valid_h          = VALID_H,
    test_h           = TEST_H,
    refit_with_valid = REFIT_WITH_VALID
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "sarima_model_summary.json"))


# ── 16. Save Fitted Model Object ─────────────────────────────
saveRDS(fit, file.path(output_dir, "sarima_model.rds"))

# ============================================================
# ── 16b. True Out-of-Sample Future Forecast (1 Week) ─────────
# ============================================================

FUTURE_H <- 168 # 7 days * 24 hours

# ── 1. Generate Future Hourly Timestamps ──────────────────────
last_date <- max(df$date_time)
future_dates <- seq(from = last_date + 3600, by = "hour", length.out = FUTURE_H)

cat(sprintf("\nGenerating future forecast from %s to %s\n",
            as.character(future_dates[1]),
            as.character(tail(future_dates, 1))))


# ── 2. Full-Data Seasonal Differencing (D=1, lag=168) ────────
full_raw  <- df$traffic_volume
diff_full <- diff(full_raw, lag = SEASON, differences = D_ORDER)
ts_full   <- ts(diff_full, frequency = 1)


# ── 3. Build Future Exogenous Features (xreg) ─────────────────
# Note: For pure SARIMA, xreg is NULL. If using weather/holiday regressors,
# future values for the next 168 hours must be supplied here.
build_future_xreg <- function(dates, type = MODEL_TYPE) {
  if (type == "pure") return(NULL)

  df_future <- data.frame(date_time = dates)
  df_future$is_holiday <- 0L # Standard assumption; update with holiday calendar if needed

  if (type == "holiday") {
    mm <- model.matrix(~ is_holiday, data = df_future)[, -1, drop = FALSE]
  } else if (type == "full") {
    # Using recent historical means as simple future weather proxies if unobserved
    df_future$temp       <- mean(tail(df$temp, 168))
    df_future$rain_1h    <- 0
    df_future$snow_1h    <- 0
    df_future$clouds_all <- mean(tail(df$clouds_all, 168))
    mm <- model.matrix(~ is_holiday + temp + rain_1h + snow_1h + clouds_all, data = df_future)[, -1, drop = FALSE]
  }
  return(mm)
}

xreg_full_fit <- build_xreg(1:nrow(df))
if (!is.null(xreg_full_fit)) {
  # Trim initial SEASON rows due to lag differencing drop
  xreg_full_fit <- xreg_full_fit[(SEASON + 1):nrow(xreg_full_fit), , drop = FALSE]
}
xreg_future <- build_future_xreg(future_dates)


# ── 4. Refit Model on Complete Historical Dataset ─────────────
cat("Refitting model on complete dataset...\n")
fit_full <- Arima(
  ts_full,
  model = fit,            # Re-uses optimized (p,d,q) orders from your training fit
  xreg  = xreg_full_fit
)


# ── 5. Generate Out-of-Sample Forecast ────────────────────────
fc_future <- forecast(fit_full, xreg = xreg_future, h = FUTURE_H)


# ── 6. Recursive Level Reconstruction (Using Last 168 Observed Hours)
# Anchor sequence starts with the last 168 actual volume observations in your dataset
recon_future <- c(tail(full_raw, SEASON), rep(NA_real_, FUTURE_H))

for (i in seq_len(FUTURE_H)) {
  recon_future[SEASON + i] <- as.numeric(fc_future$mean[i]) + recon_future[i]
}

fc_future_mean <- recon_future[(SEASON + 1):(SEASON + FUTURE_H)]
anchor_future  <- recon_future[1:FUTURE_H]

fc_future_lo80 <- as.numeric(fc_future$lower[, 1]) + anchor_future
fc_future_hi80 <- as.numeric(fc_future$upper[, 1]) + anchor_future
fc_future_lo95 <- as.numeric(fc_future$lower[, 2]) + anchor_future
fc_future_hi95 <- as.numeric(fc_future$upper[, 2]) + anchor_future


# ── 7. Package Future Forecast Results ─────────────────────────
df_future_out <- tibble(
  date_time      = future_dates,
  forecast_speed = fc_future_mean,
  lo80           = fc_future_lo80,
  hi80           = fc_future_hi80,
  lo95           = fc_future_lo95,
  hi95           = fc_future_hi95
)

# Save to output folder
write.csv(df_future_out,
          file.path(output_dir, "sarima_future_1week_forecast.csv"),
          row.names = FALSE)
cat("Saved future forecast to ->", file.path(output_dir, "sarima_future_1week_forecast.csv"), "\n")


# ── 8. Plot Future Forecast ───────────────────────────────────
p_future <- ggplot(df_future_out, aes(x = date_time, y = forecast_speed)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(colour = "#e34a33", linewidth = 0.8) +
  labs(
    title    = "SARIMA — 7-Day Out-of-Sample Traffic Volume Forecast",
    subtitle = sprintf("Projecting October 9, 2018 to October 15, 2018 (H = %d hours)", FUTURE_H),
    x        = "Date & Time",
    y        = "Forecasted Traffic Volume (vehicles / hr)",
    caption  = "Shaded regions: 80% and 95% prediction intervals"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_future_1week_forecast.png"), p_future, width = 12, height = 4, dpi = 150)


# ── 17. Execution Complete ───────────────────────────────────
cat("\n=============================================\n")
cat(" Modeling complete for type:", MODEL_TYPE, "\n")
cat(" Outputs saved to:", output_dir, "\n")
cat("=============================================\n")