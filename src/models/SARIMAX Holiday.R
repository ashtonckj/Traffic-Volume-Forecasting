# ============================================================
# SARIMA Model — Metro Interstate Traffic Volume
# Input : data/processed/traffic_volume_processed.csv
# Output: output/models/SARIMA/
#
# Key design decisions:
# - ts() with frequency=168 (weekly cycle as seasonal period)
# - Seasonal differencing D=1 (lag=168) applied to training
#  split only, consistent with all other models in this study
# - d=0: confirmed stationary after D=1 (see stationarity_summary.csv)
# - No Fourier terms: D=1 + Fourier causes near-collinearity
# - is_holiday as deterministic exogenous regressor
# - THREE-WAY SPLIT: 70% train / 15% validation / 15% test
#  using the `split` column from preprocessing.R
# - Ljung-Box at lag 48 (daily) AND lag 168 (weekly)
# - stepwise=TRUE, approximation=TRUE for speed (set FALSE
#  for exhaustive final fit before submission)
# ============================================================


# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("forecast", "tseries", "lubridate", "dplyr",
          "ggplot2", "zoo", "jsonlite", "tibble")
new <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))


# ── 1. Paths ─────────────────────────────────────────────────
input_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "output/models/SARIMAX_holiday"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)


# ── 2. Load Processed Data ───────────────────────────────────
df <- read.csv(input_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time,
                           format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
df <- df %>% arrange(date_time)

cat("Loaded:", nrow(df), "rows |",
    "Range:", as.character(min(df$date_time)), "to",
    as.character(max(df$date_time)), "\n")

if (!"is_imputed" %in% names(df)) {
  warning("is_imputed column not found — assuming all rows observed.")
  df$is_imputed <- 0L
}
cat("Imputed hours:", sum(df$is_imputed), "/", nrow(df),
    sprintf("(%.1f%%)\n", 100 * mean(df$is_imputed)))

# ── 2b. Restore day_of_week as a proper factor ───────────────
# preprocessing.R creates day_of_week as a labeled factor, but writing to
# CSV flattens it to plain text. Re-establish it as a factor with explicit
# levels here so downstream code (build_xreg's dow_f) can safely reference
# levels(df$day_of_week) and get a real, non-null level set.

df$day_of_week <- weekdays(df$date_time)
day_levels <- c("Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday")
stopifnot(all(unique(df$day_of_week) %in% day_levels))
df$day_of_week <- factor(df$day_of_week, levels = day_levels)

cat("day_of_week levels:", paste(levels(df$day_of_week), collapse = ", "), "\n")


# ── 3. Guard: complete hourly grid ───────────────────────────
expected_hours <- as.numeric(
  difftime(max(df$date_time), min(df$date_time), units = "hours")
) + 1
if (nrow(df) != expected_hours) {
  stop("Series is not a complete hourly grid. Expected ", expected_hours,
       " rows but found ", nrow(df), ". Re-run preprocessing.R first.")
}
cat("Grid check passed.\n")


# ── 4. Constants ─────────────────────────────────────────────
SEASON    <- 168 # weekly seasonal period (hours)
D_ORDER   <- 1  # seasonal differences — same as all other models
d_ORDER   <- 0  # no ordinal differencing — confirmed by stationarity tests


# ── 5. Train / Validation / Test Split ───────────────────────
# Use the `split` column from preprocessing.R so boundaries are
# identical across all models.
stopifnot("split" %in% names(df))

train_idx <- which(df$split == "train")
valid_idx <- which(df$split == "validation")
test_idx <- which(df$split == "test")

TRAIN_N <- length(train_idx)
VALID_H <- length(valid_idx)
TEST_H <- length(test_idx)
N   <- nrow(df)

stopifnot(TRAIN_N + VALID_H + TEST_H == N)

cat("\n=== Train / Validation / Test Split ===\n")
cat(sprintf("Train : %d rows | %s to %s | %.1f%%\n",
            TRAIN_N,
            as.character(df$date_time[train_idx[1]]),
            as.character(df$date_time[tail(train_idx, 1)]),
            100 * TRAIN_N / N))
cat(sprintf("Valid : %d rows | %s to %s | %.1f%%\n",
            VALID_H,
            as.character(df$date_time[valid_idx[1]]),
            as.character(df$date_time[tail(valid_idx, 1)]),
            100 * VALID_H / N))
cat(sprintf("Test : %d rows | %s to %s | %.1f%%\n",
            TEST_H,
            as.character(df$date_time[test_idx[1]]),
            as.character(df$date_time[tail(test_idx, 1)]),
            100 * TEST_H / N))

# Imputation leakage check
valid_imputed_idx <- df$is_imputed[valid_idx]
test_imputed_idx <- df$is_imputed[test_idx]
n_imputed_in_valid <- sum(valid_imputed_idx)
n_imputed_in_test <- sum(test_imputed_idx)

if (n_imputed_in_valid > 0) {
  cat(sprintf("\nNOTE: %d of %d validation hours (%.1f%%) are imputed.\n",
              n_imputed_in_valid, VALID_H,
              100 * n_imputed_in_valid / VALID_H))
} else {
  cat("\nNo imputed hours in validation — clean for tuning.\n")
}
if (n_imputed_in_test > 0) {
  cat(sprintf("WARNING: %d of %d test hours (%.1f%%) are imputed.\n",
              n_imputed_in_test, TEST_H,
              100 * n_imputed_in_test / TEST_H))
} else {
  cat("No imputed hours in test — clean for evaluation.\n")
}


# ── 6. Differencing (training split only) ────────────────────
# D=1 seasonal diff (lag=168) applied to training rows only,
# consistent with differencing.R and all other models.
# Validation and test series are differenced by carrying the last
# SEASON values of training forward as the lag reference — this
# avoids cross-boundary leakage while keeping the series aligned.
#
# auto.arima receives the pre-differenced series with d=0, D=0
# (differencing already applied manually) and searches only for
# the AR/MA orders needed to whiten the residuals.

train_raw <- df$traffic_volume[train_idx]
valid_raw <- df$traffic_volume[valid_idx]
test_raw <- df$traffic_volume[test_idx]

# Training: standard seasonal diff
diff_train <- diff(train_raw, lag = SEASON, differences = D_ORDER)
n_pad   <- TRAIN_N - length(diff_train) # leading NAs from lag lookback
cat(sprintf("\nDifferencing: D = %d (lag %d), d = %d\n",
            D_ORDER, SEASON, d_ORDER))
cat(sprintf("Training diff length: %d (%d leading rows dropped from lag)\n",
            length(diff_train), n_pad))

# Validation: diff using last SEASON obs of training as lag anchor
valid_anchor <- c(tail(train_raw, SEASON), valid_raw)
diff_valid <- diff(valid_anchor, lag = SEASON, differences = D_ORDER)
diff_valid <- diff_valid[1:VALID_H] # trim back to validation length

# Test: diff using last SEASON obs of validation as lag anchor
test_anchor <- c(tail(valid_raw, SEASON), test_raw)
diff_test <- diff(test_anchor, lag = SEASON, differences = D_ORDER)
diff_test <- diff_test[1:TEST_H]

# Build ts objects from differenced series (frequency=1 since
# D=1 already removed the weekly cycle; auto.arima searches
# non-seasonal AR/MA only)
ts_train_diff <- ts(diff_train, frequency = 1)
ts_valid_diff <- ts(diff_valid, frequency = 1)
ts_test_diff <- ts(diff_test, frequency = 1)

cat("ts_train_diff length:", length(ts_train_diff), "\n")
cat("ts_valid_diff length:", length(ts_valid_diff), "\n")
cat("ts_test_diff length:", length(ts_test_diff), "\n")

# ── 6b. Stationarity Tests: ADF and KPSS ─────────────────────
# Confirms the d=0, D=1 assumption stated in the header. Run BOTH tests
# on the RAW training series (before differencing) and the DIFFERENCED
# training series (after D=1 seasonal diff) to show the differencing
# actually achieved stationarity.
#
# ADF (Augmented Dickey-Fuller): H0 = series has a unit root (non-stationary)
# -> low p-value (< 0.05) => reject H0 => series IS stationary
# KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = series IS stationary
# -> low p-value (< 0.05) => reject H0 => series is NOT stationary
#
# Note the tests have OPPOSITE null hypotheses -- ideally you want
# ADF to reject (p < 0.05) AND KPSS to fail to reject (p > 0.05)
# for a confirmed-stationary conclusion. If they disagree, the series
# may be borderline (e.g. trend-stationary) and needs closer inspection.

cat("\n=== Stationarity Tests: RAW training series ===\n")
adf_raw <- adf.test(train_raw, alternative = "stationary")
kpss_raw <- kpss.test(train_raw, null = "Level")

cat(sprintf("ADF (raw) : stat = %.4f, p = %.4f (%s)\n",
            adf_raw$statistic, adf_raw$p.value,
            ifelse(adf_raw$p.value < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS (raw) : stat = %.4f, p = %.4f (%s)\n",
            kpss_raw$statistic, kpss_raw$p.value,
            ifelse(kpss_raw$p.value > 0.05, "stationary", "non-stationary")))

cat("\n=== Stationarity Tests: DIFFERENCED training series (D=1, lag=168) ===\n")
adf_diff <- adf.test(diff_train, alternative = "stationary")
kpss_diff <- kpss.test(diff_train, null = "Level")

cat(sprintf("ADF (diff) : stat = %.4f, p = %.4f (%s)\n",
            adf_diff$statistic, adf_diff$p.value,
            ifelse(adf_diff$p.value < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS (diff) : stat = %.4f, p = %.4f (%s)\n",
            kpss_diff$statistic, kpss_diff$p.value,
            ifelse(kpss_diff$p.value > 0.05, "stationary", "non-stationary")))

# Agreement check: flag if ADF and KPSS disagree on the differenced series
diff_adf_says_stationary <- adf_diff$p.value < 0.05
diff_kpss_says_stationary <- kpss_diff$p.value > 0.05

if (diff_adf_says_stationary && diff_kpss_says_stationary) {
  cat("\nCONCLUSION: Both tests agree — differenced series is stationary.\n")
  cat("d=0 (no further ordinal differencing needed) is supported.\n")
} else {
  cat("\nWARNING: ADF and KPSS disagree on the differenced series.\n")
  cat("Consider additional differencing or trend removal before proceeding.\n")
}

# Save results for the record (referenced in your header comment as
# stationarity_summary.csv)
stationarity_summary <- data.frame(
  series   = c("raw_train", "diff_train"),
  adf_stat  = c(adf_raw$statistic, adf_diff$statistic),
  adf_pvalue = c(adf_raw$p.value,  adf_diff$p.value),
  adf_verdict = c(ifelse(adf_raw$p.value < 0.05, "stationary", "non-stationary"),
                  ifelse(adf_diff$p.value < 0.05, "stationary", "non-stationary")),
  kpss_stat  = c(kpss_raw$statistic, kpss_diff$statistic),
  kpss_pvalue = c(kpss_raw$p.value,  kpss_diff$p.value),
  kpss_verdict = c(ifelse(kpss_raw$p.value > 0.05, "stationary", "non-stationary"),
                   ifelse(kpss_diff$p.value > 0.05, "stationary", "non-stationary"))
)
write.csv(stationarity_summary,
          file.path(output_dir, "stationarity_summary.csv"), row.names = FALSE)
cat("Saved -> ", file.path(output_dir, "stationarity_summary.csv"), "\n")


# ── 7. xreg: ALL available columns (exploratory — see caveats below) ────
# Columns available: temp, rain_1h, snow_1h, clouds_all, is_holiday,
# hour, month, day_of_week. (traffic_volume is the target; is_imputed
# and split are bookkeeping, not predictors.)
#
# hour, month, day_of_week are categorical -> dummy-encoded via model.matrix.
# Continuous weather vars are scaled using TRAIN stats only (no leakage).
#
# CAVEAT: temp/rain_1h/snow_1h/clouds_all use ACTUAL historical values in
# validation/test, which assumes perfect future weather knowledge. This is
# fine for an exploratory upper-bound test, but is NOT a deployable model
# unless these are replaced with forecasted weather at inference time.

# ── 7. xreg: is_holiday ONLY ──────────────────────────────────
build_xreg <- function(idx) {
  # Return is_holiday as a 1-column matrix with explicit column name
  matrix(
    df$is_holiday[idx], 
    ncol = 1, 
    dimnames = list(NULL, "is_holiday")
  )
}

xreg_train_full <- build_xreg(train_idx[(n_pad + 1):TRAIN_N])
xreg_valid_full <- build_xreg(valid_idx)
xreg_test_full  <- build_xreg(test_idx)

cat("Holiday-only xreg dim (train):", dim(xreg_train_full), "\n")

# ── 7b. Collinearity check BEFORE fitting ────────────────────
# Rank check: if rank < ncol, some columns are exact linear combinations
# of others and auto.arima/Arima will either drop them silently or error.
qr_rank <- qr(xreg_train_full)$rank
cat(sprintf("Design matrix: %d columns, rank = %d %s\n",
            ncol(xreg_train_full), qr_rank,
            ifelse(qr_rank < ncol(xreg_train_full), "-- RANK DEFICIENT", "-- full rank")))

# Pairwise correlation among continuous weather vars only (quick sanity check)
weather_cor <- cor(df[train_idx, c("temp","rain_1h","snow_1h","clouds_all")],
                   use = "complete.obs")
cat("\nWeather variable correlations (train):\n")
print(round(weather_cor, 2))


# ── 8. Final Model Fitting ────────────────────────────────────
# REFIT_WITH_VALID = FALSE -> fit on train only (conservative default)
# REFIT_WITH_VALID = TRUE -> fold validation into the final fit once
#               orders are selected (more data, no extra leakage)
REFIT_WITH_VALID <- FALSE

if (REFIT_WITH_VALID) {
  ts_fit_final <- ts(c(as.numeric(ts_train_diff),
                       as.numeric(ts_valid_diff)), frequency = 1)
  xreg_fit_final <- rbind(xreg_train_full, xreg_valid)
  cat(sprintf("\nFitting on TRAIN + VALIDATION (%d hours) ...\n",
              length(ts_fit_final)))
} else {
  ts_fit_final <- ts_train_diff
  xreg_fit_final <- xreg_train_full
  cat(sprintf("\nFitting on TRAIN only (%d hours) ...\n",
              length(ts_fit_final)))
}
cat("(stepwise=TRUE, approximation=TRUE — set both FALSE for exhaustive final fit)\n")

fit <- auto.arima(
  ts_fit_final,
  xreg     = xreg_fit_final,
  seasonal   = FALSE, # weekly seasonality already removed by D=1 diff
  d      = 0,   # pre-differenced; no further differencing
  D      = 0,
  stepwise   = TRUE,
  approximation = TRUE,
  trace    = TRUE
)
cat("\n--- Fitted model ---\n")
print(summary(fit))


# ── 9. Residual Diagnostics ───────────────────────────────────
resid <- residuals(fit)

lb_48 <- Box.test(resid, lag = 48, type = "Ljung-Box",
                  fitdf = sum(fit$arma[1:4]))
lb_168 <- Box.test(resid, lag = 168, type = "Ljung-Box",
                   fitdf = sum(fit$arma[1:4]))

cat(sprintf("\nLjung-Box (lag= 48): stat = %.4f, p = %.4f (%s)\n",
            lb_48$statistic, lb_48$p.value,
            ifelse(lb_48$p.value > 0.05, "white noise", "autocorrelation remains")))
cat(sprintf("Ljung-Box (lag=168): stat = %.4f, p = %.4f (%s)\n",
            lb_168$statistic, lb_168$p.value,
            ifelse(lb_168$p.value > 0.05, "white noise", "weekly autocorrelation remains")))

set.seed(42)
sw_sample <- as.numeric(resid)
if (length(sw_sample) > 5000) sw_sample <- sample(sw_sample, 5000)
sw <- shapiro.test(sw_sample)
cat(sprintf("Shapiro-Wilk   : stat = %.4f, p = %.4f\n",
            sw$statistic, sw$p.value))

png(file.path(output_dir, "sarima_residual_diagnostics.png"),
    width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(resid, main = "Residuals over Time",
     ylab = "Residual", xlab = "Time (hours)", col = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)
h_info <- hist(resid, breaks = 60, plot = FALSE)
hist(resid, breaks = 60, col = "#756bb1", border = "white",
     main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean(resid), sd(resid)) *
        length(resid) * diff(h_info$breaks[1:2]),
      col = "red", lwd = 2, add = TRUE)
acf(resid, lag.max = 200, main = "ACF of Residuals (lag 0-200)")
pacf(resid, lag.max = 200, main = "PACF of Residuals (lag 0-200)")
dev.off()
cat("Saved -> output/models/SARIMA/sarima_residual_diagnostics.png\n")


# ── 10. Forecast (test window) ───────────────────────────────
fc <- forecast(fit, xreg = xreg_test_full, h = TEST_H)
recon <- c(tail(valid_raw, SEASON), rep(NA_real_, TEST_H))
for (i in seq_len(TEST_H)) {
  # recon[i] is the anchor at lag SEASON for this step:
  # - for i <= SEASON, recon[i] is a true validation value
  # - for i > SEASON, recon[i] is our own earlier reconstructed forecast
  recon[SEASON + i] <- as.numeric(fc$mean[i]) + recon[i]
}
fc_mean_raw <- recon[(SEASON + 1):(SEASON + TEST_H)]

# The anchor series used above (recon[1:TEST_H]) is exactly the
# reference each forecast step was added to. Prediction intervals
# must be inverted against that SAME anchor sequence, not test_raw,
# for consistency with the point forecast above.
anchor_used <- recon[1:TEST_H]

fc_lo80_raw <- as.numeric(fc$lower[, 1]) + anchor_used
fc_hi80_raw <- as.numeric(fc$upper[, 1]) + anchor_used
fc_lo95_raw <- as.numeric(fc$lower[, 2]) + anchor_used
fc_hi95_raw <- as.numeric(fc$upper[, 2]) + anchor_used

cat(sprintf("\nForecast reconstructed: %d steps, anchor uses %d true validation values + %d self-referential (recursive) values\n",
            TEST_H, min(SEASON, TEST_H), max(0, TEST_H - SEASON)))


# ── 11. Accuracy Metrics ─────────────────────────────────────
actual  <- test_raw
predicted <- fc_mean_raw

mae <- mean(abs(actual - predicted))
rmse <- sqrt(mean((actual - predicted)^2))
me <- mean(actual - predicted)
mape <- mean(abs((actual - predicted) /
                   ifelse(actual == 0, NA, actual)),
             na.rm = TRUE) * 100
smape <- mean(2 * abs(actual - predicted) /
                (abs(actual) + abs(predicted) + 1e-8)) * 100
mase <- mae / mean(abs(diff(train_raw))) # naive in-sample MAE as scale

cat("\n--- Forecast Accuracy (test set — all hours) ---\n")
cat(sprintf("ME  : %10.4f\n", me))
cat(sprintf("RMSE : %10.4f\n", rmse))
cat(sprintf("MAE : %10.4f\n", mae))
cat(sprintf("MAPE : %10.4f %%\n", mape))
cat(sprintf("sMAPE : %10.4f %%\n", smape))
cat(sprintf("MASE : %10.4f\n", mase))

# ── 11b. Observed hours only ─────────────────────────────────
if (n_imputed_in_test > 0) {
  keep     <- test_imputed_idx == 0
  actual_obs  <- actual[keep]
  predicted_obs <- predicted[keep]
  rmse_obs <- sqrt(mean((actual_obs - predicted_obs)^2))
  mae_obs <- mean(abs(actual_obs - predicted_obs))
  mape_obs <- mean(abs((actual_obs - predicted_obs) /
                         ifelse(actual_obs == 0, NA, actual_obs)),
                   na.rm = TRUE) * 100
  smape_obs <- mean(2 * abs(actual_obs - predicted_obs) /
                      (abs(actual_obs) + abs(predicted_obs) + 1e-8)) * 100
  cat(sprintf("\n--- Accuracy (observed hours only, n = %d / %d) ---\n",
              sum(keep), TEST_H))
  cat(sprintf("RMSE : %10.4f\n", rmse_obs))
  cat(sprintf("MAE : %10.4f\n", mae_obs))
  cat(sprintf("MAPE : %10.4f %%\n", mape_obs))
  cat(sprintf("sMAPE : %10.4f %%\n", smape_obs))
} else {
  rmse_obs  <- rmse
  mae_obs   <- mae
  mape_obs  <- mape
  smape_obs <- smape
}


# ── 12. Forecast vs Actual Plot ──────────────────────────────
test_dates <- df$date_time[test_idx]

fc_df <- tibble(
  date_time = test_dates,
  actual  = actual,
  forecast = predicted,
  lo80   = fc_lo80_raw,
  hi80   = fc_hi80_raw,
  lo95   = fc_lo95_raw,
  hi95   = fc_hi95_raw
)

p_fc <- ggplot(fc_df, aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.5) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.6,
            linetype = "dashed") +
  labs(
    title  = sprintf("SARIMAX (is_holiday) — %d-Day Test Forecast vs Actual (D=1, d=0)",
                     TEST_H %/% 24),
    subtitle = sprintf("RMSE = %.1f | MAE = %.1f | MAPE = %.2f%% | sMAPE = %.2f%%",
                       rmse, mae, mape, smape),
    x   = "Date",
    y   = "Traffic Volume (vehicles / hr)",
    caption = "Shaded bands: 80% (dark) and 95% (light) prediction intervals"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey40"))

ggsave(file.path(output_dir, "sarima_forecast_vs_actual.png"),
       p_fc, width = 14, height = 5, dpi = 150)
cat("Saved -> output/models/SARIMA/sarima_forecast_vs_actual.png\n")


# ── 13. First-Week Zoom Plot ─────────────────────────────────
p_zoom <- ggplot(fc_df[1:min(24 * 7, nrow(fc_df)), ], aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.7,
            linetype = "dashed") +
  labs(title = "SARIMAX (is_holiday) — First 7 Days of Test Window (Zoom)",
       x = "Date", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_forecast_week1_zoom.png"),
       p_zoom, width = 12, height = 4, dpi = 150)
cat("Saved -> output/models/SARIMA/sarima_forecast_week1_zoom.png\n")


# ── 14. Save Forecast CSV ────────────────────────────────────
fc_out     <- fc_df
fc_out$date_time <- format(fc_out$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(fc_out,
          file.path(output_dir, "sarima_forecast_results.csv"),
          row.names = FALSE)
cat("Saved -> output/models/SARIMA/sarima_forecast_results.csv\n")


# ── 15. Save Model Summary JSON ──────────────────────────────
model_order <- arimaorder(fit)

summary_list <- list(
  model = list(
    order     = list(p = model_order[1],
                     d = model_order[2],
                     q = model_order[3]),
    seasonal_order = list(P = model_order[4],
                          D = model_order[5],
                          Q = model_order[6],
                          s = SEASON),
    differencing = list(d = d_ORDER, D = D_ORDER,
                        lag = SEASON,
                        note = "Pre-differenced before auto.arima; d=D=0 inside fit"),
    xreg_cols   = colnames(xreg_train_full),
    AIC      = fit$aic,
    AICc     = fit$aicc,
    BIC      = fit$bic,
    log_likelihood = fit$loglik
  ),
  stationarity = list(
    note  = "Tested in differencing.R — see stationarity_summary.csv",
    d_order = d_ORDER,
    D_order = D_ORDER
  ),
  residual_tests = list(
    Ljung_Box_lag48_p = round(lb_48$p.value, 4),
    Ljung_Box_lag168_p = round(lb_168$p.value, 4),
    Shapiro_Wilk_p  = round(sw$p.value,  4)
  ),
  accuracy = list(
    ME  = round(me,  4),
    RMSE = round(rmse, 4),
    MAE = round(mae, 4),
    MAPE = round(mape, 4),
    sMAPE = round(smape, 4),
    MASE = round(mase, 4)
  ),
  accuracy_observed_only = list(
    n_imputed_in_test = n_imputed_in_test,
    RMSE = if (!is.na(rmse_obs)) round(rmse_obs, 4) else NA,
    MAE = if (!is.na(mae_obs)) round(mae_obs, 4) else NA,
    MAPE = if (!is.na(mape_obs)) round(mape_obs, 4) else NA,
    sMAPE = if (!is.na(smape_obs)) round(smape_obs, 4) else NA
  ),
  split = list(
    total_n     = N,
    train_n     = TRAIN_N,
    valid_h     = VALID_H,
    test_h     = TEST_H,
    train_pct    = round(100 * TRAIN_N / N, 1),
    valid_pct    = round(100 * VALID_H / N, 1),
    test_pct    = round(100 * TEST_H / N, 1),
    refit_with_valid = REFIT_WITH_VALID
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "sarima_model_summary.json"))
cat("Saved -> output/models/SARIMA/sarima_model_summary.json\n")


# ── 16. Save Model Object ────────────────────────────────────
saveRDS(fit, file.path(output_dir, "sarima_model.rds"))
cat("Saved -> output/models/SARIMA/sarima_model.rds\n")
cat("(Reload with: fit <- readRDS('output/models/SARIMA/sarima_model.rds'))\n")


# ── 17. Done ─────────────────────────────────────────────────
cat("\n=============================================\n")
cat(" SARIMA modelling complete.\n")
cat(" Outputs written to:", output_dir, "\n")
cat("=============================================\n")
