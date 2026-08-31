# ============================================================
#  SARIMA Model — Metro Interstate Traffic Volume
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/models/SARIMA/  (plots + diagnostics + forecasts)
#
#  Key design decisions:
#   - ts() with frequency=24 (daily cycle via seasonal ARIMA)
#   - Weekly (168h) Fourier terms built by hand, phase-aligned
#     across train/valid/test — fourier() is NOT used because
#     it derives its period from frequency(ts)=24, not 168
#   - THREE-WAY SPLIT: 70% train / 15% validation / 15% test
#     (index-based, not calendar-month, so proportions are exact)
#   - K tuned only at multiples of 7 (7, 14, 21, 28): at those
#     values the weekly Fourier basis also recovers daily harmonics
#     (k=7 -> 24h, k=14 -> 12h, k=21 -> 8h, k=28 -> 6h), so the
#     RMSE curve has its meaningful "cliff" steps there — searching
#     intermediate K values adds runtime without decision value
#   - K selected by out-of-sample validation RMSE (not AICc)
#   - D fixed to 0: Fourier terms already absorb seasonal structure;
#     D>=1 + Fourier causes near-collinearity -> Inf AICc
#   - is_holiday as deterministic exogenous regressor
#   - Ljung-Box at lag 48 (daily) AND lag 168 (weekly)
#   - stepwise=TRUE, approximation=TRUE for speed (change to FALSE
#     for a final exhaustive fit before submission)
# ============================================================


# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("forecast", "tseries", "lubridate", "dplyr",
          "ggplot2", "zoo", "jsonlite", "tibble")
new  <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))


# ── 1. Paths ─────────────────────────────────────────────────
input_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "output/models/SARIMA"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)


# ── 2. Load Processed Data ───────────────────────────────────
df <- read.csv(input_path, stringsAsFactors = FALSE)

df$date_time <- as.POSIXct(df$date_time,
                           format = "%Y-%m-%d %H:%M:%S",
                           tz     = "UTC")

df <- df %>% arrange(date_time)

cat("Loaded:", nrow(df), "rows |",
    "Range:", as.character(min(df$date_time)), "to",
    as.character(max(df$date_time)), "\n")

# is_imputed tracks gap-filled hours from preprocessing.R
# (0 = observed, 1 = interpolated / t-168 fill).
if (!"is_imputed" %in% names(df)) {
  warning("is_imputed column not found — assuming all rows observed. ",
          "Re-run the current preprocessing.R to get imputation tracking.")
  df$is_imputed <- 0L
}
cat("Imputed hours in full series:", sum(df$is_imputed), "/", nrow(df),
    sprintf("(%.1f%%)\n", 100 * mean(df$is_imputed)))


# ── 3. Guard: verify the series is truly hourly with no gaps ─
expected_hours <- as.numeric(
  difftime(max(df$date_time), min(df$date_time), units = "hours")
) + 1

if (nrow(df) != expected_hours) {
  stop(
    "Series is not a complete hourly grid. Expected ", expected_hours,
    " rows but found ", nrow(df), ". Re-run preprocessing.R first."
  )
}
cat("Grid check passed: series is complete and hourly.\n")


# ── 4. Build ts Object ───────────────────────────────────────
SEASON        <- 24
WEEKLY_PERIOD <- 168

ts_tv  <- ts(df$traffic_volume, frequency = SEASON)
tv_vec <- as.numeric(ts_tv)

cat("ts object — length:", length(ts_tv),
    "| frequency:", frequency(ts_tv), "\n")


# ── 5. Train / Validation / Test Split (70 / 15 / 15) ────────
# Index-based proportional split so the ratios are exact
# regardless of how many rows the processed file contains.
# Time ordering is preserved — train precedes validation which
# precedes test, with no overlap.
N         <- nrow(df)
TRAIN_N   <- floor(0.70 * N)
VALID_H   <- floor(0.15 * N)
TEST_H    <- N - TRAIN_N - VALID_H   # remainder goes to test so N is exact

# Sanity checks
stopifnot(TRAIN_N + VALID_H + TEST_H == N)
stopifnot(TRAIN_N > 0, VALID_H > 0, TEST_H > 0)

ts_train <- ts(tv_vec[1:TRAIN_N],                                      frequency = SEASON)
ts_valid <- ts(tv_vec[(TRAIN_N + 1):(TRAIN_N + VALID_H)],              frequency = SEASON)
ts_test  <- ts(tv_vec[(TRAIN_N + VALID_H + 1):N],                      frequency = SEASON)

cat("\n=== Train / Validation / Test Split (70 / 15 / 15) ===\n")
cat(sprintf("Train : rows   1 – %d  |  %s  to  %s  |  %.1f%%\n",
            TRAIN_N,
            as.character(df$date_time[1]),
            as.character(df$date_time[TRAIN_N]),
            100 * TRAIN_N / N))
cat(sprintf("Valid : rows %d – %d  |  %s  to  %s  |  %.1f%%\n",
            TRAIN_N + 1,
            TRAIN_N + VALID_H,
            as.character(df$date_time[TRAIN_N + 1]),
            as.character(df$date_time[TRAIN_N + VALID_H]),
            100 * VALID_H / N))
cat(sprintf("Test  : rows %d – %d  |  %s  to  %s  |  %.1f%%\n",
            TRAIN_N + VALID_H + 1,
            N,
            as.character(df$date_time[TRAIN_N + VALID_H + 1]),
            as.character(df$date_time[N]),
            100 * TEST_H / N))

# ── 5b. Imputation-leakage check on validation and test ──────
valid_imputed_idx  <- df$is_imputed[(TRAIN_N + 1):(TRAIN_N + VALID_H)]
test_imputed_idx   <- df$is_imputed[(TRAIN_N + VALID_H + 1):N]
n_imputed_in_valid <- sum(valid_imputed_idx)
n_imputed_in_test  <- sum(test_imputed_idx)

if (n_imputed_in_valid > 0) {
  cat(sprintf(
    "\nNOTE: %d of %d validation hours (%.1f%%) are imputed. ",
    n_imputed_in_valid, VALID_H, 100 * n_imputed_in_valid / VALID_H
  ))
  cat("K is being selected by comparing forecasts against some gap-filled 'actuals'.\n")
} else {
  cat("\nNo imputed hours in the validation window — clean for K tuning.\n")
}

if (n_imputed_in_test > 0) {
  cat(sprintf(
    "\nWARNING: %d of %d test-window hours (%.1f%%) are imputed, not observed.\n",
    n_imputed_in_test, TEST_H, 100 * n_imputed_in_test / TEST_H
  ))
  cat("Supplementary accuracy on observed-only hours is computed in Section 11b.\n")
} else {
  cat("No imputed hours in the test window — clean for final evaluation.\n")
}


# ── 6. Stationarity Tests ────────────────────────────────────
# Run on training series only — no leakage into tuning/test data.
cat("\n--- Stationarity tests on training series ---\n")

adf_res  <- adf.test(as.numeric(ts_train), alternative = "stationary")
kpss_res <- kpss.test(as.numeric(ts_train))

cat(sprintf("ADF  p = %.4f  (%s)\n", adf_res$p.value,
            ifelse(adf_res$p.value  < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS p = %.4f  (%s)\n", kpss_res$p.value,
            ifelse(kpss_res$p.value > 0.05, "stationary", "non-stationary")))

# D fixed to 0 throughout — see header.
d_order <- ndiffs(ts_train)
cat("Suggested d:", d_order, "\n")
cat("D fixed to 0 (Fourier terms handle seasonal structure).\n")


# ── 7. Weekly Fourier Terms (hand-built, period = 168h) ──────
# forecast::fourier() is NOT used — see header for why.
# start_t preserves phase alignment across all three splits by
# anchoring each slice to its position in the FULL series index.

make_weekly_fourier <- function(n, K, start_t = 1) {
  t  <- start_t:(start_t + n - 1)
  X  <- matrix(NA_real_, nrow = n, ncol = 2 * K)
  cn <- character(2 * K)
  for (k in seq_len(K)) {
    X[, 2*k - 1] <- sin(2 * pi * k * t / WEEKLY_PERIOD)
    X[, 2*k    ] <- cos(2 * pi * k * t / WEEKLY_PERIOD)
    cn[2*k - 1]  <- paste0("S168_", k)
    cn[2*k    ]  <- paste0("C168_", k)
  }
  colnames(X) <- cn
  X
}

# Builds the full xreg matrix (Fourier + is_holiday) for all three
# splits simultaneously, with phase alignment guaranteed.
build_xreg <- function(K) {
  list(
    train = cbind(
      make_weekly_fourier(TRAIN_N, K, start_t = 1),
      is_holiday = df$is_holiday[1:TRAIN_N]
    ),
    valid = cbind(
      make_weekly_fourier(VALID_H, K, start_t = TRAIN_N + 1),
      is_holiday = df$is_holiday[(TRAIN_N + 1):(TRAIN_N + VALID_H)]
    ),
    test = cbind(
      make_weekly_fourier(TEST_H, K, start_t = TRAIN_N + VALID_H + 1),
      is_holiday = df$is_holiday[(TRAIN_N + VALID_H + 1):N]
    )
  )
}

MAX_P_SEASONAL <- 4
MAX_Q_SEASONAL <- 4


# ── 7a. K Tuning (multiples of 7 only: 7, 14, 21, 28) ───────
# WHY MULTIPLES OF 7 ONLY?
# The weekly Fourier basis has period 168 = 7 x 24. At every
# multiple of 7, the k-th harmonic coincides with a daily harmonic:
#   k =  7  ->  168/7  = 24h   (fundamental daily cycle)
#   k = 14  ->  168/14 = 12h   (twice-daily: AM + PM peaks)
#   k = 21  ->  168/21 =  8h   (three times daily)
#   k = 28  ->  168/28 =  6h   (four times daily)
# These are exactly the frequencies where meaningful new traffic
# structure is added. Intermediate values (k=1-6, 8-13, ...) add
# harmonics of the weekly cycle that contribute little to forecast
# accuracy but cost extra parameters and runtime. Restricting the
# search to {7, 14, 21, 28} gives the four most impactful candidate
# models and keeps the tuning loop under ~15-20 minutes.
#
# Set TUNE_K = TRUE to run the search; TUNE_K = FALSE uses K_FIXED.

TUNE_K  <- TRUE
K_FIXED <- 28      # fallback if TUNE_K = FALSE
K_GRID  <- c(7, 14, 21, 28)

if (TUNE_K) {
  cat("\nTuning K on out-of-sample validation RMSE —",
      "candidates:", paste(K_GRID, collapse = ", "), "\n")
  
  rmse_vec <- rep(NA_real_, length(K_GRID))
  aicc_vec <- rep(NA_real_, length(K_GRID))
  
  for (i in seq_along(K_GRID)) {
    k <- K_GRID[i]
    tryCatch({
      xr    <- build_xreg(k)
      fit_k <- auto.arima(
        ts_train,
        xreg          = xr$train,
        seasonal      = TRUE,
        stepwise      = TRUE,       # fast: only for ranking K
        approximation = TRUE,       # fast: only for ranking K
        d  = d_order,
        D  = 0,
        max.p = 3, max.q = 3,
        max.P = MAX_P_SEASONAL,
        max.Q = MAX_Q_SEASONAL
      )
      fc_k        <- forecast(fit_k, xreg = xr$valid, h = VALID_H)
      rmse_vec[i] <- sqrt(mean(
        (as.numeric(ts_valid) - as.numeric(fc_k$mean))^2
      ))
      aicc_vec[i] <- fit_k$aicc
      cat(sprintf("  K = %2d  |  Validation RMSE = %8.2f  |  Train AICc = %.2f\n",
                  k, rmse_vec[i], aicc_vec[i]))
    }, error = function(e) {
      cat(sprintf("  K = %2d  FAILED: %s\n", k, conditionMessage(e)))
    })
  }
  
  if (all(is.na(rmse_vec))) stop("All K candidates failed. Check your data.")
  
  best_i  <- which.min(rmse_vec)
  K_FIXED <- K_GRID[best_i]
  cat(sprintf("\nBest K by validation RMSE: %d  (RMSE = %.2f)\n",
              K_FIXED, rmse_vec[best_i]))
  
  # Save K tuning results
  tuning_df <- data.frame(K = K_GRID, Validation_RMSE = rmse_vec, AICc = aicc_vec)
  write.csv(tuning_df,
            file.path(output_dir, "sarima_k_tuning.csv"),
            row.names = FALSE)
  cat("Saved -> output/models/SARIMA/sarima_k_tuning.csv\n")
} else {
  cat(sprintf("\nTUNE_K = FALSE — using fixed K = %d\n", K_FIXED))
  rmse_vec <- rep(NA_real_, length(K_GRID))
  aicc_vec <- rep(NA_real_, length(K_GRID))
}


# ── 7b. Build final xreg matrices with the chosen K ──────────
xreg_final <- build_xreg(K_FIXED)
xreg_train <- xreg_final$train
xreg_valid <- xreg_final$valid
xreg_test  <- xreg_final$test

cat(sprintf("Using K = %d weekly Fourier pair(s) + is_holiday regressor.\n", K_FIXED))
cat("xreg_train dim:", dim(xreg_train), "\n")
cat("xreg_test  dim:", dim(xreg_test),  "\n")


# ── 8. Final Model Fitting ────────────────────────────────────
# REFIT_WITH_VALID = FALSE  → fit on train only (conservative default)
# REFIT_WITH_VALID = TRUE   → fold validation into the final fit once
#                             K is chosen (more data, no extra leakage)
REFIT_WITH_VALID <- FALSE

if (REFIT_WITH_VALID) {
  ts_fit_final   <- ts(c(as.numeric(ts_train), as.numeric(ts_valid)),
                       frequency = SEASON)
  xreg_fit_final <- rbind(xreg_train, xreg_valid)
  cat(sprintf("\nFitting final SARIMA on TRAIN + VALIDATION (%d hours) ...\n",
              length(ts_fit_final)))
} else {
  ts_fit_final   <- ts_train
  xreg_fit_final <- xreg_train
  cat(sprintf("\nFitting final SARIMA on TRAIN only (%d hours) ...\n",
              length(ts_fit_final)))
}
cat("(stepwise=TRUE, approximation=TRUE — set both to FALSE for exhaustive final fit)\n")

fit <- auto.arima(
  ts_fit_final,
  xreg          = xreg_fit_final,
  seasonal      = TRUE,
  stepwise      = TRUE,       # set FALSE for exhaustive search before submission
  approximation = TRUE,       # set FALSE for exact likelihood before submission
  d             = d_order,
  D             = 0,
  max.p = 3, max.q = 3,
  max.P = MAX_P_SEASONAL,
  max.Q = MAX_Q_SEASONAL,
  trace = TRUE
)

cat("\n--- Fitted model ---\n")
print(summary(fit))


# ── 9. Residual Diagnostics ───────────────────────────────────
resid <- residuals(fit)

# Ljung-Box at lag 48 (2 x daily) and lag 168 (weekly).
# If lb_168 fails (p < 0.05), increase K and refit.
lb_48  <- Box.test(resid, lag = 2 * SEASON,   type = "Ljung-Box",
                   fitdf = sum(fit$arma[1:4]))
lb_168 <- Box.test(resid, lag = WEEKLY_PERIOD, type = "Ljung-Box",
                   fitdf = sum(fit$arma[1:4]))

cat(sprintf("\nLjung-Box (lag=%3d): stat = %.4f, p = %.4f  (%s)\n",
            2 * SEASON, lb_48$statistic, lb_48$p.value,
            ifelse(lb_48$p.value  > 0.05,
                   "residuals ~ white noise",
                   "daily autocorrelation remains")))
cat(sprintf("Ljung-Box (lag=%3d): stat = %.4f, p = %.4f  (%s)\n",
            WEEKLY_PERIOD, lb_168$statistic, lb_168$p.value,
            ifelse(lb_168$p.value > 0.05,
                   "no weekly autocorrelation detected",
                   "weekly autocorrelation remains — consider larger K")))

set.seed(42)
sw_sample <- as.numeric(resid)
if (length(sw_sample) > 5000) sw_sample <- sample(sw_sample, 5000)
sw <- shapiro.test(sw_sample)
cat(sprintf("Shapiro-Wilk        : stat = %.4f, p = %.4f\n",
            sw$statistic, sw$p.value))

# 4-panel residual plot
png(file.path(output_dir, "sarima_residual_diagnostics.png"),
    width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(resid, main = "Residuals over Time",
     ylab = "Residual", xlab = "Time (hours)",
     col = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)

h_info <- hist(resid, breaks = 60, plot = FALSE)
hist(resid, breaks = 60, col = "#756bb1", border = "white",
     main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean(resid), sd(resid)) *
        length(resid) * diff(h_info$breaks[1:2]),
      col = "red", lwd = 2, add = TRUE)

acf(resid,  lag.max = 200, main = "ACF of Residuals (lag 0–200)")
pacf(resid, lag.max = 200, main = "PACF of Residuals (lag 0–200)")
dev.off()
cat("Saved -> output/models/SARIMA/sarima_residual_diagnostics.png\n")


# ── 10. Forecast (into the untouched test window) ────────────
fc <- forecast(fit, xreg = xreg_test, h = TEST_H)

# Realign ts_test to fc$mean's time index to prevent the
# "start cannot be after end" error in accuracy() that arises
# when ts_train was built from a plain integer index and fit$x
# carries a different clock than the ts_test built in Section 5.
ts_test <- ts(tv_vec[(TRAIN_N + VALID_H + 1):N],
              start     = stats::start(fc$mean),
              frequency = SEASON)


# ── 11. Accuracy Metrics ─────────────────────────────────────
acc <- accuracy(fc, ts_test)

actual    <- as.numeric(ts_test)
predicted <- as.numeric(fc$mean)

mape <- mean(
  abs((actual - predicted) / ifelse(actual == 0, NA, actual)),
  na.rm = TRUE
) * 100

smape <- mean(
  2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-8),
  na.rm = TRUE
) * 100

cat("\n--- Forecast Accuracy (test set — all hours) ---\n")
cat(sprintf("ME    : %10.4f\n",    acc["Test set", "ME"]))
cat(sprintf("RMSE  : %10.4f\n",    acc["Test set", "RMSE"]))
cat(sprintf("MAE   : %10.4f\n",    acc["Test set", "MAE"]))
cat(sprintf("MAPE  : %10.4f %%\n", mape))
cat(sprintf("sMAPE : %10.4f %%\n", smape))
cat(sprintf("MASE  : %10.4f\n",    acc["Test set", "MASE"]))

# ── 11b. Supplementary accuracy — observed hours only ────────
if (n_imputed_in_test > 0) {
  keep          <- test_imputed_idx == 0
  actual_obs    <- actual[keep]
  predicted_obs <- predicted[keep]
  
  rmse_obs  <- sqrt(mean((actual_obs - predicted_obs)^2))
  mae_obs   <- mean(abs(actual_obs - predicted_obs))
  mape_obs  <- mean(
    abs((actual_obs - predicted_obs) / ifelse(actual_obs == 0, NA, actual_obs)),
    na.rm = TRUE) * 100
  smape_obs <- mean(
    2 * abs(actual_obs - predicted_obs) /
      (abs(actual_obs) + abs(predicted_obs) + 1e-8),
    na.rm = TRUE) * 100
  
  cat(sprintf("\n--- Accuracy (observed hours only, n = %d / %d) ---\n",
              sum(keep), TEST_H))
  cat(sprintf("RMSE  : %10.4f\n", rmse_obs))
  cat(sprintf("MAE   : %10.4f\n", mae_obs))
  cat(sprintf("MAPE  : %10.4f %%\n", mape_obs))
  cat(sprintf("sMAPE : %10.4f %%\n", smape_obs))
} else {
  rmse_obs <- mae_obs <- mape_obs <- smape_obs <- NA_real_
}


# ── 12. Forecast vs Actual Plot ──────────────────────────────
test_dates <- df$date_time[(TRAIN_N + VALID_H + 1):N]

fc_df <- tibble(
  date_time = test_dates,
  actual    = actual,
  forecast  = predicted,
  lo80      = as.numeric(fc$lower[, 1]),
  hi80      = as.numeric(fc$upper[, 1]),
  lo95      = as.numeric(fc$lower[, 2]),
  hi95      = as.numeric(fc$upper[, 2])
)

p_fc <- ggplot(fc_df, aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual),   colour = "#333333", linewidth = 0.5) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.6,
            linetype = "dashed") +
  labs(
    title    = sprintf("SARIMA — %d-Day Test Forecast vs Actual  (K = %d)",
                       TEST_H %/% 24, K_FIXED),
    subtitle = sprintf("RMSE = %.1f  |  MAE = %.1f  |  MAPE = %.2f%%  |  sMAPE = %.2f%%",
                       acc["Test set", "RMSE"], acc["Test set", "MAE"],
                       mape, smape),
    x       = "Date",
    y       = "Traffic Volume (vehicles / hr)",
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
  geom_line(aes(y = actual),   colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.7,
            linetype = "dashed") +
  labs(title = "SARIMA — First 7 Days of Test Window (Zoom)",
       x = "Date", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_forecast_week1_zoom.png"),
       p_zoom, width = 12, height = 4, dpi = 150)
cat("Saved -> output/models/SARIMA/sarima_forecast_week1_zoom.png\n")


# ── 14. Save Forecast CSV ────────────────────────────────────
fc_out           <- fc_df
fc_out$date_time <- format(fc_out$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(fc_out,
          file.path(output_dir, "sarima_forecast_results.csv"),
          row.names = FALSE)
cat("Saved -> output/models/SARIMA/sarima_forecast_results.csv\n")


# ── 15. Save Model Summary JSON ──────────────────────────────
model_order <- arimaorder(fit)

summary_list <- list(
  model = list(
    order          = list(p = model_order[1],
                          d = model_order[2],
                          q = model_order[3]),
    seasonal_order = list(P = model_order[4],
                          D = model_order[5],
                          Q = model_order[6],
                          s = SEASON),
    fourier_K      = K_FIXED,
    fourier_period = WEEKLY_PERIOD,
    xreg_cols      = colnames(xreg_train),
    AIC            = fit$aic,
    AICc           = fit$aicc,
    BIC            = fit$bic,
    log_likelihood = fit$loglik
  ),
  tuning = list(
    tuned_on_validation = TUNE_K,
    k_grid_searched     = K_GRID,
    chosen_K            = K_FIXED,
    validation_rmse     = round(rmse_vec, 4),
    n_imputed_in_valid  = n_imputed_in_valid
  ),
  stationarity = list(
    ADF_pvalue  = round(adf_res$p.value,  4),
    KPSS_pvalue = round(kpss_res$p.value, 4),
    d_order     = d_order,
    D_order     = 0
  ),
  residual_tests = list(
    Ljung_Box_lag48_p  = round(lb_48$p.value,  4),
    Ljung_Box_lag168_p = round(lb_168$p.value, 4),
    Shapiro_Wilk_p     = round(sw$p.value,     4)
  ),
  accuracy = list(
    ME    = round(acc["Test set", "ME"],   4),
    RMSE  = round(acc["Test set", "RMSE"], 4),
    MAE   = round(acc["Test set", "MAE"],  4),
    MAPE  = round(mape,  4),
    sMAPE = round(smape, 4),
    MASE  = round(acc["Test set", "MASE"], 4)
  ),
  accuracy_observed_only = list(
    n_imputed_in_test = n_imputed_in_test,
    RMSE  = if (!is.na(rmse_obs))  round(rmse_obs,  4) else NA,
    MAE   = if (!is.na(mae_obs))   round(mae_obs,   4) else NA,
    MAPE  = if (!is.na(mape_obs))  round(mape_obs,  4) else NA,
    sMAPE = if (!is.na(smape_obs)) round(smape_obs, 4) else NA
  ),
  split = list(
    total_n          = N,
    train_n          = TRAIN_N,
    valid_h          = VALID_H,
    test_h           = TEST_H,
    train_pct        = round(100 * TRAIN_N / N, 1),
    valid_pct        = round(100 * VALID_H / N, 1),
    test_pct         = round(100 * TEST_H  / N, 1),
    season           = SEASON,
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
cat("  SARIMA modelling complete.\n")
cat("  Outputs written to:", output_dir, "\n")
cat("=============================================\n")