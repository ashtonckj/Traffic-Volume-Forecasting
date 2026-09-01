# ============================================================
#  SARIMA Model (Standard) — Metro Interstate Traffic Volume
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/models/SARIMA_standard/
#
#  Standard Box-Jenkins SARIMA — no Fourier terms, no external
#  regressors. Seasonality is handled entirely by the seasonal
#  ARIMA (P,D,Q)[24] component. D is determined by nsdiffs()
#  rather than being forced to 0, because there is no Fourier
#  collinearity concern here.
#
#  Use this alongside the Fourier-augmented SARIMA to compare
#  whether the added complexity of Fourier terms is justified
#  by a meaningful improvement in test-set forecast accuracy.
# ============================================================


# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("forecast", "tseries", "lubridate", "dplyr",
          "ggplot2", "jsonlite", "tibble")
new  <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))


# ── 1. Paths ─────────────────────────────────────────────────
input_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "output/models/SARIMA_standard"
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

if (!"is_imputed" %in% names(df)) {
  warning("is_imputed column not found — assuming all rows observed. ",
          "Re-run preprocessing.R to get imputation tracking.")
  df$is_imputed <- 0L
}
cat("Imputed hours in full series:", sum(df$is_imputed), "/", nrow(df),
    sprintf("(%.1f%%)\n", 100 * mean(df$is_imputed)))


# ── 3. Guard: verify complete hourly grid ────────────────────
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


# ── 4. Constants & ts Object ─────────────────────────────────
SEASON <- 168

ts_tv  <- ts(df$traffic_volume, frequency = SEASON)
tv_vec <- as.numeric(ts_tv)

cat("ts object — length:", length(ts_tv),
    "| frequency:", frequency(ts_tv), "\n")


# ── 5. Train / Validation / Test Split (70 / 15 / 15) ────────
N       <- nrow(df)
TRAIN_N <- floor(0.70 * N)
VALID_H <- floor(0.15 * N)
TEST_H  <- N - TRAIN_N - VALID_H

stopifnot(TRAIN_N + VALID_H + TEST_H == N)
stopifnot(TRAIN_N > 0, VALID_H > 0, TEST_H > 0)

ts_train <- ts(tv_vec[1:TRAIN_N],                          frequency = SEASON)
ts_valid <- ts(tv_vec[(TRAIN_N + 1):(TRAIN_N + VALID_H)],  frequency = SEASON)

cat("\n=== Train / Validation / Test Split (70 / 15 / 15) ===\n")
cat(sprintf("Train : rows   1 – %d  |  %s  to  %s  |  %.1f%%\n",
            TRAIN_N,
            as.character(df$date_time[1]),
            as.character(df$date_time[TRAIN_N]),
            100 * TRAIN_N / N))
cat(sprintf("Valid : rows %d – %d  |  %s  to  %s  |  %.1f%%\n",
            TRAIN_N + 1, TRAIN_N + VALID_H,
            as.character(df$date_time[TRAIN_N + 1]),
            as.character(df$date_time[TRAIN_N + VALID_H]),
            100 * VALID_H / N))
cat(sprintf("Test  : rows %d – %d  |  %s  to  %s  |  %.1f%%\n",
            TRAIN_N + VALID_H + 1, N,
            as.character(df$date_time[TRAIN_N + VALID_H + 1]),
            as.character(df$date_time[N]),
            100 * TEST_H / N))

# ── 5b. Imputation-leakage check ─────────────────────────────
test_imputed_idx  <- df$is_imputed[(TRAIN_N + VALID_H + 1):N]
n_imputed_in_test <- sum(test_imputed_idx)

if (n_imputed_in_test > 0) {
  cat(sprintf(
    "\nWARNING: %d of %d test-window hours (%.1f%%) are imputed.\n",
    n_imputed_in_test, TEST_H, 100 * n_imputed_in_test / TEST_H
  ))
  cat("Supplementary accuracy on observed-only hours computed in Section 11b.\n")
} else {
  cat("\nNo imputed hours in the test window — clean for final evaluation.\n")
}


# ── 6. Stationarity Tests ────────────────────────────────────
# Run on training series only.
cat("\n--- Stationarity tests on training series ---\n")

adf_res  <- adf.test(as.numeric(ts_train), alternative = "stationary")
kpss_res <- kpss.test(as.numeric(ts_train))

cat(sprintf("ADF  p = %.4f  (%s)\n", adf_res$p.value,
            ifelse(adf_res$p.value  < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS p = %.4f  (%s)\n", kpss_res$p.value,
            ifelse(kpss_res$p.value > 0.05, "stationary", "non-stationary")))

# Both d and D are determined from the data — no Fourier terms
# means no collinearity concern, so nsdiffs() is fully trusted.
d_order <- ndiffs(ts_train)
D_order <- nsdiffs(ts_train)
cat("Suggested d:", d_order, "\n")
cat("Suggested D:", D_order, "\n")


# ── 7. Model Fitting ─────────────────────────────────────────
# Pure seasonal ARIMA — no xreg, no Fourier terms.
# auto.arima() searches over (p,d,q)(P,D,Q)[24].
# Set stepwise=FALSE and approximation=FALSE for an exhaustive
# exact-likelihood search before final submission.
cat("\nFitting standard SARIMA model — please wait ...\n")
cat("(stepwise=TRUE, approximation=TRUE — set both FALSE for exhaustive fit)\n")

fit <- auto.arima(
  ts_train,
  seasonal      = TRUE,
  stepwise      = TRUE,
  approximation = TRUE,
  d             = d_order,
  D             = D_order,
  trace = TRUE
)

cat("\n--- Fitted model ---\n")
print(summary(fit))


# ── 8. Residual Diagnostics ───────────────────────────────────
resid <- residuals(fit)

# Ljung-Box at lag 48 (2 x daily) and lag 168 (weekly).
lb_48  <- Box.test(resid, lag = 2 * SEASON, type = "Ljung-Box",
                   fitdf = sum(fit$arma[1:4]))
lb_168 <- Box.test(resid, lag = 168,        type = "Ljung-Box",
                   fitdf = sum(fit$arma[1:4]))

cat(sprintf("\nLjung-Box (lag=%3d): stat = %.4f, p = %.4f  (%s)\n",
            2 * SEASON, lb_48$statistic, lb_48$p.value,
            ifelse(lb_48$p.value  > 0.05,
                   "residuals ~ white noise",
                   "daily autocorrelation remains")))
cat(sprintf("Ljung-Box (lag=%3d): stat = %.4f, p = %.4f  (%s)\n",
            168, lb_168$statistic, lb_168$p.value,
            ifelse(lb_168$p.value > 0.05,
                   "no weekly autocorrelation detected",
                   "weekly autocorrelation remains")))

set.seed(42)
sw_sample <- as.numeric(resid)
if (length(sw_sample) > 5000) sw_sample <- sample(sw_sample, 5000)
sw <- shapiro.test(sw_sample)
cat(sprintf("Shapiro-Wilk        : stat = %.4f, p = %.4f\n",
            sw$statistic, sw$p.value))

# 4-panel residual plot
png(file.path(output_dir, "sarima_std_residual_diagnostics.png"),
    width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(resid, main = "Residuals over Time",
     ylab = "Residual", xlab = "Time (hours)",
     col = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)

hist(resid, breaks = 60, col = "#756bb1", border = "white",
     main = "Residual Distribution", xlab = "Residual",
     freq = FALSE)
curve(dnorm(x, mean(resid), sd(resid)), col = "red", lwd = 2, add = TRUE)

acf(resid,  lag.max = 200, main = "ACF of Residuals (lag 0-200)")
pacf(resid, lag.max = 200, main = "PACF of Residuals (lag 0-200)")
dev.off()
cat("Saved -> output/models/SARIMA_standard/sarima_std_residual_diagnostics.png\n")


# ── 9. Forecast (into the untouched test window) ─────────────
fc <- forecast(fit, h = TEST_H)

# Realign ts_test to fc$mean's time index to prevent a clock
# mismatch error in accuracy().
ts_test <- ts(tv_vec[(TRAIN_N + VALID_H + 1):N],
              start     = stats::start(fc$mean),
              frequency = SEASON)


# ── 10. Accuracy Metrics ─────────────────────────────────────
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

# ── 10b. Supplementary accuracy — observed hours only ────────
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


# ── 11. Forecast vs Actual Plot ──────────────────────────────
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
    title    = sprintf("Standard SARIMA — %d-Day Test Forecast vs Actual",
                       TEST_H %/% 24),
    subtitle = sprintf("RMSE = %.1f  |  MAE = %.1f  |  MAPE = %.2f%%  |  sMAPE = %.2f%%",
                       acc["Test set", "RMSE"], acc["Test set", "MAE"],
                       mape, smape),
    x       = "Date",
    y       = "Traffic Volume (vehicles / hr)",
    caption = "Shaded bands: 80% (dark) and 95% (light) prediction intervals"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey40"))

ggsave(file.path(output_dir, "sarima_std_forecast_vs_actual.png"),
       p_fc, width = 14, height = 5, dpi = 150)
cat("Saved -> output/models/SARIMA_standard/sarima_std_forecast_vs_actual.png\n")


# ── 12. First-Week Zoom Plot ─────────────────────────────────
p_zoom <- ggplot(fc_df[1:min(24 * 7, nrow(fc_df)), ], aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual),   colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.7,
            linetype = "dashed") +
  labs(title = "Standard SARIMA — First 7 Days of Test Window (Zoom)",
       x = "Date", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_std_forecast_week1_zoom.png"),
       p_zoom, width = 12, height = 4, dpi = 150)
cat("Saved -> output/models/SARIMA_standard/sarima_std_forecast_week1_zoom.png\n")


# ── 13. Save Forecast CSV ────────────────────────────────────
fc_df$date_time <- format(fc_df$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(fc_df,
          file.path(output_dir, "sarima_std_forecast_results.csv"),
          row.names = FALSE)
cat("Saved -> output/models/SARIMA_standard/sarima_std_forecast_results.csv\n")


# ── 14. Save Model Summary JSON ──────────────────────────────
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
    fourier_terms  = FALSE,
    xreg           = FALSE,
    AIC            = fit$aic,
    AICc           = fit$aicc,
    BIC            = fit$bic,
    log_likelihood = fit$loglik
  ),
  stationarity = list(
    ADF_pvalue  = round(adf_res$p.value,  4),
    KPSS_pvalue = round(kpss_res$p.value, 4),
    d_order     = d_order,
    D_order     = D_order
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
    total_n  = N,
    train_n  = TRAIN_N,
    valid_h  = VALID_H,
    test_h   = TEST_H,
    train_pct = round(100 * TRAIN_N / N, 1),
    valid_pct = round(100 * VALID_H / N, 1),
    test_pct  = round(100 * TEST_H  / N, 1),
    season   = SEASON
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "sarima_std_model_summary.json"))
cat("Saved -> output/models/SARIMA_standard/sarima_std_model_summary.json\n")


# ── 15. Save Model Object ────────────────────────────────────
saveRDS(fit, file.path(output_dir, "sarima_std_model.rds"))
cat("Saved -> output/models/SARIMA_standard/sarima_std_model.rds\n")
cat("(Reload with: fit <- readRDS('output/models/SARIMA_standard/sarima_std_model.rds'))\n")


# ── 16. Done ─────────────────────────────────────────────────
cat("\n=============================================\n")
cat("  Standard SARIMA modelling complete.\n")
cat("  Outputs written to:", output_dir, "\n")
cat("=============================================\n")