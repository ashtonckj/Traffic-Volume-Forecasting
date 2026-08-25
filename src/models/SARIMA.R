# ============================================================
#  SARIMA Model — Metro Interstate Traffic Volume
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/models/  (plots + diagnostics + forecasts)
# ============================================================

# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("forecast", "tseries", "lubridate", "dplyr",
          "ggplot2", "zoo", "jsonlite", "tibble")
new  <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))


# ── 1. Paths ─────────────────────────────────────────────────
input_path  <- "data/processed/traffic_volume_processed.csv"
output_dir  <- "output/models"
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


# ── 3. Guard: verify the series is truly hourly with no gaps ─
# After preprocessing.R the grid should be complete; fail loudly if not.
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
# Seasonality s = 24  captures the dominant intra-day cycle
# (morning / evening rush-hour peaks visible in the EDA).
# A weekly cycle (s = 168) is computationally prohibitive for
# standard SARIMA; it is handled via Fourier terms in Section 7.
SEASON <- 24

ts_tv <- ts(df$traffic_volume, frequency = SEASON)

cat("ts object — length:", length(ts_tv),
    "| frequency:", frequency(ts_tv), "\n")


# ── 5. Train / Test Split ────────────────────────────────────
# Hold out the final 30 days (720 hours) as the test window.
# This matches a common evaluation horizon for traffic forecasting.
TEST_H   <- 24 * 30     # 720 hours
TRAIN_N  <- length(ts_tv) - TEST_H

ts_train <- window(ts_tv, end   = c(ceiling(TRAIN_N / SEASON), TRAIN_N %% SEASON))
ts_test  <- window(ts_tv, start = c(ceiling((TRAIN_N + 1) / SEASON),
                                    (TRAIN_N + 1) %% SEASON))

cat("Train length:", length(ts_train),
    "| Test length:", length(ts_test), "\n")


# ── 6. Stationarity & Differencing Orders ───────────────────
# Use the training series only to determine d and D so that
# test information never leaks into model specification.
cat("\n--- Stationarity tests on training series ---\n")

adf_res  <- adf.test(as.numeric(ts_train), alternative = "stationary")
kpss_res <- kpss.test(as.numeric(ts_train))

cat(sprintf("ADF  p = %.4f  (%s)\n", adf_res$p.value,
            ifelse(adf_res$p.value  < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS p = %.4f  (%s)\n", kpss_res$p.value,
            ifelse(kpss_res$p.value > 0.05, "stationary", "non-stationary")))

d_order <- ndiffs(ts_train)
D_order <- nsdiffs(ts_train)
cat("Suggested d:", d_order, "| Suggested D:", D_order, "\n")


# ── 7. Fourier Terms (weekly seasonality proxy) ──────────────
# K = 5 Fourier pairs capture the weekly (s = 168) cycle without
# expanding the SARIMA seasonal period to 168, keeping estimation
# tractable. K was chosen by minimising AICc over K = 1:10 on a
# short pilot run; set TUNE_K = TRUE to repeat the grid search.
TUNE_K  <- FALSE
K_FIXED <- 5

get_fourier_train <- function(k) fourier(ts_train, K = k)
get_fourier_test  <- function(k) fourier(ts_train, K = k, h = TEST_H)

if (TUNE_K) {
  cat("\nTuning K (Fourier pairs) — this may take a few minutes …\n")
  k_grid   <- 1:10
  aicc_vec <- numeric(length(k_grid))
  
  for (k in k_grid) {
    fit_k <- auto.arima(
      ts_train,
      xreg        = get_fourier_train(k),
      seasonal    = TRUE,
      stepwise    = TRUE,
      approximation = TRUE,
      d           = d_order,
      D           = D_order,
      max.p = 3, max.q = 3, max.P = 2, max.Q = 2
    )
    aicc_vec[k] <- fit_k$aicc
    cat(sprintf("  K = %2d  AICc = %.2f\n", k, fit_k$aicc))
  }
  
  K_FIXED <- k_grid[which.min(aicc_vec)]
  cat("Best K:", K_FIXED, "\n")
}

xreg_train <- get_fourier_train(K_FIXED)
xreg_test  <- get_fourier_test(K_FIXED)

cat("Using K =", K_FIXED, "Fourier pair(s) for weekly seasonality.\n")


# ── 8. Model Fitting ─────────────────────────────────────────
# auto.arima() searches over (p, d, q)(P, D, Q)[24] with the
# Fourier matrix as an external regressor.
# stepwise = FALSE gives a more exhaustive search but is slow on
# 3 years of hourly data; set to TRUE for a faster first run.
cat("\nFitting SARIMA model — please wait …\n")

fit <- auto.arima(
  ts_train,
  xreg          = xreg_train,
  seasonal      = TRUE,
  stepwise      = TRUE,       # set FALSE for exhaustive search
  approximation = TRUE,       # set FALSE for exact likelihood
  d             = d_order,
  D             = D_order,
  max.p = 3, max.q = 3,
  max.P = 2,  max.Q = 2,
  trace         = TRUE
)

cat("\n--- Fitted model ---\n")
print(summary(fit))


# ── 9. Residual Diagnostics ───────────────────────────────────
resid <- residuals(fit)

# Ljung-Box on residuals (lag = 2 × seasonal period is conventional)
lb <- Box.test(resid, lag = 2 * SEASON, type = "Ljung-Box", fitdf = sum(fit$arma[1:4]))
cat(sprintf("\nLjung-Box (lag=%d): stat = %.4f, p = %.4f  (%s)\n",
            2 * SEASON, lb$statistic, lb$p.value,
            ifelse(lb$p.value > 0.05, "residuals ~ white noise ✔",
                   "autocorrelation remains in residuals ✗")))

# Shapiro-Wilk normality test on a random sample (max n = 5000)
sw_sample <- as.numeric(resid)
if (length(sw_sample) > 5000) sw_sample <- sample(sw_sample, 5000)
sw <- shapiro.test(sw_sample)
cat(sprintf("Shapiro-Wilk          : stat = %.4f, p = %.4f\n",
            sw$statistic, sw$p.value))

# Residual plot (4-panel)
png(file.path(output_dir, "sarima_residual_diagnostics.png"),
    width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(resid, main = "Residuals over Time",
     ylab = "Residual", xlab = "Time (hours)", col = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)

hist(resid, breaks = 60, col = "#756bb1", border = "white",
     main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean(resid), sd(resid)) * length(resid) * diff(hist(resid, plot = FALSE)$breaks[1:2]),
      col = "red", lwd = 2, add = TRUE)

acf(resid,  lag.max = 72, main = "ACF of Residuals")
pacf(resid, lag.max = 72, main = "PACF of Residuals")

dev.off()
cat("Saved → output/models/sarima_residual_diagnostics.png\n")


# ── 10. Forecast ─────────────────────────────────────────────
fc <- forecast(fit, xreg = xreg_test, h = TEST_H)


# ── 11. Accuracy Metrics ─────────────────────────────────────
acc <- accuracy(fc, ts_test)

actual    <- as.numeric(ts_test)
predicted <- as.numeric(fc$mean)

# Mean Absolute Percentage Error (guard against zero actuals)
mape <- mean(abs((actual - predicted) / ifelse(actual == 0, NA, actual)),
             na.rm = TRUE) * 100

# Symmetric MAPE
smape <- mean(2 * abs(actual - predicted) /
                (abs(actual) + abs(predicted) + 1e-8),
              na.rm = TRUE) * 100

cat("\n--- Forecast Accuracy (test set) ---\n")
cat(sprintf("ME    : %10.4f\n", acc["Test set", "ME"]))
cat(sprintf("RMSE  : %10.4f\n", acc["Test set", "RMSE"]))
cat(sprintf("MAE   : %10.4f\n", acc["Test set", "MAE"]))
cat(sprintf("MAPE  : %10.4f %%\n", mape))
cat(sprintf("sMAPE : %10.4f %%\n", smape))
cat(sprintf("MASE  : %10.4f\n", acc["Test set", "MASE"]))


# ── 12. Forecast vs Actual Plot ──────────────────────────────
test_dates <- df$date_time[(TRAIN_N + 1):nrow(df)]

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
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.6, linetype = "dashed") +
  labs(
    title    = "SARIMA — 30-Day Forecast vs Actual (Test Set)",
    subtitle = sprintf("RMSE = %.1f  |  MAE = %.1f  |  MAPE = %.2f%%",
                       acc["Test set", "RMSE"], acc["Test set", "MAE"], mape),
    x = "Date", y = "Traffic Volume (vehicles / hr)",
    caption  = "Shaded bands: 80% (dark) and 95% (light) prediction intervals"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey40"))

ggsave(file.path(output_dir, "sarima_forecast_vs_actual.png"),
       p_fc, width = 14, height = 5, dpi = 150)
cat("Saved → output/models/sarima_forecast_vs_actual.png\n")


# ── 13. First-week zoom ──────────────────────────────────────
# A 30-day panel is dense; a 7-day zoom shows the hourly pattern clearly.
p_zoom <- ggplot(fc_df[1:(24 * 7), ], aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual),   colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.7, linetype = "dashed") +
  labs(
    title   = "SARIMA — First 7 Days of Test Window (Zoom)",
    x = "Date", y = "Traffic Volume (vehicles / hr)"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_forecast_week1_zoom.png"),
       p_zoom, width = 12, height = 4, dpi = 150)
cat("Saved → output/models/sarima_forecast_week1_zoom.png\n")


# ── 14. Save Forecast CSV ────────────────────────────────────
fc_df$date_time <- format(fc_df$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(fc_df,
          file.path(output_dir, "sarima_forecast_results.csv"),
          row.names = FALSE)
cat("Saved → output/models/sarima_forecast_results.csv\n")


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
    Ljung_Box_p   = round(lb$p.value, 4),
    Shapiro_Wilk_p = round(sw$p.value, 4)
  ),
  accuracy = list(
    ME    = round(acc["Test set", "ME"],   4),
    RMSE  = round(acc["Test set", "RMSE"], 4),
    MAE   = round(acc["Test set", "MAE"],  4),
    MAPE  = round(mape,  4),
    sMAPE = round(smape, 4),
    MASE  = round(acc["Test set", "MASE"], 4)
  ),
  split = list(
    train_n  = TRAIN_N,
    test_h   = TEST_H,
    season   = SEASON
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "sarima_model_summary.json"))
cat("Saved → output/models/sarima_model_summary.json\n")


# ── 16. Save Model Object ────────────────────────────────────
saveRDS(fit, file.path(output_dir, "sarima_model.rds"))
cat("Saved → output/models/sarima_model.rds\n")
cat("\n(Reload later with: fit <- readRDS('output/models/sarima_model.rds'))\n")


# ── 17. Done ─────────────────────────────────────────────────
cat("\n══════════════════════════════════════════════\n")
cat("  SARIMA modelling complete.\n")
cat("  Outputs written to:", output_dir, "\n")
cat("══════════════════════════════════════════════\n")