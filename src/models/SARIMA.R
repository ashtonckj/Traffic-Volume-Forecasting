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
input_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "output/models"
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
# s = 24 captures the dominant intra-day (daily) seasonal cycle.
# The weekly cycle (s = 168) is handled via Fourier terms in
# Section 7 rather than expanding the seasonal period, which
# would make SARIMA estimation computationally intractable.
SEASON <- 24

ts_tv <- ts(df$traffic_volume, frequency = SEASON)

cat("ts object — length:", length(ts_tv),
    "| frequency:", frequency(ts_tv), "\n")


# ── 5. Train / Test Split ────────────────────────────────────
# Hold out the final 30 days (720 hours) as the test window.
#
# NOTE: head()/tail() are used instead of window() to avoid an
# edge case where TRAIN_N %% SEASON == 0 produces a cycle index
# of 0, which window() treats as the last cycle of the prior
# period and silently returns the wrong observations.
TEST_H  <- 24 * 56    # 24 * 30 for 30 days // 24 * 56 for 56 days
TRAIN_N <- length(ts_tv) - TEST_H

ts_train <- head(ts_tv, TRAIN_N)
ts_test  <- tail(ts_tv, TEST_H)

cat("Train length:", length(ts_train),
    "| Test length:", length(ts_test), "\n")


# ── 6. Stationarity Tests ────────────────────────────────────
# Tests are run on the training series only — no test leakage.
cat("\n--- Stationarity tests on training series ---\n")

adf_res  <- adf.test(as.numeric(ts_train), alternative = "stationary")
kpss_res <- kpss.test(as.numeric(ts_train))

cat(sprintf("ADF  p = %.4f  (%s)\n", adf_res$p.value,
            ifelse(adf_res$p.value  < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS p = %.4f  (%s)\n", kpss_res$p.value,
            ifelse(kpss_res$p.value > 0.05, "stationary", "non-stationary")))

# Determine non-seasonal differencing order only.
# D is fixed to 0 below because Fourier regressors already absorb
# the seasonal structure — combining D >= 1 with Fourier terms
# double-handles seasonality and causes near-collinearity between
# the differenced series and the regressors, which blows up the
# likelihood and produces Inf AICc for every candidate model.
d_order <- ndiffs(ts_train)
cat("Suggested d:", d_order, "\n")
cat("D fixed to 0 (Fourier terms handle seasonal structure).\n")


# ── 7. Fourier Terms (weekly seasonality proxy) ──────────────
# K Fourier pairs model the weekly (period = 168 h) cycle within
# the s = 24 ts object. The optimal K is found via a two-phase
# strategy:
#
#   Phase 1 — FAST pilot search (TUNE_K = TRUE):
#     auto.arima() runs on a subsampled pilot series (PILOT_WEEKS
#     of recent data) with stepwise = TRUE and approximation = TRUE.
#     This takes ~2-5 minutes for K = 1:12 instead of hours.
#     The K with the lowest AICc on the pilot is selected.
#
#   Phase 2 — FINAL fit (Section 8):
#     The full training series is fitted once with the best K.
#     stepwise / approximation settings in Section 8 control
#     how thorough that single final fit is.
#
# Set TUNE_K = FALSE to skip the search and use K_FIXED directly.
# Max K for frequency 24 is 12 (= floor(24 / 2)).

TUNE_K     <- TRUE
K_FIXED    <- 5       # used directly when TUNE_K = FALSE,
# overwritten by the search when TUNE_K = TRUE

# Number of recent weeks to use for the pilot search.
# 26 weeks (~6 months) captures enough seasonal cycles to
# rank K values reliably while keeping each fit fast.
PILOT_WEEKS <- 26
PILOT_N     <- min(PILOT_WEEKS * 7 * 24, length(ts_train))

# Pilot series: take the most recent PILOT_N observations so the
# series end matches the training cutoff (avoids look-ahead bias
# in the pilot — we never use test data here).
ts_pilot <- tail(ts_train, PILOT_N)

get_fourier_train <- function(k) fourier(ts_train, K = k)
get_fourier_pilot <- function(k) fourier(ts_pilot, K = k)
get_fourier_test  <- function(k) fourier(ts_train, K = k, h = TEST_H)

if (TUNE_K) {
  cat("\nTuning K on", PILOT_N, "obs pilot (~", round(PILOT_N / 168, 1),
      "weeks) — stepwise + approximation for speed ...\n")
  
  k_grid   <- 1:12
  aicc_vec <- rep(NA_real_, length(k_grid))
  
  for (k in k_grid) {
    tryCatch({
      fit_k <- auto.arima(
        ts_pilot,
        xreg          = get_fourier_pilot(k),
        seasonal      = TRUE,
        stepwise      = TRUE,   # fast: pilot only needs to rank K values
        approximation = TRUE,   # fast: pilot only needs to rank K values
        d             = d_order,
        D             = 0,      # must stay 0 with Fourier xreg
        max.p = 3, max.q = 3,
        max.P = 2,  max.Q = 2
      )
      aicc_vec[k] <- fit_k$aicc
      cat(sprintf("  K = %2d  AICc = %.2f\n", k, fit_k$aicc))
    }, error = function(e) {
      cat(sprintf("  K = %2d  FAILED: %s\n", k, conditionMessage(e)))
    })
  }
  
  if (all(is.na(aicc_vec))) stop("All K values failed during pilot search.")
  K_FIXED <- k_grid[which.min(aicc_vec)]
  cat("Best K from pilot search:", K_FIXED, "\n")
}

# Build the final regressor matrices using the FULL training series.
xreg_train <- get_fourier_train(K_FIXED)
xreg_test  <- get_fourier_test(K_FIXED)

cat("Using K =", K_FIXED, "Fourier pair(s) for weekly seasonality.\n")


# ── 8. Model Fitting ─────────────────────────────────────────
# auto.arima() searches over ARIMA(p,d,q)(P,0,Q)[24] with the
# Fourier matrix as an external regressor.
# D = 0 is enforced here — see explanation in Section 6.
# stepwise = TRUE for a faster first run; set FALSE before
# final submission for an exhaustive search.
cat("\nFitting SARIMA model — please wait ...\n")

fit <- auto.arima(
  ts_train,
  xreg          = xreg_train,
  seasonal      = TRUE,
  stepwise      = TRUE,       # set FALSE for exhaustive search
  approximation = TRUE,       # set FALSE for exact likelihood
  d             = d_order,
  D             = 0,          # must stay 0 with Fourier xreg
  max.p = 3, max.q = 3,
  max.P = 2,  max.Q = 2,
  trace         = TRUE
)

cat("\n--- Fitted model ---\n")
print(summary(fit))


# ── 9. Residual Diagnostics ───────────────────────────────────
resid <- residuals(fit)

# Ljung-Box: lag = 2 x seasonal period is the conventional choice;
# fitdf adjusts for the estimated ARMA parameters.
lb <- Box.test(
  resid,
  lag   = 2 * SEASON,
  type  = "Ljung-Box",
  fitdf = sum(fit$arma[1:4])
)

cat(sprintf("\nLjung-Box (lag=%d): stat = %.4f, p = %.4f  (%s)\n",
            2 * SEASON, lb$statistic, lb$p.value,
            ifelse(lb$p.value > 0.05,
                   "residuals ~ white noise",
                   "autocorrelation remains in residuals")))

# Shapiro-Wilk on a random subsample (test requires n <= 5000)
set.seed(42)
sw_sample <- as.numeric(resid)
if (length(sw_sample) > 5000) sw_sample <- sample(sw_sample, 5000)
sw <- shapiro.test(sw_sample)
cat(sprintf("Shapiro-Wilk: stat = %.4f, p = %.4f\n",
            sw$statistic, sw$p.value))

# 4-panel residual plot
png(file.path(output_dir, "sarima_residual_diagnostics.png"),
    width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(resid,
     main = "Residuals over Time",
     ylab = "Residual", xlab = "Time (hours)",
     col  = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)

h_info <- hist(resid, breaks = 60, plot = FALSE)
hist(resid, breaks = 60,
     col = "#756bb1", border = "white",
     main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean(resid), sd(resid)) *
        length(resid) * diff(h_info$breaks[1:2]),
      col = "red", lwd = 2, add = TRUE)

acf(resid,  lag.max = 72, main = "ACF of Residuals")
pacf(resid, lag.max = 72, main = "PACF of Residuals")

dev.off()
cat("Saved -> output/models/sarima_residual_diagnostics.png\n")


# ── 10. Forecast ─────────────────────────────────────────────
fc <- forecast(fit, xreg = xreg_test, h = TEST_H)


# ── 11. Accuracy Metrics ─────────────────────────────────────
acc <- accuracy(fc, ts_test)

actual    <- as.numeric(ts_test)
predicted <- as.numeric(fc$mean)

# MAPE — guard against zero actuals to avoid division by zero
mape <- mean(
  abs((actual - predicted) / ifelse(actual == 0, NA, actual)),
  na.rm = TRUE
) * 100

# Symmetric MAPE — bounded and robust to near-zero actuals
smape <- mean(
  2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-8),
  na.rm = TRUE
) * 100

cat("\n--- Forecast Accuracy (test set) ---\n")
cat(sprintf("ME    : %10.4f\n",    acc["Test set", "ME"]))
cat(sprintf("RMSE  : %10.4f\n",    acc["Test set", "RMSE"]))
cat(sprintf("MAE   : %10.4f\n",    acc["Test set", "MAE"]))
cat(sprintf("MAPE  : %10.4f %%\n", mape))
cat(sprintf("sMAPE : %10.4f %%\n", smape))
cat(sprintf("MASE  : %10.4f\n",    acc["Test set", "MASE"]))


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
  geom_ribbon(aes(ymin = lo95, ymax = hi95),
              fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80),
              fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual),
            colour = "#333333", linewidth = 0.5) +
  geom_line(aes(y = forecast),
            colour = "#e34a33", linewidth = 0.6, linetype = "dashed") +
  labs(
    title    = "SARIMA — 30-Day Forecast vs Actual (Test Set)",
    subtitle = sprintf("RMSE = %.1f  |  MAE = %.1f  |  MAPE = %.2f%%",
                       acc["Test set", "RMSE"],
                       acc["Test set", "MAE"],
                       mape),
    x       = "Date",
    y       = "Traffic Volume (vehicles / hr)",
    caption = "Shaded bands: 80% (dark) and 95% (light) prediction intervals"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey40"))

ggsave(file.path(output_dir, "sarima_forecast_vs_actual.png"),
       p_fc, width = 14, height = 5, dpi = 150)
cat("Saved -> output/models/sarima_forecast_vs_actual.png\n")


# ── 13. First-Week Zoom Plot ─────────────────────────────────
# A 30-day panel is dense at hourly resolution; a 7-day zoom
# makes the daily pattern and forecast quality clearly visible.
p_zoom <- ggplot(fc_df[1:(24 * 7), ], aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95),
              fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80),
              fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual),
            colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast),
            colour = "#e34a33", linewidth = 0.7, linetype = "dashed") +
  labs(
    title = "SARIMA — First 7 Days of Test Window (Zoom)",
    x     = "Date",
    y     = "Traffic Volume (vehicles / hr)"
  ) +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_forecast_week1_zoom.png"),
       p_zoom, width = 12, height = 4, dpi = 150)
cat("Saved -> output/models/sarima_forecast_week1_zoom.png\n")


# ── 14. Save Forecast CSV ────────────────────────────────────
fc_out <- fc_df
fc_out$date_time <- format(fc_out$date_time, "%Y-%m-%d %H:%M:%S")

write.csv(fc_out,
          file.path(output_dir, "sarima_forecast_results.csv"),
          row.names = FALSE)
cat("Saved -> output/models/sarima_forecast_results.csv\n")


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
    D_order     = 0
  ),
  residual_tests = list(
    Ljung_Box_p    = round(lb$p.value, 4),
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
    train_n = TRAIN_N,
    test_h  = TEST_H,
    season  = SEASON
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "sarima_model_summary.json"))
cat("Saved -> output/models/sarima_model_summary.json\n")


# ── 16. Save Model Object ────────────────────────────────────
saveRDS(fit, file.path(output_dir, "sarima_model.rds"))
cat("Saved -> output/models/sarima_model.rds\n")
cat("(Reload later with: fit <- readRDS('output/models/sarima_model.rds'))\n")


# ── 17. Done ─────────────────────────────────────────────────
cat("\n=============================================\n")
cat("  SARIMA modelling complete.\n")
cat("  Outputs written to:", output_dir, "\n")
cat("=============================================\n")