# ============================================================
#  SARIMA Model — Metro Interstate Traffic Volume
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/models/  (plots + diagnostics + forecasts)
#
#  Key changes vs previous version:
#   - ts() with frequency=24 (not msts) to avoid head()/tail()
#     silently stripping multi-period attributes
#   - Weekly (168h) Fourier terms built by hand, phase-aligned
#     across pilot/train/test slices — fourier() is NOT used
#     because it derives its period from frequency(ts), not 168
#   - Separate K tuning for daily (via SARIMA P/Q) vs weekly
#     (via Fourier) — K_MAX raised to 20 for 168h period
#   - is_holiday added as exogenous regressor (known in advance,
#     no leakage)
#   - Ljung-Box also checked at lag 168 to verify weekly leak
#     is gone after fix
#   - stepwise=FALSE, approximation=FALSE for final fit
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
# Plain ts with frequency=24: the seasonal ARIMA (P,D,Q)[24]
# component handles the daily cycle. The weekly cycle (168h) is
# handled via hand-built Fourier regressors in Section 7.
#
# We deliberately avoid msts here: head()/tail() (used for the
# train/test split) dispatch to head.ts/tail.ts, which know
# about frequency/start but NOT about the msts seasonal.periods
# attribute. The result is that ts_train can silently downgrade
# to a plain single-frequency ts — no error, just wrong behavior
# downstream. Using ts(freq=24) avoids this entirely.
SEASON       <- 24
WEEKLY_PERIOD <- 168

ts_tv <- ts(df$traffic_volume, frequency = SEASON)

cat("ts object — length:", length(ts_tv),
    "| frequency:", frequency(ts_tv), "\n")


# ── 5. Train / Test Split ────────────────────────────────────
# Hold out the final 56 days (1344 hours) as the test window.
#
# NOTE: head()/tail() are used instead of window() to avoid an
# edge case where TRAIN_N %% SEASON == 0 produces a cycle index
# of 0, which window() treats as the last cycle of the prior
# period and silently returns the wrong observations.
TEST_H  <- 24 * 56
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
# D is fixed to 0: Fourier regressors already absorb seasonal
# structure, so combining D>=1 with Fourier terms double-handles
# seasonality and causes near-collinearity that blows up the
# likelihood (Inf AICc for every candidate model).
d_order <- ndiffs(ts_train)
cat("Suggested d:", d_order, "\n")
cat("D fixed to 0 (Fourier terms handle seasonal structure).\n")


# ── 7. Weekly Fourier Terms (hand-built, period = 168h) ──────
# WHY NOT fourier()? forecast::fourier() builds sin/cos terms
# using frequency(x) as the period — for a ts with freq=24 that
# gives period=24, not 168. Every "weekly" term in the previous
# version was actually a redundant harmonic of the daily cycle,
# which is why K pinned to 12 (the maximum) and the weekly
# signal leaked into the residuals.
#
# Here we build the terms manually with a fixed period of 168.
# start_t tracks where each slice begins in the FULL series so
# that the sin/cos phase stays consistent across pilot → train
# → test. If start_t were always reset to 1, the phase would
# shift and the regressors would be misaligned at the forecast
# origin.
#
# K_MAX: floor(168/2) = 84 is the mathematical limit, but past
# ~20-30 pairs you are mostly fitting noise. We cap at 20 for
# the pilot search; raise if AICc is still falling at K=20.

make_weekly_fourier <- function(n, K, start_t = 1) {
  # n       : number of rows to generate
  # K       : number of sin/cos pairs
  # start_t : index of the first row in the FULL series
  #           (1-based; ensures phase alignment across slices)
  t <- start_t:(start_t + n - 1)
  X <- matrix(NA_real_, nrow = n, ncol = 2 * K)
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

# ── 7a. Pilot K search ───────────────────────────────────────
# Phase 1: fast pilot search over K=1:K_MAX using a recent
# sub-series. stepwise=TRUE and approximation=TRUE keep each
# fit to seconds rather than minutes.
# Phase 2: full training series is fitted once (Section 8)
# with the winning K.
# Set TUNE_K=FALSE to skip the search and use K_FIXED directly.

TUNE_K      <- FALSE
K_FIXED     <- 20      # used when TUNE_K=FALSE; overwritten otherwise
K_MAX       <- 20     # raise to 30 if AICc is still falling at 20

PILOT_WEEKS <- 26
PILOT_N     <- min(PILOT_WEEKS * 7 * 24, length(ts_train))

# pilot_start_t: the position of the pilot's first observation
# within the FULL series (so phase is correct)
pilot_start_t <- TRAIN_N - PILOT_N + 1
ts_pilot      <- tail(ts_train, PILOT_N)

if (TUNE_K) {
  cat("\nTuning weekly Fourier K on", PILOT_N, "obs pilot (~",
      round(PILOT_N / 168, 1), "weeks) ...\n")
  
  k_grid   <- 1:K_MAX
  aicc_vec <- rep(NA_real_, length(k_grid))
  
  for (k in k_grid) {
    tryCatch({
      fit_k <- auto.arima(
        ts_pilot,
        xreg          = make_weekly_fourier(PILOT_N, k, start_t = pilot_start_t),
        seasonal      = TRUE,
        stepwise      = FALSE,      # fast: pilot only needs to rank K values
        approximation = TRUE,      # fast: pilot only needs to rank K values
        d  = d_order, D = 0,
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
  cat("Best weekly K from pilot search:", K_FIXED, "\n")
}

# ── 7b. Build final xreg matrices ───────────────────────────
# train slice: starts at t=1 in the full series
# test  slice: starts at t=TRAIN_N+1
# is_holiday is appended as an exogenous regressor: it is
# determined by the calendar (known in advance), so including
# it in xreg_test causes no data leakage.

xreg_train <- cbind(
  make_weekly_fourier(TRAIN_N, K_FIXED, start_t = 1),
  is_holiday = df$is_holiday[1:TRAIN_N]
)

xreg_test <- cbind(
  make_weekly_fourier(TEST_H, K_FIXED, start_t = TRAIN_N + 1),
  is_holiday = df$is_holiday[(TRAIN_N + 1):nrow(df)]
)

cat("Using K =", K_FIXED, "weekly Fourier pair(s) + is_holiday regressor.\n")
cat("xreg_train dim:", dim(xreg_train), "\n")
cat("xreg_test  dim:", dim(xreg_test),  "\n")


# ── 8. Model Fitting ─────────────────────────────────────────
# auto.arima() searches over ARIMA(p,d,q)(P,0,Q)[24] with the
# Fourier + holiday matrix as external regressors.
# D=0 is enforced — see explanation in Section 6.
# stepwise=FALSE + approximation=FALSE for the final fit:
# exhaustive search over the full candidate space with exact
# likelihood evaluation. This is slow (~hours on the full
# training set) but produces a better-calibrated model order.
# Set both back to TRUE only if you need a quick sanity check.
cat("\nFitting SARIMA model on full training set — please wait ...\n")
cat("(stepwise=FALSE, approximation=FALSE: this may take a while)\n")

fit <- auto.arima(
  ts_train,
  xreg          = xreg_train,
  seasonal      = TRUE,
  stepwise      = TRUE,      # exhaustive search
  approximation = TRUE,      # exact likelihood
  d  = d_order,
  D  = 0,                     # must stay 0 with Fourier xreg
  max.p = 3, max.q = 3,
  max.P = 2,  max.Q = 2,
  trace = TRUE
)

cat("\n--- Fitted model ---\n")
print(summary(fit))


# ── 9. Residual Diagnostics ───────────────────────────────────
checkresiduals(fit)
resid <- residuals(fit)

# Ljung-Box at 2*SEASON (48h): standard check for daily residual
# autocorrelation.
lb_48 <- Box.test(
  resid,
  lag   = 2 * SEASON,
  type  = "Ljung-Box",
  fitdf = sum(fit$arma[1:4])
)

# Ljung-Box at lag 168: specifically checks whether the weekly
# seasonal signal has been absorbed. A p-value < 0.05 here
# indicates weekly autocorrelation still leaking through —
# increase K_MAX or check that xreg phase alignment is correct.
lb_168 <- Box.test(
  resid,
  lag   = WEEKLY_PERIOD,
  type  = "Ljung-Box",
  fitdf = sum(fit$arma[1:4])
)

cat(sprintf("\nLjung-Box (lag=%d ): stat = %.4f, p = %.4f  (%s)\n",
            2 * SEASON, lb_48$statistic, lb_48$p.value,
            ifelse(lb_48$p.value > 0.05,
                   "residuals ~ white noise",
                   "autocorrelation remains in residuals")))

cat(sprintf("Ljung-Box (lag=%d): stat = %.4f, p = %.4f  (%s)\n",
            WEEKLY_PERIOD, lb_168$statistic, lb_168$p.value,
            ifelse(lb_168$p.value > 0.05,
                   "no weekly autocorrelation detected",
                   "weekly autocorrelation remains — consider increasing K")))

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

acf(resid,  lag.max = 200, main = "ACF of Residuals (lag up to 200)")
pacf(resid, lag.max = 200, main = "PACF of Residuals (lag up to 200)")

dev.off()
cat("Saved -> output/models/sarima_residual_diagnostics.png\n")


# ── 10. Forecast ─────────────────────────────────────────────
fc <- forecast(fit, xreg = xreg_test, h = TEST_H)


# ── 11. Accuracy Metrics ─────────────────────────────────────
acc <- accuracy(fc, ts_test)

actual    <- as.numeric(ts_test)
predicted <- as.numeric(fc$mean)

# MAPE — guard against zero actuals to avoid division by zero.
# Note: MAPE is inflated by near-zero overnight traffic counts
# regardless of model quality; sMAPE and MASE are more reliable
# for this series.
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
    title    = "SARIMA — 56-Day Forecast vs Actual (Test Set)",
    subtitle = sprintf("RMSE = %.1f  |  MAE = %.1f  |  MAPE = %.2f%%  |  sMAPE = %.2f%%",
                       acc["Test set", "RMSE"],
                       acc["Test set", "MAE"],
                       mape, smape),
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
# A 56-day panel is dense at hourly resolution; a 7-day zoom
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
    fourier_K         = K_FIXED,
    fourier_period    = WEEKLY_PERIOD,
    xreg_cols         = colnames(xreg_train),
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