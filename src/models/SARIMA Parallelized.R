# ============================================================
#  SARIMA Model — Metro Interstate Traffic Volume
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/models/  (plots + diagnostics + forecasts)
#
#  Key changes vs previous version:
#   - ts() with frequency=24 (not msts) to avoid head()/tail()
#     silently stripping multi-period attributes
#   - Weekly (168h) Fourier terms built by hand, phase-aligned
#     across train/valid/test slices — fourier() is NOT used
#     because it derives its period from frequency(ts), not 168
#   - THREE-WAY SPLIT (train / validation / test) instead of a
#     two-way split with an AICc-based pilot search:
#       * Train      = everything except the last 6 months
#       * Validation = 3 months, used ONLY to pick weekly
#                      Fourier K by genuine out-of-sample
#                      forecast RMSE (fit on train, forecast
#                      into validation) — not by AICc on a
#                      sub-slice of train, and not by touching
#                      the test set at all.
#       * Test       = final 3 months, touched exactly once,
#                      for final reporting only.
#     This directly targets forecast accuracy during tuning
#     (rather than in-sample likelihood) and removes any risk
#     of tuning decisions leaking into the final test metrics.
#   - Seasonal search bounds (max.P/max.Q) widened to 4 after
#     residual ACF showed banded (not single-spike) leftover
#     autocorrelation around lag 24 with the original max.P=2,
#     max.Q=2 — see MAX_P_SEASONAL/MAX_Q_SEASONAL below.
#   - is_holiday added as exogenous regressor (known in advance,
#     no leakage)
#   - Ljung-Box also checked at lag 168 to verify weekly leak
#     is gone after fix
#   - stepwise=TRUE, approximation=TRUE throughout (final fit
#     included) for speed. Because auto.arima's own `parallel=`
#     argument only has an effect when stepwise=FALSE (it
#     parallelizes the exhaustive grid search), it does nothing
#     useful here — so instead the K-tuning loop in Section 7a
#     (which fits K_MAX independent models) is parallelized by
#     hand across K values with doParallel + foreach.
# ============================================================


# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("forecast", "tseries", "lubridate", "dplyr",
          "ggplot2", "zoo", "jsonlite", "tibble",
          "doParallel", "foreach")
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

# is_imputed comes from preprocessing.R's gap-fill tiers
# (0 = observed, 1 = interpolated / t-168 / seasonal-average fill).
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

ts_tv <- ts(df$traffic_volume, frequency = SEASON)
tv_vec <- as.numeric(ts_tv)

cat("ts object — length:", length(ts_tv),
    "| frequency:", frequency(ts_tv), "\n")


# ── 5. Train / Validation / Test Split (date-based, 3-way) ──
TEST_MONTHS  <- 3
VALID_MONTHS <- 3

last_time <- max(df$date_time)
test_cut  <- last_time %m-% months(TEST_MONTHS)
valid_cut <- test_cut  %m-% months(VALID_MONTHS)

train_idx <- df$date_time <= valid_cut
valid_idx <- df$date_time >  valid_cut & df$date_time <= test_cut
test_idx  <- df$date_time >  test_cut

TRAIN_N <- sum(train_idx)
VALID_H <- sum(valid_idx)
TEST_H  <- sum(test_idx)

stopifnot(TRAIN_N + VALID_H + TEST_H == nrow(df))

ts_train <- ts(tv_vec[1:TRAIN_N], frequency = SEASON)
ts_valid <- ts(tv_vec[(TRAIN_N + 1):(TRAIN_N + VALID_H)], frequency = SEASON)
ts_test  <- ts(tv_vec[(TRAIN_N + VALID_H + 1):length(tv_vec)], frequency = SEASON)

cat("\n=== Train / Validation / Test Split ===\n")
cat("Train:", as.character(min(df$date_time)), "to",
    as.character(df$date_time[TRAIN_N]), "(", TRAIN_N, "hours )\n")
cat("Valid:", as.character(df$date_time[TRAIN_N + 1]), "to",
    as.character(df$date_time[TRAIN_N + VALID_H]), "(", VALID_H, "hours )\n")
cat("Test :", as.character(df$date_time[TRAIN_N + VALID_H + 1]), "to",
    as.character(last_time), "(", TEST_H, "hours )\n")

# ── 5b. Imputation-leakage check on validation and test ──────
valid_imputed_idx  <- df$is_imputed[(TRAIN_N + 1):(TRAIN_N + VALID_H)]
test_imputed_idx   <- df$is_imputed[(TRAIN_N + VALID_H + 1):nrow(df)]
n_imputed_in_valid <- sum(valid_imputed_idx)
n_imputed_in_test  <- sum(test_imputed_idx)

if (n_imputed_in_valid > 0) {
  cat(sprintf("\nNOTE: %d of %d validation hours (%.1f%%) are imputed. K is being ",
              n_imputed_in_valid, VALID_H, 100 * n_imputed_in_valid / VALID_H))
  cat("selected by comparing forecasts against some gap-filled 'actuals' there.\n")
}
if (n_imputed_in_test > 0) {
  cat(sprintf(
    "\nWARNING: %d of %d test-window hours (%.1f%%) are imputed, not observed.\n",
    n_imputed_in_test, TEST_H, 100 * n_imputed_in_test / TEST_H
  ))
  cat("Final accuracy on those hours is scored against gap-fill values, not\n",
      "real traffic counts. A supplementary accuracy excluding these hours is\n",
      "computed in Section 11b.\n", sep = "")
} else {
  cat("\nNo imputed hours in the test window — clean for final evaluation.\n")
}


# ── 6. Stationarity Tests ────────────────────────────────────
cat("\n--- Stationarity tests on training series ---\n")

adf_res  <- adf.test(as.numeric(ts_train), alternative = "stationary")
kpss_res <- kpss.test(as.numeric(ts_train))

cat(sprintf("ADF  p = %.4f  (%s)\n", adf_res$p.value,
            ifelse(adf_res$p.value  < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS p = %.4f  (%s)\n", kpss_res$p.value,
            ifelse(kpss_res$p.value > 0.05, "stationary", "non-stationary")))

d_order <- ndiffs(ts_train)
cat("Suggested d:", d_order, "\n")
cat("D fixed to 0 (Fourier terms handle seasonal structure).\n")


# ── 7. Weekly Fourier Terms (hand-built, period = 168h) ──────
make_weekly_fourier <- function(n, K, start_t = 1) {
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
      is_holiday = df$is_holiday[(TRAIN_N + VALID_H + 1):nrow(df)]
    )
  )
}

MAX_P_SEASONAL <- 4
MAX_Q_SEASONAL <- 4

# ── 7a. K tuning via out-of-sample validation RMSE (PARALLEL) ─
# Each K is an independent auto.arima fit + forecast — embarrassingly
# parallel. Uses doParallel/foreach with a PSOCK cluster (works on
# Windows/Mac/Linux, unlike mclapply which is Unix-only).
TUNE_K  <- TRUE
K_FIXED <- 20   # used when TUNE_K=FALSE; overwritten by the loop otherwise
K_MAX   <- 30

if (TUNE_K) {
  n_cores <- max(1, parallel::detectCores(logical = TRUE) - 1)
  cat("\nTuning weekly Fourier K on out-of-sample validation forecasts (K = 1:",
      K_MAX, ") using ", n_cores, " parallel workers ...\n", sep = "")
  
  cl <- parallel::makeCluster(n_cores)
  doParallel::registerDoParallel(cl)
  # Ensure the cluster is torn down even if the loop below errors out,
  # so a failed run doesn't leave orphan worker processes.
  on.exit({
    if (exists("cl")) {
      tryCatch(parallel::stopCluster(cl), error = function(e) NULL)
      foreach::registerDoSEQ()
    }
  }, add = TRUE)
  
  k_grid <- 1:K_MAX
  
  tune_results <- foreach(
    k = k_grid,
    .combine  = rbind,
    .packages = c("forecast")
  ) %dopar% {
    tryCatch({
      xr <- build_xreg(k)
      fit_k <- auto.arima(
        ts_train,
        xreg          = xr$train,
        seasonal      = TRUE,
        stepwise      = TRUE,
        approximation = TRUE,
        d  = d_order, D = 0,
        max.p = 3, max.q = 3,
        max.P = MAX_P_SEASONAL, max.Q = MAX_Q_SEASONAL
      )
      fc_k <- forecast(fit_k, xreg = xr$valid, h = VALID_H)
      rmse_k <- sqrt(mean((as.numeric(ts_valid) - as.numeric(fc_k$mean))^2))
      data.frame(K = k, rmse = rmse_k, aicc = fit_k$aicc,
                 error = NA_character_, stringsAsFactors = FALSE)
    }, error = function(e) {
      data.frame(K = k, rmse = NA_real_, aicc = NA_real_,
                 error = conditionMessage(e), stringsAsFactors = FALSE)
    })
  }
  
  parallel::stopCluster(cl)
  foreach::registerDoSEQ()
  rm(cl)  # so the on.exit cleanup above is a no-op now that it's done
  
  tune_results <- tune_results[order(tune_results$K), ]
  for (i in seq_len(nrow(tune_results))) {
    row <- tune_results[i, ]
    if (is.na(row$rmse)) {
      cat(sprintf("  K = %2d  FAILED: %s\n", row$K, row$error))
    } else {
      cat(sprintf("  K = %2d  Validation RMSE = %8.2f   (train AICc = %.2f)\n",
                  row$K, row$rmse, row$aicc))
    }
  }
  
  if (all(is.na(tune_results$rmse))) stop("All K values failed during validation tuning.")
  K_FIXED <- tune_results$K[which.min(tune_results$rmse)]
  cat(sprintf("Best weekly K by validation RMSE: %d (RMSE = %.2f)\n",
              K_FIXED, min(tune_results$rmse, na.rm = TRUE)))
}

# ── 7b. Build final xreg matrices with the chosen K ──────────
xreg_final <- build_xreg(K_FIXED)
xreg_train <- xreg_final$train
xreg_valid <- xreg_final$valid
xreg_test  <- xreg_final$test

cat("Using K =", K_FIXED, "weekly Fourier pair(s) + is_holiday regressor.\n")
cat("xreg_train dim:", dim(xreg_train), "\n")
cat("xreg_test  dim:", dim(xreg_test),  "\n")


# ── 8. Model Fitting ─────────────────────────────────────────
REFIT_WITH_VALID <- FALSE

if (REFIT_WITH_VALID) {
  ts_fit_final   <- ts(c(as.numeric(ts_train), as.numeric(ts_valid)), frequency = SEASON)
  xreg_fit_final <- rbind(xreg_train, xreg_valid)
  cat("\nFitting final SARIMA model on TRAIN + VALIDATION (", length(ts_fit_final),
      "hours) — please wait ...\n")
} else {
  ts_fit_final   <- ts_train
  xreg_fit_final <- xreg_train
  cat("\nFitting final SARIMA model on TRAIN only (", length(ts_fit_final),
      "hours) — please wait ...\n")
}
# stepwise=TRUE + approximation=TRUE: fastest setting. Note that
# auto.arima's own parallel=/num.cores= arguments are a no-op when
# stepwise=TRUE (they only parallelize the exhaustive stepwise=FALSE
# grid search), so there's nothing to add here — the loop above is
# where parallelism actually helps.
cat("(stepwise=TRUE, approximation=TRUE: fast greedy search)\n")

fit <- auto.arima(
  ts_fit_final,
  xreg          = xreg_fit_final,
  seasonal      = TRUE,
  stepwise      = TRUE,
  approximation = TRUE,
  d  = d_order,
  D  = 0,
  max.p = 3, max.q = 3,
  max.P = MAX_P_SEASONAL, max.Q = MAX_Q_SEASONAL,
  trace = TRUE
)

cat("\n--- Fitted model ---\n")
print(summary(fit))


# ── 9. Residual Diagnostics ───────────────────────────────────
checkresiduals(fit)
resid <- residuals(fit)

lb_48 <- Box.test(
  resid, lag = 2 * SEASON, type = "Ljung-Box",
  fitdf = sum(fit$arma[1:4])
)
lb_168 <- Box.test(
  resid, lag = WEEKLY_PERIOD, type = "Ljung-Box",
  fitdf = sum(fit$arma[1:4])
)

cat(sprintf("\nLjung-Box (lag=%d ): stat = %.4f, p = %.4f  (%s)\n",
            2 * SEASON, lb_48$statistic, lb_48$p.value,
            ifelse(lb_48$p.value > 0.05, "residuals ~ white noise", "autocorrelation remains in residuals")))
cat(sprintf("Ljung-Box (lag=%d): stat = %.4f, p = %.4f  (%s)\n",
            WEEKLY_PERIOD, lb_168$statistic, lb_168$p.value,
            ifelse(lb_168$p.value > 0.05, "no weekly autocorrelation detected", "weekly autocorrelation remains — consider increasing K")))

set.seed(42)
sw_sample <- as.numeric(resid)
if (length(sw_sample) > 5000) sw_sample <- sample(sw_sample, 5000)
sw <- shapiro.test(sw_sample)
cat(sprintf("Shapiro-Wilk: stat = %.4f, p = %.4f\n", sw$statistic, sw$p.value))

png(file.path(output_dir, "sarima_residual_diagnostics.png"),
    width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

plot(resid, main = "Residuals over Time", ylab = "Residual", xlab = "Time (hours)",
     col = "#2c7fb8", cex = 0.3)
abline(h = 0, col = "red", lty = 2)

h_info <- hist(resid, breaks = 60, plot = FALSE)
hist(resid, breaks = 60, col = "#756bb1", border = "white",
     main = "Residual Distribution", xlab = "Residual")
curve(dnorm(x, mean(resid), sd(resid)) *
        length(resid) * diff(h_info$breaks[1:2]),
      col = "red", lwd = 2, add = TRUE)

acf(resid,  lag.max = 200, main = "ACF of Residuals (lag up to 200)")
pacf(resid, lag.max = 200, main = "PACF of Residuals (lag up to 200)")

dev.off()
cat("Saved -> output/models/sarima_residual_diagnostics.png\n")


# ── 10. Forecast (into the untouched test window) ────────────
fc <- forecast(fit, xreg = xreg_test, h = TEST_H)


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

cat("\n--- Forecast Accuracy (test set, ALL hours incl. any imputed) ---\n")
cat(sprintf("ME    : %10.4f\n",    acc["Test set", "ME"]))
cat(sprintf("RMSE  : %10.4f\n",    acc["Test set", "RMSE"]))
cat(sprintf("MAE   : %10.4f\n",    acc["Test set", "MAE"]))
cat(sprintf("MAPE  : %10.4f %%\n", mape))
cat(sprintf("sMAPE : %10.4f %%\n", smape))
cat(sprintf("MASE  : %10.4f\n",    acc["Test set", "MASE"]))

# ── 11b. Supplementary accuracy excluding imputed test hours ─
if (n_imputed_in_test > 0) {
  keep <- test_imputed_idx == 0
  actual_obs    <- actual[keep]
  predicted_obs <- predicted[keep]
  
  rmse_obs <- sqrt(mean((actual_obs - predicted_obs)^2))
  mae_obs  <- mean(abs(actual_obs - predicted_obs))
  mape_obs <- mean(
    abs((actual_obs - predicted_obs) / ifelse(actual_obs == 0, NA, actual_obs)),
    na.rm = TRUE
  ) * 100
  smape_obs <- mean(
    2 * abs(actual_obs - predicted_obs) / (abs(actual_obs) + abs(predicted_obs) + 1e-8),
    na.rm = TRUE
  ) * 100
  
  cat(sprintf("\n--- Forecast Accuracy (test set, OBSERVED hours only, n=%d) ---\n",
              sum(keep)))
  cat(sprintf("RMSE  : %10.4f\n", rmse_obs))
  cat(sprintf("MAE   : %10.4f\n", mae_obs))
  cat(sprintf("MAPE  : %10.4f %%\n", mape_obs))
  cat(sprintf("sMAPE : %10.4f %%\n", smape_obs))
} else {
  rmse_obs <- mae_obs <- mape_obs <- smape_obs <- NA_real_
}


# ── 12. Forecast vs Actual Plot ──────────────────────────────
test_dates <- df$date_time[(TRAIN_N + VALID_H + 1):nrow(df)]

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
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.5) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.6, linetype = "dashed") +
  labs(
    title    = sprintf("SARIMA — %d-Day Test Forecast vs Actual", TEST_H %/% 24),
    subtitle = sprintf("RMSE = %.1f  |  MAE = %.1f  |  MAPE = %.2f%%  |  sMAPE = %.2f%%",
                       acc["Test set", "RMSE"], acc["Test set", "MAE"], mape, smape),
    x = "Date", y = "Traffic Volume (vehicles / hr)",
    caption = "Shaded bands: 80% (dark) and 95% (light) prediction intervals"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.subtitle = element_text(size = 9, colour = "grey40"))

ggsave(file.path(output_dir, "sarima_forecast_vs_actual.png"),
       p_fc, width = 14, height = 5, dpi = 150)
cat("Saved -> output/models/sarima_forecast_vs_actual.png\n")


# ── 13. First-Week Zoom Plot ─────────────────────────────────
p_zoom <- ggplot(fc_df[1:min(24 * 7, nrow(fc_df)), ], aes(x = date_time)) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = "#a6cee3", alpha = 0.4) +
  geom_ribbon(aes(ymin = lo80, ymax = hi80), fill = "#1f78b4", alpha = 0.3) +
  geom_line(aes(y = actual), colour = "#333333", linewidth = 0.7) +
  geom_line(aes(y = forecast), colour = "#e34a33", linewidth = 0.7, linetype = "dashed") +
  labs(title = "SARIMA — First 7 Days of Test Window (Zoom)",
       x = "Date", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "sarima_forecast_week1_zoom.png"),
       p_zoom, width = 12, height = 4, dpi = 150)
cat("Saved -> output/models/sarima_forecast_week1_zoom.png\n")


# ── 14. Save Forecast CSV ────────────────────────────────────
fc_out <- fc_df
fc_out$date_time <- format(fc_out$date_time, "%Y-%m-%d %H:%M:%S")
write.csv(fc_out, file.path(output_dir, "sarima_forecast_results.csv"), row.names = FALSE)
cat("Saved -> output/models/sarima_forecast_results.csv\n")


# ── 15. Save Model Summary JSON ──────────────────────────────
model_order <- arimaorder(fit)

summary_list <- list(
  model = list(
    order          = list(p = model_order[1], d = model_order[2], q = model_order[3]),
    seasonal_order = list(P = model_order[4], D = model_order[5], Q = model_order[6], s = SEASON),
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
    chosen_K            = K_FIXED,
    valid_rmse          = if (TUNE_K) round(min(tune_results$rmse, na.rm = TRUE), 4) else NA,
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
    RMSE  = if (n_imputed_in_test > 0) round(rmse_obs, 4)  else NA,
    MAE   = if (n_imputed_in_test > 0) round(mae_obs, 4)   else NA,
    MAPE  = if (n_imputed_in_test > 0) round(mape_obs, 4)  else NA,
    sMAPE = if (n_imputed_in_test > 0) round(smape_obs, 4) else NA
  ),
  split = list(
    train_n           = TRAIN_N,
    valid_h           = VALID_H,
    test_h            = TEST_H,
    season            = SEASON,
    refit_with_valid  = REFIT_WITH_VALID
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