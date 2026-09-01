# ============================================================
#  Exponential Smoothing (Holt-Winters, additive, weekly season)
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/models/  (plot + scoring row)
#
#  Python equivalent being matched:
#    ExponentialSmoothing(seasonal="add", sp=24*7)
#  i.e. additive seasonality, NO trend (seasonal="add" alone in
#  sktime implies no trend term), weekly period = 168 hours.
#
#  stats::HoltWinters(beta=FALSE, gamma=TRUE, seasonal="additive")
#  is the direct match: beta=FALSE turns off the trend component,
#  gamma=TRUE + seasonal="additive" turns on additive seasonality.
#  This is also the more faithful R equivalent for another reason:
#  HoltWinters() is the same underlying Holt-Winters method sktime's
#  ExponentialSmoothing wraps (via statsmodels) -- unlike forecast::ets(),
#  which builds an m x m seasonal state matrix internally and gets very
#  slow/memory-heavy once m (168 here) goes much past ~24.
#
#  Split: 70/15/15 by ROW COUNT (not calendar months), to match the
#  preprocessing used across the other models being compared.
# ============================================================


# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("forecast", "dplyr", "ggplot2", "tibble", "jsonlite")
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


# ── 3. Build ts Object at Weekly Frequency (168h) ─────────────
WEEKLY_PERIOD <- 168

ts_tv_168 <- ts(df$traffic_volume, frequency = WEEKLY_PERIOD)
tv_vec    <- as.numeric(ts_tv_168)


# ── 4. Train / Validation / Test Split (70/15/15, row-count) ─
# Same split logic as the SARIMA script's Section 5, applied here
# independently and at frequency=168 instead of 24, so both models
# see identical row boundaries despite using different ts frequencies.
TRAIN_PCT <- 0.70
VALID_PCT <- 0.15

TRAIN_N <- floor(TRAIN_PCT * nrow(df))
VALID_H <- floor(VALID_PCT * nrow(df))
TEST_H  <- nrow(df) - TRAIN_N - VALID_H

stopifnot(TRAIN_N + VALID_H + TEST_H == nrow(df))

# Built via window() on the full-length ts_tv_168 (not separate ts()
# calls) so train/valid/test share one continuous time index -- same
# reasoning as the SARIMA script: HoltWinters()/forecast() continue
# the time index from where the fitted series ends, so a fresh ts()
# object that restarts its own clock at time 1 would misalign against
# a downstream accuracy()/window() call.
train_times <- time(ts_tv_168)[1:TRAIN_N]
test_times  <- time(ts_tv_168)[(TRAIN_N + VALID_H + 1):length(tv_vec)]

ts_train_168 <- window(ts_tv_168, start = train_times[1], end = train_times[length(train_times)])

cat("\n=== Train / Validation / Test Split (70/15/15) ===\n")
cat("Train:", TRAIN_N, "hours (", round(100 * TRAIN_N / nrow(df), 1), "%)\n")
cat("Valid:", VALID_H, "hours (", round(100 * VALID_H / nrow(df), 1), "%)\n")
cat("Test :", TEST_H,  "hours (", round(100 * TEST_H  / nrow(df), 1), "%)\n")

# Guard: HoltWinters() errors if the training series is shorter than
# 2 full seasonal cycles (2 * 168 = 336 hours here).
if (TRAIN_N < 2 * WEEKLY_PERIOD) {
  stop("Training set (", TRAIN_N, " hours) is shorter than 2 seasonal ",
       "cycles (", 2 * WEEKLY_PERIOD, " hours) -- HoltWinters() requires more data.")
}


# ── 5. Fit Holt-Winters (additive seasonal, no trend) ─────────
fit_es <- HoltWinters(ts_train_168, beta = FALSE, gamma = NULL, seasonal = "additive")


# ── 6. Forecast into the Test Window ───────────────────────────
fc_es <- forecast(fit_es, h = TEST_H)

# Realign the test actuals to fc_es$mean's own time index (same fix
# as the SARIMA script's Section 10 -- fit_es$x's clock is whatever
# ts_train_168 happened to start at, so anchor to the forecast's own
# start rather than trusting a separately-built ts_test to line up).
test_vec  <- tv_vec[(TRAIN_N + VALID_H + 1):length(tv_vec)]
ts_test_168 <- ts(test_vec, start = stats::start(fc_es$mean), frequency = WEEKLY_PERIOD)

y_pred_es <- as.numeric(fc_es$mean)
y_test_es <- as.numeric(ts_test_168)


# ── 7. Plot (mirrors plot_series(y_test, y_pred)) ──────────────
es_df <- tibble(
  date_time = df$date_time[(TRAIN_N + VALID_H + 1):nrow(df)],
  y_test    = y_test_es,
  y_pred    = y_pred_es
)

p_es <- ggplot(es_df, aes(x = date_time)) +
  geom_line(aes(y = y_test, colour = "y_test")) +
  geom_line(aes(y = y_pred, colour = "y_pred")) +
  scale_colour_manual(values = c(y_test = "#333333", y_pred = "#e34a33")) +
  labs(title = "ExponentialSmoothing, strategy = last",
       x = NULL, y = "Traffic Volume (vehicles / hr)", colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(size = 20, colour = "black"))

print(p_es)

ggsave(file.path(output_dir, "exp_smoothing_forecast_vs_actual.png"),
       p_es, width = 14, height = 5, dpi = 150)
cat("Saved -> output/models/exp_smoothing_forecast_vs_actual.png\n")


# ── 8. Score row (mirrors your `scoring` DataFrame pattern) ────
# pandas' .append() is deprecated anyway -- dplyr::bind_rows is the
# direct replacement, not just an R translation quirk.
if (!exists("scoring")) {
  scoring <- tibble(Test = character(), RMSE = double(), MAE = double(),
                    MAPE = double(), R2_score = double())
}

new_row_es <- tibble(
  Test     = "Exponent_smooth",
  RMSE     = round(sqrt(mean((y_test_es - y_pred_es)^2)), 2),
  MAE      = round(mean(abs(y_test_es - y_pred_es)), 2),
  MAPE     = round(mean(abs((y_test_es - y_pred_es) / y_test_es)), 4),
  R2_score = round(1 - sum((y_test_es - y_pred_es)^2) /
                     sum((y_test_es - mean(y_test_es))^2), 4)
)

scoring <- bind_rows(scoring, new_row_es)
print(scoring %>% arrange(desc(R2_score)))


# ── 9. Save Model Summary JSON ──────────────────────────────
# sMAPE alongside MAPE, since MAPE alone can be dominated by a
# handful of low-traffic (near-zero) hours -- see the diagnostic
# notes above. Bounded 0-200%, doesn't blow up the same way.
smape_es <- mean(
  2 * abs(y_test_es - y_pred_es) / (abs(y_test_es) + abs(y_pred_es) + 1e-8)
) * 100

summary_list <- list(
  model = list(
    method   = "HoltWinters",
    trend    = FALSE,
    seasonal = "additive",
    period   = WEEKLY_PERIOD,
    alpha    = fit_es$alpha,
    beta     = fit_es$beta,
    gamma    = fit_es$gamma,
    SSE      = fit_es$SSE
  ),
  split = list(
    train_n    = TRAIN_N,
    valid_h    = VALID_H,
    test_h     = TEST_H,
    train_pct  = TRAIN_PCT,
    valid_pct  = VALID_PCT,
    test_pct   = round(TEST_H / nrow(df), 4),
    weekly_cycles_in_test = round(TEST_H / WEEKLY_PERIOD, 2)
  ),
  accuracy = list(
    RMSE  = new_row_es$RMSE,
    MAE   = new_row_es$MAE,
    MAPE  = new_row_es$MAPE,
    sMAPE = round(smape_es, 4),
    R2_score = new_row_es$R2_score
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "exp_smoothing_summary.json"))
cat("Saved -> output/models/exp_smoothing_summary.json\n")