# ============================================================
#  Metro Interstate Traffic Volume — Statistical Analysis
#  Input : data/processed/traffic_volume_processed.csv
#  Output: output/  (plots + diagnostics + summary)
# ============================================================
#  Columns in processed dataset:
#    date_time, temp, rain_1h, snow_1h, clouds_all,
#    is_holiday, hour, month, day_of_week, traffic_volume
# ============================================================

# ── 0. Packages ──────────────────────────────────────────────
pkgs <- c("dplyr", "lubridate", "forecast", "tseries",
          "ggplot2", "zoo", "moments", "scales", "gridExtra",
          "jsonlite")
new  <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))
cat("Packages loaded.\n")


# ── 1. Paths ─────────────────────────────────────────────────
input_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "output"
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
cat("Columns:", paste(names(df), collapse = ", "), "\n")


# ── 3. Derive Additional Features ────────────────────────────
# day_of_week is already in the processed data as a character
# label. Re-derive numeric versions and is_weekend here so
# all EDA plots have consistent grouping variables.
df <- df %>%
  mutate(
    year       = year(date_time),
    month_lbl  = month(date_time, label = TRUE, abbr = TRUE),
    dow_lbl    = wday(date_time, label = TRUE, abbr = TRUE),
    dow_num    = wday(date_time),               # 1 = Sun … 7 = Sat
    is_weekend = ifelse(dow_num %in% c(1, 7), 1L, 0L),
    day_type   = ifelse(is_weekend == 1, "Weekend", "Weekday"),
    temp_C     = temp - 273.15                  # Kelvin -> Celsius
  )

cat("Feature derivation complete.\n")


# ── 4. Descriptive Statistics ─────────────────────────────────
cat("\n=============================================\n")
cat("  DESCRIPTIVE STATISTICS — traffic_volume\n")
cat("=============================================\n")

tv <- df$traffic_volume

stats_tbl <- data.frame(
  Statistic = c(
    "N", "Mean", "Median", "Std Dev",
    "Min", "Max", "Q1 (25%)", "Q3 (75%)",
    "IQR", "Skewness", "Kurtosis", "Missing (NAs)"
  ),
  Value = c(
    length(tv),
    round(mean(tv,                    na.rm = TRUE), 2),
    round(median(tv,                  na.rm = TRUE), 2),
    round(sd(tv,                      na.rm = TRUE), 2),
    round(min(tv,                     na.rm = TRUE), 2),
    round(max(tv,                     na.rm = TRUE), 2),
    round(quantile(tv, 0.25,          na.rm = TRUE), 2),
    round(quantile(tv, 0.75,          na.rm = TRUE), 2),
    round(IQR(tv,                     na.rm = TRUE), 2),
    round(skewness(tv,                na.rm = TRUE), 4),
    round(kurtosis(tv,                na.rm = TRUE), 4),
    sum(is.na(tv))
  )
)

print(stats_tbl, row.names = FALSE)

# Save to JSON for downstream use
write(
  toJSON(setNames(as.list(stats_tbl$Value), stats_tbl$Statistic),
         pretty = TRUE, auto_unbox = TRUE),
  file.path(output_dir, "descriptive_stats.json")
)
cat("Saved -> output/descriptive_stats.json\n")


# ── 5. Stationarity Tests ────────────────────────────────────
cat("\n=============================================\n")
cat("  STATIONARITY TESTS\n")
cat("=============================================\n")

# Run on the full series — 26k observations is manageable for
# ADF and KPSS; PP uses the full series too.
tv_clean <- na.omit(tv)

adf_res  <- adf.test(tv_clean, alternative = "stationary")
kpss_res <- kpss.test(tv_clean)
pp_res   <- pp.test(tv_clean)

cat(sprintf("\nADF  : stat = %8.4f,  p = %.4f  (%s)\n",
            adf_res$statistic,  adf_res$p.value,
            ifelse(adf_res$p.value  < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS : stat = %8.4f,  p = %.4f  (%s)\n",
            kpss_res$statistic, kpss_res$p.value,
            ifelse(kpss_res$p.value > 0.05, "stationary", "non-stationary")))
cat(sprintf("PP   : stat = %8.4f,  p = %.4f  (%s)\n",
            pp_res$statistic,   pp_res$p.value,
            ifelse(pp_res$p.value   < 0.05, "stationary", "non-stationary")))

ts_for_diff <- ts(tv_clean, frequency = 24)
d_val <- ndiffs(ts_for_diff)
D_val <- nsdiffs(ts_for_diff)
cat(sprintf("ndiffs (d) : %d\n", d_val))
cat(sprintf("nsdiffs (D): %d  (s = 24)\n", D_val))

# Save stationarity results
stat_list <- list(
  ADF  = list(statistic = round(adf_res$statistic,  4),
              p_value   = round(adf_res$p.value,    4)),
  KPSS = list(statistic = round(kpss_res$statistic, 4),
              p_value   = round(kpss_res$p.value,   4)),
  PP   = list(statistic = round(pp_res$statistic,   4),
              p_value   = round(pp_res$p.value,     4)),
  ndiffs  = d_val,
  nsdiffs = D_val
)
write(toJSON(stat_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "stationarity_tests.json"))
cat("Saved -> output/stationarity_tests.json\n")


# ── 6. ACF / PACF ────────────────────────────────────────────
# Use a 5,000-observation window so the plots are readable;
# the full 26k series produces an identical picture but slower.
tv_sub <- tv_clean[1:min(5000, length(tv_clean))]

png(file.path(output_dir, "acf_pacf_original.png"),
    width = 1200, height = 500, res = 130)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
acf(tv_sub,  lag.max = 72,
    main = "ACF — Traffic Volume (original)")
pacf(tv_sub, lag.max = 72,
     main = "PACF — Traffic Volume (original)")
dev.off()
cat("Saved -> output/acf_pacf_original.png\n")

# Differenced series ACF/PACF (d=1 for illustration even if d=0
# is suggested — useful to confirm the series is already stationary)
tv_diff <- diff(tv_sub, differences = 1)

png(file.path(output_dir, "acf_pacf_differenced.png"),
    width = 1200, height = 500, res = 130)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
acf(tv_diff,  lag.max = 72,
    main = "ACF — Traffic Volume (1st difference)")
pacf(tv_diff, lag.max = 72,
     main = "PACF — Traffic Volume (1st difference)")
dev.off()
cat("Saved -> output/acf_pacf_differenced.png\n")


# ── 7. STL Decomposition ─────────────────────────────────────
# Use 10 weeks of data so the plot is readable (s=24 daily cycle).
stl_n   <- min(24 * 7 * 10, nrow(df))
ts_stl  <- ts(df$traffic_volume[1:stl_n], frequency = 24)
stl_fit <- stl(ts_stl, s.window = "periodic")

png(file.path(output_dir, "stl_decomposition.png"),
    width = 1100, height = 750, res = 130)
plot(stl_fit,
     main = "STL Decomposition — Traffic Volume (s=24, first 10 weeks)")
dev.off()
cat("Saved -> output/stl_decomposition.png\n")


# ── 8. EDA Plots ─────────────────────────────────────────────
cat("\nGenerating EDA plots ...\n")

# 8a. Full time-series line
p1 <- ggplot(df, aes(date_time, traffic_volume)) +
  geom_line(colour = "#2c7fb8", linewidth = 0.25, alpha = 0.7) +
  labs(title = "Hourly Traffic Volume — I-94 Westbound (Oct 2015 – Sep 2018)",
       x = "Date", y = "Traffic Volume (vehicles / hr)") +
  scale_x_datetime(labels = date_format("%b %Y"),
                   date_breaks = "6 months") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# 8b. Average by hour of day
p2 <- df %>%
  group_by(hour) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(hour, avg_vol)) +
  geom_line(colour = "#e34a33", linewidth = 1.1) +
  geom_point(colour = "#e34a33", size = 2) +
  scale_x_continuous(breaks = seq(0, 23, 2)) +
  labs(title = "Average Traffic by Hour of Day",
       x = "Hour (0–23)", y = "Mean Volume") +
  theme_minimal(base_size = 10)

# 8c. Average by day of week (ordered Mon–Sun)
dow_order <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
p3 <- df %>%
  group_by(dow_lbl) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  mutate(dow_lbl = factor(dow_lbl, levels = dow_order)) %>%
  ggplot(aes(dow_lbl, avg_vol, fill = dow_lbl)) +
  geom_col(show.legend = FALSE, alpha = 0.85) +
  labs(title = "Average Traffic by Day of Week",
       x = "Day", y = "Mean Volume") +
  theme_minimal(base_size = 10)

# 8d. Average by month
p4 <- df %>%
  group_by(month_lbl) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  mutate(month_lbl = factor(month_lbl,
                            levels = month.abb)) %>%
  ggplot(aes(month_lbl, avg_vol, group = 1)) +
  geom_line(colour = "#31a354", linewidth = 1.1) +
  geom_point(colour = "#31a354", size = 2.5) +
  labs(title = "Average Traffic by Month",
       x = "Month", y = "Mean Volume") +
  theme_minimal(base_size = 10)

# 8e. Distribution histogram
p5 <- ggplot(df, aes(traffic_volume)) +
  geom_histogram(bins = 60, fill = "#756bb1",
                 colour = "white", alpha = 0.85) +
  labs(title = "Distribution of Hourly Traffic Volume",
       x = "Traffic Volume (vehicles / hr)", y = "Count") +
  theme_minimal(base_size = 10)

# 8f. Weekday vs Weekend hourly profile
p6 <- df %>%
  group_by(hour, day_type) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(hour, avg_vol, colour = day_type)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = c("Weekday" = "#2c7fb8",
                                 "Weekend" = "#e34a33")) +
  scale_x_continuous(breaks = seq(0, 23, 2)) +
  labs(title = "Weekday vs Weekend Hourly Profile",
       x = "Hour (0–23)", y = "Mean Volume", colour = NULL) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom")

# 8g. Holiday vs Non-holiday hourly profile
p7 <- df %>%
  mutate(holiday_type = ifelse(is_holiday == 1, "Holiday", "Non-holiday")) %>%
  group_by(hour, holiday_type) %>%
  summarise(avg_vol = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(hour, avg_vol, colour = holiday_type)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = c("Holiday"     = "#e34a33",
                                 "Non-holiday" = "#2c7fb8")) +
  scale_x_continuous(breaks = seq(0, 23, 2)) +
  labs(title = "Holiday vs Non-Holiday Hourly Profile",
       x = "Hour (0–23)", y = "Mean Volume", colour = NULL) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom")

# 8h. Temperature vs traffic scatter (sample 3,000 for speed)
set.seed(42)
df_samp <- df %>% slice_sample(n = min(3000, nrow(df)))

p8 <- ggplot(df_samp, aes(temp_C, traffic_volume)) +
  geom_point(alpha = 0.25, colour = "#2c7fb8", size = 0.8) +
  geom_smooth(method = "loess", colour = "#e34a33",
              se = FALSE, linewidth = 1) +
  labs(title = "Temperature vs Traffic Volume",
       x = "Temperature (°C)", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 10)

# Save 6-panel grid (core EDA)
grid_core <- gridExtra::grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 2)
ggsave(file.path(output_dir, "eda_plots_core.png"),
       grid_core, width = 14, height = 18, dpi = 150)
cat("Saved -> output/eda_plots_core.png\n")

# Save supplementary panel (holiday + temperature)
grid_supp <- gridExtra::grid.arrange(p7, p8, ncol = 2)
ggsave(file.path(output_dir, "eda_plots_supplementary.png"),
       grid_supp, width = 14, height = 6, dpi = 150)
cat("Saved -> output/eda_plots_supplementary.png\n")


# ── 9. Boxplots ───────────────────────────────────────────────
# 9a. Traffic by day of week
bp1 <- df %>%
  mutate(dow_lbl = factor(dow_lbl, levels = dow_order)) %>%
  ggplot(aes(dow_lbl, traffic_volume, fill = dow_lbl)) +
  geom_boxplot(show.legend = FALSE, outlier.size = 0.4,
               outlier.alpha = 0.3, alpha = 0.8) +
  labs(title = "Traffic Volume by Day of Week",
       x = "Day", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "boxplot_day_of_week.png"),
       bp1, width = 10, height = 5, dpi = 150)
cat("Saved -> output/boxplot_day_of_week.png\n")

# 9b. Traffic by holiday flag
bp2 <- df %>%
  mutate(Holiday = ifelse(is_holiday == 1, "Holiday", "Non-holiday")) %>%
  ggplot(aes(Holiday, traffic_volume, fill = Holiday)) +
  geom_boxplot(show.legend = FALSE, outlier.size = 0.5,
               outlier.alpha = 0.3, alpha = 0.8) +
  scale_fill_manual(values = c("Holiday"     = "#e34a33",
                               "Non-holiday" = "#2c7fb8")) +
  labs(title = "Traffic Volume: Holiday vs Non-Holiday",
       x = NULL, y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "boxplot_holiday.png"),
       bp2, width = 7, height = 5, dpi = 150)
cat("Saved -> output/boxplot_holiday.png\n")

# 9c. Traffic by month (seasonal pattern)
bp3 <- df %>%
  mutate(month_lbl = factor(month_lbl, levels = month.abb)) %>%
  ggplot(aes(month_lbl, traffic_volume, fill = month_lbl)) +
  geom_boxplot(show.legend = FALSE, outlier.size = 0.4,
               outlier.alpha = 0.3, alpha = 0.8) +
  labs(title = "Traffic Volume by Month",
       x = "Month", y = "Traffic Volume (vehicles / hr)") +
  theme_minimal(base_size = 11)

ggsave(file.path(output_dir, "boxplot_month.png"),
       bp3, width = 12, height = 5, dpi = 150)
cat("Saved -> output/boxplot_month.png\n")


# ── 10. Correlation of Numeric Variables ─────────────────────
num_cols <- c("traffic_volume", "temp_C", "rain_1h",
              "snow_1h", "clouds_all", "is_holiday", "is_weekend")

cor_mat  <- cor(df[, num_cols], use = "complete.obs")
cor_df   <- as.data.frame(as.table(cor_mat))
names(cor_df) <- c("Var1", "Var2", "Correlation")

p_cor <- ggplot(cor_df, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.2f", Correlation)),
            size = 3.2, colour = "black") +
  scale_fill_gradient2(low  = "#d73027", mid  = "white",
                       high = "#1a9850", midpoint = 0,
                       limits = c(-1, 1)) +
  labs(title = "Correlation Heatmap — Numeric Variables",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

ggsave(file.path(output_dir, "correlation_heatmap.png"),
       p_cor, width = 9, height = 8, dpi = 150)
cat("Saved -> output/correlation_heatmap.png\n")


# ── 11. Summary JSON ─────────────────────────────────────────
summary_list <- list(
  dataset = list(
    rows        = nrow(df),
    columns     = ncol(df),
    date_start  = as.character(min(df$date_time)),
    date_end    = as.character(max(df$date_time)),
    holiday_hrs = sum(df$is_holiday),
    weekend_hrs = sum(df$is_weekend)
  ),
  traffic_volume = list(
    mean     = round(mean(tv,                 na.rm = TRUE), 2),
    median   = round(median(tv,               na.rm = TRUE), 2),
    sd       = round(sd(tv,                   na.rm = TRUE), 2),
    min      = round(min(tv,                  na.rm = TRUE), 2),
    max      = round(max(tv,                  na.rm = TRUE), 2),
    skewness = round(skewness(tv,             na.rm = TRUE), 4),
    kurtosis = round(kurtosis(tv,             na.rm = TRUE), 4)
  ),
  stationarity = list(
    ADF_p  = round(adf_res$p.value,  4),
    KPSS_p = round(kpss_res$p.value, 4),
    PP_p   = round(pp_res$p.value,   4),
    d      = d_val,
    D      = D_val
  )
)

write(toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE),
      file.path(output_dir, "analysis_summary.json"))
cat("Saved -> output/analysis_summary.json\n")


# ── 12. Done ─────────────────────────────────────────────────
cat("\n=============================================\n")
cat("  STATISTICAL ANALYSIS COMPLETE\n")
cat("  Output files:\n")
cat("    output/descriptive_stats.json\n")
cat("    output/stationarity_tests.json\n")
cat("    output/analysis_summary.json\n")
cat("    output/acf_pacf_original.png\n")
cat("    output/acf_pacf_differenced.png\n")
cat("    output/stl_decomposition.png\n")
cat("    output/eda_plots_core.png\n")
cat("    output/eda_plots_supplementary.png\n")
cat("    output/boxplot_day_of_week.png\n")
cat("    output/boxplot_holiday.png\n")
cat("    output/boxplot_month.png\n")
cat("    output/correlation_heatmap.png\n")
cat("=============================================\n")