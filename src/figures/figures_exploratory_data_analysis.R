# ============================================================
# Figures for documentation: Metro Interstate Traffic Volume
# ============================================================

library(ggplot2)
library(dplyr)
library(reshape2)   # melt(), used by the correlation heatmap below
library(forecast)    # ggseasonplot
library(lubridate)   # floor_date, year, month

processed_path <- "data/processed/traffic_volume_processed.csv"
output_dir <- "output/exploratory"
if (!dir.exists(output_dir)) dir.create(output_dir)

df <- read.csv(processed_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

# hour and day_of_week are no longer columns in the processed dataset, because
# the models are restricted to the raw columns. They are derived here purely as
# plot axes -- no model ever sees them. %H and %u are locale-independent, unlike
# weekdays(), which returns translated names on a non-English system.
df$hour <- as.integer(format(df$date_time, "%H"))
df$day_of_week <- weekdays(df$date_time)
df$day_of_week <- factor( df$day_of_week,
  levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
)

df$holiday_label <- ifelse(df$is_holiday == 1, "Holiday", "No Holiday")


# ---- 1. Boxplot of traffic volume by day of week ----
p1 <- ggplot(df, aes(x = day_of_week, y = traffic_volume, fill = day_of_week)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Day of Week",
       x = "", y = "Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(hjust = 0.5))
ggsave(file.path(output_dir, "boxplot_day_of_week.png"), p1, width = 7, height = 3, dpi = 300)


# ---- 2. Boxplot of traffic volume by holiday status ----
p2 <- ggplot(df, aes(x = holiday_label, y = traffic_volume, fill = holiday_label)) +
  geom_boxplot() +
  labs(title = "Traffic Volume: Holiday vs. Non-Holiday",
       x = "", y = "Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))
ggsave(file.path(output_dir, "boxplot_holiday.png"), p2, width = 7, height = 3, dpi = 300)


# ---- 3. Average traffic volume by hour of day ----
hourly_avg <- df %>%
  group_by(hour) %>%
  summarise(avg_traffic = mean(traffic_volume, na.rm = TRUE), .groups = "drop")

p3 <- ggplot(hourly_avg, aes(x = hour, y = avg_traffic)) +
  geom_line(color = "#2166AC", linewidth = 1) +
  geom_point(color = "#2166AC") +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Average Traffic Volume by Hour of Day",
       x = "Hour (24h, local CST)", y = "Average Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, size = 8), plot.title = element_text(hjust = 0.5))
ggsave(file.path(output_dir, "line_avg_traffic_by_hour.png"), p3, width = 7, height = 3, dpi = 300)


# ---- 4. Correlation heatmap (numeric variables only) ----
# Pulls out the numeric columns and computes pairwise correlation.
numeric_df <- df %>% select(temp, rain_1h, snow_1h, clouds_all, traffic_volume)

cor_matrix <- cor(numeric_df, use = "complete.obs")

# Melt into long format for ggplot
cor_melted <- melt(cor_matrix)

p4 <- ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 4) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limit = c(-1, 1),
                       name = "Correlation") +
  labs(title = "Correlation Heatmap of Numeric Variables",
       x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(hjust = 0.5))
ggsave(file.path(output_dir, "correlation_heatmap.png"), p4, width = 6, height = 3.6, dpi = 300)

cat("\nAll figures saved to '", output_dir, "/':\n", sep = "")
cat(" - hist_traffic_volume.png\n")
cat(" - boxplot_day_of_week.png\n")
cat(" - boxplot_holiday.png\n")
cat(" - line_avg_traffic_by_hour.png\n")
cat(" - correlation_heatmap.png\n")

# ---- 6. Hourly profile split by weekday and weekend ----
# Separates the daily cycle by day type, showing that weekday and weekend
# profiles differ substantially and that one weekly seasonal period has to
# absorb both shapes implicitly.
day_profile <- df %>%
  mutate(day_type = ifelse(day_of_week %in% c("Saturday", "Sunday"), "Weekend", "Weekday")) %>%
  group_by(day_type, hour) %>%
  summarise(avg_volume = mean(traffic_volume, na.rm = TRUE), .groups = "drop")

p8 <- ggplot(day_profile, aes(x = hour, y = avg_volume, colour = day_type)) +
  geom_line(linewidth = 1) +
  scale_x_continuous(breaks = seq(0, 23, 2)) +
  labs(title = "Average Hourly Traffic Profile: Weekday vs Weekend",
       x = "Hour of Day", y = "Average Traffic Volume (vehicles/hr)", colour = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5))
ggsave(file.path(output_dir, "hourly_profile_by_day_type.png"), p8, width = 7, height = 3, dpi = 300)


# ---- 7. Monthly-aggregated seasonal plot (annual pattern) ----
# Each line is one calendar year, so overlapping lines show whether the
# annual shape repeats from year to year.
monthly_ts <- df %>%
  mutate(ym = floor_date(date_time, "month")) %>%
  group_by(ym) %>%
  summarise(avg_volume = mean(traffic_volume, na.rm = TRUE), .groups = "drop") %>%
  pull(avg_volume) %>%
  ts(start = c(year(min(df$date_time)), month(min(df$date_time))), frequency = 12)

p9 <- ggseasonplot(monthly_ts, year.labels = TRUE, year.labels.left = TRUE) +
  labs(title = "Monthly-Aggregated Seasonal Plot",
       x = "Month", y = "Average Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
ggsave(file.path(output_dir, "monthly_seasonal_plot.png"), p9, width = 7, height = 3, dpi = 300)


# ---- 8. Summary statistics ----
cat("\n=== Summary Statistics: traffic_volume ===\n")
summary_stats <- df %>%
  summarise(n = n(),
            Min = min(traffic_volume), Median = median(traffic_volume),
            Mean = round(mean(traffic_volume), 1), Max = max(traffic_volume),
            SD = round(sd(traffic_volume), 1))
write.csv(summary_stats, file.path(output_dir, "summary_statistics.csv"), row.names = FALSE)

cat(" - hourly_profile_by_day_type.png\n")
cat(" - monthly_seasonal_plot.png\n")
cat(" - summary_statistics.csv\n")