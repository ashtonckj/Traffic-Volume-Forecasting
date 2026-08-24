# ============================================================
# Figures for documentation: Metro Interstate Traffic Volume
# Reads the PROCESSED csv (output of preprocessing.R)
# ============================================================

# install.packages(c("ggplot2", "dplyr"))
library(ggplot2)
library(dplyr)

processed_path <- "data/processed/traffic_volume_processed.csv"  # <-- EDIT if needed
output_dir <- "output"
if (!dir.exists(output_dir)) dir.create(output_dir)

df <- read.csv(processed_path, stringsAsFactors = FALSE)
df$date_time <- as.POSIXct(
  df$date_time,
  format = "%Y-%m-%d %H:%M:%S"
)
df$day_of_week <- weekdays(df$date_time)
df$day_of_week <- factor(
  df$day_of_week,
  levels = c(
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
  )
)

df$holiday_label <- ifelse(df$is_holiday == 1, "Holiday", "No Holiday")

# ---- 1. Histogram of traffic volume (shows multimodal daily pattern) ----
p1 <- ggplot(df, aes(x = traffic_volume)) +
  geom_histogram(binwidth = 200, fill = "#2166AC", color = "white") +
  labs(title = "Distribution of Hourly Traffic Volume",
       subtitle = "Multimodal shape reflects overnight lull vs. AM/PM commute peaks",
       x = "Traffic Volume (vehicles/hr)", y = "Count") +
  theme_minimal()
print(p1)
ggsave(file.path(output_dir, "hist_traffic_volume.png"), p1, width = 7, height = 5)

# ---- 2. Boxplot of traffic volume by day of week ----
p2 <- ggplot(df, aes(x = day_of_week, y = traffic_volume, fill = day_of_week)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Day of Week",
       subtitle = "Weekday volumes cluster higher and tighter than weekends",
       x = "", y = "Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))
print(p2)
ggsave(file.path(output_dir, "boxplot_day_of_week.png"), p2, width = 7, height = 5)

# ---- 3. Boxplot of traffic volume by holiday status ----
p3 <- ggplot(df, aes(x = holiday_label, y = traffic_volume, fill = holiday_label)) +
  geom_boxplot() +
  labs(title = "Traffic Volume: Holiday vs. Non-Holiday",
       x = "", y = "Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(legend.position = "none")
print(p3)
ggsave(file.path(output_dir, "boxplot_holiday.png"), p3, width = 6, height = 5)

# ---- 4. Average traffic volume by hour of day ----
hourly_avg <- df %>%
  group_by(hour) %>%
  summarise(avg_traffic = mean(traffic_volume, na.rm = TRUE), .groups = "drop")

p4 <- ggplot(hourly_avg, aes(x = hour, y = avg_traffic)) +
  geom_line(color = "#2166AC", linewidth = 1) +
  geom_point(color = "#2166AC") +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Average Traffic Volume by Hour of Day",
       subtitle = "Morning peak ~6-7am, afternoon peak ~4-5pm",
       x = "Hour (24h, local CST)", y = "Average Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, size = 8))
print(p4)
ggsave(file.path(output_dir, "line_avg_traffic_by_hour.png"), p4, width = 8, height = 5)

cat("\nAll figures saved to '", output_dir, "/':\n", sep = "")
cat(" - hist_traffic_volume.png\n")
cat(" - boxplot_day_of_week.png\n")
cat(" - boxplot_holiday.png\n")
cat(" - line_avg_traffic_by_hour.png\n")