# ============================================================
# Figures for documentation: Metro Interstate Traffic Volume
# Reads the PROCESSED csv (output of preprocessing.R)
# ============================================================

library(ggplot2)
library(dplyr)
library(reshape2)

processed_path <- "data/processed/traffic_volume_processed.csv"
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
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
  )
)

df$holiday_label <- ifelse(df$is_holiday == 1, "Holiday", "No Holiday")


# ---- 1. Histogram of traffic volume (shows multimodal daily pattern) ----
p1 <- ggplot(df, aes(x = traffic_volume)) +
  geom_histogram(binwidth = 200, fill = "#2166AC", color = "white") +
  labs(title = "Distribution of Hourly Traffic Volume",
       x = "Traffic Volume (vehicles/hr)", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
print(p1)
ggsave(file.path(output_dir, "hist_traffic_volume.png"), p1, width = 7, height = 5, dpi = 300)


# ---- 2. Boxplot of traffic volume by day of week ----
p2 <- ggplot(df, aes(x = day_of_week, y = traffic_volume, fill = day_of_week)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Day of Week",
       x = "", y = "Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(hjust = 0.5))
print(p2)
ggsave(file.path(output_dir, "boxplot_day_of_week.png"), p2, width = 7, height = 3, dpi = 300)


# ---- 3. Boxplot of traffic volume by holiday status ----
p3 <- ggplot(df, aes(x = holiday_label, y = traffic_volume, fill = holiday_label)) +
  geom_boxplot() +
  labs(title = "Traffic Volume: Holiday vs. Non-Holiday",
       x = "", y = "Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))
print(p3)
ggsave(file.path(output_dir, "boxplot_holiday.png"), p3, width = 7, height = 3, dpi = 300)


# ---- 4. Average traffic volume by hour of day ----
hourly_avg <- df %>%
  group_by(hour) %>%
  summarise(avg_traffic = mean(traffic_volume, na.rm = TRUE), .groups = "drop")

p4 <- ggplot(hourly_avg, aes(x = hour, y = avg_traffic)) +
  geom_line(color = "#2166AC", linewidth = 1) +
  geom_point(color = "#2166AC") +
  scale_x_continuous(breaks = 0:23) +
  labs(title = "Average Traffic Volume by Hour of Day",
       x = "Hour (24h, local CST)", y = "Average Traffic Volume (vehicles/hr)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, size = 8), plot.title = element_text(hjust = 0.5))
print(p4)
ggsave(file.path(output_dir, "line_avg_traffic_by_hour.png"), p4, width = 7, height = 3, dpi = 300)


# ---- 5. Correlation heatmap (numeric variables only) ----
# Pulls out the numeric columns and computes pairwise correlation.
numeric_df <- df %>% select(temp, rain_1h, snow_1h, clouds_all, traffic_volume)

cor_matrix <- cor(numeric_df, use = "complete.obs")

# Melt into long format for ggplot
cor_melted <- melt(cor_matrix)

p7 <- ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 4) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limit = c(-1, 1),
                       name = "Correlation") +
  labs(title = "Correlation Heatmap of Numeric Variables",
       x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(hjust = 0.5))

print(p7)
ggsave(file.path(output_dir, "correlation_heatmap.png"), p7, width = 6, height = 3.6, dpi = 300)

cat("\nAll figures saved to '", output_dir, "/':\n", sep = "")
cat(" - hist_traffic_volume.png\n")
cat(" - boxplot_day_of_week.png\n")
cat(" - boxplot_holiday.png\n")
cat(" - line_avg_traffic_by_hour.png\n")
cat(" - correlation_heatmap.png\n")