library(dplyr)
library(zoo)

# ---- 0. Paths ----
raw_path <- "data/raw/Metro_Interstate_Traffic_Volume.csv"
output_dir <- "data/processed"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ---- 1. Read CSV and format dates ----
# ---- 1. Read CSV and format dates ----
df <- read.csv(raw_path, stringsAsFactors = FALSE)
names(df) <- trimws(names(df))
df$holiday <- tolower(trimws(df$holiday))
df$holiday <- tolower(trimws(df$holiday))
df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

cat("Raw rows read:", nrow(df), "\n")

# ---- 2. Remove fully duplicated rows ----
n_rows_before <- nrow(df)
n_rows_before <- nrow(df)
df <- distinct(df)
cat("Removed", n_rows_before - nrow(df), "duplicate rows.\n")
cat("Removed", n_rows_before - nrow(df), "duplicate rows.\n")

# ---- 3. Collapse remaining duplicate timestamps ----
# Some hours have several rows (multiple simultaneous weather readings).
# holiday and traffic_volume are identical within a timestamp; temp, rain, snow, and clouds are averaged.
n_rows_before <- nrow(df)
# Some hours have several rows (multiple simultaneous weather readings).
# holiday and traffic_volume are identical within a timestamp; temp, rain, snow, and clouds are averaged.
n_rows_before <- nrow(df)
df <- df %>%
  group_by(date_time) %>%
  summarise(
    holiday        = first(holiday),
    temp           = round(mean(temp, na.rm = TRUE), 2),
    rain_1h        = mean(rain_1h, na.rm = TRUE),
    snow_1h        = mean(snow_1h, na.rm = TRUE),
    clouds_all     = mean(clouds_all, na.rm = TRUE),
    traffic_volume = first(traffic_volume),
    .groups = "drop"
  ) %>%
  arrange(date_time)
cat("Collapsed", n_rows_before - nrow(df), "duplicate-timestamp rows into", nrow(df), "unique hours.\n")
cat("Collapsed", n_rows_before - nrow(df), "duplicate-timestamp rows into", nrow(df), "unique hours.\n")

# ---- 4. Fix and normalize known mislabeled holiday date ----
# "christmas day" is mislabeled on 2016-12-26 in the raw data; it belongs on 12-25.
# ---- 4. Fix and normalize known mislabeled holiday date ----
# "christmas day" is mislabeled on 2016-12-26 in the raw data; it belongs on 12-25.
df$holiday <- tolower(df$holiday)
df$holiday[df$holiday == "christmas day" & as.Date(df$date_time) == as.Date("2016-12-26")] <- "none"
df$holiday[df$date_time == as.POSIXct("2016-12-25 00:00:00", tz = "UTC")] <- "christmas day"
df$holiday[df$holiday == "christmas day" & as.Date(df$date_time) == as.Date("2016-12-26")] <- "none"
df$holiday[df$date_time == as.POSIXct("2016-12-25 00:00:00", tz = "UTC")] <- "christmas day"

# ---- 5. Fix "holiday labeled only on first hour" bug ----
holiday_dates <- unique(as.Date(df$date_time[df$holiday != "none"]))
df$is_holiday <- as.integer(as.Date(df$date_time) %in% holiday_dates)
cat("Holiday days:", length(holiday_dates), "->", sum(df$is_holiday), "holiday hours.\n")
# ---- 5. Fix "holiday labeled only on first hour" bug ----
holiday_dates <- unique(as.Date(df$date_time[df$holiday != "none"]))
df$is_holiday <- as.integer(as.Date(df$date_time) %in% holiday_dates)
cat("Holiday days:", length(holiday_dates), "->", sum(df$is_holiday), "holiday hours.\n")

# ---- 6. Binary holiday feature ----
# ---- 6. Binary holiday feature ----
df$is_holiday <- ifelse(df$holiday != "none", 1L, 0L)
df$holiday <- NULL

# ---- 6b. Outlier Detection Visualization ----
# Create output directory for figures
fig_dir <- "output/figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# Export a 4-panel diagnostic grid showing physical anomalies before cleaning
png(file.path(fig_dir, "outlier_detection.png"), width = 1200, height = 800, res = 130)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# 1. Temperature 0 K anomaly plot
plot(df$date_time, df$temp, type = "l", col = "#2c7fb8",
     main = "Temperature (0 K Measurement Error)", xlab = "Date", ylab = "Temperature (Kelvin)")
points(df$date_time[df$temp <= 0], df$temp[df$temp <= 0], col = "red", pch = 19, cex = 1.5)
legend("bottomleft", legend = "Physical Error (0 K)", col = "red", pch = 19, bty = "n")

# 2. Rainfall extreme spike plot
plot(df$date_time, df$rain_1h, type = "l", col = "#2c7fb8",
     main = "Rainfall Extreme Sensor Spike", xlab = "Date", ylab = "Rain (mm/h)")
points(df$date_time[df$rain_1h > 300], df$rain_1h[df$rain_1h > 300], col = "red", pch = 19, cex = 1.5)
legend("topright", legend = "Sensor Failure (>300 mm/h)", col = "red", pch = 19, bty = "n")

# 3. Traffic Volume Distribution (Boxplot check)
boxplot(df$traffic_volume, main = "Traffic Volume Distribution",
        ylab = "Vehicles / Hour", col = "#756bb1")

# 4. Raw Traffic Volume Time Series
plot(df$date_time, df$traffic_volume, type = "l", col = "#333333",
     main = "Raw Traffic Volume Series", xlab = "Date", ylab = "Vehicles / Hour")

dev.off()
cat("Saved outlier diagnostic plot ->", file.path(fig_dir, "outlier_detection.png"), "\n")

# ---- 7. Fix physically impossible values via linear interpolation ----
# temp of 0 K, and one 9831 mm/h rain reading
# (next-highest valid value is 55.63, so 300 isolates just that one).
n_bad <- sum(df$temp <= 0) + sum(df$rain_1h > 300)
df$temp[df$temp <= 0] <- NA
df$rain_1h[df$rain_1h > 300] <- NA
df <- df %>% arrange(date_time)
df$temp <- na.approx(df$temp, x = as.numeric(df$date_time), rule = 2)
df$rain_1h <- na.approx(df$rain_1h, x = as.numeric(df$date_time), rule = 2)
cat("Interpolated", n_bad, "physically impossible weather value(s).\n")
cat("Interpolated", n_bad, "physically impossible weather value(s).\n")

# ---- 8. Restrict to a clean, continuous date range ----
# ---- 8. Restrict to a clean, continuous date range ----
CUTOFF_DATE <- as.POSIXct("2015-10-01 00:00:00", tz = "UTC")
n_rows_before <- nrow(df)
n_rows_before <- nrow(df)
df <- df %>% filter(date_time >= CUTOFF_DATE)
cat("Dropped", n_rows_before - nrow(df), "rows before", format(CUTOFF_DATE), "|", nrow(df), "rows remain.\n")
cat("Dropped", n_rows_before - nrow(df), "rows before", format(CUTOFF_DATE), "|", nrow(df), "rows remain.\n")

# ---- 9. Fill remaining internal hourly gaps ----
# traffic_volume:  <= 2 h  linear interpolation
#                  3-24 h  same hour one week earlier (t-168)
#                  > 24 h  average of the same hour-of-week across +/- 4 weeks
# weather:         linear interpolation regardless of gap length
WEATHER <- c("temp", "rain_1h", "snow_1h", "clouds_all")
SHORT_MAX <- 2
LONG_MAX <- 24
grid <- data.frame(date_time = seq(min(df$date_time), max(df$date_time), by = "hour"))
df <- grid %>%
  left_join(df %>% select(date_time, all_of(WEATHER), traffic_volume), by = "date_time") %>%
# ---- 9. Fill remaining internal hourly gaps ----
# traffic_volume:  <= 2 h  linear interpolation
#                  3-24 h  same hour one week earlier (t-168)
#                  > 24 h  average of the same hour-of-week across +/- 4 weeks
# weather:         linear interpolation regardless of gap length
WEATHER <- c("temp", "rain_1h", "snow_1h", "clouds_all")
SHORT_MAX <- 2
LONG_MAX <- 24
grid <- data.frame(date_time = seq(min(df$date_time), max(df$date_time), by = "hour"))
df <- grid %>%
  left_join(df %>% select(date_time, all_of(WEATHER), traffic_volume), by = "date_time") %>%
  arrange(date_time)

df$is_holiday <- as.integer(as.Date(df$date_time) %in% holiday_dates)
df$is_holiday <- as.integer(as.Date(df$date_time) %in% holiday_dates)

run <- rle(is.na(df$traffic_volume))
df$gap_length <- ifelse(is.na(df$traffic_volume), rep(run$lengths, run$lengths), 0)

for (c in WEATHER)
  df[[c]] <- round(na.approx(df[[c]], x = as.numeric(df$date_time), rule = 2), 2)

df$gap_length <- ifelse(is.na(df$traffic_volume), rep(run$lengths, run$lengths), 0)

for (c in WEATHER)
  df[[c]] <- round(na.approx(df[[c]], x = as.numeric(df$date_time), rule = 2), 2)

tv <- df$traffic_volume
med <- df$gap_length > SHORT_MAX & df$gap_length <= LONG_MAX
tv[med] <- lag(tv, 168)[med]
tv <- na.approx(tv, x = as.numeric(df$date_time), na.rm = FALSE)
tv[df$gap_length > LONG_MAX] <- NA

long_idx <- which(df$gap_length > LONG_MAX)
seasonal_avg <- function(i, n_weeks = 4) {
  j <- i + c(-n_weeks:-1, 1:n_weeks) * 168L
  j <- j[j >= 1 & j <= length(tv)]
  v <- tv[j][!is.na(tv[j])]
  if (length(v) == 0) NA_real_ else mean(v)
}
tv[long_idx] <- sapply(long_idx, seasonal_avg)
df$traffic_volume <- round(tv)

n_imputed <- sum(df$gap_length > 0)
cat(sprintf("Gap fill: %d short, %d medium, %d long | %d imputed hours (%.2f%%)\n",
            sum(df$gap_length > 0 & df$gap_length <= SHORT_MAX),
            sum(med), length(long_idx), n_imputed, 100 * n_imputed / nrow(df)))
stopifnot("unresolved NA remains" = !anyNA(df[c(WEATHER, "traffic_volume")]))

# ---- 10. Chronological 70 / 15 / 15 split ----
TRAIN_FRAC <- 0.70
VAL_FRAC <- 0.15
n <- nrow(df)
train_n <- floor(TRAIN_FRAC * n)
val_n <- floor(VAL_FRAC * n)

df$split <- c(rep("train", train_n),
              rep("validation", val_n),
              rep("test", n - train_n - val_n))

for (s in c("train", "validation", "test")) {
  i <- which(df$split == s)
  cat(sprintf("%-10s rows %6d - %6d  (%s to %s)  %.1f%%\n", s, min(i), max(i),
              as.character(df$date_time[min(i)]), as.character(df$date_time[max(i)]),
              100 * length(i) / n))
}

# Imputed hours in validation or test are values the pipeline generated, so a
# seasonal-naive model would partly be scored against its own output there.
for (s in c("validation", "test")) {
  k <- sum(df$gap_length[df$split == s] > 0)
  if (k > 0) cat("WARNING:", k, "imputed hour(s) in", s,
                 "- keep an is_imputed column and exclude them when scoring.\n")
}

# ---- 11. Save processed dataset ----
out <- df %>%
  transmute(date_time = format(date_time, "%Y-%m-%d %H:%M:%S"),
            temp, rain_1h, snow_1h, clouds_all, is_holiday, traffic_volume, split)

write.csv(out, file.path(output_dir, "traffic_volume_processed.csv"), row.names = FALSE)
cat("\nSaved:", file.path(output_dir, "traffic_volume_processed.csv"), "\n")
cat("Columns:", paste(names(out), collapse = ", "), "| Rows:", nrow(out), "\n")
med <- df$gap_length > SHORT_MAX & df$gap_length <= LONG_MAX
tv[med] <- lag(tv, 168)[med]
tv <- na.approx(tv, x = as.numeric(df$date_time), na.rm = FALSE)
tv[df$gap_length > LONG_MAX] <- NA

long_idx <- which(df$gap_length > LONG_MAX)
seasonal_avg <- function(i, n_weeks = 4) {
  j <- i + c(-n_weeks:-1, 1:n_weeks) * 168L
  j <- j[j >= 1 & j <= length(tv)]
  v <- tv[j][!is.na(tv[j])]
  if (length(v) == 0) NA_real_ else mean(v)
}
tv[long_idx] <- sapply(long_idx, seasonal_avg)
df$traffic_volume <- round(tv)

n_imputed <- sum(df$gap_length > 0)
cat(sprintf("Gap fill: %d short, %d medium, %d long | %d imputed hours (%.2f%%)\n",
            sum(df$gap_length > 0 & df$gap_length <= SHORT_MAX),
            sum(med), length(long_idx), n_imputed, 100 * n_imputed / nrow(df)))
stopifnot("unresolved NA remains" = !anyNA(df[c(WEATHER, "traffic_volume")]))

# ---- 10. Chronological 70 / 15 / 15 split ----
TRAIN_FRAC <- 0.70
VAL_FRAC <- 0.15
n <- nrow(df)
train_n <- floor(TRAIN_FRAC * n)
val_n <- floor(VAL_FRAC * n)

df$split <- c(rep("train", train_n),
              rep("validation", val_n),
              rep("test", n - train_n - val_n))

for (s in c("train", "validation", "test")) {
  i <- which(df$split == s)
  cat(sprintf("%-10s rows %6d - %6d  (%s to %s)  %.1f%%\n", s, min(i), max(i),
              as.character(df$date_time[min(i)]), as.character(df$date_time[max(i)]),
              100 * length(i) / n))
}

# Imputed hours in validation or test are values the pipeline generated, so a
# seasonal-naive model would partly be scored against its own output there.
for (s in c("validation", "test")) {
  k <- sum(df$gap_length[df$split == s] > 0)
  if (k > 0) cat("WARNING:", k, "imputed hour(s) in", s,
                 "- keep an is_imputed column and exclude them when scoring.\n")
}

# ---- 11. Save processed dataset ----
out <- df %>%
  transmute(date_time = format(date_time, "%Y-%m-%d %H:%M:%S"),
            temp, rain_1h, snow_1h, clouds_all, is_holiday, traffic_volume, split)

write.csv(out, file.path(output_dir, "traffic_volume_processed.csv"), row.names = FALSE)
cat("\nSaved:", file.path(output_dir, "traffic_volume_processed.csv"), "\n")
cat("Columns:", paste(names(out), collapse = ", "), "| Rows:", nrow(out), "\n")