library(dplyr)
library(lubridate)
library(zoo)

# ---- 0. Paths ----
raw_path <- "data/raw/Metro_Interstate_Traffic_Volume.csv"
output_dir <- "data/processed"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
output_path <- file.path(output_dir, "traffic_volume_processed.csv")


# ---- 1. Read and normalize column names / whitespace ----
# The raw file may have padded/whitespace column names and string values
# (this happens with some re-exports of the dataset) - strip them defensively.
df <- read.csv(raw_path, stringsAsFactors = FALSE)
names(df) <- trimws(names(df))
char_cols <- names(df)[sapply(df, is.character)]
df[char_cols] <- lapply(df[char_cols], trimws)

df$date_time <- as.POSIXct(df$date_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

cat("Raw rows read:", nrow(df), "\n")


# ---- 2. Remove fully duplicated rows ----
n_before <- nrow(df)
df <- distinct(df)
cat("Removed", n_before - nrow(df), "fully duplicated rows.\n")


# ---- 3. Collapse remaining duplicate timestamps ----
# Some hours have multiple rows because OpenWeatherMap logged more than one
# simultaneous weather reading (e.g. "rain" and "mist" for the same hour).
# holiday and traffic_volume are identical within a timestamp group (verified),
# so we keep the first value; temp/rain_1h/snow_1h/clouds_all are averaged
# across the group; weather_main keeps the first reported condition.
n_before <- nrow(df)
df <- df %>%
  group_by(date_time) %>%
  summarise(
    holiday        = first(holiday),
    temp           = round(mean(temp, na.rm = TRUE), 2),
    rain_1h        = mean(rain_1h, na.rm = TRUE),
    snow_1h        = mean(snow_1h, na.rm = TRUE),
    clouds_all     = mean(clouds_all, na.rm = TRUE),
    weather_main   = first(weather_main),
    traffic_volume = first(traffic_volume),
    .groups = "drop"
  ) %>%
  arrange(date_time)
cat("Collapsed", n_before - nrow(df), "duplicate-timestamp rows into", nrow(df), "unique hourly rows.\n")


# ---- 4. Text normalization ----
# Standardize case so e.g. "Sky is Clear" and "sky is clear" are not treated
# as different categories.
df$holiday <- tolower(df$holiday)
df$weather_main <- tolower(df$weather_main)


# ---- 5. Fix known mislabeled holiday date ----
# "christmas day" is mislabeled on 2016-12-26 in the raw data; it belongs on
# 2016-12-25. Verified against this dataset directly.
mis_idx <- which(df$holiday == "christmas day" & as.Date(df$date_time) == as.Date("2016-12-26"))
df$holiday[mis_idx] <- "none"
fix_idx <- which(df$date_time == as.POSIXct("2016-12-25 00:00:00", tz = "UTC"))
if (length(fix_idx) > 0) df$holiday[fix_idx] <- "christmas day"


# ---- 6. Fix "holiday labeled only on first hour" bug ----
# In the raw data, a holiday name is only attached to the first available
# hour of that calendar date; every other hour of the same date reads "none".
# Propagate the holiday name to every hour of its calendar date.
df$cal_date <- as.Date(df$date_time)
holiday_lookup <- df %>%
  filter(holiday != "none") %>%
  group_by(cal_date) %>%
  summarise(holiday_name = first(holiday), .groups = "drop")

n_before_holiday <- sum(df$holiday != "none")
df <- df %>%
  left_join(holiday_lookup, by = "cal_date") %>%
  mutate(holiday = ifelse(!is.na(holiday_name), holiday_name, "none")) %>%
  select(-holiday_name)
cat("Holiday-labeled rows: ", n_before_holiday, " (before fix) -> ",
    sum(df$holiday != "none"), " (after propagating to full day)\n", sep = "")


# ---- 7. Binary holiday feature ----
df$is_holiday <- ifelse(df$holiday != "none", 1L, 0L)
df$cal_date <- NULL
df$holiday <- NULL


# ---- 8. Fix physically impossible values via linear interpolation ----
# temp == 0 Kelvin is not physically possible (never observed on Earth).
# rain_1h has one extreme outlier (9831.3 mm/hr); the next-highest value is
# 55.63 mm/hr, which is physically plausible, so we use 300 mm/hr as a safe
# cutoff that isolates only the one erroneous record.
# Rather than dropping these 11 rows (which would punch holes in the hourly
# series), they are flagged as sensor/logging errors, set to NA, and linearly
# interpolated from the nearest valid neighbouring hours on either side.
n_before <- nrow(df)

n_bad_temp <- sum(df$temp <= 0)
n_bad_rain <- sum(df$rain_1h > 300)

df$temp[df$temp <= 0] <- NA
df$rain_1h[df$rain_1h > 300] <- NA

df <- df %>% arrange(date_time)
df$temp <- na.approx(df$temp, x = as.numeric(df$date_time), rule = 2)
df$rain_1h <- na.approx(df$rain_1h, x = as.numeric(df$date_time), rule = 2)

cat("Interpolated", n_bad_temp, "temp value(s) and", n_bad_rain, "rain_1h value(s) flagged as physically impossible.\n")


# ---- 9. Restrict to a clean, continuous date range ----
# The dataset has a ~10-month gap between 2014-08-08 and 2015-06-11. Rather
# than impute across a 10-month hole, we restrict to the continuous period
# from 2015-10-01 onward, giving an exact 3-year window (2015-10-01 to the
# dataset's end, 2018-09-30) with no large gaps.
# Adjust CUTOFF_DATE below if your group decides on a different window.
CUTOFF_DATE <- as.POSIXct("2015-10-01 00:00:00", tz = "UTC")
n_before <- nrow(df)
df <- df %>% filter(date_time >= CUTOFF_DATE)
cat("Removed", n_before - nrow(df), "rows before", format(CUTOFF_DATE),
    "; final range:", as.character(min(df$date_time)), "to", as.character(max(df$date_time)),
    "(", nrow(df), "rows )", "\n")


# ---- 10. Calendar-derived features (deterministic, safe for forecasting) ----
df$hour <- hour(df$date_time)
df$month <- month(df$date_time)
df$day_of_week <- wday(df$date_time, label = TRUE, abbr = FALSE)


# ---- 11. Keep only the finalized set of columns ----
# weather_main and the raw holiday name are dropped here; is_holiday already
# captures the information needed for modeling, and weather_main was already
# excluded from forecasting inputs per the group report (Section II-C3).
df <- df %>%
  select(date_time, temp, rain_1h, snow_1h, clouds_all, is_holiday, hour, month, day_of_week, traffic_volume)


# ---- 12. Force a consistent date_time string before writing ----
# write.csv() on a POSIXct column silently drops "00:00:00" for any row that
# falls exactly at midnight, while keeping the full timestamp on every other
# row - this is a documented R quirk (format.POSIXct() decides the time
# portion per element, not per column). Converting explicitly to a fixed
# character format avoids inconsistent timestamps in the output file.
df$date_time <- format(df$date_time, "%Y-%m-%d %H:%M:%S")


# ---- 13. Save processed dataset ----
write.csv(df, output_path, row.names = FALSE)
cat("\nSaved processed dataset to:", output_path, "\n")
cat("Final dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
str(df)