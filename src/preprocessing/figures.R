# ============================================================
# Feature Selection via Boxplots
# Dataset: Metro-style traffic data
# Target variable: traffic_volume
# ============================================================

# ---- 0. Setup ----
# Install packages once if you don't have them:
# install.packages(c("ggplot2", "dplyr", "reshape2", "rcompanion"))

library(ggplot2)
library(dplyr)
library(reshape2)   # for melting the correlation matrix for ggplot

# ---- 1. Point this to your dataset ----
# Change this path to wherever your CSV lives on your machine
file_path <- "data/raw/Metro_Interstate_Traffic_Volume.csv"   # <-- EDIT THIS

# All plots will be saved into this folder (created automatically if missing)
output_dir <- "output"
if (!dir.exists(output_dir)) dir.create(output_dir)

df <- read.csv(file_path, stringsAsFactors = FALSE)

# Quick sanity check
str(df)
summary(df)

# ---- 2. Clean up known quirks ----
# (a) Case inconsistency: "Sky is Clear" vs "sky is clear" are the same
# condition but get treated as separate categories if left as-is. This
# happens in the raw data (a handful of rows use title case). Standardize
# ALL text/categorical columns to lowercase before any analysis, or every
# count/boxplot/statistic downstream will silently double-count categories.
text_cols <- c("holiday", "weather_main", "weather_description")
df[text_cols] <- lapply(df[text_cols], tolower)

# (b) "None" as a string should really be NA / a real "No Holiday" category
df$holiday[df$holiday == "none"] <- "No Holiday"
df$holiday <- ifelse(df$holiday == "No Holiday", "No Holiday", "Holiday")

# Make categorical columns factors
df$holiday <- as.factor(df$holiday)
df$weather_main <- as.factor(df$weather_main)

# ---- 3. Boxplot: traffic_volume by holiday ----
p1 <- ggplot(df, aes(x = holiday, y = traffic_volume, fill = holiday)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Holiday",
       x = "Holiday", y = "Traffic Volume") +
  theme_minimal() +
  theme(legend.position = "none")

print(p1)
ggsave(file.path(output_dir, "boxplot_holiday.png"), p1, width = 6, height = 5)

# ---- 4. Boxplot: traffic_volume by weather_main ----
p2 <- ggplot(df, aes(x = reorder(weather_main, traffic_volume, FUN = median),
                     y = traffic_volume, fill = weather_main)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Weather Condition",
       x = "Weather Main", y = "Traffic Volume") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

print(p2)
ggsave(file.path(output_dir, "boxplot_weather_main.png"), p2, width = 8, height = 5)

# ---- 5. Boxplot: traffic_volume by weather_description, colored by weather_main ----
# This is the "do they work together" view: each individual description is
# shown on the x-axis (sorted by median traffic_volume), but colored by its
# PARENT weather_main category. If descriptions cluster by color rather than
# spreading traffic_volume further apart, that visually demonstrates
# weather_description is largely redundant with weather_main.
# Keep top N most frequent descriptions so the plot stays readable.
top_desc <- df %>%
  count(weather_description, sort = TRUE) %>%
  slice_head(n = 12) %>%
  pull(weather_description)

df_top_desc <- df %>% filter(weather_description %in% top_desc)

p3 <- ggplot(df_top_desc, aes(x = reorder(weather_description, traffic_volume, FUN = median),
                              y = traffic_volume, fill = weather_main)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Weather Description (colored by Weather Main)",
       subtitle = "Descriptions clustering by color = weather_description adds little beyond weather_main",
       x = "Weather Description", y = "Traffic Volume", fill = "Weather Main") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3)
ggsave(file.path(output_dir, "boxplot_weather_description.png"), p3, width = 10, height = 6)

# ---- 6. Numeric features: bucket into ranges so we can still boxplot ----
# clouds_all (0-100) -> buckets of 20
df$clouds_bucket <- cut(df$clouds_all,
                        breaks = seq(0, 100, by = 20),
                        include.lowest = TRUE)

p4 <- ggplot(df, aes(x = clouds_bucket, y = traffic_volume, fill = clouds_bucket)) +
  geom_boxplot() +
  labs(title = "Traffic Volume by Cloud Cover Bucket",
       x = "Clouds All (%)", y = "Traffic Volume") +
  theme_minimal() +
  theme(legend.position = "none")

print(p4)
ggsave(file.path(output_dir, "boxplot_clouds_all.png"), p4, width = 7, height = 5)

# rain_1h - most values are 0, so bucket into "No rain" vs "Rain"
df$rain_flag <- ifelse(df$rain_1h > 0, "Rain", "No Rain")

p5 <- ggplot(df, aes(x = rain_flag, y = traffic_volume, fill = rain_flag)) +
  geom_boxplot() +
  labs(title = "Traffic Volume: Rain vs No Rain",
       x = "", y = "Traffic Volume") +
  theme_minimal() +
  theme(legend.position = "none")

print(p5)
ggsave(file.path(output_dir, "boxplot_rain_flag.png"), p5, width = 6, height = 5)

# snow_1h - same idea
df$snow_flag <- ifelse(df$snow_1h > 0, "Snow", "No Snow")

p6 <- ggplot(df, aes(x = snow_flag, y = traffic_volume, fill = snow_flag)) +
  geom_boxplot() +
  labs(title = "Traffic Volume: Snow vs No Snow",
       x = "", y = "Traffic Volume") +
  theme_minimal() +
  theme(legend.position = "none")

print(p6)
ggsave(file.path(output_dir, "boxplot_snow_flag.png"), p6, width = 6, height = 5)

# ---- 7. Correlation heatmap (numeric variables only) ----
# Pulls out the numeric columns and computes pairwise correlation.
# Note: rain_1h and snow_1h are often almost all zeros, so their correlation
# can look weak/unstable - that's expected, not a bug.
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
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p7)
ggsave(file.path(output_dir, "correlation_heatmap.png"), p7, width = 7, height = 6)

# ---- 8. Statistical justification for dropping weather_description ----
# A correlation heatmap only works for NUMERIC variables, so it can't be used
# to justify dropping a categorical column like weather_description. Instead
# we use two proper statistical tools - both give a single clean number you
# can drop straight into a documentation table.

# install.packages("rcompanion")  # only needed for cramerV()
library(rcompanion)

# (a) Cramer's V: how REDUNDANT is weather_description with weather_main?
# Ranges 0 (unrelated) to 1 (perfectly redundant / one determines the other).
cramers_v <- cramerV(table(df$weather_main, df$weather_description))

# (b) Eta-squared: how much of the VARIANCE in traffic_volume does each
# categorical variable explain on its own? This is the categorical
# equivalent of R-squared, from a one-way ANOVA. Ranges 0 (explains nothing)
# to 1 (explains everything).
eta_squared <- function(data, group_col, value_col) {
  aov_model <- aov(as.formula(paste(value_col, "~", group_col)), data = data)
  ss <- summary(aov_model)[[1]][["Sum Sq"]]
  ss[1] / sum(ss)
}

eta_weather_main <- eta_squared(df, "weather_main", "traffic_volume")
eta_weather_desc <- eta_squared(df, "weather_description", "traffic_volume")

# Build a small summary table - this is the table to paste into your report
justification_table <- data.frame(
  Metric = c("Cramer's V (weather_main vs weather_description)",
             "Eta-squared: weather_main -> traffic_volume",
             "Eta-squared: weather_description -> traffic_volume"),
  Value = round(c(cramers_v, eta_weather_main, eta_weather_desc), 3)
)

print(justification_table)
write.csv(justification_table, file.path(output_dir, "weather_description_justification.csv"),
          row.names = FALSE)

cat("\n--- Interpretation for documentation ---\n")
cat(sprintf("Cramer's V = %.3f: weather_description is %s redundant with weather_main.\n",
            cramers_v, ifelse(cramers_v > 0.6, "highly", ifelse(cramers_v > 0.3, "moderately", "weakly"))))
cat(sprintf("eta-squared (weather_main) = %.3f vs eta-squared (weather_description) = %.3f\n",
            eta_weather_main, eta_weather_desc))
cat("If these two eta-squared values are close, weather_description explains\n")
cat("essentially the same share of traffic_volume variance as weather_main,\n")
cat("despite having far more categories - i.e. it adds cardinality without\n")
cat("adding explanatory power. This, combined with the boxplot showing\n")
cat("descriptions clustering by their parent weather_main color, supports\n")
cat("dropping weather_description and keeping only weather_main.\n")

# ---- 9. How to read these boxplots for feature selection ----
# - Look at the MEDIAN LINE position across groups: if it shifts a lot between
#   categories, that feature likely has predictive power.
# - Look at the SPREAD (IQR box height) and OVERLAP between groups: heavily
#   overlapping boxes with similar medians suggest the feature doesn't separate
#   traffic_volume well and may be less useful for the model.
# - Watch for OUTLIERS (dots beyond the whiskers) - a feature with wildly
#   different outlier patterns per group can still be informative.
#
# Rule of thumb: prioritize features whose boxplots show clearly different
# medians and IQRs across categories (e.g., holiday tends to show a strong
# split), and consider dropping/de-prioritizing features where all groups
# look nearly identical.
#
# For the correlation heatmap: values close to +1 or -1 (darker red/blue)
# mean a strong linear relationship with traffic_volume - those numeric
# features are good candidates to keep. Values near 0 (white) suggest weak
# linear relationship - though note correlation only captures LINEAR
# relationships, so a weak correlation doesn't automatically rule out a
# feature with a non-linear effect.

cat("\nAll plots saved to the '", output_dir, "/' folder:\n", sep = "")
cat(" - boxplot_holiday.png\n")
cat(" - boxplot_weather_main.png\n")
cat(" - boxplot_weather_description.png (colored by weather_main)\n")
cat(" - boxplot_clouds_all.png\n")
cat(" - boxplot_rain_flag.png\n")
cat(" - boxplot_snow_flag.png\n")
cat(" - correlation_heatmap.png\n")
cat(" - weather_description_justification.csv\n")