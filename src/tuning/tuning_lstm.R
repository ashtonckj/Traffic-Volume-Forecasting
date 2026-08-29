# ============================================================
# Hyperparameter Tuning for LSTM Forecasting of Hourly Traffic Volume
# Input: data/processed/traffic_volume_processed.csv (output of preprocessing.R)
#
# Grid-searches the LSTM's key tunable parameters, scores each combination
# on the VALIDATION set (never the test set -- that stays held out until
# the very end), and saves the search results plus a comparison figure.
# The final tuned model itself (training curve, test metrics,
# actual-vs-predicted) is built separately in lstm.R using the best
# configuration found here.
# ============================================================
source("src/models/lstm_common.R")

library(keras3)
library(tensorflow)
library(ggplot2)

set.seed(42)
tensorflow::set_random_seed(42)

if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR, recursive = TRUE)


# ---- 1. Load data and engineer features ----
df <- load_processed()
df <- engineer_features(df)

cat("Loaded", nrow(df), "rows spanning", as.character(min(df$date_time)),
    "to", as.character(max(df$date_time)), "\n")

model_df <- df[, FEATURE_COLS]


# ---- 2. Chronological train / validation / test split ----
# Boundaries come from lstm_common.R so tuning, final training, and
# forecasting cannot disagree about which rows are which.
train_idx <- df$date_time <= TRAIN_END
val_idx   <- df$date_time > TRAIN_END & df$date_time <= VAL_END
test_idx  <- df$date_time > VAL_END

cat("Train:", sum(train_idx), " Val:", sum(val_idx), " Test:", sum(test_idx), "\n")


# ---- 3. Scaling (fitted on TRAIN ONLY) ----
# The whole frame is scaled in one pass with the training parameters, so the
# context rows handed to make_sequences() are on the same scale as the split
# they precede.
scale_params <- fit_scaler(model_df[train_idx, ])
scaled_all   <- apply_scale(model_df, scale_params)

train_scaled <- scaled_all[train_idx, ]
val_scaled   <- scaled_all[val_idx, ]


# ---- 4. Define the search grid ----
# Each row trains a full model, so the grid is kept deliberately small.
# look_back is the parameter of real interest: 24 asks whether one day of
# history is enough, 168 whether the model needs a full week to see the
# weekday/weekend contrast.
grid <- expand.grid(
  look_back     = c(24, 168),
  lstm_units    = c(32, 64),
  dropout_rate  = c(0.2, 0.3),
  learning_rate = c(0.001),
  batch_size    = c(64),
  stringsAsFactors = FALSE
)

cat("\nSearching", nrow(grid), "hyperparameter combinations...\n")
print(grid)

# Sequences depend only on look_back, so build each look_back's sequences
# once and reuse them across the other hyperparameters.
seq_cache <- list()
get_sequences <- function(look_back) {
  key <- as.character(look_back)
  if (is.null(seq_cache[[key]])) {
    seq_cache[[key]] <<- list(
      train = make_sequences(train_scaled, look_back),
      # Validation is windowed with the tail of training as context, so every
      # validation hour is scored instead of the first look_back hours being
      # consumed as warm-up.
      val   = make_sequences(val_scaled, look_back,
                             context = tail(train_scaled, look_back))
    )
  }
  seq_cache[[key]]
}


# ---- 5. Grid search loop (scored on VALIDATION, never test) ----
build_model <- function(look_back, n_features, lstm_units, dropout_rate, learning_rate) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = lstm_units, input_shape = c(look_back, n_features),
               return_sequences = FALSE) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = lstm_units / 2, activation = "relu") %>%
    layer_dense(units = 1)

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = "mse",
    metrics = list("mae")
  )
  model
}

results_list <- list()

for (i in seq_len(nrow(grid))) {
  cfg <- grid[i, ]
  cat("\n---- Config", i, "/", nrow(grid), "----\n")
  print(cfg)

  seqs <- get_sequences(cfg$look_back)
  n_features <- dim(seqs$train$x)[3]

  # Re-seeded per configuration so the comparison reflects the
  # hyperparameters rather than a lucky weight initialization.
  set.seed(42)
  tensorflow::set_random_seed(42)

  model <- build_model(cfg$look_back, n_features, cfg$lstm_units,
                       cfg$dropout_rate, cfg$learning_rate)

  early_stop <- callback_early_stopping(
    monitor = "val_loss", patience = 8, restore_best_weights = TRUE
  )

  history <- model %>% fit(
    x = seqs$train$x, y = seqs$train$y,
    validation_data = list(seqs$val$x, seqs$val$y),
    epochs = 50,
    batch_size = cfg$batch_size,
    callbacks = list(early_stop),
    verbose = 0
  )

  best_epoch <- which.min(history$metrics$val_loss)
  val_loss   <- history$metrics$val_loss[[best_epoch]]
  val_mae    <- history$metrics$val_mae[[best_epoch]]

  results_list[[i]] <- data.frame(
    look_back     = cfg$look_back,
    lstm_units    = cfg$lstm_units,
    dropout_rate  = cfg$dropout_rate,
    learning_rate = cfg$learning_rate,
    batch_size    = cfg$batch_size,
    val_loss_mse  = val_loss,
    val_mae       = val_mae,
    # Recorded so lstm_forecast.R can refit on all observed data for a
    # matching number of epochs, where no validation set is left to early-stop on.
    best_epoch    = best_epoch,
    n_epochs_run  = length(history$metrics$val_loss)
  )

  cat("Val MSE:", round(val_loss, 5), " Val MAE:", round(val_mae, 5),
      " best epoch:", best_epoch, "\n")

  rm(model, history)
  gc()
}

tuning_results <- do.call(rbind, results_list) %>% arrange(val_loss_mse)

cat("\n==== Tuning results (sorted by validation MSE) ====\n")
print(tuning_results)
write.csv(tuning_results, file.path(RESULTS_DIR, "lstm_tuning_results.csv"), row.names = FALSE)


# ---- 6. Tuning comparison figure ----
# One bar per configuration tried, sorted from best (lowest validation MSE)
# to worst, with the selected configuration highlighted -- this shows which
# hyperparameter combination won and by how much.
plot_df <- tuning_results %>%
  mutate(
    config_label = sprintf("LB%d U%d D%.1f", look_back, lstm_units, dropout_rate),
    config_label = factor(config_label, levels = config_label[order(-val_loss_mse)]),
    is_best = row_number() == 1
  )

p <- ggplot(plot_df, aes(x = config_label, y = val_loss_mse, fill = is_best)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = signif(val_loss_mse, 3)), hjust = -0.15, size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("FALSE" = "grey65", "TRUE" = "#1f77b4"), guide = "none") +
  labs(title = "LSTM Hyperparameter Search: Validation MSE by Configuration",
       x = "Configuration (Look-back, Units, Dropout)",
       y = "Validation MSE (scaled units)") +
  expand_limits(y = max(plot_df$val_loss_mse) * 1.15) +
  theme_minimal(base_size = 11)

ggsave(file.path(RESULTS_DIR, "lstm_tuning_comparison.png"), plot = p,
       width = 7, height = 4.5, dpi = 300)

cat("\nBest config (lowest validation MSE):\n")
print(tuning_results[1, ])

cat("\nSaved tuning results and comparison figure to:", RESULTS_DIR, "\n")
