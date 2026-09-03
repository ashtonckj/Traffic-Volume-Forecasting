# ================================================================
# BMMS2094 STATISTICS FOR DATA SCIENCE
# XGBOOST TRAFFIC VOLUME FORECASTING
#
# Complete group + individual report reproducibility version
#
# Final selected model: XGB2
# Common seasonal differencing: d = 0, D = 1, s = 168
# Primary evaluation criterion: RMSE
#
# Full workflow:
#
# Processed data
# -> Verify data
# -> Common D=1, s=168 seasonal differencing
# -> Create XGBoost lag/calendar features
# -> Train XGB1-XGB8 using TRAIN only
# -> Evaluate recursively on VALIDATION
# -> Select lowest validation RMSE
# -> Freeze XGB2
# -> Refit XGB2 using TRAIN + VALIDATION
# -> Forecast held-out TEST once
# -> Compute RMSE, MAE, MAPE, sMAPE, MASE
# -> Residual diagnostics
# -> Refit same frozen XGB2 on ALL observed data
# -> 168-hour future forecast
#
# IMPORTANT:
# Do not tune XGB2 using the locked test results.
# ================================================================


# ================================================================
# 1. PACKAGE
# ================================================================

library(xgboost)

set.seed(2094)


# ================================================================
# 2. LOAD PROCESSED DATA
# ================================================================

data_path <- "data/processed/traffic_volume_processed1.csv"


traffic <- read.csv(
  data_path,
  stringsAsFactors = FALSE
)


# UTC preserves the regular hourly index without
# daylight-saving-time conversion.

traffic$date_time <- as.POSIXct(
  traffic$date_time,
  format = "%Y-%m-%d %H:%M:%S",
  tz = "UTC"
)


# ================================================================
# 3. VERIFY DATA STRUCTURE
# ================================================================

required_columns <- c(
  "date_time",
  "traffic_volume",
  "split",
  "hour",
  "month",
  "day_of_week",
  "is_holiday"
)


missing_columns <- setdiff(
  required_columns,
  names(traffic)
)


if (length(missing_columns) > 0) {
  
  stop(
    paste(
      "Missing required columns:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )
}


if (any(is.na(traffic$date_time))) {
  stop("Some date_time values could not be parsed.")
}


if (anyDuplicated(traffic$date_time) > 0) {
  stop("Duplicate timestamps detected.")
}


hour_differences <- as.numeric(
  diff(
    traffic$date_time
  ),
  units = "hours"
)


if (any(abs(hour_differences - 1) > 1e-9)) {
  stop("Dataset is not a continuous hourly sequence.")
}


if (any(is.na(traffic))) {
  stop("Missing values detected in processed dataset.")
}


if (!all(
  traffic$split %in%
  c(
    "train",
    "validation",
    "test"
  )
)) {
  
  stop("Unexpected split labels detected.")
}


split_code <- match(
  traffic$split,
  c(
    "train",
    "validation",
    "test"
  )
)


if (any(diff(split_code) < 0)) {
  stop("Train/validation/test splits are not chronological.")
}


# Dataset checks used throughout the reports.

stopifnot(
  nrow(traffic) == 26304
)


cat(
  "\n============================================================\n"
)

cat(
  "DATASET VERIFICATION\n"
)

cat(
  "============================================================\n"
)


cat(
  "Rows:",
  nrow(traffic),
  "\n"
)


cat(
  "Columns:",
  ncol(traffic),
  "\n"
)


cat(
  "Date range:",
  format(
    min(traffic$date_time)
  ),
  "to",
  format(
    max(traffic$date_time)
  ),
  "\n"
)


cat(
  "Zero traffic observations:",
  sum(
    traffic$traffic_volume == 0
  ),
  "\n"
)


cat(
  "Chronological data: TRUE\n"
)

cat(
  "Exactly hourly: TRUE\n"
)

cat(
  "Duplicate timestamps: 0\n"
)

cat(
  "Missing values: 0\n"
)


# ================================================================
# 4. IDENTIFY CHRONOLOGICAL SPLITS
# ================================================================

train_idx <- which(
  traffic$split == "train"
)


validation_idx <- which(
  traffic$split == "validation"
)


test_idx <- which(
  traffic$split == "test"
)


train_data <- traffic[
  train_idx,
]


validation_data <- traffic[
  validation_idx,
]


test_data <- traffic[
  test_idx,
]


stopifnot(
  length(train_idx) == 18412,
  length(validation_idx) == 3945,
  length(test_idx) == 3947
)


cat(
  "\n============================================================\n"
)

cat(
  "CHRONOLOGICAL SPLITS\n"
)

cat(
  "============================================================\n"
)


cat(
  "Training:",
  nrow(train_data),
  "observations |",
  format(
    min(train_data$date_time)
  ),
  "to",
  format(
    max(train_data$date_time)
  ),
  "\n"
)


cat(
  "Validation:",
  nrow(validation_data),
  "observations |",
  format(
    min(validation_data$date_time)
  ),
  "to",
  format(
    max(validation_data$date_time)
  ),
  "\n"
)


cat(
  "Testing:",
  nrow(test_data),
  "observations |",
  format(
    min(test_data$date_time)
  ),
  "to",
  format(
    max(test_data$date_time)
  ),
  "\n"
)


# ================================================================
# 5. CALENDAR VARIABLES
# ================================================================

# IMPORTANT:
# This is the same weekday coding used in the correct final model.
#
# Monday = 1
# Tuesday = 2
# ...
# Sunday = 7


weekday_levels <- c(
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday"
)


if (is.numeric(traffic$day_of_week)) {
  
  traffic$day_of_week_num <- as.numeric(
    traffic$day_of_week
  )
  
} else {
  
  traffic$day_of_week_num <- match(
    traffic$day_of_week,
    weekday_levels
  )
}


if (any(is.na(traffic$day_of_week_num))) {
  stop("Unable to convert day_of_week.")
}


if (!all(
  traffic$day_of_week_num %in% 1:7
)) {
  
  stop("Unexpected numeric weekday coding.")
}


# ------------------------------------------------
# Holiday indicator
# ------------------------------------------------

if (is.logical(traffic$is_holiday)) {
  
  traffic$is_holiday_num <- as.integer(
    traffic$is_holiday
  )
  
} else if (is.numeric(traffic$is_holiday)) {
  
  traffic$is_holiday_num <- as.numeric(
    traffic$is_holiday
  )
  
} else {
  
  holiday_text <- tolower(
    trimws(
      as.character(
        traffic$is_holiday
      )
    )
  )
  
  
  traffic$is_holiday_num <- ifelse(
    
    holiday_text %in% c(
      "1",
      "true",
      "yes",
      "holiday"
    ),
    
    1,
    
    0
  )
}


if (anyNA(traffic$is_holiday_num)) {
  stop("Unable to convert is_holiday.")
}

# ================================================================
# REFRESH SPLIT DATA AFTER CALENDAR CONVERSION
# ================================================================

# train_data, validation_data and test_data were originally created
# before day_of_week_num and is_holiday_num were added to traffic.
#
# Recreate the split data frames here so they contain the final
# numeric calendar variables required by XGBoost.

train_data <- traffic[
  train_idx,
]

validation_data <- traffic[
  validation_idx,
]

test_data <- traffic[
  test_idx,
]


# Verify required XGBoost calendar columns are present.

stopifnot(
  "day_of_week_num" %in% names(train_data),
  "is_holiday_num" %in% names(train_data),
  "day_of_week_num" %in% names(validation_data),
  "is_holiday_num" %in% names(validation_data)
)

stopifnot(
  !anyNA(train_data$day_of_week_num),
  !anyNA(train_data$is_holiday_num),
  !anyNA(validation_data$day_of_week_num),
  !anyNA(validation_data$is_holiday_num)
)

# ================================================================
# 6. COMMON SEASONAL DIFFERENCING / MASE
# ================================================================

# Lecturer/group specification:
#
# d = 0
# D = 1
# s = 168 hours


m <- 168L


train_y <-
  traffic$traffic_volume[
    train_idx
  ]


# MASE denominator:
#
# Calculated ONCE from original-scale TRAINING observations.
# This definition must be common across the group models.


mase_scale <- mean(
  abs(
    train_y[
      (m + 1L):length(train_y)
    ] -
      train_y[
        1L:(length(train_y) - m)
      ]
  ),
  na.rm = TRUE
)


if (
  !is.finite(mase_scale) ||
  mase_scale <= 0
) {
  
  stop("Invalid MASE scale.")
}


cat(
  "\n============================================================\n"
)

cat(
  "COMMON DIFFERENCING / MASE SCALE\n"
)

cat(
  "============================================================\n"
)


cat(
  "d = 0\n"
)

cat(
  "D = 1\n"
)

cat(
  "s = 168\n"
)


cat(
  "Training MASE scale:",
  round(
    mase_scale,
    4
  ),
  "vehicles/hour\n"
)


# Expected approximately:
#
# MASE scale = 334.7018


# ================================================================
# 7. HELPER FUNCTIONS
# ================================================================


# ------------------------------------------------
# 7.1 Weekly seasonal differencing
#
# W_t = Y_t - Y_(t-168)
# ------------------------------------------------

seasonal_difference <- function(
    y,
    m = 168L
) {
  
  w <- rep(
    NA_real_,
    length(y)
  )
  
  
  if (length(y) > m) {
    
    w[
      (m + 1):length(y)
    ] <-
      
      y[
        (m + 1):length(y)
      ] -
      
      y[
        1:(length(y) - m)
      ]
  }
  
  
  return(w)
}


# ------------------------------------------------
# 7.2 Create XGBoost features
# ------------------------------------------------

create_xgb_features <- function(
    w,
    hour,
    day_of_week,
    month,
    is_holiday
) {
  
  
  lag_value <- function(
    x,
    lag_n
  ) {
    
    c(
      rep(
        NA_real_,
        lag_n
      ),
      
      x[
        1:(length(x) - lag_n)
      ]
    )
  }
  
  
  features <- data.frame(
    
    target =
      w,
    
    
    lag_1 =
      lag_value(
        w,
        1L
      ),
    
    
    lag_24 =
      lag_value(
        w,
        24L
      ),
    
    
    lag_168 =
      lag_value(
        w,
        168L
      ),
    
    
    hour =
      as.numeric(
        hour
      ),
    
    
    day_of_week =
      as.numeric(
        day_of_week
      ),
    
    
    month =
      as.numeric(
        month
      ),
    
    
    is_holiday =
      as.numeric(
        is_holiday
      )
  )
  
  
  features <- features[
    complete.cases(features),
  ]
  
  
  return(features)
}


# ------------------------------------------------
# 7.3 Fixed-origin recursive forecast on W_t
# ------------------------------------------------

recursive_w_forecast <- function(
    model,
    w_history,
    future_data
) {
  
  
  history <-
    as.numeric(
      w_history
    )
  
  
  horizon <-
    nrow(
      future_data
    )
  
  
  predictions <-
    numeric(
      horizon
    )
  
  
  for (
    i in seq_len(
      horizon
    )
  ) {
    
    
    n_hist <-
      length(
        history
      )
    
    
    if (n_hist < 168L) {
      
      stop(
        "Insufficient differenced history."
      )
    }
    
    
    x_new <- matrix(
      
      c(
        
        history[
          n_hist
        ],
        
        
        history[
          n_hist - 23L
        ],
        
        
        history[
          n_hist - 167L
        ],
        
        
        as.numeric(
          future_data$hour[i]
        ),
        
        
        as.numeric(
          future_data$day_of_week_num[i]
        ),
        
        
        as.numeric(
          future_data$month[i]
        ),
        
        
        as.numeric(
          future_data$is_holiday_num[i]
        )
      ),
      
      nrow = 1
    )
    
    
    colnames(
      x_new
    ) <- c(
      
      "lag_1",
      "lag_24",
      "lag_168",
      "hour",
      "day_of_week",
      "month",
      "is_holiday"
    )
    
    
    w_hat <-
      as.numeric(
        
        predict(
          model,
          x_new
        )
      )
    
    
    predictions[i] <-
      w_hat
    
    
    # IMPORTANT:
    #
    # Use the MODEL'S previous prediction for future
    # lag values.
    #
    # Never append actual validation/test W_t.
    
    history <- c(
      history,
      w_hat
    )
  }
  
  
  return(
    predictions
  )
}


# ------------------------------------------------
# 7.4 Recursive inverse seasonal differencing
# ------------------------------------------------

inverse_seasonal_difference <- function(
    w_forecast,
    y_history,
    m = 168L
) {
  
  
  history <-
    as.numeric(
      y_history
    )
  
  
  horizon <-
    length(
      w_forecast
    )
  
  
  y_forecast <-
    numeric(
      horizon
    )
  
  
  for (
    i in seq_len(
      horizon
    )
  ) {
    
    
    n_hist <-
      length(
        history
      )
    
    
    seasonal_reference <-
      history[
        n_hist - m + 1L
      ]
    
    
    y_hat <-
      w_forecast[i] +
      seasonal_reference
    
    
    y_forecast[i] <-
      y_hat
    
    
    # Append prediction.
    #
    # For horizons beyond 168 hours,
    # reconstructed forecasts become the seasonal references.
    
    history <- c(
      history,
      y_hat
    )
  }
  
  
  return(
    y_forecast
  )
}


# ------------------------------------------------
# 7.5 Common forecast metrics
# ------------------------------------------------

evaluate_forecast <- function(
    actual,
    predicted,
    mase_scale
) {
  
  
  valid_pair <-
    !is.na(actual) &
    !is.na(predicted)
  
  
  if (!any(valid_pair)) {
    
    stop(
      "No valid actual/predicted pairs."
    )
  }
  
  
  errors <-
    actual[
      valid_pair
    ] -
    predicted[
      valid_pair
    ]
  
  
  rmse <-
    sqrt(
      mean(
        errors^2
      )
    )
  
  
  mae <-
    mean(
      abs(
        errors
      )
    )
  
  
  # -----------------------------
  # MAPE
  #
  # Actual traffic = 0 excluded.
  # -----------------------------
  
  valid_mape <-
    valid_pair &
    actual != 0
  
  
  mape <-
    mean(
      abs(
        
        (
          actual[
            valid_mape
          ] -
            
            predicted[
              valid_mape
            ]
        ) /
          
          actual[
            valid_mape
          ]
      )
    ) *
    100
  
  
  # -----------------------------
  # sMAPE
  #
  # 2|A-F| / (|A| + |F|)
  # -----------------------------
  
  smape_denominator <-
    abs(
      actual
    ) +
    abs(
      predicted
    )
  
  
  valid_smape <-
    valid_pair &
    smape_denominator != 0
  
  
  smape <-
    mean(
      
      2 *
        
        abs(
          
          actual[
            valid_smape
          ] -
            
            predicted[
              valid_smape
            ]
        ) /
        
        smape_denominator[
          valid_smape
        ]
    ) *
    100
  
  
  mase <-
    mae /
    mase_scale
  
  
  return(
    
    data.frame(
      
      RMSE =
        rmse,
      
      MAE =
        mae,
      
      MAPE =
        mape,
      
      sMAPE =
        smape,
      
      MASE =
        mase,
      
      MAPE_zero_actuals_excluded =
        sum(
          valid_pair &
            actual == 0
        )
    )
  )
}


# ================================================================
# 8. VALIDATION MODEL SELECTION
# ================================================================

# Eight configurations were defined BEFORE test evaluation.
#
# Model selection criterion:
#
# LOWEST VALIDATION RMSE


# ------------------------------------------------
# 8.1 Training-only differenced target
# ------------------------------------------------

w_train <-
  seasonal_difference(
    train_y,
    m
  )


xgb_validation_training_data <-
  create_xgb_features(
    
    w =
      w_train,
    
    hour =
      train_data$hour,
    
    day_of_week =
      train_data$day_of_week_num,
    
    month =
      train_data$month,
    
    is_holiday =
      train_data$is_holiday_num
  )


validation_x_train <- as.matrix(
  
  xgb_validation_training_data[
    ,
    c(
      "lag_1",
      "lag_24",
      "lag_168",
      "hour",
      "day_of_week",
      "month",
      "is_holiday"
    )
  ]
)


validation_y_train <-
  xgb_validation_training_data$target


validation_dtrain <-
  xgb.DMatrix(
    
    data =
      validation_x_train,
    
    label =
      validation_y_train
  )


cat(
  "\nUsable XGBoost TRAINING rows:",
  nrow(
    validation_x_train
  ),
  "\n"
)


# Expected:
#
# 18,076 usable training rows.


stopifnot(
  nrow(
    validation_x_train
  ) == 18076
)


# ------------------------------------------------
# 8.2 Prespecified XGBoost candidate grid
# ------------------------------------------------

xgb_candidates <- data.frame(
  
  Configuration =
    paste0(
      "XGB",
      1:8
    ),
  
  
  nrounds =
    c(
      250L,
      400L,
      300L,
      500L,
      250L,
      400L,
      300L,
      500L
    ),
  
  
  max_depth =
    c(
      3L,
      3L,
      5L,
      5L,
      4L,
      4L,
      6L,
      6L
    ),
  
  
  eta =
    c(
      0.05,
      0.03,
      0.05,
      0.03,
      0.05,
      0.03,
      0.05,
      0.03
    ),
  
  
  min_child_weight =
    c(
      5,
      5,
      5,
      5,
      10,
      10,
      10,
      10
    ),
  
  
  subsample =
    c(
      0.8,
      0.8,
      0.8,
      0.8,
      0.9,
      0.9,
      0.9,
      0.9
    ),
  
  
  colsample_bytree =
    c(
      0.9,
      0.9,
      0.9,
      0.9,
      1.0,
      1.0,
      1.0,
      1.0
    ),
  
  
  stringsAsFactors =
    FALSE
)


# ------------------------------------------------
# 8.3 Validation evaluation
# ------------------------------------------------

xgb_validation_results <-
  vector(
    "list",
    nrow(
      xgb_candidates
    )
  )


for (
  i in seq_len(
    nrow(
      xgb_candidates
    )
  )
) {
  
  
  candidate <-
    xgb_candidates[
      i,
    ]
  
  
  cat(
    "\nEvaluating",
    candidate$Configuration,
    "...\n"
  )
  
  
  candidate_params <- list(
    
    objective =
      "reg:squarederror",
    
    eval_metric =
      "rmse",
    
    max_depth =
      candidate$max_depth,
    
    eta =
      candidate$eta,
    
    min_child_weight =
      candidate$min_child_weight,
    
    subsample =
      candidate$subsample,
    
    colsample_bytree =
      candidate$colsample_bytree,
    
    nthread =
      1,
    
    seed =
      2094
  )
  
  
  set.seed(
    2094
  )
  
  
  candidate_model <-
    xgb.train(
      
      params =
        candidate_params,
      
      data =
        validation_dtrain,
      
      nrounds =
        candidate$nrounds,
      
      verbose =
        0
    )
  
  
  # --------------------------------
  # Fixed-origin validation forecast
  #
  # Start with TRAIN W only.
  # Validation actual W is NEVER fed back.
  # --------------------------------
  
  validation_w_hat <-
    recursive_w_forecast(
      
      model =
        candidate_model,
      
      w_history =
        w_train,
      
      future_data =
        validation_data
    )
  
  
  # --------------------------------
  # Reconstruct original traffic scale
  #
  # Start with TRAIN Y only.
  # --------------------------------
  
  validation_y_hat <-
    inverse_seasonal_difference(
      
      w_forecast =
        validation_w_hat,
      
      y_history =
        train_y,
      
      m =
        m
    )
  
  
  validation_metrics <-
    evaluate_forecast(
      
      actual =
        validation_data$traffic_volume,
      
      predicted =
        validation_y_hat,
      
      mase_scale =
        mase_scale
    )
  
  
  xgb_validation_results[[i]] <-
    cbind(
      candidate,
      validation_metrics
    )
}


xgb_validation_results <-
  do.call(
    rbind,
    xgb_validation_results
  )


xgb_validation_results <-
  xgb_validation_results[
    order(
      xgb_validation_results$RMSE
    ),
  ]


row.names(
  xgb_validation_results
) <- NULL


cat(
  "\n============================================================\n"
)

cat(
  "XGBOOST VALIDATION RESULTS - SORTED BY RMSE\n"
)

cat(
  "============================================================\n"
)


print(
  xgb_validation_results,
  digits = 8
)


# ================================================================
# 9. SELECT AND FREEZE XGB2
# ================================================================

selected_configuration <-
  xgb_validation_results[
    1L,
  ]


cat(
  "\n============================================================\n"
)

cat(
  "SELECTED XGBOOST CONFIGURATION\n"
)

cat(
  "============================================================\n"
)


print(
  selected_configuration,
  digits = 8
)


# XGB2 must remain the selected configuration
# when this exact workflow is reproduced.

if (
  selected_configuration$Configuration != "XGB2"
) {
  
  stop(
    paste(
      "Unexpected validation winner:",
      selected_configuration$Configuration,
      "- expected XGB2."
    )
  )
}


# Expected XGB2 validation result:
#
# RMSE  = 806.4218
# MAE   = 547.3086
# MAPE  = 34.1679 %
# sMAPE = 25.5065 %
# MASE  = 1.6352


# ------------------------------------------------
# Frozen final configuration
# ------------------------------------------------

xgb_params <- list(
  
  objective =
    "reg:squarederror",
  
  eval_metric =
    "rmse",
  
  max_depth =
    3L,
  
  eta =
    0.03,
  
  min_child_weight =
    5,
  
  subsample =
    0.8,
  
  colsample_bytree =
    0.9,
  
  nthread =
    1,
  
  seed =
    2094
)


xgb_nrounds <-
  400L


# ================================================================
# 10. FINAL REFIT — TRAIN + VALIDATION
# ================================================================

final_idx <- c(
  train_idx,
  validation_idx
)


final_data <- traffic[
  final_idx,
]


final_y <-
  final_data$traffic_volume


w_final <-
  seasonal_difference(
    final_y,
    m
  )


xgb_training_data <-
  create_xgb_features(
    
    w =
      w_final,
    
    hour =
      final_data$hour,
    
    day_of_week =
      final_data$day_of_week_num,
    
    month =
      final_data$month,
    
    is_holiday =
      final_data$is_holiday_num
  )


x_train <- as.matrix(
  
  xgb_training_data[
    ,
    c(
      "lag_1",
      "lag_24",
      "lag_168",
      "hour",
      "day_of_week",
      "month",
      "is_holiday"
    )
  ]
)


y_train <-
  xgb_training_data$target


dtrain <-
  xgb.DMatrix(
    
    data =
      x_train,
    
    label =
      y_train
  )


stopifnot(
  length(
    final_y
  ) == 22357,
  
  nrow(
    x_train
  ) == 22021
)


set.seed(
  2094
)


xgb_final_model <-
  xgb.train(
    
    params =
      xgb_params,
    
    data =
      dtrain,
    
    nrounds =
      xgb_nrounds,
    
    verbose =
      0
  )


cat(
  "\nFinal XGB2 model fitted using Train + Validation.\n"
)


# ================================================================
# 11. LOCKED TEST FORECAST
# ================================================================

test_future <- traffic[
  test_idx,
]


# ------------------------------------------------
# Recursive differenced forecast
# ------------------------------------------------

xgb_test_w_hat <-
  recursive_w_forecast(
    
    model =
      xgb_final_model,
    
    w_history =
      w_final,
    
    future_data =
      test_future
  )


# ------------------------------------------------
# Recursive inverse differencing
# ------------------------------------------------

xgb_test_y_hat <-
  inverse_seasonal_difference(
    
    w_forecast =
      xgb_test_w_hat,
    
    y_history =
      final_y,
    
    m =
      m
  )


xgb_test_actual <-
  test_future$traffic_volume


# ================================================================
# 12. FINAL LOCKED TEST METRICS
# ================================================================

xgb_test_metrics <-
  evaluate_forecast(
    
    actual =
      xgb_test_actual,
    
    predicted =
      xgb_test_y_hat,
    
    mase_scale =
      mase_scale
  )


cat(
  "\n============================================================\n"
)

cat(
  "FINAL LOCKED XGBOOST TEST RESULTS\n"
)

cat(
  "============================================================\n"
)


print(
  xgb_test_metrics
)


# Group comparison object.

xgb_group_result <- data.frame(
  
  Model =
    "XGBoost",
  
  RMSE =
    xgb_test_metrics$RMSE,
  
  MAE =
    xgb_test_metrics$MAE,
  
  MAPE =
    xgb_test_metrics$MAPE,
  
  sMAPE =
    xgb_test_metrics$sMAPE,
  
  MASE =
    xgb_test_metrics$MASE
)


cat(
  "\nGROUP COMPARISON ROW:\n"
)


print(
  xgb_group_result,
  row.names = FALSE
)


# ================================================================
# OFFICIAL XGBOOST TEST RESULTS
#
# RMSE  = 540.2577
# MAE   = 378.7486
# MAPE  = 17.93068 %
# sMAPE = 18.9864 %
# MASE  = 1.1316
#
# Test horizon = 3947 hours
# ================================================================


# ================================================================
# 13. TEST FORECAST SANITY CHECK
# ================================================================

negative_forecasts <-
  sum(
    xgb_test_y_hat < 0
  )


negative_percentage <-
  
  negative_forecasts /
  
  length(
    xgb_test_y_hat
  ) *
  
  100


cat(
  "\n============================================================\n"
)

cat(
  "TEST FORECAST SANITY CHECK\n"
)

cat(
  "============================================================\n"
)


cat(
  "Minimum predicted traffic:",
  min(
    xgb_test_y_hat
  ),
  "\n"
)


cat(
  "Maximum predicted traffic:",
  max(
    xgb_test_y_hat
  ),
  "\n"
)


cat(
  "Mean predicted traffic:",
  mean(
    xgb_test_y_hat
  ),
  "\n"
)


cat(
  "Negative forecasts:",
  negative_forecasts,
  "of",
  length(
    xgb_test_y_hat
  ),
  "\n"
)


cat(
  "Negative forecast percentage:",
  round(
    negative_percentage,
    2
  ),
  "%\n"
)


# Expected:
#
# Negative test forecasts = 40
# Percentage = approximately 1.01%


# ================================================================
# 14. MODEL-SPECIFIC RESIDUAL DIAGNOSTIC
# ================================================================

# Residual:
#
# e_t = actual - forecast


xgb_test_residuals <-
  xgb_test_actual -
  xgb_test_y_hat


# Ljung-Box tests used in the individual report.

xgb_lb_24 <-
  Box.test(
    
    xgb_test_residuals,
    
    lag =
      24,
    
    type =
      "Ljung-Box",
    
    fitdf =
      0
  )


xgb_lb_48 <-
  Box.test(
    
    xgb_test_residuals,
    
    lag =
      48,
    
    type =
      "Ljung-Box",
    
    fitdf =
      0
  )


xgb_lb_168 <-
  Box.test(
    
    xgb_test_residuals,
    
    lag =
      168,
    
    type =
      "Ljung-Box",
    
    fitdf =
      0
  )


cat(
  "\n============================================================\n"
)

cat(
  "XGBOOST RESIDUAL LJUNG-BOX TESTS\n"
)

cat(
  "============================================================\n"
)


cat(
  "Lag 24  | Q =",
  round(
    as.numeric(
      xgb_lb_24$statistic
    ),
    2
  ),
  "| p-value =",
  xgb_lb_24$p.value,
  "\n"
)


cat(
  "Lag 48  | Q =",
  round(
    as.numeric(
      xgb_lb_48$statistic
    ),
    2
  ),
  "| p-value =",
  xgb_lb_48$p.value,
  "\n"
)


cat(
  "Lag 168 | Q =",
  round(
    as.numeric(
      xgb_lb_168$statistic
    ),
    2
  ),
  "| p-value =",
  xgb_lb_168$p.value,
  "\n"
)


# Expected diagnostic:
#
# lag 24:
# Q = 6002.54
#
# lag 48:
# Q = 6648.53
#
# lag 168:
# Q = 11310.67
#
# p < 0.001 for all three tests.
#
# Interpretation:
# Significant residual temporal dependence remains.
#
# IMPORTANT:
# This is diagnostic only.
# Do NOT retune XGB2 after seeing these test residuals.


# ================================================================
# 15. OUTPUT DIRECTORY
# ================================================================

output_dir <- "output/models/xgboost"


dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)


# ================================================================
# 16. SAVE VALIDATION + TEST OUTPUTS
# ================================================================

write.csv(
  
  xgb_validation_results,
  
  file.path(
    output_dir,
    "xgboost_validation_results.csv"
  ),
  
  row.names =
    FALSE
)


xgb_predictions <- data.frame(
  
  date_time =
    test_future$date_time,
  
  actual =
    xgb_test_actual,
  
  predicted =
    xgb_test_y_hat,
  
  residual =
    xgb_test_residuals,
  
  differenced_forecast =
    xgb_test_w_hat
)


write.csv(
  
  xgb_predictions,
  
  file.path(
    output_dir,
    "final_test_predictions.csv"
  ),
  
  row.names =
    FALSE
)


write.csv(
  
  xgb_test_metrics,
  
  file.path(
    output_dir,
    "final_test_metrics.csv"
  ),
  
  row.names =
    FALSE
)


# ================================================================
# 17. ONE-WEEK FORWARD FORECAST
# ================================================================

# Dataset ends:
#
# 2018-09-30 23:00
#
# Future forecast:
#
# 2018-10-01 00:00
# through
# 2018-10-07 23:00
#
# Total = 168 hours.
#
# No actual observations exist for this period,
# therefore NO accuracy metrics are calculated.


FORECAST_HOURS <-
  168L


forecast_start <-
  
  max(
    traffic$date_time
  ) +
  
  3600


future_dates <- seq(
  
  from =
    forecast_start,
  
  by =
    "hour",
  
  length.out =
    FORECAST_HOURS
)


future_lt <- as.POSIXlt(
  future_dates,
  tz = "UTC"
)


future_hour <-
  future_lt$hour


future_month <-
  future_lt$mon +
  1L


# Monday = 1, ..., Sunday = 7

future_day_of_week <-
  
  (
    (
      future_lt$wday +
        6L
    ) %%
      7L
  ) +
  
  1L


# 1-7 October 2018 contains no
# US federal holiday in the forecast period.

future_is_holiday <-
  rep(
    0,
    FORECAST_HOURS
  )


xgb_future_data <- data.frame(
  
  date_time =
    future_dates,
  
  hour =
    future_hour,
  
  day_of_week_num =
    future_day_of_week,
  
  month =
    future_month,
  
  is_holiday_num =
    future_is_holiday
)


# ================================================================
# 18. REFIT FROZEN XGB2 ON ALL OBSERVED DATA
# ================================================================

# This is done ONLY after completion of the
# official locked test evaluation.
#
# It does not change the official test metrics.


all_y <-
  traffic$traffic_volume


w_all <-
  seasonal_difference(
    all_y,
    m
  )


xgb_all_training_data <-
  create_xgb_features(
    
    w =
      w_all,
    
    hour =
      traffic$hour,
    
    day_of_week =
      traffic$day_of_week_num,
    
    month =
      traffic$month,
    
    is_holiday =
      traffic$is_holiday_num
  )


x_all <- as.matrix(
  
  xgb_all_training_data[
    ,
    c(
      "lag_1",
      "lag_24",
      "lag_168",
      "hour",
      "day_of_week",
      "month",
      "is_holiday"
    )
  ]
)


y_all <-
  xgb_all_training_data$target


dall <-
  xgb.DMatrix(
    
    data =
      x_all,
    
    label =
      y_all
  )


set.seed(
  2094
)


xgb_future_model <-
  xgb.train(
    
    params =
      xgb_params,
    
    data =
      dall,
    
    nrounds =
      xgb_nrounds,
    
    verbose =
      0
  )


# ================================================================
# 19. RECURSIVE 168-HOUR FUTURE FORECAST
# ================================================================

xgb_future_w_hat <-
  recursive_w_forecast(
    
    model =
      xgb_future_model,
    
    w_history =
      w_all,
    
    future_data =
      xgb_future_data
  )


xgb_future_y_hat <-
  inverse_seasonal_difference(
    
    w_forecast =
      xgb_future_w_hat,
    
    y_history =
      all_y,
    
    m =
      m
  )


# ================================================================
# 20. FUTURE FORECAST DATAFRAME
# ================================================================

future_weekday_names <-
  weekday_levels[
    future_day_of_week
  ]


xgb_future_forecast <- data.frame(
  
  date_time =
    future_dates,
  
  forecast =
    xgb_future_y_hat,
  
  differenced_forecast =
    xgb_future_w_hat,
  
  day_of_week =
    future_weekday_names,
  
  hour =
    future_hour,
  
  is_holiday =
    future_is_holiday
)


write.csv(
  
  xgb_future_forecast,
  
  file.path(
    output_dir,
    "xgboost_future_forecast.csv"
  ),
  
  row.names =
    FALSE
)


# ================================================================
# 21. FUTURE FORECAST FACTS
# ================================================================

weekday_flag <-
  future_day_of_week <= 5L


weekend_flag <-
  future_day_of_week >= 6L


future_negative_count <-
  sum(
    xgb_future_y_hat < 0
  )


peak_index <-
  which.max(
    xgb_future_y_hat
  )


minimum_index <-
  which.min(
    xgb_future_y_hat
  )


cat(
  "\n============================================================\n"
)

cat(
  "ONE-WEEK XGBOOST FORECAST FACTS\n"
)

cat(
  "============================================================\n"
)


cat(
  "Forecast period:",
  format(
    min(
      future_dates
    ),
    "%Y-%m-%d %H:%M"
  ),
  "to",
  format(
    max(
      future_dates
    ),
    "%Y-%m-%d %H:%M"
  ),
  "\n"
)


cat(
  "Mean forecast:",
  round(
    mean(
      xgb_future_y_hat
    ),
    2
  ),
  "vehicles/hour\n"
)


cat(
  "Weekday mean:",
  round(
    mean(
      xgb_future_y_hat[
        weekday_flag
      ]
    ),
    2
  ),
  "vehicles/hour\n"
)


cat(
  "Weekend mean:",
  round(
    mean(
      xgb_future_y_hat[
        weekend_flag
      ]
    ),
    2
  ),
  "vehicles/hour\n"
)


cat(
  "Peak forecast:",
  round(
    xgb_future_y_hat[
      peak_index
    ],
    2
  ),
  "vehicles/hour at",
  format(
    future_dates[
      peak_index
    ],
    "%a %Y-%m-%d %H:%M"
  ),
  "\n"
)


cat(
  "Minimum forecast:",
  round(
    xgb_future_y_hat[
      minimum_index
    ],
    2
  ),
  "vehicles/hour at",
  format(
    future_dates[
      minimum_index
    ],
    "%a %Y-%m-%d %H:%M"
  ),
  "\n"
)


cat(
  "Negative future forecasts:",
  future_negative_count,
  "\n"
)


cat(
  "Last observed week mean:",
  round(
    mean(
      tail(
        traffic$traffic_volume,
        168
      )
    ),
    2
  ),
  "vehicles/hour\n"
)


cat(
  "Forecast week mean:",
  round(
    mean(
      xgb_future_y_hat
    ),
    2
  ),
  "vehicles/hour\n"
)


# ================================================================
# EXPECTED FUTURE FORECAST FACTS
#
# Mean             = 3373.16
# Weekday mean     = 3604.27
# Weekend mean     = 2795.40
#
# Peak             = 6606.52
#                    Tue 2018-10-02 16:00
#
# Minimum          = 231.93
#                    Tue 2018-10-02 02:00
#
# Negative         = 0
#
# Last observed
# week mean        = 3387.93
# ================================================================


# ================================================================
# 22. FIGURE 1 — FULL TEST PERIOD
# ================================================================

png(
  
  file.path(
    output_dir,
    "fig1_test_full.png"
  ),
  
  width =
    1800,
  
  height =
    900,
  
  res =
    150
)


plot(
  
  test_future$date_time,
  
  xgb_test_actual,
  
  type =
    "l",
  
  lwd =
    1,
  
  col =
    "blue",
  
  xlab =
    "Date",
  
  ylab =
    "Traffic Volume (vehicles/hour)",
  
  main =
    "XGBoost: Actual vs Forecast - Full Test Period"
)


lines(
  
  test_future$date_time,
  
  xgb_test_y_hat,
  
  lwd =
    1,
  
  lty =
    2,
  
  col =
    "red"
)


legend(
  
  "topright",
  
  legend =
    c(
      "Actual Traffic",
      "XGBoost Forecast"
    ),
  
  col =
    c(
      "blue",
      "red"
    ),
  
  lty =
    c(
      1,
      2
    ),
  
  lwd =
    c(
      1,
      1
    ),
  
  bty =
    "n"
)


dev.off()


# ================================================================
# 23. FIGURE 2 — FIRST FOUR WEEKS OF TEST
# ================================================================

four_weeks <- min(
  
  24L *
    7L *
    4L,
  
  length(
    xgb_test_actual
  )
)


png(
  
  file.path(
    output_dir,
    "fig2_test_first_4_weeks.png"
  ),
  
  width =
    1800,
  
  height =
    900,
  
  res =
    150
)


plot(
  
  test_future$date_time[
    1:four_weeks
  ],
  
  xgb_test_actual[
    1:four_weeks
  ],
  
  type =
    "l",
  
  lwd =
    1,
  
  col =
    "blue",
  
  xlab =
    "Date",
  
  ylab =
    "Traffic Volume (vehicles/hour)",
  
  main =
    "XGBoost: Actual vs Forecast - First Four Weeks"
)


lines(
  
  test_future$date_time[
    1:four_weeks
  ],
  
  xgb_test_y_hat[
    1:four_weeks
  ],
  
  lwd =
    1,
  
  lty =
    2,
  
  col =
    "red"
)


legend(
  
  "topright",
  
  legend =
    c(
      "Actual Traffic",
      "XGBoost Forecast"
    ),
  
  col =
    c(
      "blue",
      "red"
    ),
  
  lty =
    c(
      1,
      2
    ),
  
  lwd =
    c(
      1,
      1
    ),
  
  bty =
    "n"
)


dev.off()


# ================================================================
# 24. FIGURE 3 — ONE-WEEK FUTURE FORECAST
# ================================================================

observed_tail_hours <-
  24L *
  7L *
  2L


observed_tail_idx <-
  
  (
    nrow(
      traffic
    ) -
      
      observed_tail_hours +
      
      1L
  ):
  
  nrow(
    traffic
  )


observed_tail_dates <-
  traffic$date_time[
    observed_tail_idx
  ]


observed_tail_values <-
  traffic$traffic_volume[
    observed_tail_idx
  ]


combined_dates <- c(
  
  observed_tail_dates,
  
  future_dates
)


combined_values <- c(
  
  observed_tail_values,
  
  xgb_future_y_hat
)


y_range <- range(
  combined_values,
  na.rm = TRUE
)


png(
  
  file.path(
    output_dir,
    "fig3_future_forecast.png"
  ),
  
  width =
    1800,
  
  height =
    900,
  
  res =
    150
)


plot(
  
  observed_tail_dates,
  
  observed_tail_values,
  
  type =
    "l",
  
  lwd =
    1,
  
  col =
    "blue",
  
  xlim =
    range(
      combined_dates
    ),
  
  ylim =
    y_range,
  
  xlab =
    "Date",
  
  ylab =
    "Traffic Volume (vehicles/hour)",
  
  main =
    "XGBoost: One-Week Forward Traffic Forecast"
)


lines(
  
  future_dates,
  
  xgb_future_y_hat,
  
  lwd =
    1.5,
  
  lty =
    2,
  
  col =
    "red"
)


abline(
  
  v =
    forecast_start,
  
  lty =
    3,
  
  col =
    "darkgray"
)


legend(
  
  "topright",
  
  legend =
    c(
      "Observed Traffic",
      "Future XGBoost Forecast",
      "Forecast Origin"
    ),
  
  col =
    c(
      "blue",
      "red",
      "darkgray"
    ),
  
  lty =
    c(
      1,
      2,
      3
    ),
  
  lwd =
    c(
      1,
      1.5,
      1
    ),
  
  bty =
    "n"
)


dev.off()


# ================================================================
# 25. FIGURE 4 — RESIDUAL ANALYSIS
# ================================================================

png(
  
  file.path(
    output_dir,
    "fig4_xgboost_residual_analysis.png"
  ),
  
  width =
    1800,
  
  height =
    1100,
  
  res =
    150
)


layout(
  
  matrix(
    c(
      1,
      1,
      2,
      3
    ),
    
    nrow =
      2,
    
    byrow =
      TRUE
  ),
  
  heights =
    c(
      1.1,
      1
    )
)


# ------------------------------------------------
# Residual time series
# ------------------------------------------------

par(
  mar =
    c(
      4,
      4,
      3,
      1
    )
)


plot(
  
  xgb_test_residuals,
  
  type =
    "l",
  
  lwd =
    0.7,
  
  xlab =
    "Test Observation",
  
  ylab =
    "Residual",
  
  main =
    "XGBoost Test Forecast Residuals"
)


abline(
  
  h =
    0,
  
  lty =
    2,
  
  col =
    "darkgray"
)


# ------------------------------------------------
# Residual ACF
# ------------------------------------------------

par(
  mar =
    c(
      4,
      4,
      3,
      1
    )
)


acf(
  
  xgb_test_residuals,
  
  lag.max =
    168,
  
  main =
    "ACF of Residuals",
  
  xlab =
    "Lag (hours)",
  
  ylab =
    "ACF"
)


# ------------------------------------------------
# Residual distribution
# ------------------------------------------------

par(
  mar =
    c(
      4,
      4,
      3,
      1
    )
)


hist(
  
  xgb_test_residuals,
  
  breaks =
    60,
  
  probability =
    TRUE,
  
  main =
    "Distribution of Residuals",
  
  xlab =
    "Residuals",
  
  ylab =
    "Density"
)


lines(
  
  density(
    xgb_test_residuals
  ),
  
  lwd =
    2,
  
  col =
    "red"
)


rug(
  xgb_test_residuals
)


dev.off()


# ================================================================
# 26. FINAL XGBOOST SUMMARY
# ================================================================

cat(
  "\n============================================================\n"
)

cat(
  "FINAL XGBOOST SUMMARY\n"
)

cat(
  "============================================================\n"
)


cat(
  "Selected model: XGB2\n"
)


cat(
  "Selection method: Lowest Validation RMSE\n"
)


cat(
  "Validation XGB2 RMSE:",
  round(
    selected_configuration$RMSE,
    4
  ),
  "\n"
)


cat(
  "Final model fit: Train + Validation\n"
)


cat(
  "Test method: Fixed-origin recursive\n"
)


cat(
  "Differencing: d = 0, D = 1, s = 168\n"
)


cat(
  "Features: lag_1, lag_24, lag_168,",
  "hour, day_of_week, month, is_holiday\n"
)


cat(
  "Weather predictors: Excluded\n"
)


cat(
  "\nFinal test RMSE:",
  round(
    xgb_test_metrics$RMSE,
    4
  ),
  "\n"
)


cat(
  "Final test MAE:",
  round(
    xgb_test_metrics$MAE,
    4
  ),
  "\n"
)


cat(
  "Final test MAPE:",
  round(
    xgb_test_metrics$MAPE,
    4
  ),
  "%\n"
)


cat(
  "Final test sMAPE:",
  round(
    xgb_test_metrics$sMAPE,
    4
  ),
  "%\n"
)


cat(
  "Final test MASE:",
  round(
    xgb_test_metrics$MASE,
    4
  ),
  "\n"
)


cat(
  "Negative test forecasts:",
  negative_forecasts,
  "of",
  length(
    xgb_test_y_hat
  ),
  "\n"
)


cat(
  "\nResidual Ljung-Box:",
  "lags 24, 48, 168 all expected p < 0.001\n"
)


cat(
  "\nFuture model:",
  "same frozen XGB2 refitted on all observed data\n"
)


cat(
  "Future horizon:",
  FORECAST_HOURS,
  "hours\n"
)


cat(
  "Future period:",
  format(
    min(
      future_dates
    ),
    "%Y-%m-%d %H:%M"
  ),
  "to",
  format(
    max(
      future_dates
    ),
    "%Y-%m-%d %H:%M"
  ),
  "\n"
)


cat(
  "\nIMPORTANT:\n"
)


cat(
  "The future forecast is separate from the locked test evaluation.\n"
)


cat(
  "No future accuracy metrics are calculated because actual values are unavailable.\n"
)


cat(
  "DO NOT RETUNE XGB2 USING THE LOCKED TEST RESULTS.\n"
)
