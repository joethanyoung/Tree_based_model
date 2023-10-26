library(tictoc())

tic()
# import required packages
library(dplyr)
library(readxl)
library(caret)
library(xgboost)
library(fastDummies)


# import required dataframe
df1 <-
  read_excel('L01105基础表个险承保清单.xlsx',
             skip = 1)

# check the null values
df1 %>% is.na %>% sum

# take glimpse on dataframe
df1 %>% glimpse

# filter out the dataframe
df <-
  df1 %>% select(业务员姓名,
                 营销经理名称,
                 分区经理姓名,
                 险种名称,
                 缴费周期,
                 自保互保标志,
                 业务来源,
                 保单保费,
                 受益人与被保险人关系) %>%
  mutate(保单保费 = as.numeric(保单保费))

# Set up 5-fold cross-validation
cv_control <- trainControl(method = "cv", number = 5)

# Find factor columns with only one level
single_level_columns <-
  sapply(df, function(x)
    is.factor(x) && length(levels(x)) < 2)

# Remove single-level factor columns
df_filtered <- df[, !single_level_columns]

# Perform one-hot encoding on the filtered dataset
dummies <- dummyVars( ~ ., data = df)

# Perform one-hot encoding on the filtered dataset
df_encoded <-
  dummy_cols(df,
             remove_first_dummy = TRUE,
             remove_selected_columns = TRUE)

df_encoded %>% head
# Train the XGBoost model with 5-fold cross-validation
# set seed number to 1234 to make code productive
set.seed(1234)

# train the xgboost model with specified parameters
xgb_model_cv <-
  train(
    保单保费 ~ .,
    data = df_encoded,
    method = "xgbTree",
    trControl = cv_control,
    tuneGrid = expand.grid(
      nrounds = c(50, 100, 500),
      max_depth = c(3, 5, 7),
      eta = c(0.01, 0.025, 0.05, 0.075, 0.1),
      gamma = c(0, 0.25, 0.75, 1),
      colsample_bytree = c(0.2, 0.4, 0.6, 0.8, 1),
      min_child_weight = c(1, 5, 10),
      subsample = c(0.25, 0.5, 0.6, 0.8, 1)
    ),
    verbose = TRUE,
    # set the progress
    verbosity = 1
  )

# print the results
xgb_model_cv$results[which.min(xgb_model_cv$results$RMSE), ]

# find the importance
final_model <- xgb_model_cv$finalModel

# Get the feature names without the target variable
feature_names <-
  colnames(df_encoded[, !names(df_encoded) %in% "保单保费"])

# Get the feature importance matrix
importance_matrix <-
  xgb.importance(feature_names, model = final_model)

# Print the importance matrix to check for any issues
print(importance_matrix)

# Plot feature importance
xgb.plot.importance(importance_matrix)


###############################################################################
toc()

