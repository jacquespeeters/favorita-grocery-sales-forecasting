library(matrixStats)
library(data.table)
library(tidyverse)
library(dtplyr)
library(lubridate)
library(lightgbm)
# library(xgboost)
library(RcppRoll)
library(hrbrthemes)
library(caret)
library(CatEncoders)

source("src/utils.R")

# Constant ---------------------------------
TRAINING_SAMPLE = F
TRAINING_SIZE = 2000000 # Used only if TRAINING_SAMPLE == TRUE
PROTOTYPING = F

#-------------------------------------------------------
print("Read data")
# train = fread("./data/train.csv")
train = fread("./data/train.csv", sep=",", na.strings="",
              skip=66458909, # >= "2016-01-01"
              col.names=c("id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"))

test = fread("./data/test.csv")

# holidays_events = read_csv("data/holidays_events.csv")
oil = read_csv("data/oil.csv")
stores = read_csv("data/stores.csv")
# sample_submission = fread("data/sample_submission.csv")
items = fread("./data/items.csv")
transactions = read_csv("data/transactions.csv")

print("Casting columns")
Sys.setenv(TZ="Europe/Paris")

TEST_SIZE = test$date %>% unique %>% length()
END_TEST = test$date %>% max()
BEGIN_TEST = test$date %>% min()
BEGIN_VALID = ymd(BEGIN_TEST) - days(TEST_SIZE)
BEGIN_TRAIN = ymd(BEGIN_VALID) - days(5*TEST_SIZE) # Change parameter here accordingly of what you want to do

df_infos = train %>%  
  copy() %>% 
  select(-id)

rm(train)

df_infos = df_infos %>% 
  mutate(date = ymd(date))

df_infos_test = test %>%  
  copy() %>% 
  select(-id)

df_infos_test = df_infos_test %>% 
  mutate(date = ymd(date))

###
MAX_BACKWARD_FE = 120
print(paste0("Keep last MAX_BACKWARD_FE : ", MAX_BACKWARD_FE, " days, before BEGIN_TRAIN :", BEGIN_TRAIN))
print(paste0("Starting of learning data is ", ymd(BEGIN_TRAIN) - days(MAX_BACKWARD_FE)))

df_infos = df_infos %>% 
  filter(date >= ymd(BEGIN_TRAIN) - days(MAX_BACKWARD_FE))

transactions = transactions %>% 
  filter(date >= ymd(BEGIN_TRAIN) - days(MAX_BACKWARD_FE))

df_infos = df_infos %>% 
  mutate(onpromotion = as.integer(onpromotion))

df_infos_test = df_infos_test %>% 
  mutate(onpromotion = as.integer(onpromotion))

# Keep only
# df_infos = df_infos %>% 
#   filter(date < BEGIN_TEST)

# Do it after FE?
# TODO losing information here, negatives sales means problem with the product probably!
df_infos = df_infos %>% 
  mutate(unit_sales = ifelse(!is.na(unit_sales), pmax(unit_sales, 0), unit_sales))

print("Complete missing (date, store_nbr, item_nbr) where missing means unit_sales = 0 and onpromotion = 0")

purchased_product = df_infos %>% 
  select(item_nbr, store_nbr) %>% 
  distinct()

df_infos = df_infos %>% 
  complete(date = seq(.$date %>% min(), .$date %>% max(), by="days"), 
           nesting(item_nbr, store_nbr), # Only combinations that appear in the data
           fill = list(unit_sales = 0, onpromotion = 0))

print("Restrict df_infos_test to purchased_product")
df_infos_test = df_infos_test %>% 
  inner_join(purchased_product, by = c("store_nbr", "item_nbr")) %>% 
  mutate(unit_sales = NA_integer_)

print("Restrict df_infos to product in test set")
product_in_test = df_infos_test %>% 
  select(store_nbr, item_nbr) %>% 
  distinct()

df_infos = df_infos %>% 
  inner_join(product_in_test, by = c("store_nbr", "item_nbr"))

df_infos = bind_rows(df_infos, df_infos_test)

rm(df_infos_test, purchased_product, product_in_test)

###
print("Change unit_sales to log(1 + unit_sales)")
df_infos = df_infos %>% 
  mutate(unit_sales = log(1 + unit_sales))

###
if(PROTOTYPING){
  print("Downsample for fast prototyping : Keep only 1/100 groups (store_nbr, item_nbr)")
  df_infos = df_infos %>%
    as_tibble() %>%
    filter(group_indices(., store_nbr, item_nbr) %% 100 == 0)
}

###
print("Feature engineering")

###
print("Feature engineering : date_fe ")
date_fe = df_infos %>% 
  select(date) %>% 
  distinct() %>% 
  mutate(year = year(date), 
         month = month(date), 
         day = day(date), 
         wday = wday(date, week_start = getOption("lubridate.week.start", 1))) %>% 
  mutate(day_shift = (day + 15) %% 30)

print("Feature engineering : store_fe ")
# There is a small leak here, should be backward moving average on last K weeks

# Shops ativity given wday
store_fe = transactions %>%
  inner_join(date_fe %>% 
               filter(date > ymd("2017-01-15")), by = "date") %>% 
  group_by(store_nbr, wday) %>% 
  summarise(store_wday_transactions = mean(transactions)) %>% 
  group_by(store_nbr) %>% 
  mutate(store_wday_transactions = store_wday_transactions / sum(store_wday_transactions)) %>% 
  ungroup()

### 
print("Feature engineering : Add unit_sales_rt")
# df_infos = df_infos %>%  
#   group_by(date, store_nbr) %>% 
#   # Ratio of sales per 100 * 100 = 10000 units to make it a bit more readable
#   mutate(unit_sales_rt = (unit_sales / sum(unit_sales))*100) %>% 
#   ungroup()

###
print("Feature engineering : backward_fe")

# mutate_at not working with data.table :(
df_infos = df_infos %>% 
  as_tibble()

print("Compute backward_funs")

# Avoid to apply function on parts of data which we won't use

# Create FE about zero_sales
backward_mean = df_infos %>% 
  mutate(zero_sales = as.integer(unit_sales == 0)) %>% 
  get_backward_fun(iter = c(3, 7, 14, 28, 56, 112), no_mutate = c("date","store_nbr", "item_nbr"), 
                   name_add = "_mean_", fun = roll_mean)

# Add median 
# Idea from https://www.kaggle.com/paulorzp/log-ma-and-days-of-week-means-lb-0-529, kind of ensembling
median_row = backward_mean %>% select(contains("unit_sales_mean_")) %>% 
  as.matrix %>% 
  rowMedians(na.rm = T)

backward_mean = backward_mean %>%
  mutate(unit_sales_mean_median = median_row)

# Add "trend"
# Solve firstly the problem of division by 0 :) 
backward_mean = backward_mean %>%
  mutate(unit_sales_mean_7_28 = unit_sales_mean_7 / ifelse(unit_sales_mean_28 == 0, NA, unit_sales_mean_28),
         unit_sales_mean_28_56 = unit_sales_mean_28 / ifelse(unit_sales_mean_56 == 0, NA, unit_sales_mean_56),
         unit_sales_mean_28_112 = unit_sales_mean_28 / ifelse(unit_sales_mean_112 == 0, NA, unit_sales_mean_112)) 

backward_sd = df_infos %>%
  select(-onpromotion) %>%  # sd on binary variable makes no sense
  get_backward_fun(iter = c(3, 7, 14, 28, 56, 112),
                   no_mutate = c("date","store_nbr", "item_nbr"),
                   name_add = "_sd_", fun = roll_sd)

backward_fe = bind_cols(backward_mean %>% select( -date, -item_nbr, -store_nbr),
                        backward_sd %>% select( -date, -item_nbr, -store_nbr))

rm(backward_mean, backward_sd)

# Add median unit_sales_mean
# TODO

cols_to_delta = backward_fe %>% colnames()

df_learning = bind_cols(df_infos, backward_fe)

rm(backward_fe)

print("backward_mean_dow")
# Rolling function in R don't have Freq option
# therefore frequency (here wday) has to be included in cols_group
backward_mean_dow = df_infos %>% 
  mutate(wday = wday(date)) %>% 
  mutate(zero_sales = as.integer(unit_sales == 0)) %>% 
  get_backward_fun(iter = c(1, 4, 8, 12, 16), no_mutate = c("date","store_nbr", "item_nbr", "wday"), 
                   name_add = "_mean_dow_", fun = roll_mean, 
                   cols_group = c("store_nbr", "item_nbr", "wday"))

median_row = backward_mean_dow %>% select(contains("unit_sales_mean_dow_")) %>% 
  as.matrix %>% 
  rowMedians(na.rm = T)

backward_mean_dow = backward_mean_dow %>%
  mutate(unit_sales_mean_dow_median = median_row)

rm(median_row)

backward_mean_dow = backward_mean_dow %>% select(-date, -store_nbr, -item_nbr)
cols_to_delta_week = backward_mean_dow %>% colnames()

df_learning = bind_cols(df_learning, backward_mean_dow)

rm(backward_mean_dow)

###
print("Create forward_fe")

# Avoid further join by directly working on df_infos
forward_fe = df_infos %>%
  select(date, item_nbr, store_nbr, onpromotion)

forward_fe = forward_fe %>% 
  group_by(store_nbr, item_nbr)

for (i in 1:15){
  forward_fe = forward_fe %>% 
    # mutate(!!paste0("onpromotion", "_", i) := lead(onpromotion, n = i))
    mutate(!!paste0("onpromotion", "_", i) := onpromotion - 0.5*lead(onpromotion, n = i)) 
  # -0.5 if promotion stops
  # 0 if same no promotion
  # 0.5 if promotion keep going on
  # 1 is promotion starts
}
rm(i)

forward_fe = forward_fe %>% 
  ungroup()

forward_fe = forward_fe %>% select(-date, -item_nbr, -store_nbr)

cols_to_forward_drop = forward_fe %>% colnames()

df_learning = bind_cols(df_learning, 
                        forward_fe)

rm(forward_fe)

### 
print("Label encoding : items")
le_family = LabelEncoder.fit(items$family)
le_class = LabelEncoder.fit(items$class)

items = items %>%
  # -1 because lightgbm index starts from 0
  mutate(family = transform(le_family, family) - 1,
         class = transform(le_class, class) - 1)

rm(le_family, le_class)

###
print("Label encoding : stores")

le_city = LabelEncoder.fit(stores$city)
le_state = LabelEncoder.fit(stores$state)
le_type = LabelEncoder.fit(stores$type)
le_cluster = LabelEncoder.fit(stores$cluster)

stores = stores %>% 
  mutate(city = transform(le_city, city) - 1,
         state = transform(le_state, state) - 1,
         type = transform(le_type, type) - 1,
         cluster = transform(le_cluster, cluster) - 1)

rm(le_city, le_state, le_type, le_cluster)

print("Feature engineering on oil")

# Complete missing date values
oil = oil %>% 
  complete(date = seq(min(date), max(date), by="days"))

# Replace NA by linear Interpolation
oil = oil %>% 
  mutate(dcoilwtico = zoo::na.approx(dcoilwtico, na.rm = F))

# Make it stationnary
oil = oil %>%
  mutate(dcoilwtico_diff = dcoilwtico - lag(dcoilwtico))

for (i in c(3, 7, 14, 28, 56, 112)){
  oil = oil %>% 
    mutate(!!paste0("dcoilwtico_diff", "_", i) := dcoilwtico - lag(dcoilwtico, n = i),
           !!paste0("dcoilwtico_rt", "_", i) := dcoilwtico / lag(dcoilwtico, n = i))
}
rm(i)

# oil_now is kind of a dataleak because we wouldn't get it in reality
no_rename = c("date")
oil_now = oil %>% 
  select(-dcoilwtico) %>% 
  rename_at(setdiff(colnames(.), no_rename), .funs = funs(paste0(., "_now")))

oil_old = oil %>% 
  select(-dcoilwtico) %>% 
  rename_at(setdiff(colnames(.), no_rename), .funs = funs(paste0(., "_old")))
rm(no_rename)

cols_to_delta = c(cols_to_delta, oil_old %>% select(-date) %>% colnames())

oil_fe = bind_cols(oil_old, oil_now %>% select(-date))
rm(oil_now, oil_old)

# Finish to construct df_learning
df_date = date_fe %>% 
  left_join(oil_fe, by = "date")

rm(date_fe, oil_fe)

df_learning = df_learning %>%
  left_join(items, by = c("item_nbr")) %>%
  left_join(stores, by = c("store_nbr")) %>%
  left_join(df_date, by = c("date")) %>% 
  left_join(store_fe, by = c("store_nbr", "wday"))

# Multiply unit_sales_mean_median by expected coefficient sales of store_wday_transactions 
# df_learning = df_learning %>% 
#   mutate(unit_sales_mean_median_normed = unit_sales_mean_median * store_wday_transactions * 7)
# 
# cols_to_delta = c(cols_to_delta, "unit_sales_mean_median_normed")

###
submission = tibble()
df_valid_pred = tibble()
metrics  = tibble()

params1 = list(num_leaves = 2^5
               ,objective = "regression_l2"
               ,max_depth = 4
               ,min_data_in_leaf = 200
               ,learning_rate = 0.1
               ,metric = "l2_root")

params2 = list(num_leaves = 2^5-2
               ,objective = "regression_l2"
               ,max_depth = 8
               ,min_data_in_leaf = 200
               ,learning_rate = 0.1
               ,metric = "l2_root")

params3 = list(num_leaves = 2^8
               ,objective = "regression_l2"
               ,max_depth = 12
               ,min_data_in_leaf = 200
               ,learning_rate = 0.1
               ,metric = "l2_root")

params4 = list(num_leaves = 2^10
               ,objective = "regression_l2"
               ,max_depth = 14
               ,min_data_in_leaf = 200
               ,learning_rate = 0.1
               ,metric = "l2_root")

list_params = list(params1,
                   params2,
                   params3,
                   params4)

# Train classifiers
for(delta in 1:16){
  df_learning_tmp = df_learning
  
  print(paste0("Delta step : ", delta))
  
  # Common grouping
  df_learning_tmp = df_learning_tmp %>% 
    group_by(store_nbr, item_nbr)
  
  print("Adjust cols_to_delta")
  df_learning_tmp = df_learning_tmp %>% 
    mutate_at(cols_to_delta, lag, delta)
  
  print("Adjust cols_to_delta_week")
  WEEK_BEFORE = ceiling(delta / 7)
  df_learning_tmp = df_learning_tmp %>% 
    mutate_at(cols_to_delta_week, lag, WEEK_BEFORE*7)
  
  df_learning_tmp = df_learning_tmp %>% 
    ungroup()
  
  print("Adjust cols_to_forward_drop")
  
  # Update forward_fe given available data according to delta
  all_forward = paste0("onpromotion", "_", seq_len(15)) # Should be outside the loop
  keep_forward = paste0("onpromotion", "_", seq_len(16 - delta))
  drop_forward = setdiff(all_forward, keep_forward)
  df_learning_tmp = df_learning_tmp %>% 
    select(-one_of(drop_forward))
  
  # Compute mean
  if(length(keep_forward) > 1 ){
    df_learning_tmp = df_learning_tmp %>% 
      mutate(onpromotion_forward_mean = rowMeans(select(., one_of(keep_forward)), na.rm = T))
  }
  
  # Avoid items which have never been sold yet
  df_learning_tmp = df_learning_tmp %>%     
    filter(unit_sales_mean_112 > 0)
  
  df_train = df_learning_tmp %>% 
    filter(date < ymd(BEGIN_VALID) & date >= ymd(BEGIN_TRAIN))
  
  df_valid = df_learning_tmp %>% 
    filter(date >= ymd(BEGIN_VALID) & date < ymd(BEGIN_TEST))
  
  df_test = df_learning_tmp %>% 
    filter(date == ymd(BEGIN_TEST) + (delta-1))
  
  print("Modeling")
  to_drop = c("date", "item_nbr", "store_nbr", "year", "month", 
              "city", "state", "class")
  
  categorical_feature = c("family", "type", "cluster") # "city", "state"
  
  target = "unit_sales"
  
  if(TRAINING_SAMPLE){
    set.seed(1)
    df_train = df_train %>%
      sample_n(TRAINING_SIZE)
  }
  
  X_train = df_train %>%
    select(-one_of(c(to_drop, target))) %>% 
    as.matrix()
  
  X_valid = df_valid %>% 
    select(-one_of(c(to_drop, target))) %>% 
    as.matrix()
  
  X_test = df_test %>% 
    select(-one_of(c(to_drop, target))) %>% 
    as.matrix()
  
  y_train = df_train %>% pull(unit_sales)
  y_valid = df_valid %>% pull(unit_sales)
  
  weight_train = df_train %>% pull(perishable) * 0.25 + 1
  weight_valid = df_valid %>% pull(perishable) * 0.25 + 1
  
  lgb_train = lgb.Dataset(X_train, label = y_train, weight = weight_train)
  lgb_valid = lgb.Dataset(X_valid, label = y_valid, weight = weight_valid)
  # lgb_test = lgb.Dataset(X_test) # predict is done on X_test with lightgbm
  
  for(i in seq_along(list_params)){
    print(paste0("Step ", i, " of list_params"))
    params = list_params[[i]]
    system.time(
      model_lgb <- lgb.train(params, lgb_train, nrounds = 100000, 
                             categorical_feature = categorical_feature,
                             valids = list(train = lgb_train, valid=lgb_valid), 
                             early_stopping_rounds = 400, verbose = -1, eval_freq = 100)
    ) %>% print()
    
    # importance_matrix = lgb.importance(model = model_lgb)
    # print(importance_matrix[])
    
    ### 
    print("Predict")
    submission_tmp = df_test %>%
      mutate(unit_sales = predict(model_lgb, X_test)) %>%
      mutate(unit_sales = exp(unit_sales) - 1) %>% 
      select(date, store_nbr, item_nbr, unit_sales) %>% 
      mutate(delta = delta,
             params = i)
    
    df_valid_pred_tmp = df_valid %>% 
      rename(unit_sales_true = unit_sales) %>% 
      mutate(unit_sales = predict(model_lgb, X_valid)) %>%
      mutate(unit_sales = exp(unit_sales) - 1) %>% 
      select(date, store_nbr, item_nbr, unit_sales_true, unit_sales) %>% 
      mutate(delta = delta,
             params = i)
    
    print("Best iter : ")
    print(model_lgb$best_iter)
    print("Best score : ")
    print(model_lgb$best_score %>% abs())
    
    metrics_tmp  = tibble(delta = delta, params = i, best_iteration = model_lgb$best_iter, 
                          best_score = model_lgb$best_score)
    metrics = bind_rows(metrics, metrics_tmp)
    submission = bind_rows(submission, submission_tmp)
    df_valid_pred = bind_rows(df_valid_pred, df_valid_pred_tmp)
    
  }
  
  rm(df_learning_tmp, df_train, df_valid, df_test, X_train, X_valid, X_test, 
     lgb_train, lgb_valid, model_lgb, submission_tmp , df_valid_pred_tmp)
}

"Mean RMSE on all delta : " %>% print()
metrics$best_score %>% mean() %>% print()

print("Ensemble error")
df_valid_pred %>% 
  group_by(date, store_nbr, item_nbr) %>% 
  mutate(unit_sales_ensemble = mean(unit_sales)) %>% 
  ungroup() %>% 
  summarise(rmse = RMSE(unit_sales_true, log(unit_sales_ensemble + 1)))

print("No ensemble error")
df_valid_pred %>% 
  summarise(rmse = RMSE(unit_sales_true, log(unit_sales + 1)))

model_performance = df_valid_pred %>% 
  group_by(delta, params) %>% 
  summarise(rmse = RMSE(unit_sales_true, log(unit_sales + 1))) %>% 
  ungroup()

print("Weighted ensemble error")
df_valid_pred %>% 
  left_join(model_performance) %>% 
  mutate(weight = 1 / rmse) %>% 
  group_by(date, store_nbr, item_nbr) %>% 
  summarise(unit_sales_weight = sum(unit_sales * weight)/sum(weight),
            unit_sales_true = first(unit_sales_true)) %>% 
  ungroup() %>% 
  summarise(rmse = RMSE(unit_sales_true, log(unit_sales_weight + 1)))


metrics %>% 
  write_csv(paste0("output/metrics_multiple", Sys.time(), ".csv"))

df_valid_pred %>% 
  write_csv("output/lgb_valid_pred_multiple.csv")

submission = submission %>%
  right_join(test %>% as_tibble() %>% mutate(date = ymd(date)), by = c("date", "store_nbr", "item_nbr")) %>%
  select(-onpromotion)

# Avoid negative and NA predictions
submission = submission %>% 
  mutate(unit_sales = ifelse(unit_sales < 0 | is.na(unit_sales), 0, unit_sales))

submission %>% 
  write_csv("./output/submission_multiple.csv")

submission %>% 
  group_by(id, date, store_nbr, item_nbr) %>% 
  summarise(unit_sales = mean(unit_sales)) %>% 
  ungroup() %>% 
  select(id, unit_sales) %>%
  write_csv("./output/submission_multiple_submit.csv")

submission %>% 
  left_join(model_performance) %>% 
  mutate(weight = 1 / rmse) %>% 
  mutate(weight = ifelse(is.na(weight), 1, weight)) %>% 
  group_by(id, date, store_nbr, item_nbr) %>% 
  summarise(unit_sales = sum(unit_sales * weight)/sum(weight)) %>% 
  ungroup() %>% 
  select(id, unit_sales) %>% 
  write_csv("./output/submission_multiple_weight_submit.csv")
