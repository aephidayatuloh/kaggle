# Kaggle Titanic
# Improved from : https://www.benjaminsorensen.me/post/modeling-with-parsnip-and-tidymodels/

library(tidyverse)
library(tidymodels)
library(skimr)
library(vip)
library(janitor)
library(lime)

training_titanic <- read_csv("titanic/input/train.csv") %>% 
  janitor::clean_names() 
testing_titanic <- read_csv("titanic/input/test.csv") %>% 
  janitor::clean_names() 
submission_titanic <- read_csv("titanic/input/gender_submission.csv")

training_titanic %>% str()
testing_titanic %>% str()

training_titanic %>% skim()
testing_titanic %>% skim()

# Impute embarked variable
count(training_titanic, embarked)

training_titanic %>% 
  filter(is.na(embarked)) %>% 
  select(cabin, pclass, fare)

training_titanic %>% 
  filter(!is.na(embarked)) %>%
  ggplot(aes(x = embarked, y = fare, fill = factor(pclass))) +
  geom_boxplot()

# Impute fare variable
training_titanic %>% 
  group_by(pclass, embarked) %>% 
  summarise(med = median(fare))

custom_palette <- c("#32a852", "#a4e5f5", "#f0c481", "#a8f0c7")

training_titanic %>% 
  ggplot(aes(x = age)) +
  geom_histogram(color = "white", fill = custom_palette[1]) +
  theme_minimal()

age_na <- training_titanic %>% 
  mutate(age_na = case_when(is.na(age) ~ "NA",
                            TRUE ~ "Filled"))

age_na %>% 
  ggplot(aes(sex, fill = age_na)) +
  geom_bar()

# Preprocessing Data untuk training dan testing
# Disebut jg feature engineering
preproc <- function(dataset, outcome = NULL, level = NULL){
  d1 <- dataset %>% 
    mutate(#sex = factor(sex),
           title = str_remove_all(name, '(.*, )|(\\..*)'),
           title = case_when(title %in% c("Ms", "Mlle") ~ "Miss",
                             title ==  "Mme" ~ "Mrs",
                             title %in% c("Master", "Miss", "Mr", "Mrs") ~ as.character(title),
                             TRUE ~ "Rare Title"),
           # title = factor(title),
           family_name = str_sub(name, 1, str_locate(name, ",")[,2] - 1),
           # family_name = factor(family_name),
           family_size = sib_sp + parch + 1,
           family_size_category = case_when(family_size == 1 ~ "singleton",
                                          between(family_size, 1, 5) ~ "small",
                                          family_size > 5 ~ "large"),
           family_size_category = factor(family_size_category),
           deck = str_sub(cabin, 1, 1),
           deck = case_when(is.na(deck) ~ "Regular",
                            TRUE ~ as.character(deck)),
           # deck = factor(deck),
           embarked = case_when(is.na(embarked) ~ "C",
                                TRUE ~ embarked),
           # embarked = factor(embarked),
           # pclass = factor(pclass),
           fare = case_when(is.na(fare) & pclass == 1 & embarked == "C" ~ 78.3,
                            is.na(fare) & pclass == 1 & embarked == "Q" ~ 90,
                            is.na(fare) & pclass == 1 & embarked == "S" ~ 52,
                            is.na(fare) & pclass == 2 & embarked == "C" ~ 24,
                            is.na(fare) & pclass == 2 & embarked == "Q" ~ 12.4,
                            is.na(fare) & pclass == 2 & embarked == "S" ~ 13.5,
                            is.na(fare) & pclass == 3 & embarked == "C" ~ 7.90,
                            is.na(fare) & pclass == 3 & embarked == "Q" ~ 7.75,
                            is.na(fare) & pclass == 3 & embarked == "S" ~ 8.05,
                            TRUE ~ fare),
           age = case_when(is.na(age) ~ median(age, na.rm = TRUE),
                           TRUE ~ as.numeric(age))
    ) %>% 
    mutate_if(is.character, as_factor)
  
  if(is.null(outcome)){
    return(d1 %>% 
             select(passenger_id, sex, age, pclass, fare, embarked, title, family_name, family_size, family_size_category, deck)
           )
  } else if(!is.null(outcome) & outcome %in% names(d1)){
    d1[[outcome]] <- factor(d1[[outcome]], levels = level)
    return(d1 %>% 
             select(passenger_id, survived, sex, age, pclass, fare, embarked, title, family_name, family_size, family_size_category, deck))
  }
}

train_data <- preproc(training_titanic, outcome = "survived", level = c(0,1))
test_data <- preproc(testing_titanic)

skim(train_data)
train_data %>% 
  ggplot(aes(x = age)) +
  geom_histogram(color = "white", fill = custom_palette[1]) +
  theme_minimal()
count(train_data, deck)

set.seed(1001)
splits <- initial_split(train_data, prop = 0.8)
training_set <- splits %>% training()
testing_set <- splits %>% testing()
## v-fold cross validation
folds <- vfold_cv(training_set, v = 5, repeats = 3, strata = "survived")

# Pre-processing
rec <- train_data %>% 
  recipe() %>% 
  update_role(pclass, sex, age, fare, embarked, title, family_size, new_role = "predictor") %>% 
  update_role(survived, new_role = "outcome") %>% 
  step_rm(-has_role("outcome"), -has_role("predictor")) %>% 
  step_string2factor(all_nominal(), -all_outcomes()) #%>%
  # step_center(all_predictors()) %>% 
  # step_scale(all_predictors())

# Prepping
prepped <- rec %>% 
  prep(retain = TRUE)

# Baking
train <- prepped %>% 
  juice()

test <- prepped %>% 
  bake(new_data = testing_set)

###########################################################################
#                                                                         #
#                      Logistic Regression Modeling                       #
#                                                                         #
###########################################################################

## Model specification
logreg_mod <- logistic_reg(
  mode = "classification"
  ) %>% 
  set_engine("glm")

## Fitting
logreg_fit <- fit(
  object = logreg_mod,
  formula = formula(prepped),
  data = train
)

# Variable Important
vip(logreg_fit) + 
  ggtitle("Variable Important: logreg")

## Predicting!
predicted <- tibble(actual = test$survived) %>% 
  bind_cols(pred_class = predict(logreg_fit, test, type = "class")$.pred_class) %>% 
  bind_cols(pred_prob = predict(logreg_fit, test, type = "prob"))

## Assessment -- test error
metrics(predicted, truth = actual, .pred_1, estimate = pred_class) %>% 
  knitr::kable()

conf_mat(predicted, truth = actual, estimate = pred_class)[[1]] %>% 
  as_tibble() %>% 
  mutate(pct = n/sum(n)) %>% 
  ggplot(aes(Prediction, Truth, alpha = n)) + 
  geom_tile(fill = custom_palette[4], show.legend = FALSE) +
  geom_text(aes(label = paste0(n, "\n", round(pct*100, 2), "%")), 
            color = "coral", alpha = 1, size = 6) +
  theme_minimal() +
  labs(
    title = "Confusion matrix"
  )

metrics_tbl <- tibble("accuracy" = yardstick::metrics(predicted, actual, pred_class) %>%
                        dplyr::select(-.estimator) %>%
                        dplyr::filter(.metric == "accuracy") %>% 
                        dplyr::select(.estimate),
                      "precision" = yardstick::precision(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "recall" = yardstick::recall(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "f_meas" = yardstick::f_meas(predicted, actual, pred_class) %>%
                        dplyr::select(.estimate),
                      "auc" = yardstick::metrics(predicted, actual, .pred_1, estimate = pred_class) %>% 
                        filter(.metric == "roc_auc") %>% 
                        select(.estimate)) %>%
  tidyr::unnest(cols = c(accuracy, precision, recall, f_meas, auc))
metrics_tbl
# autoplot(roc_curve(predicted, actual, .pred_1))

roc_curve(predicted, actual, .pred_1) %>%
  mutate(.threshold = case_when(is.infinite(.threshold) ~ 0,
                                TRUE ~ as.numeric(.threshold)),
         specificity = 1 - specificity) %>% 
  arrange(specificity, sensitivity) %>% 
  ggplot(aes(x = specificity, y = sensitivity)) +
  geom_line(color = "red") +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = "longdash") +
  labs(x = "1 - specificity",
       subtitle = sprintf("AUC: %s", round(metrics_tbl$auc, 5))) + 
  theme_light() 

# LIME explaination
explanation <- lime(train %>% select(-survived), logreg_fit)
explanations <- explain(x = test[1:3,-1], 
                        explainer = explanation, 
                        labels = "1", 
                        n_permutations  = 5000,
                        dist_fun        = "manhattan",
                        kernel_width    = 3,
                        n_features = 7)

# Get an overview with the standard plot
plot_explanations(explanations)
plot_features(explanations, ncol = 2)

test_data_baked <- prepped %>% 
  bake(new_data = test_data)

submission_titanic %>% 
  mutate(Survived = predict(logreg_fit, test_data_baked, type = "class")$.pred_class) %>% 
  write_csv("06-modeling/submission_titanic_logreg_eng_glm.csv")

# Save model
# saveRDS(object = logreg_fit, "model/logreg_fit.rds")

############################ Cross Validation #############################

folded <- folds %>% 
  mutate(
    recipes = splits %>%
      # Prepper is a wrapper for `prep()` which handles `split` objects
      map(prepper, recipe = rec), 
    training_data = splits %>% map(analysis),
    logreg_fits = map2(
      recipes,
      training_data, 
      ~ fit(
        logreg_mod, 
        formula(.x), 
        data = bake(object = .x, new_data = .y)
      )
    )
  )

## Predict 
predict_logreg <- function(split, rec, model) {
  test_set <- bake(rec, assessment(split))
  tibble(actual = test_set$survived) %>% 
    bind_cols(pred_class = predict(model, test_set, type = "class")$.pred_class) %>% 
    bind_cols(pred_prob = predict(model, test_set, type = "prob"))
}

predictions <- folded %>% 
  mutate(pred = list(
    splits,
    recipes,
    logreg_fits
  ) %>% 
    pmap(predict_logreg)
  )

## Evaluate
eval <- predictions %>% 
  transmute(
    metrics = pred %>% map(~ metrics(., truth = actual, starts_with(".pred")[1], estimate = pred_class))
  ) %>% 
  unnest(metrics)

eval %>% knitr::kable()

eval %>% 
  group_by(.metric) %>% 
  summarise_at(.vars = vars(.estimate), .funs = list(min = min, median = median, mean = mean, sd = sd, max = max)) %>% 
  knitr::kable()


###########################################################################
#                                                                         #
#                         Random Forest Modeling                          #
#                                                                         #
###########################################################################

## Model specification
rf_mod <- rand_forest(
  mode = "classification",
  trees = 500
) %>% 
  set_engine("randomForest")

## Fitting
rf_fit <- fit(
  object = rf_mod,
  formula = formula(prepped),
  data = train
)

## Predicting!
predicted <- tibble(actual = test$survived) %>% 
  bind_cols(pred_class = predict(rf_fit, test, type = "class")$.pred_class) %>% 
  bind_cols(pred_prob = predict(rf_fit, test, type = "prob"))

## Assessment -- test error
metrics(predicted, truth = actual, .pred_1, estimate = pred_class) %>% 
  knitr::kable()

conf_mat(predicted, truth = actual, estimate = pred_class)[[1]] %>% 
  as_tibble() %>% 
  mutate(pct = n/sum(n)) %>% 
  ggplot(aes(Prediction, Truth, alpha = n)) + 
  geom_tile(fill = custom_palette[4], show.legend = FALSE) +
  geom_text(aes(label = paste0(n, "\n", round(pct*100, 2), "%")), 
            color = "coral", alpha = 1, size = 6) +
  theme_minimal() +
  labs(
    title = "Confusion matrix"
  )

metrics_tbl <- tibble("accuracy" = yardstick::metrics(predicted, actual, pred_class) %>%
                        dplyr::select(-.estimator) %>%
                        dplyr::filter(.metric == "accuracy") %>% 
                        dplyr::select(.estimate),
                      "precision" = yardstick::precision(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "recall" = yardstick::recall(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "f_meas" = yardstick::f_meas(predicted, actual, pred_class) %>%
                        dplyr::select(.estimate),
                      "auc" = yardstick::metrics(predicted, actual, .pred_1, estimate = pred_class) %>% 
                        filter(.metric == "roc_auc") %>% 
                        select(.estimate)) %>%
  tidyr::unnest(cols = c(accuracy, precision, recall, f_meas, auc))
metrics_tbl
# autoplot(roc_curve(predicted, actual, .pred_1))

roc_curve(predicted, actual, .pred_1) %>%
  mutate(.threshold = case_when(is.infinite(.threshold) ~ 0,
                                TRUE ~ as.numeric(.threshold)),
         specificity = 1 - specificity) %>% 
  arrange(specificity, sensitivity) %>% 
  ggplot(aes(x = specificity, y = sensitivity)) +
  geom_line(color = "red") +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = "longdash") +
  labs(x = "1 - specificity",
       subtitle = sprintf("AUC: %s", round(metrics_tbl$auc, 5))) + 
  theme_light() 

# LIME explaination
explanation <- lime(train %>% select(-survived), rf_fit)
explanations <- explain(x = test[1:3,-1], 
                        explainer = explanation, 
                        labels = "1", 
                        n_permutations  = 5000,
                        dist_fun        = "manhattan",
                        kernel_width    = 3,
                        n_features = 7)

# Get an overview with the standard plot
plot_explanations(explanations)
plot_features(explanations, ncol = 2)

test_data_baked <- prepped %>% 
  bake(new_data = test_data)

submission_titanic %>% 
  mutate(Survived = predict(rf_fit, test_data_baked, type = "class")$.pred_class) %>% 
  write_csv("06-modeling/submission_titanic.csv")

# Save model
# saveRDS(object = rf_fit, "model/rf.rds")

############################ Cross Validation #############################

## v-fold cross validation
folds <- vfold_cv(train_data, v = 5)

folded <- folds %>% 
  mutate(
    recipes = splits %>%
      # Prepper is a wrapper for `prep()` which handles `split` objects
      map(prepper, recipe = rec), 
    training_data = splits %>% map(analysis),
    rf_fits = map2(
      recipes,
      training_data, 
      ~ fit(
        rf_mod, 
        formula(.x), 
        data = bake(object = .x, new_data = .y)
      )
    )
  )

## Predict 
predict_rf <- function(split, rec, model) {
  test_set <- bake(rec, assessment(split))
  tibble(actual = test_set$survived) %>% 
    bind_cols(pred_class = predict(model, test_set, type = "class")$.pred_class) %>% 
    bind_cols(pred_prob = predict(model, test_set, type = "prob"))
}

predictions <- folded %>% 
  mutate(pred = list(
    splits,
    recipes,
    rf_fits
  ) %>% 
    pmap(predict_rf)
  )

## Evaluate
eval <- predictions %>% 
  transmute(
    metrics = pred %>% map(~ metrics(., truth = actual, starts_with(".pred")[1], estimate = pred_class))
  ) %>% 
  unnest(metrics)

eval %>% knitr::kable()

eval %>% 
  group_by(.metric) %>% 
  summarise_at(.vars = vars(.estimate), .funs = list(min = min, median = median, mean = mean, sd = sd, max = max)) %>% 
  knitr::kable()

###########################################################################
#                                                                         #
#                            XGBoost Modeling                             #
#                                                                         #
###########################################################################

## Model specification
xgb_mod <- boost_tree(
  mode = "classification",
  trees = 500
) %>%
  set_engine("xgboost")

## Fitting
xgb_fit <- fit(
  object = xgb_mod,
  formula = formula(prepped),
  data = train
)

## Predicting!
predicted <- tibble(actual = test$survived) %>% 
  bind_cols(pred_class = predict(xgb_fit, test, type = "class")$.pred_class) %>% 
  bind_cols(pred_prob = predict(xgb_fit, test, type = "prob"))

## Assessment -- test error
metrics(predicted, truth = actual, .pred_1, estimate = pred_class) %>% 
  knitr::kable()

conf_mat(predicted, truth = actual, estimate = pred_class)[[1]] %>% 
  as_tibble() %>% 
  mutate(pct = n/sum(n)) %>% 
  ggplot(aes(Prediction, Truth, alpha = n)) + 
  geom_tile(fill = custom_palette[4], show.legend = FALSE) +
  geom_text(aes(label = paste0(n, "\n", round(pct*100, 2), "%")), 
            color = "coral", alpha = 1, size = 6) +
  theme_minimal() +
  labs(
    title = "Confusion matrix"
  )

metrics_tbl <- tibble("accuracy" = yardstick::metrics(predicted, actual, pred_class) %>%
                        dplyr::select(-.estimator) %>%
                        dplyr::filter(.metric == "accuracy") %>% 
                        dplyr::select(.estimate),
                      "precision" = yardstick::precision(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "recall" = yardstick::recall(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "f_meas" = yardstick::f_meas(predicted, actual, pred_class) %>%
                        dplyr::select(.estimate),
                      "auc" = yardstick::metrics(predicted, actual, .pred_1, estimate = pred_class) %>% 
                        filter(.metric == "roc_auc") %>% 
                        select(.estimate)) %>%
  tidyr::unnest(cols = c(accuracy, precision, recall, f_meas, auc))
metrics_tbl
# autoplot(roc_curve(predicted, actual, .pred_1))

roc_curve(predicted, actual, .pred_1) %>%
  mutate(.threshold = case_when(is.infinite(.threshold) ~ 0,
                                TRUE ~ as.numeric(.threshold)),
         specificity = 1 - specificity) %>% 
  arrange(specificity, sensitivity) %>% 
  ggplot(aes(x = specificity, y = sensitivity)) +
  geom_line(color = "red") +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = "longdash") +
  labs(x = "1 - specificity",
       subtitle = sprintf("AUC: %s", round(metrics_tbl$auc, 5))) + 
  theme_light() 

# LIME explaination
explanation <- lime(train %>% select(-survived), xgb_fit)
explanations <- explain(x = test[1:3,-1], 
                        explainer = explanation, 
                        labels = "1", 
                        n_permutations  = 5000,
                        dist_fun        = "manhattan",
                        kernel_width    = 3,
                        n_features = 7)

# Get an overview with the standard plot
plot_explanations(explanations)
plot_features(explanations, ncol = 2)

test_data_baked <- prepped %>% 
  bake(new_data = test_data)

submission_titanic %>% 
  mutate(Survived = predict(xgb_fit, test_data_baked, type = "class")$.pred_class) %>% 
  write_csv("06-modeling/submission_titanic_xgb_eng_xgb.csv")

# Save model
# saveRDS(object = xgb_fit, "model/xgb.rds")

############################ Cross Validation #############################

## v-fold cross validation
# folds <- vfold_cv(train_data, v = 5)

folded <- folds %>% 
  mutate(
    recipes = splits %>%
      # Prepper is a wrapper for `prep()` which handles `split` objects
      map(prepper, recipe = rec), 
    training_data = splits %>% map(analysis),
    xgb_fits = map2(
      recipes,
      training_data, 
      ~ fit(
        xgb_mod, 
        formula(.x), 
        data = bake(object = .x, new_data = .y)
      )
    )
  )

## Predict 
predict_rf <- function(split, rec, model) {
  test_set <- bake(rec, assessment(split))
  tibble(actual = test_set$survived) %>% 
    bind_cols(pred_class = predict(model, test_set, type = "class")$.pred_class) %>% 
    bind_cols(pred_prob = predict(model, test_set, type = "prob"))
}

predictions <- folded %>% 
  mutate(pred = list(
    splits,
    recipes,
    xgb_fits
  ) %>% 
    pmap(predict_rf)
  )

## Evaluate
eval <- predictions %>% 
  transmute(
    metrics = pred %>% map(~ metrics(., truth = actual, starts_with(".pred")[1], estimate = pred_class))
  ) %>% 
  unnest(metrics)

eval %>% knitr::kable()

eval %>% 
  group_by(.metric) %>% 
  summarise_at(.vars = vars(.estimate), .funs = list(min = min, median = median, mean = mean, sd = sd, max = max)) %>% 
  knitr::kable()


###########################################################################
#                                                                         #
#                            Ensemble Modeling                             #
#                                                                         #
###########################################################################

# Ensemble Model
ens_data_prep <- function(dataset, logreg, rf, xgb){
  dataset %>% 
    bind_cols(pred_logreg = predict(logreg, dataset, type = "class")$.pred_class) %>% 
    bind_cols(pred_rf = predict(rf, dataset, type = "class")$.pred_class) %>% 
    bind_cols(pred_xgb = predict(xgb, dataset, type = "class")$.pred_class)
}
ens_train <- ens_data_prep(train, logreg_fit, rf_fit, xgb_fit)

ens_test <- ens_data_prep(test, logreg_fit, rf_fit, xgb_fit)

## Model specification
logregkeras_mod <- logistic_reg(
  mode = "classification"
  ) %>%
  set_engine("keras")

## Fitting
logregkeras_fit <- fit(
  object = logregkeras_mod,
  formula = formula(prepped),
  data = ens_train
)

## Predicting!
predicted <- tibble(actual = ens_test$survived) %>% 
  bind_cols(pred_class = predict(logregkeras_fit, ens_test, type = "class")$.pred_class) %>% 
  bind_cols(pred_prob = predict(logregkeras_fit, ens_test, type = "prob"))

## Assessment -- test error
metrics(predicted, truth = actual, .pred_1, estimate = pred_class) %>% 
  knitr::kable()

conf_mat(predicted, truth = actual, estimate = pred_class)[[1]] %>% 
  as_tibble() %>% 
  mutate(pct = n/sum(n)) %>% 
  ggplot(aes(Prediction, Truth, alpha = n)) + 
  geom_tile(fill = custom_palette[4], show.legend = FALSE) +
  geom_text(aes(label = paste0(n, "\n", round(pct*100, 2), "%")), 
            color = "coral", alpha = 1, size = 6) +
  theme_minimal() +
  labs(
    title = "Confusion matrix"
  )

metrics_tbl <- tibble("accuracy" = yardstick::metrics(predicted, actual, pred_class) %>%
                        dplyr::select(-.estimator) %>%
                        dplyr::filter(.metric == "accuracy") %>% 
                        dplyr::select(.estimate),
                      "precision" = yardstick::precision(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "recall" = yardstick::recall(predicted, actual, pred_class) %>% 
                        dplyr::select(.estimate),
                      "f_meas" = yardstick::f_meas(predicted, actual, pred_class) %>%
                        dplyr::select(.estimate),
                      "auc" = yardstick::metrics(predicted, actual, .pred_1, estimate = pred_class) %>% 
                        filter(.metric == "roc_auc") %>% 
                        select(.estimate)) %>%
  tidyr::unnest(cols = c(accuracy, precision, recall, f_meas, auc))
metrics_tbl
# autoplot(roc_curve(predicted, actual, .pred_1))

roc_curve(predicted, actual, .pred_1) %>%
  mutate(.threshold = case_when(is.infinite(.threshold) ~ 0,
                                TRUE ~ as.numeric(.threshold)),
         specificity = 1 - specificity) %>% 
  arrange(specificity, sensitivity) %>% 
  ggplot(aes(x = specificity, y = sensitivity)) +
  geom_line(color = "red") +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = "longdash") +
  labs(x = "1 - specificity",
       subtitle = sprintf("AUC: %s", round(metrics_tbl$auc, 5))) + 
  theme_light() 

# LIME explaination
explanation <- lime(ens_train %>% select(-survived), logregkeras_fit)
explanations <- explain(x = ens_test[1:3,-1], 
                        explainer = explanation, 
                        labels = "1", 
                        n_permutations  = 5000,
                        dist_fun        = "manhattan",
                        kernel_width    = 3,
                        n_features = 7)

# Get an overview with the standard plot
plot_explanations(explanations)
plot_features(explanations, ncol = 2)

test_data_baked <- prepped %>% 
  bake(new_data = test_data) %>% 
  ens_data_prep(logreg_fit, rf_fit, xgb_fit)

submission_titanic %>% 
  mutate(Survived = predict(logregkeras_fit, test_data_baked, type = "class")$.pred_class) %>%
  write_csv("06-modeling/submission_titanic_logreg_eng_keras.csv")

# Save model
# saveRDS(object = xgb_fit, "model/xgb.rds")

############################ Cross Validation #############################

## v-fold cross validation
# folds <- vfold_cv(train_data, v = 5)

folded <- folds %>% 
  mutate(
    recipes = splits %>%
      # Prepper is a wrapper for `prep()` which handles `split` objects
      map(prepper, recipe = rec), 
    training_data = splits %>% map(analysis),
    xgb_fits = map2(
      recipes,
      training_data, 
      ~ fit(
        xgb_mod, 
        formula(.x), 
        data = bake(object = .x, new_data = .y)
      )
    )
  )

## Predict 
predict_rf <- function(split, rec, model) {
  test_set <- bake(rec, assessment(split))
  tibble(actual = test_set$survived) %>% 
    bind_cols(pred_class = predict(model, test_set, type = "class")$.pred_class) %>% 
    bind_cols(pred_prob = predict(model, test_set, type = "prob"))
}

predictions <- folded %>% 
  mutate(pred = list(
    splits,
    recipes,
    xgb_fits
  ) %>% 
    pmap(predict_rf)
  )

## Evaluate
eval <- predictions %>% 
  transmute(
    metrics = pred %>% map(~ metrics(., truth = actual, starts_with(".pred")[1], estimate = pred_class))
  ) %>% 
  unnest(metrics)

eval %>% knitr::kable()

eval %>% 
  group_by(.metric) %>% 
  summarise_at(.vars = vars(.estimate), .funs = list(min = min, median = median, mean = mean, sd = sd, max = max)) %>% 
  knitr::kable()
