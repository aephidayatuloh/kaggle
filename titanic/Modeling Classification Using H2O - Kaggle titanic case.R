################################################################################
#                                                                              #
# Purpose:       Titanic Data Classification                                   #
#                                                                              #
# Author:        aephidayatuloh                                                #
# Contact:       aephidayatuloh.mail@gmail.com                                 #
# Client:        NDSC 2020 Shopee                                              #
#                                                                              #
# Code created:  2020-03-05                                                    #
# Last updated:  2020-03-05                                                    #
#                                                                              #
# Comment:       Titanic Data Classification Using H2O and R                   #
#                                                                              #
#                                                                              #
################################################################################

library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16G")
# h2o.removeAll() ## clean slate - just in case the cluster was already running


################################################################################
#                                                                              #
#                                 Import Data                                  #
#                                                                              #
################################################################################

training_titanic <- h2o.importFile("06-modeling/train.csv")
testing_titanic <- h2o.importFile("06-modeling/test.csv")
submission_titanic <- h2o.importFile("06-modeling/gender_submission.csv")

################################################################################
#                                                                              #
#                         Exploration & Data Processing                        #
#                                                                              #
################################################################################

library(tidyverse)
library(skimr)
library(janitor)
skim(training_titanic)
skim(testing_titanic)

preproc <- function(dataset, outcome = NULL, level = NULL){
  d1 <- dataset %>% 
    as.data.frame() %>%
    mutate(Sex = as.character(Sex),
           Title = str_remove_all(Name, '(.*, )|(\\..*)'),
           Title = case_when(Title %in% c("Ms", "Mlle") ~ "Miss",
                             Title ==  "Mme" ~ "Mrs",
                             Title %in% c("Master", "Miss", "Mr", "Mrs") ~ as.character(Title),
                             TRUE ~ "Rare Title"),
           FamilyName = str_sub(Name, 1, str_locate(Name, ",")[,2] - 1),
           FamilySize = SibSp + Parch + 1,
           FamilySizeCategory = case_when(FamilySize == 1 ~ "singleton",
                                            between(FamilySize, 1, 5) ~ "small",
                                          FamilySize > 5 ~ "large"),
           Deck = str_sub(Cabin, 1, 1),
           Deck = case_when(is.na(Deck) ~ "Regular",
                            TRUE ~ as.character(Deck)),
           Embarked = case_when(is.na(Embarked) ~ "C",
                                TRUE ~ as.character(Embarked)),
           Fare = case_when(is.na(Fare) & Pclass == 1 & Embarked == "C" ~ 78.3,
                            is.na(Fare) & Pclass == 1 & Embarked == "Q" ~ 90,
                            is.na(Fare) & Pclass == 1 & Embarked == "S" ~ 52,
                            is.na(Fare) & Pclass == 2 & Embarked == "C" ~ 24,
                            is.na(Fare) & Pclass == 2 & Embarked == "Q" ~ 12.4,
                            is.na(Fare) & Pclass == 2 & Embarked == "S" ~ 13.5,
                            is.na(Fare) & Pclass == 3 & Embarked == "C" ~ 7.90,
                            is.na(Fare) & Pclass == 3 & Embarked == "Q" ~ 7.75,
                            is.na(Fare) & Pclass == 3 & Embarked == "S" ~ 8.05,
                            TRUE ~ Fare),
           Age = case_when(is.na(Age) ~ median(Age, na.rm = TRUE),
                           TRUE ~ as.numeric(Age))
    )
  
  if(is.null(outcome)){
    return(d1 %>% 
             select(PassengerId, Sex, Age, Fare, Embarked, Title, FamilyName, FamilySizeCategory, Deck) %>% 
             as.h2o())
  } else if(!is.null(outcome) & outcome %in% names(d1)){
    d1[[outcome]] <- factor(d1[[outcome]], levels = level)
    return(d1 %>% 
             select(PassengerId, Survived, Sex, Age, Fare, Embarked, Title, FamilyName, FamilySizeCategory, Deck) %>% 
             as.h2o())
  }
}

train_data <- preproc(training_titanic, outcome = "Survived", level = c(0,1))
test_data <- preproc(testing_titanic)
str(train_data)
str(test_data)

titanic_split <- h2o.splitFrame(1:nrow(train_data) %>% as.h2o(), ratios = 0.8, 
                                destination_frames = c("train", "valid"), seed = 1001)
names(titanic_split) <- c("Train", "Valid")
train_titanic_data <- train_data[as.vector(titanic_split$Train),-1]
valid_titanic_data <- train_data[as.vector(titanic_split$Valid),-1]
test_titanic_data <- test_data


################################################################################
#                                                                              #
#                           Build Model : H2O AutoML                           #
#                                                                              #
################################################################################
# H2O AutoML
aml.titanic <- h2o.automl(x = names(train_titanic_data), y = "Survived", 
                           training_frame = train_titanic_data, 
                           validation_frame = valid_titanic_data,
                           nfolds = 5, max_models = 10, max_runtime_secs = 3600, seed = 1001)
# h2o.saveModel(aml.titanic@leader, "./models", force = TRUE)
# aml_leader <- h2o.loadModel("./models/GBM_3_AutoML_20191003_211337")
h2o.performance(aml.titanic@leader, train = TRUE) # get confusion matrix in the training data
# MSE: (Extract with `h2o.mse`) 0.003369673
# RMSE: (Extract with `h2o.rmse`) 0.05804889
# Logloss: (Extract with `h2o.logloss`) 0.01714247
# Mean Per-Class Error: 0.00423002
# R^2: (Extract with `h2o.r2`) 0.9999664
h2o.performance(aml.titanic@leader, valid = TRUE)  # get confusion matrix in the validation data
# MSE: (Extract with `h2o.mse`) 0.07809565
# RMSE: (Extract with `h2o.rmse`) 0.279456
# Logloss: (Extract with `h2o.logloss`) 0.3970213
# Mean Per-Class Error: 0.292924
# R^2: (Extract with `h2o.r2`) 0.9991995

test.perf <- h2o.predict(aml.titanic@leader, valid_titanic_data)$predict
test.data <- valid_titanic_data

# Accuracy Test Dataset
mean(test.data$Survived == test.perf$predict)
# [1] 0.4028832
mean(test.data$Survived == test.perf$predict_modif)
# [1] 0.4044006
# aml_leader <- h2o.saveModel(aml.titanic@leader, path = "./models", force = TRUE)

################################################################################
#                                                                              #
#                    Build Model : Gradient Boosting Method                    #
#                                                                              #
################################################################################

cat("Build a basic GBM model\n")
gbm.titanic <- h2o.gbm(model_id = "gbm_titanic", 
                        y = "Survived",
                        x = names(train_titanic_data),
                        training_frame = train_titanic_data, 
                        validation_frame = valid_titanic_data, 
                        nfolds = 5, ntrees = 63, max_depth = 8, 
                        seed = 1001)

h2o.performance(gbm.titanic, train = TRUE)

h2o.performance(gbm.titanic, valid = TRUE)

test.perf <- h2o.predict(gbm.titanic, valid_titanic_data)$predict
test.data <- valid_titanic_data

# Accuracy Test Dataset
mean(test.data$Survived == test.perf$predict)

################################################################################
#                                                                              #
#              Build Model : Multinomial Generalized Linear Model              #
#                                                                              #
################################################################################
# GLM Multinomial
nbm.titanic <- h2o.naiveBayes(model_id = "nbm_titanic", 
                              y = "Survived",
                              x = names(train_titanic_data),
                              training_frame = train_titanic_data, 
                              validation_frame = valid_titanic_data, 
                               nfolds = 5, laplace = 3, 
                               seed = 1001)
# h2o.performance(nbm.titanic, train = TRUE) # get confusion matrix in the training data
# # MSE: (Extract with `h2o.mse`) 0.5255349
# # RMSE: (Extract with `h2o.rmse`) 0.7249379
# # Logloss: (Extract with `h2o.logloss`) 1.796833
# # Mean Per-Class Error: 0.9487179
# # Null Deviance: (Extract with `h2o.nulldeviance`) 37880.84
# # Residual Deviance: (Extract with `h2o.residual_deviance`) 37880.84
# # R^2: (Extract with `h2o.r2`) 0.9947407
# h2o.performance(nbm.titanic, valid = TRUE)  # get confusion matrix in the validation data
# # MSE: (Extract with `h2o.mse`) 0.5199741
# # RMSE: (Extract with `h2o.rmse`) 0.7210923
# # Logloss: (Extract with `h2o.logloss`) 1.769447
# # Mean Per-Class Error: 0.6923077
# # Null Deviance: (Extract with `h2o.nulldeviance`) 4448.39
# # Residual Deviance: (Extract with `h2o.residual_deviance`) 4448.39
# # R^2: (Extract with `h2o.r2`) 0.9946242
# 
# test.perf <- h2o.predict(nbm.titanic, data.split$Test)$predict
# test.data <- test.text.data
# test.data <- test.data %>% 
#   mutate(predict = as.vector(test.perf),#as.character(),
#          predict_modif = 
#            case_when(
#              str_detect(str_to_lower(DESCRIPTION), "lipstick") | str_detect(str_to_lower(DESCRIPTION), "lipstik") ~ "Banded Pack",
#              TRUE ~ as.character(predict))
#   )
# # Accuracy Test Dataset
# mean(test.data$CATEGORY == test.data$predict)
# # [1] 0.5197269
# mean(test.data$CATEGORY == test.data$predict_modif)
# # [1] 0.5212443

################################################################################
#                                                                              #
#                    Build Model : Xtreme Gradient Boosting                    #
#                                                                              #
################################################################################
# H2O XGBoost
xgb.titanic <- h2o.xgboost(model_id = "xgb_titanic",
                           y = "Survived",
                           x = names(train_titanic_data),
                           training_frame = train_titanic_data, 
                           validation_frame = valid_titanic_data, 
                            nfolds = 5, 
                            seed = 1001)
# h2o.performance(xgb.titanic, train = TRUE) # get confusion matrix in the training data
# # MSE: (Extract with `h2o.mse`) 0.003256575
# # RMSE: (Extract with `h2o.rmse`) 0.05706641
# # Logloss: (Extract with `h2o.logloss`) 0.01584924
# # Mean Per-Class Error: 0.005965252
# h2o.performance(xgb.titanic, valid = TRUE)  # get confusion matrix in the validation data
# # MSE: (Extract with `h2o.mse`) 0.07031497
# # RMSE: (Extract with `h2o.rmse`) 0.2651697
# # Logloss: (Extract with `h2o.logloss`) 0.3095258
# # Mean Per-Class Error: 0.2307275
# 
# test.perf <- h2o.predict(xgb.titanic, data.split$Test)$predict
# test.data <- test.text.data
# test.data <- test.data %>% 
#   mutate(predict = as.vector(test.perf),#as.character(),
#          predict_modif = 
#            case_when(
#              str_detect(str_to_lower(DESCRIPTION), "lipstick") | str_detect(str_to_lower(DESCRIPTION), "lipstik") ~ "Banded Pack",
#              TRUE ~ as.character(predict))
#   )
# # writexl::write_xlsx(test.data, "test data predicted.xlsx")
# # Accuracy Test Dataset
# mean(test.data$CATEGORY == test.data$predict)
# # [1] 0.9241275
# mean(test.data$CATEGORY == test.data$predict_modif)
# # [1] 0.9241275

################################################################################
#                                                                              #
#                 Build Model : Deep Learning (Neural Network)                 #
#                                                                              #
################################################################################
# H2O Deep Learning (Neural Network)
dlm.titanic <- h2o.deeplearning(model_id = "dlm_titanic",
                                y = "Survived",
                                x = names(train_titanic_data),
                                training_frame = train_titanic_data, 
                                validation_frame = valid_titanic_data, 
                                 nfolds = 5, 
                                 seed = 1001)
# h2o.performance(dlm.titanic, train = TRUE) # get confusion matrix in the training data
# # MSE: (Extract with `h2o.mse`) 0.109942
# # RMSE: (Extract with `h2o.rmse`) 0.331575
# # Logloss: (Extract with `h2o.logloss`) 0.5011667
# # Mean Per-Class Error: 0.1618064
# h2o.performance(dlm.titanic, valid = TRUE)  # get confusion matrix in the validation data
# # MSE: (Extract with `h2o.mse`) 0.1317676
# # RMSE: (Extract with `h2o.rmse`) 0.362998
# # Logloss: (Extract with `h2o.logloss`) 0.7355825
# # Mean Per-Class Error: 0.1486196
# 
# test.perf <- h2o.predict(dlm.titanic, data.split$Test)$predict
# test.data <- test.text.data
# test.data <- test.data %>% 
#   mutate(predict = as.vector(test.perf),#as.character(),
#          predict_modif = 
#            case_when(
#              str_detect(str_to_lower(DESCRIPTION), "lipstick") | str_detect(str_to_lower(DESCRIPTION), "lipstik") ~ "Banded Pack",
#              TRUE ~ as.character(predict))
#   )
# # Accuracy Test Dataset
# mean(test.data$CATEGORY == test.data$predict)
# # [1] 0.8467375
# mean(test.data$CATEGORY == test.data$predict_modif)
# # [1] 0.8467375

################################################################################
#                                                                              #
#                         Build Model : Random Forest                          #
#                                                                              #
################################################################################

drf.titanic <- h2o.randomForest(model_id = "drf_titanic",
                                y = "Survived",
                                x = names(train_titanic_data),
                                training_frame = train_titanic_data, 
                                validation_frame = valid_titanic_data, 
                                 nfolds = 5, ntrees = 500, 
                                 seed = 1001)
# h2o.performance(drf.titanic, train = TRUE) # get confusion matrix in the training data
# # MSE: (Extract with `h2o.mse`) 0.09060449
# # RMSE: (Extract with `h2o.rmse`) 0.3010058
# # Logloss: (Extract with `h2o.logloss`) 0.3704996
# # Mean Per-Class Error: 0.484957
# # R^2: (Extract with `h2o.r2`) 0.9990933
# h2o.performance(drf.titanic, valid = TRUE)  # get confusion matrix in the validation data
# # MSE: (Extract with `h2o.mse`) 0.09133326
# # RMSE: (Extract with `h2o.rmse`) 0.3022139
# # Logloss: (Extract with `h2o.logloss`) 0.3540979
# # Mean Per-Class Error: 0.2176975
# # R^2: (Extract with `h2o.r2`) 0.9990557
# 
# test.perf <- h2o.predict(drf.titanic, data.split$Test)$predict
# test.data <- test.text.data
# test.data <- test.data %>% 
#   mutate(predict = as.vector(test.perf),#as.character(),
#          predict_modif = 
#            case_when(
#              str_detect(str_to_lower(DESCRIPTION), "lipstick") | str_detect(str_to_lower(DESCRIPTION), "lipstik") ~ "Banded Pack",
#              TRUE ~ as.character(predict))
#   )
# # Accuracy Test Dataset
# mean(test.data$CATEGORY == test.data$predict)
# # [1] 0.9241275
# mean(test.data$CATEGORY == test.data$predict_modif)
# # [1] 0.9241275


# # Save best model
h2o.saveModel(gbm.titanic, path = "./models", force = TRUE) # define your path here
h2o.saveModel(nbm.titanic, path = "./models", force = TRUE) # define your path here
h2o.saveModel(dlm.titanic, path = "./models", force = TRUE) # define your path here
h2o.saveModel(drf.titanic, path = "./models", force = TRUE) # define your path here
h2o.saveModel(xgb.titanic, path = "./models", force = TRUE) # define your path here


################################################################################
#                                                                              #
#                         Build Model : Ensemble Model                         #
#                                                                              #
################################################################################

test.data <- test.text.data
test.data$pred_gbm <- as.vector(unlist(h2o.predict(gbm.titanic, data.split$Test)$predict))
test.data$pred_nbm <- as.vector(unlist(h2o.predict(nbm.titanic, data.split$Test)$predict))
test.data$pred_dlm <- as.vector(unlist(h2o.predict(dlm.titanic, data.split$Test)$predict))
test.data$pred_drf <- as.vector(unlist(h2o.predict(drf.titanic, data.split$Test)$predict))
test.data$pred_xgb <- as.vector(unlist(h2o.predict(xgb.titanic, data.split$Test)$predict))

test.data$predict_final <- apply(test.data[,c("pred_gbm", "pred_nbm", "pred_dlm", "pred_drf", "pred_xgb")], 1, function(x)names(sort(table(unlist(x)), decreasing = TRUE))[1])
mean(ifelse(test.data$CATEGORY == test.data$pred_gbm, 1, 0)) # Accuracy GBM ~0.90
mean(ifelse(test.data$CATEGORY == test.data$pred_nbm, 1, 0)) # Accuracy Naive Bayes ~0.85
mean(ifelse(test.data$CATEGORY == test.data$pred_dlm, 1, 0)) # Accuracy Deep Learning ~0.85
mean(ifelse(test.data$CATEGORY == test.data$pred_drf, 1, 0)) # Accuracy Disributed Rand. Forest ~0.918
mean(ifelse(test.data$CATEGORY == test.data$pred_xgb, 1, 0)) # Accuracy XGBoost ~0.913
mean(ifelse(test.data$CATEGORY == test.data$predict_final, 1, 0)) # Accuracy Ensemble Model ~0.925
writexl::write_xlsx(test.data, "test_data.xlsx")
