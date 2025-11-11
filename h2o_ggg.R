library(tidyverse)
library(patchwork)
library(ggplot2)
library(tidymodels)
library(vroom)
library(GGally)
library(rpart)
library(glmnet)
library(bonsai)
library(lightgbm)
library(agua)
library(h2o)
library(kknn)
library(discrim)
library(kernlab)
library(naivebayes)
library(themis)
library(beepr)

# Call Java and H2O ai
Sys.setenv(JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-25.0.0.36-hotspot")
h2o::h2o.init()

# Import train and test data
trainData <- vroom("GitHub/GhostsGhoulsGoblins/train.csv")
testData <- vroom("GitHub/GhostsGhoulsGoblins/test.csv")
trainData$type <- as.factor(trainData$type)

# Recipe
ggg_recipe <- recipe(type ~ bone_length + rotting_flesh + hair_length + has_soul + color,  
                     data = trainData) %>%
  step_dummy(all_nominal_predictors()) %>%  
  step_normalize(all_numeric_predictors())
  
# --- Define H2O AutoML model ---
auto_model <- 
  auto_ml() %>%
  set_engine("h2o",
             max_runtime_secs = 300,   # run for up to 30 minutes
             max_models = 50,
             seed = 17,
             stopping_metric = "logloss") %>%  # "auc" isn’t ideal for multiclass
  set_mode("classification")

# --- Workflow ---
automl_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(auto_model)

# --- Fit model ---
final_fit <- automl_wf %>%
  fit(data = trainData)

# --- Predict on test data ---
predictions <- predict(final_fit, new_data = testData)

# --- Prepare submission ---
kaggle_submission <- bind_cols(testData, predictions) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

# --- Write to CSV ---
vroom_write(kaggle_submission,
            file = "GitHub/GhostsGhoulsGoblins/H2O_GGG.csv",
            delim = ",")
