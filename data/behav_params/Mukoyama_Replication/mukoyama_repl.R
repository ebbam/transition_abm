library(tidyverse)
library(here)
library(haven)
library(janitor)


read_dta(here("data/behav_params/Mukoyama_Replication/final_data/full_CPS_data.dta"))       

# Load necessary libraries
library(dplyr)
library(tidyr)
library(lme4)
library(mice)
library(lubridate)
library(haven)
library(here)

# Set seed for reproducibility
set.seed(12345)

# Step 1: Load the CPS data
cps_data <- read_dta(here("data/behav_params/Mukoyama_Replication/final_data/full_CPS_data_no_time.dta"))  # Replace with your CPS dataset path

# Step 2: Clean the CPS data (filter out unemployed and specific conditions)
cps_data_clean <- cps_data %>%
  filter(mlr == 4) %>%  # Only keep unemployed
  mutate(age2 = age^2,
         age3 = age^3,
         age4 = age^4) %>%
  select(-c(nlfwant, ernhr, ernwk, ind, whenlj, lfs, dwwant, prernhly, statenum))

# Step 3: Load and clean the auxiliary datasets (JOLTS, Barnichon, HWOL, etc.)
jolts_data <- read.csv(here("data/behav_params/Mukoyama_Replication/int_data/theta/JOLTS_agg_theta_monthly.dta"))  # Replace with JOLTS data path
barnichon_data <- read.csv(here("data/behav_params/Mukoyama_Replication/int_data/theta/BARNICHON_agg_theta_monthly.dta"))  # Replace with Barnichon data path
hwol_data <- read.csv(here("data/behav_params/Mukoyama_Replication/int_data/theta/HWOL_state_theta_monthly.dta"))  # Replace with HWOL data path
sp500_data <- read.csv(here("data/behav_params/Mukoyama_Replication/raw_data/other/sp_500.dta"))  # Replace with SP500 data path
houseprice_data <- read.csv(here("data/behav_params/Mukoyama_Replication/raw_data/other/houseprice_index.dta"))  # Replace with house price data path
payroll_data <- read.csv(here("data/behav_params/Mukoyama_Replication/raw_data/other/payroll_employment.dta"))  # Replace with payroll data path

# Merge all the additional datasets with the CPS data
cps_data_clean <- cps_data_clean %>%
  left_join(jolts_data, by = c("year", "month")) %>%
  left_join(barnichon_data, by = c("year", "month")) %>%
  left_join(hwol_data, by = c("year", "month", "statefips")) %>%
  left_join(sp500_data, by = c("year", "month")) %>%
  left_join(houseprice_data, by = c("year", "month")) %>%
  left_join(payroll_data, by = c("year", "month"))

# Step 4: Prepare for multiple imputation
# List of variables to include in imputation
variables_for_imputation <- c("age", "age2", "age3", "age4", "female", "hs", "somecol", "college", "black", "married", "marriedfemale")

# Impute missing values using mice
imputed_data <- mice(cps_data_clean[, variables_for_imputation], method = "pmm", m = 5, seed = 123)

# Step 5: Imputation of the minutes spent on job search (example variable `time_less8`)
# Use imputed data to predict job search minutes for each draw
imputed_results <- complete(imputed_data, "long", include = TRUE)

# Step 6: Fit regression models using the imputed data
# Regression model 1: Model for probability of searching
search_model <- glm(dummy_search ~ age + age2 + age3 + age4 + female + hs + somecol + college + black + married + marriedfemale,
                    data = imputed_results, family = binomial)

# Store the predicted probabilities
imputed_results$prob_search <- predict(search_model, type = "response")

# Regression model 2: Model for time spent on search (given that searching > 0)
time_model <- lm(time_less8 ~ age + age2 + age3 + age4 + female + hs + somecol + college + black + married + marriedfemale,
                 data = imputed_results)

# Store the residuals
imputed_results$residuals_search <- residuals(time_model)

# Step 7: Apply the imputation to the original dataset
# Impute the missing time spent on job search using the imputation results
cps_data_clean$imputed_time_search <- predict(time_model, newdata = cps_data_clean)

# Step 8: Combine results for analysis
# After imputation and modeling, the dataset will include imputed time for job search and other necessary features.

# Final dataset for analysis
final_data <- cps_data_clean %>%
  select(age, age2, age3, age4, female, hs, somecol, college, black, married, marriedfemale, imputed_time_search)

# Example: Fit a regression model using the final dataset
final_model <- lm(imputed_time_search ~ age + age2 + age3 + age4 + female + hs + somecol + college + black + married + marriedfemale, 
                  data = final_data)

# Display the results of the final model
summary(final_model)
