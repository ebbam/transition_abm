# Cleaning the Survey on Consumer Expectations Supplement###

library(here)
library(tidyverse)
library(readxl)
library(ggformula)
library(lubridate)

sce_job_search <- read_xlsx(here("data/behav_params/SCE Job Search Survey/SCE-Public-LM-Quarterly-Microdata.xlsx"), sheet = 3) %>% 
  select(year, userid, responseid, survey_weight,
         l7_days_spent_searching, 
         l7b_looked_for_work_last8wks,
         l8_months_no_work) %>% 
  filter(!is.na(l7_days_spent_searching) & !is.na(l8_months_no_work) & l8_months_no_work != 999995) %>% 
  mutate(l8_days_no_work = l8_months_no_work*30.5,
         l7_days_spent_searching_transformed = ifelse(l8_days_no_work >= l7_days_spent_searching, l7_days_spent_searching, l8_days_no_work))
  