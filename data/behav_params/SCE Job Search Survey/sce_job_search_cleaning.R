# Cleaning the Survey on Consumer Expectations Supplement###

library(here)
library(tidyverse)
library(readxl)
library(ggformula)
library(lubridate)

sce_job_search_full <- read_xlsx(here("data/behav_params/SCE Job Search Survey/SCE-Public-LM-Quarterly-Microdata.xlsx"), sheet = 3) 

sce_job_search <- sce_job_search_full %>% 
  select(year, userid, responseid, survey_weight,
         l1a_lfs_rc,
         l1_lfs_rc,
         l7_days_spent_searching,
         l7b_looked_for_work_last8wks,
         l8_months_no_work) %>%
  # Include only those that are currently unemployed and exclude those that have never been employed
  filter(l1a_lfs_rc == 3 & l8_months_no_work != 999995) %>%
  mutate(l8_days_no_work = l8_months_no_work*30.5,
         l7_days_spent_searching_transformed = ifelse(l8_days_no_work >= l7_days_spent_searching, l7_days_spent_searching, l8_days_no_work))  


sce_job_search_emp <- sce_job_search_full %>% 
  select(year, userid, responseid, survey_weight,
         l1a_lfs_rc,
         l1_lfs_rc,
         l6_emp_looked_for_work,
         l7_days_spent_searching,
         l7b_looked_for_work_last8wks,
         l8_months_no_work) %>%
  # Include only those that are currently employed
  filter(l1a_lfs_rc %in% c(1,2,3) & l7_days_spent_searching < 2000) %>% 
  mutate(l1a_text = case_when(l1a_lfs_rc == 1 ~ "FT",
                              l1a_lfs_rc == 2 ~ "PT",
                              l1a_lfs_rc == 3 ~ "UE"))


