# Cleaning the Survey on Consumer Expectations Labour Market Survey Supplement###
# Source: https://www.newyorkfed.org/microeconomics/sce/labor#/
# Codebook/questionnaire: https://www.newyorkfed.org/medialibrary/media/research/microeconomics/interactive/downloads/sce-labor-questionnaire.pdf?sc_lang=en
# rm(list = ls())
library(here)
library(tidyverse)
library(readxl)
library(lubridate)
library(stargazer)
library(lfe) # for regressions with clustering
library(weights)
library(diagis)
library(fixest) # For regressions with clustering and fixed effects
library(patchwork)
library(broom) # For extracting model coefficients
library(modelsummary)
library(flextable)
final = TRUE

var_names <- read_xlsx(here('data/behav_params/SCE Labour Market Survey/sce_labour_questionnaire_codebook.xlsx'))
sce_lab <- read_xlsx(here("data/behav_params/SCE Labour Market Survey/sce-labor-microdata-public.xlsx"), 
                     sheet = 3, skip = 1, col_types = var_names$type)
# sce_raw <- readRDS(paste0("data/behav_params/Mueller_Replication/sce_datafile_13_24_w_lab_survey.RDS")) 

# 1. Create glossary of the variable names
# sce_lab %>% 
#   names %>% 
#   data.frame("var" = .) %>% 
#  #write.xlsx(here('data/behav_params/SCE Labour Market Survey/sce_labour_questionnaire_codebook.xlsx'))

pull_name <- function(nm){
  var_names %>% 
    filter(var == nm) %>% 
    pull(short_var) %>% return(.)
}

sce_lab <- sce_lab %>% 
  rename_with(., .fn = pull_name) %>%
  mutate(sce_lab_survey = 1,
         date = ceiling_date(ym(date), 'month') - days(1)) 

# sce_full <- readRDS(here("data/behav_params/Mueller_Replication/sce_datafile_13_24.RDS")) 

# 3191 cases are matched in the original file - ie. 3191 unemployed people!
# sce_13_24 %>% 
#   select(userid, date) %>% 
#   arrange(date, userid) %>% 
#   semi_join(select(arrange(sce_lab, date, userid), userid, date), ., by = c("userid", "date")) %>% nrow(.) == nrow(sce_lab)
# 
# sce_full %>% 
#   select(userid, date) %>% 
#   arrange(date, userid) %>% 
#   anti_join(sce_lab, ., by = c("userid", "date")) 

# Read in raw SCE files which now include all observations
# source(here(paste0("data/behav_params/Mueller_Replication/mueller_repl_sce_raw_data_cleaning.R")))
# rm(sce_13_19, sce_13_19_same_t, sce_20_24)
# saveRDS(sce_13_24, here("data/behav_params/SCE Labour Market Survey/sce_13_24_raw.rds"))
sce_13_24 <- readRDS(here("data/behav_params/SCE Labour Market Survey/sce_13_24_raw.rds"))
sce_13_24_raw <- sce_13_24
sce_13_24 <- sce_13_24_raw %>% 
   select(-names(sce_lab)[!(names(sce_lab) %in% c('userid', 'date'))])

sce_13_24 %>%
  select(userid, date) %>%
  arrange(date, userid) %>%
  semi_join(select(arrange(sce_lab, date, userid), userid, date), ., 
            by = c("userid", "date")) %>% nrow(.) == nrow(sce_lab)


################################################################################
transform_fun <- function(df){
  df <- df  %>% 
  #readRDS(here('data/behav_params/Mueller_Replication/sce_datafile_13_24_w_lab_survey_new.RDS'))%>% 
  rename(reservation_wage_orig = reservation_wage) %>% 
  mutate(year = year(date), 
         month = month(date),
         never_worked = ifelse(is.na(never_worked), 0 , 1),
                   #   agesq = age^2,
         self_employed = case_when(self_employed == 1 ~ 0,
                                   self_employed == 2 ~ 1, 
                                   TRUE ~ NA),
         looked_for_work_l4_wks = case_when(looked_for_work_l4_wks == 1 ~ 1,
                                            looked_for_work_l4_wks == 2 ~ 0, 
                                            TRUE ~ NA),
         across(c(reservation_wage_orig, wage_most_recent_job, current_wage_annual), ~as.integer(.)),
         reservation_wage_unit_scale = case_when(reservation_wage_unit == 1 ~ 2080, # Hourly - calculated 40 hours x 52 weeks # & nchar(reservation_wage_orig) %in% c(2,3) 
                                                 reservation_wage_unit == 2 ~ 52, # Weekly- 52 weeks # & nchar(reservation_wage_orig) %in% c(3,4)
                                                 reservation_wage_unit == 3 ~ 26, # Bi-weekly ~ 52/2 # & nchar(reservation_wage_orig) %in% c(4)
                                                 reservation_wage_unit == 4 ~ 12, # Monthly - 12 # & nchar(reservation_wage_orig) %in% c(4,5)
                                                 reservation_wage_unit == 5  ~ 1, # Annual - 1 #& nchar(reservation_wage_orig) >= 5
                                                 date >= "2017-03-01" ~ 1),  
         # Can somewhat safely assume that single-digit or two-digit reservation wage is a reported hourly wage
         reservation_wage_unit_scale = ifelse(is.na(reservation_wage_unit) & nchar(reservation_wage_orig) %in% c(1,2), 2080, reservation_wage_unit_scale),
         reservation_wage_unit_scale = ifelse(nchar(reservation_wage_orig) >= 5, 1, reservation_wage_unit_scale),
         reservation_wage = reservation_wage_orig * reservation_wage_unit_scale, 
         # Reservation wage is sometimes filled in as 0 - will replace with NA values
         reservation_wage = ifelse(reservation_wage == 0, NA, reservation_wage), 
         current_wage_annual_cat = case_when(current_wage_annual < 10000 ~ 1, # tested: sce_lab %>% filter(!is.na(current_wage_annual) & !is.na(current_wage_annual_cat)) %>% nrow(.) == 0
                                             current_wage_annual  >= 10000 & current_wage_annual <= 19999 ~ 2,
                                             current_wage_annual  >= 20000 & current_wage_annual <= 29999 ~ 3,
                                             current_wage_annual  >= 30000 & current_wage_annual <= 39999 ~ 4,
                                             current_wage_annual  >= 40000 & current_wage_annual <= 49999 ~ 5,
                                             current_wage_annual  >= 50000 & current_wage_annual <= 59999 ~ 6,
                                             current_wage_annual  >= 60000 & current_wage_annual <= 74999 ~ 7,
                                             current_wage_annual  >= 75000 & current_wage_annual <= 99999 ~ 8,
                                             current_wage_annual  >= 100000 & current_wage_annual <= 149999 ~ 9,
                                             current_wage_annual  >= 150000 ~ 10,
                                             TRUE ~ current_wage_annual_cat),
         current_wage_annual = case_when(is.na(current_wage_annual) & current_wage_annual_cat == 1 ~ 10000, # If I only have the category (happens in 144 observations), I replace with the mean if a range is provided or the upper/lower range in the bounding categories, respectively.
                                         is.na(current_wage_annual) & current_wage_annual_cat == 2 ~ 15000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 3 ~ 25000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 4 ~ 35000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 5 ~ 45000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 6 ~ 55000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 7 ~ 67500,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 8 ~ 87500,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 9 ~ 125000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 10 ~ 150000,
                                         TRUE ~ current_wage_annual),
         wage_most_recent_job = ifelse(wage_most_recent_job == 0, NA, wage_most_recent_job),
         current_wage_annual = ifelse(current_wage_annual == 0, NA, current_wage_annual)) %>% 
  rename(wage_most_recent_job_orig = wage_most_recent_job,
         current_wage_annual_orig = current_wage_annual) %>% 
         mutate(wage_most_recent_job = case_when(nchar(wage_most_recent_job_orig) <= 2 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 2080,
                                                   nchar(wage_most_recent_job_orig) == 4 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 12, 
                                                   nchar(wage_most_recent_job_orig) == 3 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 52, 
                                                   TRUE ~ wage_most_recent_job_orig),
         current_wage_annual = case_when(nchar(current_wage_annual_orig) <= 2 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 2080,
                                                        nchar(current_wage_annual_orig) == 4 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 12, 
                                                        nchar(current_wage_annual_orig) == 3 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 52, 
                                                        TRUE ~ current_wage_annual_orig),
         expbest4mos_rel_res = exp_salary_best_offer_4mos/reservation_wage, 
         expbest4mos_rel_current = exp_salary_best_offer_4mos/current_wage_annual, 
         expbest4mos_rel_most_recent = exp_salary_best_offer_4mos/wage_most_recent_job,
         res_wage_to_current = reservation_wage/current_wage_annual,
         res_wage_to_latest = reservation_wage/wage_most_recent_job) %>%
    filter(age >= 20 & age <= 65) %>% 
    mutate(accepted_salary_1 = job_offer_1_salary*(job_offer_1_accepted %in% c(1,2)),
           accepted_salary_2 = job_offer_2_salary*(job_offer_2_accepted %in% c(1,2)),
           accepted_salary_3 = job_offer_3_salary*(job_offer_3_accepted %in% c(1,2))) %>%
   rowwise %>%
   mutate(accepted_salary = max(accepted_salary_1, accepted_salary_2, accepted_salary_3, na.rm = TRUE),
          reservation_wage_latest = max(reservation_wage, na.rm = TRUE)) %>%
   ungroup %>%
           mutate(accepted_salary = ifelse(accepted_salary == -Inf, NA, accepted_salary),
                  accepted_salary = ifelse(accepted_salary == 0, NA, accepted_salary),
           reservation_wage_latest = ifelse(reservation_wage_latest == -Inf, NA, reservation_wage_latest),
           salary_prop_reswage = accepted_salary/reservation_wage_latest,
           salary_to_latest = accepted_salary/wage_most_recent_job) %>%
 # Exclude anyone who sets a reservation wage below the minimum annual salary or above 1million USD
  filter(reservation_wage >= 14000 & reservation_wage < 1000000) %>% 
   filter(!(res_wage_to_latest > 2 & is.na(res_wage_to_current))) %>% 
   filter(!(res_wage_to_current > 2 & is.na(res_wage_to_latest))) %>% 
   # exclude part-time workers
   filter(working_pt == 0)
 return(df)
}

# Define base directory (replace this with the correct path later)
base <- here("data/behav_params/Mueller_Replication/")

# Load data file
data_13_24 <- #readRDS(paste0(base, "sce_datafile_13_24_w_lab_survey.RDS")) %>% 
  sce_lab %>% 
  left_join(., sce_13_24, by = c("userid", "date")) %>% 
  rename(weight = rim_4_original) %>% 
  transform_fun(.)

if(!final){
  # 10,629 people are interviewed more than once!
  panel_ids <- sce_lab %>% select(userid) %>% group_by(userid) %>% mutate(n = n()) %>% filter(n != 1) %>% pull(userid) %>% unique
  # 1,119 observations are in the unemployed sample in sce
  sce_13_24 %>% group_by(userid) %>% filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
    pull(userid) %>% unique %>% intersect(panel_ids, .) %>% length
  # 9,520 observations are not in the unemployed sample in sce
  sce_13_24 %>% group_by(userid) %>% filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
    pull(userid) %>% unique %>% setdiff(panel_ids, .) %>% length
}

unemp_only <- readRDS(here('data/behav_params/Mueller_Replication/sce_datafile_13_24_w_lab_survey_new.RDS')) %>% 
  transform_fun(.)


