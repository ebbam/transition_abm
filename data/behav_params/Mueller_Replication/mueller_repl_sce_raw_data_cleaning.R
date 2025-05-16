## Mueller et al. Job Seekers' Perceptions and Employment Prospects: Heterogeneity, Duration Dependence and Bias
#[Mueller et al: Job Seekers' Perceptions and Employment Prospects](https://www.aeaweb.org/articles?id=10.1257/aer.20190808)

# Data downloaded directly from: https://www.newyorkfed.org/microeconomics/sce#/
# Survey of Consumer Expectations
# Codebook: https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/FRBNY-SCE-Survey-Core-Module-Public-Questionnaire.pdf?sc_lang=en


# Translating stata cleaning code to R such that one can easily process new data in R!

library(here)
library(tidyverse)
library(assertthat)
library(haven)   # For reading and writing .dta files
library(dplyr)   # For data manipulation
library(lubridate) # For date handling
library(openxlsx)


# Set base path
base <- here("data/behav_params/Mueller_Replication/120501-V1/MST/EMPIRICAL_ANALYSIS/Codes_and_Data/SCE/")

# Load data
sce_raw <- read_dta(paste0(base, "Raw_data/sce.dta"))
cleaned_sce <- read_dta(paste0(base, "sce_datafile.dta"))

#### New Data
# Just need to find the 36 variables that are necessary
raw_1316 <- read.xlsx(here("data/behav_params/Mueller_Replication/new_data/FRBNY-SCE-Public-Microdata-Complete-13-16.xlsx"), startRow = 2, colNames = TRUE, detectDates = TRUE) %>% 
  tibble

raw_1719 <- read.xlsx(here("data/behav_params/Mueller_Replication/new_data/FRBNY-SCE-Public-Microdata-Complete-17-19.xlsx"), startRow = 2, colNames = TRUE, detectDates = TRUE) %>% 
  tibble

identical(names(raw_1316), names(raw_1719))

raw_latest <- read.xlsx(here("data/behav_params/Mueller_Replication/new_data/frbny-sce-public-microdata-latest.xlsx"), startRow = 2, colNames = TRUE, detectDates = TRUE) %>% 
  tibble %>% 
  select(names(raw_1719))

identical(names(raw_1316), names(raw_latest))

# Cleaning the Survey on Consumer Expectations Labour Market Survey Supplement###
# Source: https://www.newyorkfed.org/microeconomics/sce/labor#/
# Codebook/questionnaire: https://www.newyorkfed.org/medialibrary/media/research/microeconomics/interactive/downloads/sce-labor-questionnaire.pdf?sc_lang=en
var_names <- read_xlsx(here('data/behav_params/SCE Labour Market Survey/sce_labour_questionnaire_codebook.xlsx'))
sce_lab <- read_xlsx(here("data/behav_params/SCE Labour Market Survey/sce-labor-microdata-public.xlsx"), sheet = 3, skip = 1,  
                     col_types = var_names$type)

pull_name <- function(nm){
  var_names %>% 
    filter(var == nm) %>% 
    pull(short_var) %>% return(.)
}

sce_lab <- sce_lab %>% 
  rename_with(., .fn = pull_name) %>%
  mutate(sce_lab_survey = 1)

### Transformation function
raw_transform <- function(df){
  df %>% 
    fill(contains("Q35"), .direction = "down") %>% 
    select(userid	= userid,
         date	= date,
         start_datetime = survey_date,
         state	= `_STATE`,
         find_job_12mon = Q17new,
         find_job_3mon	= Q18new,
         find_newjob_3mon	= Q22new,
         USunemployment_higher	= Q4new,
         USstocks_higher	= Q6new,
         working_ft = Q10_1,
         working_pt = Q10_2,
         not_working_wouldlike =	Q10_3,
         temp_laid_off =	Q10_4,
         sick_leave =	Q10_5,
         looking_for_job =	Q15,
         unemployment_duration =	Q16,
         selfemployed = Q12new,
         age =	Q32,
         female = Q33,
         edu_cat	= Q36,
         hh_income = Q47, #Also possibly D6 - same question - different response levels
         white	= Q35_1,
         black	= Q35_2,
         other_race = Q35_6,
         hispanic = Q34,
         rim_4_original = weight,
         Q35_3, Q35_4, Q35_5) %>%
    group_by(userid) %>% 
    # Careful if time since last survey date is greater than 365..none are for now so I ignore
    fill(age, .direction = "down") %>% 
    fill(edu_cat, .direction = "down") %>% 
    fill(hh_income, .direction = "down") %>% 
    fill(female, .direction = "down") %>% 
    fill(hispanic, .direction = "down") %>% 
    ungroup %>% 
    mutate(
      #date = ceiling_date(ym(date), 'month') - days(1),
      across(c(find_job_12mon, find_job_3mon, find_newjob_3mon), ~ ./100),
      education_1 = edu_cat == 1,
      education_2 = edu_cat == 2,
      education_3 = edu_cat == 3,
      education_4 = edu_cat == 4,
      education_5 = edu_cat == 5,
      education_6 = edu_cat == 6,
      education_7 = edu_cat == 7,
      education_8 = edu_cat == 8,
      education_9 = edu_cat == 9,
      edu_cat = case_when(education_1 | education_2 ~ "Up to HS grad",
                          education_3 | education_4 | education_9 ~ "Some college, including associate\x92s degree", # Possibly not education_6 - this also results in an NA most times...not sure about education_9 would have to check
                          education_5 | education_6 | education_7 | education_8 ~ "College grad plus"),
      hhinc_1 = hh_income <= 3,
      hhinc_2 = hh_income > 3 & hh_income <= 6,
      hhinc_3 = hh_income > 6 & hh_income <= 8,
      hhinc_4 = hh_income > 8,
      r_asoth	= as.numeric(Q35_3 | Q35_4 | Q35_5),
      selfemployed = case_when(selfemployed == 1 ~ 0, 
                               selfemployed == 2 ~ 1,
                               is.na(selfemployed) ~ NA),
      hispanic = case_when(hispanic == 1 ~ 1, 
                           hispanic == 2 ~ 0,
                           is.na(hispanic) ~ NA),
      female = case_when(female == 1 ~ 1, 
                           female == 2 ~ 0,
                           is.na(female) ~ NA),
      across(contains("education_"), ~as.numeric(.)),
      across(contains("hhinc_"), ~as.numeric(.))) %>% 
    # This is the closest I could get to the categorisation they did in their original data - applied when dataframes are made immediately below this function
    #filter(!(temp_laid_off & (working_ft | selfemployed | working_pt)))  %>% 
    select(-c("hh_income", "Q35_3", "Q35_4", "Q35_5", "education_7", "education_8", "education_9")) %>% return(.)
}

# Dataset along same data range as sce_datafile.dta and sce_datafile_em_2025.RDS - it seems the first half of 2013 is missing from the data on the SCE website
sce_13_19_same_t <- raw_1316 %>% 
  rbind(raw_1719) %>% 
  filter(date <= 201906) %>% 
  raw_transform(.) %>% 
  left_join(., sce_lab, by = c('userid', 'date')) %>% 
  mutate(date = ceiling_date(ym(date), 'month') - days(1)) 

# Full dataset from 2013-2019
sce_13_19 <- raw_1316 %>% 
  rbind(raw_1719) %>% 
  raw_transform(.) %>% 
  left_join(., sce_lab, by = c('userid', 'date')) %>% 
  mutate(date = ceiling_date(ym(date), 'month') - days(1)) 


sce_13_24_no_lab <- raw_1316 %>% 
  rbind(raw_1719) %>% 
  rbind(raw_latest) %>% 
  raw_transform(.) %>% 
  select(userid, date, "female", "hispanic", "black", "r_asoth", "other_race",
    "age", #"agesq", 
    "hhinc_2", "hhinc_3", "hhinc_4", 
    "education_2", "education_3", "education_4", 
    "education_5", "education_6", "unemployment_duration", "rim_4_original")

# All available data 2013-2014
sce_13_24 <- raw_1316 %>% 
  rbind(raw_1719) %>% 
  rbind(raw_latest) %>% 
  raw_transform(.) %>% 
  left_join(., sce_lab, by = c('userid', 'date')) %>% 
  mutate(date = ceiling_date(ym(date), 'month') - days(1))

# Following test passes
# raw_1316 %>% 
# rbind(raw_1719) %>% 
#   rbind(raw_latest) %>% 
#   raw_transform(.) %>% left_join(., sce_lab, by = c('userid', 'date')) %>% select(all_of(names(sce_lab))) %>% filter(!if_all(-c(userid, date), is.na)) %>% arrange(userid, date) %>% identical(arrange(sce_lab, userid, date))


# Only data from 2020-2024 (date range outside of Mueller paper)
sce_20_24 <- raw_latest %>% 
  raw_transform(.) %>% 
  left_join(., sce_lab, by = c('userid', 'date')) %>% 
  mutate(date = ceiling_date(ym(date), 'month') - days(1)) 

#rm(raw_1316, raw_1719, raw_latest, sce_raw, cleaned_sce, raw_transform)

# # 1,079 when not filtering by age
# sce_raw %>% 
#   group_by(userid) %>% 
#   # Same without this condition
#   #filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
#   n_groups()
# 
# sce_13_19_same_t %>% 
#   group_by(userid) %>% 
#   filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
#   n_groups()
# 
# # 948 when filtering by age
# sce_raw %>% 
#   group_by(userid) %>% 
#   filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
#   filter(age >= 20 & age <= 65) %>% n_groups()
# 
# sce_13_19_same_t %>% 
#   group_by(userid) %>% 
#   filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
#   filter(age >= 20 & age <= 65) %>% n_groups()

# vars_to_test <- c("working_ft", "working_pt", "not_working_wouldlike", "temp_laid_off", "sick_leave", "looking_for_job", "selfemployed")
# # Testing conditions
# cond_test_new <- test_sce %>% 
#   group_by(userid) %>% 
#   filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
#   ungroup %>% 
#   select(all_of(vars_to_test)) %>% 
#   distinct %>% 
#   arrange_(.cots = vars_to_test)
# 
# sce_raw %>% 
#   group_by(userid) %>% 
#   filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
#   ungroup %>% 
#   select(all_of(vars_to_test)) %>% 
#   distinct %>% 
#   arrange_(.cots = vars_to_test) %>% 
#   anti_join(.,cond_test_new)

# Load and merge additional data
# IGNORE FOR NOW
ignore_addfiles = TRUE
if(!ignore_addfiles){
  urjr <- read_dta(paste0(base, "Raw_data/Addfiles/urjr.dta"))
  sce <- left_join(sce, urjr, by = "date")
  
  stateur <- read_dta(paste0(base, "Raw_data/Addfiles/stateur.dta"))
  sce <- left_join(sce, stateur, by = c("date", "state")) #%>% select(-_merge)
  
  rgdp <- read_dta(paste0(base, "Raw_data/Addfiles/rgdp.dta"))
  sce <- left_join(sce, rgdp, by = "date") #%>% select(-_merge)
  sce <- sce %>% select(-statefips) #filter(sce, x == 1) %>% select(-statefips)
}

unemp_temp <- sce_lab %>% 
  left_join(., sce_13_24_no_lab, by = c(userid, date)) %>% 
  mutate(date = ceiling_date(ym(date), 'month') - days(1))
