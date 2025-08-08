# Information about the education and experience requirements of specific occupations
library(tidyverse)
library(here)
library(readxl)

dat <- read_xlsx(here("data/occ_macro_vars/BLS_entry_exit_reqs/occ_entry_level_education_reqs.xlsx"), 
                 sheet = 5, skip = 1) %>% 
  rename(occ_title = 1,
         occ_code = 2,
         ed_req = 3, 
         experience_req = 4, 
         otj_training = 5) %>% 
  select(-6) %>% 
  slice(-nrow(.)) 

dat %>% 
  mutate(entry_level = experience_req == "None",
         entry_age = case_when(ed_req == "Bachelor's degree" ~ 21,
                               ed_req == "High school diploma or equivalent" ~ 18,
                               ed_req == "Master's degree" ~ 23,
                               ed_req == "Associate's degree" ~ 23,
                               ed_req == "Postsecondary nondegree award" ~ 23,
                               ed_req == "No formal educational credential" ~ 18,
                               ed_req == "Some college, no degree" ~ 21,
                               ed_req == "Doctoral or professional degree" ~ 30))

         