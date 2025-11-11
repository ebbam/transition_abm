# Information about the education and experience requirements of specific occupations
# Source : https://www.bls.gov/emp/tables/education-and-training-by-occupation.htm
library(tidyverse)
library(here)
library(readxl)
library(assertthat)

# ONET
cw <- read.csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/ONET/acs_onet_2010_soc_cw.csv") %>% tibble %>% 
  rename(SOC2010 = soc_2010_code,
         acs_occ_code = acs_2010_code,
         label = title)

temp <- read.csv(here("/Users/ebbamark/Dropbox/GenerateOccMobNets/ONET/acs_onet_2010_ipums_vars.csv")) %>% tibble %>% 
  rename(acs_occ_code = occ)

# Crosswalk here:
# https://www.bls.gov/emp/classifications-crosswalks/NEM-OccCode-ACS-Crosswalk.xlsx

# Load crosswalk from experience occupational categories to ACS codes (which are our occupational codes in the model)
cw_ed_acs <- read_xlsx(here("data/occ_macro_vars/BLS_entry_exit_reqs/NEM-OccCode-ACS-Crosswalk.xlsx"), skip = 4) %>%
  rename(sort_order = 1,
         SOC2010 = 2,
         occ_title = 3,
         acs_code = 4,
         acs_title = 5)

dat <- read_xlsx(here("data/occ_macro_vars/BLS_entry_exit_reqs/occ_entry_level_education_reqs.xlsx"),
                 sheet = 5, skip = 1) %>%
  rename(occ_title = 1,
         SOC2010 = 2,
         ed_req = 3,
         experience_req = 4,
         otj_training = 5) %>%
  select(-6) %>%
  slice(-nrow(.)) %>%
  mutate(entry_level = experience_req == "None",
         entry_age = case_when(ed_req == "Bachelor's degree" ~ 21,
                               ed_req == "High school diploma or equivalent" ~ 18,
                               ed_req == "Master's degree" ~ 23,
                               ed_req == "Associate's degree" ~ 23,
                               ed_req == "Postsecondary nondegree award" ~ 23,
                               ed_req == "No formal educational credential" ~ 18,
                               ed_req == "Some college, no degree" ~ 21,
                               ed_req == "Doctoral or professional degree" ~ 30),
         experience_req = factor(experience_req, levels = c("5 years or more", "Less than 5 years", "None"), ordered = TRUE),
         otj_training = factor(otj_training, levels = c("None",    
                                                        "Internship/residency",
                                                        "Apprenticeship",
                                                        "Short-term on-the-job training",
                                                        "Moderate-term on-the-job training",
                                                        "Long-term on-the-job training"), 
                               ordered = TRUE)) %>%
  left_join(., cw_ed_acs, by = "SOC2010") %>% 
  mutate(SOC_broad = paste0(substr(SOC2010, 1, 6), 0),
         SOC_minor = paste0(substr(SOC2010, 1, 4), "000"),
         SOC_major = paste0(substr(SOC2010, 1, 2)))

assert_that(dat %>% filter(is.na(entry_level)) %>% nrow(.) == 0)

cw %>% 
  select(acs_occ_code) %>% 
  distinct %>%
  #mutate(id = id - 1) %>% 
  all_equal(select(temp, acs_occ_code))

test_missing <- function(df, total_df){
  df %>% 
    filter(!is.na(ed_req)) %>% 
    pull(acs_occ_code) %>% 
    n_distinct %>% 
    paste0("Matched codes: ", .) %>% 
    print(.)
  
  df %>% 
    filter(is.na(ed_req)) %>% 
    nrow(.) %>% 
    paste0("Still missing ", ., " codes!") %>% 
    print(.)
}

# match on ACS Code
# Match 369 codes (369/516)
acs_matches <- cw %>% 
  left_join(., select(dat, -any_of(names(cw))), by = c("acs_occ_code" = "acs_code"), multiple = "first") %>% 
  select(-contains(".y"))

test_missing(acs_matches)

# # Match 2 codes (371/464)
# acs_cps_matches <- acs_matches %>% 
#   filter(is.na(ed_req)) %>% 
#   select(-any_of(names(dat)[!(names(dat) %in% c("SOC2010", "SOC_broad", "SOC_minor"))])) %>% 
#   left_join(., select(dat, -any_of(names(cw))), by = c("OCC2010_cps" = "acs_code")) 
# 
# test_missing(acs_cps_matches)

# Match 82 codes (451/516)
soc2010_matches <- acs_matches %>% 
  filter(is.na(ed_req)) %>% 
  select(-any_of(names(dat)[!(names(dat) %in% c("SOC2010"))])) %>%   
  left_join(., select(dat, -any_of(names(cw)[!(names(cw) %in% c("SOC2010"))])), by = c("SOC2010" = "SOC2010"))

test_missing(soc2010_matches)

# Match 45 codes (441 total/464)
soc_broad_matches <- soc2010_matches %>% 
  filter(is.na(ed_req)) %>% 
  select(-any_of(names(dat)[!(names(dat) %in% c("SOC2010", "SOC_broad"))])) %>% 
  mutate(SOC_broad = paste0(substr(SOC2010, 1, 6), 0)) %>% 
  left_join(., select(dat, -any_of(names(cw)[!(names(cw) %in% c("SOC_broad"))])), by = c('SOC_broad'), relationship = "many-to-many")

test_missing(soc_broad_matches)

# Matched 23 codes
soc_minor_matches <- soc_broad_matches %>% 
  filter(is.na(ed_req)) %>% 
  select(-any_of(names(dat)[!(names(dat) %in% c("SOC2010", "SOC_broad", "SOC_minor"))])) %>% 
  mutate(SOC_minor = paste0(substr(SOC2010, 1, 4), "000")) %>% 
  left_join(., select(dat, -any_of(names(cw)[!(names(cw) %in% c("SOC2010", "SOC_broad", "SOC_minor"))])), by = c('SOC_minor'), relationship = "many-to-many")

test_missing(soc_minor_matches)

total_df <- bind_rows(acs_matches,
                      soc2010_matches,
                      soc_broad_matches,
                      soc_minor_matches) %>% 
  filter(!is.na(ed_req)) %>% 
  select(acs_occ_code, label, ed_req, experience_req, otj_training, entry_level, entry_age) %>% 
  distinct(.)

assert_that(cw %>% filter(!acs_occ_code %in% total_df$acs_occ_code) %>% nrow(.) == 0)

final_exp <- total_df %>% 
  group_by(acs_occ_code, label) %>% 
  summarise(ed_req = first(ed_req), 
            experience_req = min(experience_req), 
            otj_training = max(otj_training), 
            entry_level = max(entry_level),
            entry_age = mean(entry_age)) %>% 
  ungroup %>% 
  mutate(experience_age = case_when(experience_req == "None" ~ entry_age + 0,
                                    experience_req == "Less than 5 years" ~ entry_age + 4,
                                    experience_req == "5 years or more" ~ entry_age + 7)) %>% 
  select(-label)


assert_that(temp %>% arrange(X) %>% identical(temp))
assert_that(temp %>% filter(!(acs_occ_code %in% unique(final_exp$acs_occ_code))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(!(acs_occ_code %in% unique(temp$acs_occ_code))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(any(is.na(.))) %>% nrow(.) == 0)

temp %>%
  left_join(., final_exp, by = "acs_occ_code") %>%
  mutate(entry_level = as.numeric(experience_req == "None"),
         acs_occ_code = as.integer(acs_occ_code)) -> tosave

stopifnot(identical(temp$acs_occ_code, tosave$acs_occ_code))

tosave %>% 
 write.csv(here("calibration_remote/dRC_Replication/data/acs_onet_2010_ipums_vars_w_exp.csv"))

