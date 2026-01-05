# Information about the education and experience requirements of specific occupations
# Source : https://www.bls.gov/emp/tables/education-and-training-by-occupation.htm
library(tidyverse)
library(here)
library(readxl)
library(assertthat)
library(readxl)
library(janitor)
source(here('data/occ_soc_cw_2010.R'))

# cw <- read.csv(here('data/crosswalk_occ_soc_cps_codes_full_omn.csv')) %>% 
#   tibble %>% 
#   mutate(SOC2010_cleaned = gsub("X", "0", SOC2010)) %>% 
#   mutate(OCC2010_match_cps = as.character(OCC2010_cps))

# Crosswalk here:
# https://www.bls.gov/emp/classifications-crosswalks/NEM-OccCode-ACS-Crosswalk.xlsx

# # # Load crosswalk from experience occupational categories to ACS codes (which are our occupational codes in the model)
# cw_ed_acs <- read_xlsx(here("data/occ_macro_vars/BLS_entry_exit_reqs/NEM-OccCode-ACS-Crosswalk.xlsx"), skip = 4) %>%
#   rename(sort_order = 1,
#          SOC2010 = 2,
#          occ_title = 3,
#          acs_code = 4,
#          acs_title = 5)

# Need crosswalk from 2018 to 2010 SOC codes
# Source for soc_2010_to_2018_crosswalk.xlsx
# https://www.bls.gov/soc/2018/crosswalks_used_by_agencies.htm
cw <- read_xlsx(here("data/occ_macro_vars/OEWS/soc_2010_to_2018_crosswalk.xlsx"), skip = 6) %>% 
  clean_names() %>% 
  rename(SOC2010 = x2010_soc_code,
         SOC2010_title = x2010_soc_title,
         SOC2018 = x2018_soc_code,
         SOC2018_title = x2018_soc_title)

dat <- read_xlsx(here("data/occ_macro_vars/BLS_entry_exit_reqs/occ_entry_level_education_reqs.xlsx"),
                 sheet = 5, skip = 1) %>%
  rename(occ_title = 1,
         SOC2018 = 2,
         ed_req = 3,
         experience_req = 4,
         otj_training = 5) %>% 
  slice(-nrow(.)) %>% 
  select(-6) %>% 
  left_join(., cw, by = "SOC2018") %>% 
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
  mutate(SOC_broad = paste0(substr(SOC2010, 1, 6), 0),
         # Create Minor groupings with exceptions for 15-11XX and 51-51XX
         # For most cases: first 4 chars + '000' (e.g., 29-1XXX -> 29-1000)
         # Exceptions: 15-11XX -> 15-1100 and 51-51XX -> 51-5100
         SOC_minor = case_when(
           substr(SOC2010, 1, 5) %in% c('15-11', '51-51') ~ paste0(substr(SOC2010, 1, 5), "00"),
           TRUE ~ paste0(substr(SOC2010, 1, 4), "000")
         ),
         SOC_major = paste0(substr(SOC2010, 1, 2)))

assert_that(dat %>% filter(is.na(entry_level)) %>% nrow(.) == 0) 

# includes average employment level so that we can properly weight across entry_level characteristics
################################################################################
########################### SOC2010 Detailed ###################################
################################################################################
soc_all <- read.csv("/Users/ebbamark/Library/CloudStorage/OneDrive-Nexus365/GenerateOccMobNets/data/soc_2010_codes_employment_asec.csv") %>% 
  rename(SOC2010 = SOC) %>% 
  mutate(SOC_broad = as.character(paste0(substr(SOC2010, 1, 6), 0)),
         SOC_minor = case_when(
           substr(SOC2010, 1, 5) %in% c('15-11', '51-51') ~ paste0(substr(SOC2010, 1, 5), "00"),
           TRUE ~ paste0(substr(SOC2010, 1, 4), "000")
         ))

setdiff(unique(dat$SOC2010), unique(soc_all$SOC2010))
setdiff(unique(soc_all$SOC2010), unique(dat$SOC2010))

test_missing <- function(df, total_df){
  df %>%
    filter(!is.na(ed_req)) %>%
    pull(SOC2010) %>%
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
# Match 397 codes (133 remaining)
acs_matches <- soc_all %>%
  left_join(., select(dat, -any_of(names(soc_all)[!(names(soc_all) %in% c("SOC2010"))])), by = "SOC2010") %>%
  select(-contains(".y"))

test_missing(acs_matches)

# Match 115 codes (18 remaining)
acs_cps_matches <- acs_matches %>%
  filter(is.na(ed_req)) %>%
  select(-any_of(names(dat)[!(names(dat) %in% c("SOC2010", "SOC_broad", "SOC_minor"))])) %>%
  left_join(., select(dat, -any_of(names(soc_all)[!(names(soc_all) %in% c("SOC_broad"))])), by = "SOC_broad")

test_missing(acs_cps_matches)

# Match 17 codes (1 remaining)
soc2010_matches <- acs_cps_matches %>%
  filter(is.na(ed_req)) %>%
  select(-any_of(names(dat)[!(names(dat) %in% c("SOC2010", "SOC_broad", "SOC_minor"))])) %>%
  left_join(., select(dat, -any_of(names(soc_all)[!(names(soc_all) %in% c("SOC_minor"))])), by = "SOC_minor", relationship = "many-to-many")

test_missing(soc2010_matches)

# Match 1 codes (0 remaining)
soc_broad_matches <- soc2010_matches %>%
  filter(is.na(ed_req)) %>% 
  select(-any_of(names(dat)[!(names(dat) %in% c("SOC2010", "SOC_broad", "SOC_minor"))])) %>% 
  # We match the closest possible code to 39-7010 (tour guides) which is concierges "39-6012"
  mutate(SOC2010_temp = ifelse(SOC2010 == "39-7010", "39-6012", NA)) %>% 
  left_join(., select(dat, -any_of(names(soc_all)[!(names(soc_all) %in% c("SOC2010"))])), by = c("SOC2010_temp"="SOC2010")) %>% 
  select(-SOC2010_temp)

test_missing(soc_broad_matches)

total_df <- bind_rows(acs_matches,
                      acs_cps_matches,
                      soc2010_matches,
                      soc_broad_matches) %>%
  filter(!is.na(ed_req)) %>%
  select(SOC2010, ed_req, experience_req, otj_training, entry_level, entry_age) %>%
  distinct(.)

assert_that(soc_all %>% filter(!SOC2010 %in% total_df$SOC2010) %>% nrow(.) == 0)

final_exp <- total_df %>%
  group_by(SOC2010) %>%
  summarise(ed_req = first(ed_req),
            experience_req = min(experience_req),
            otj_training = max(otj_training),
            entry_level = max(entry_level),
            entry_age = mean(entry_age)) %>%
  ungroup %>%
  mutate(experience_age = case_when(experience_req == "None" ~ entry_age + 0,
                                    experience_req == "Less than 5 years" ~ entry_age + 4,
                                    experience_req == "5 years or more" ~ entry_age + 7))

temp <- soc_all
assert_that(temp %>% filter(!(SOC2010 %in% unique(final_exp$SOC2010))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(!(SOC2010 %in% unique(temp$SOC2010))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(any(is.na(.))) %>% nrow(.) == 0)

temp %>%
  left_join(., final_exp, by = "SOC2010") %>%
  mutate(entry_level = as.numeric(experience_req == "None")) -> tosave

stopifnot(identical(temp$SOC2010, soc_all$SOC2010))

tosave %>%
  write.csv(here("calibration_remote/dRC_Replication/data/ipums_variables_soc2010_w_exp.csv"))

assert_that(temp %>% filter(!(SOC2010 %in% unique(final_exp$SOC2010))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(!(SOC2010 %in% unique(temp$SOC2010))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(any(is.na(.))) %>% nrow(.) == 0)

# # testing that matched occs have the same values in both:
# original_omn_exp <- read_csv(here("calibration_remote/dRC_Replication/data/ipums_variables_w_exp.csv"))
# full_omn_exp <- read_csv(here("calibration_remote/dRC_Replication/data/ipums_variables_full_omn_w_exp.csv"))
# 
# full_omn_exp %>%
#   filter(SOC2010 %in% unique(original_omn_exp$SOC2010)) %>%
#   select(any_of(names(original_omn_exp)), -id, -X) %>%
#   select(-1) %>%
#   mutate(SOC2010 = as.integer(SOC2010))  %>%
#   arrange(SOC2010) -> full_test
# 
# original_omn_exp %>%
#   filter(SOC2010 %in% unique(full_omn_exp$SOC2010)) %>%
#   select(any_of(names(full_omn_exp)), -id) %>%
#   select(-1) %>%
#   mutate(SOC2010 = as.integer(SOC2010)) %>%
#   arrange(SOC2010) -> original_test
# 
# stopifnot(identical(full_test, original_test))

################################################################################
########################### SOC2010 Detailed ###################################
################################################################################

soc_all <- read.csv("/Users/ebbamark/Library/CloudStorage/OneDrive-Nexus365/GenerateOccMobNets/ONET/occ_names_employment_asec_soc_2010_minor_ipums_vars.csv") %>% 
  rename(SOC_minor = soc)# %>% 
  #mutate(SOC_broad = as.character(paste0(substr(SOC2010, 1, 6), 0)),
         #SOC_minor = paste0(substr(SOC2010, 1, 4), "000"))

dat_minor <- dat %>% 
  group_by(SOC_minor) %>%
  summarise(ed_req = first(ed_req),
            experience_req = min(experience_req),
            otj_training = max(otj_training),
            entry_level = max(entry_level),
            entry_age = mean(entry_age))

setdiff(unique(dat_minor$SOC_minor), unique(soc_all$SOC_minor))
setdiff(unique(soc_all$SOC_minor), unique(dat_minor$SOC_minor))

test_missing <- function(df, total_df){
  df %>%
    filter(!is.na(ed_req)) %>%
    pull(SOC_minor) %>%
    n_distinct %>%
    paste0("Matched codes: ", .) %>%
    print(.)
  
  df %>%
    filter(is.na(ed_req)) %>%
    nrow(.) %>%
    paste0("Still missing ", ., " codes!") %>%
    print(.)
}

# match on SOC minor
# Match 93 codes (1 remaining)
first_matches <- soc_all %>%
  left_join(., dat_minor, by = "SOC_minor") %>%
  select(-contains(".y"))

test_missing(first_matches)

# Match 1 code (0 remaining)
second_matches <- first_matches %>%
  filter(is.na(ed_req)) %>%
  select(-any_of(names(dat_minor)[!(names(dat_minor) %in% c("SOC_minor"))])) %>%
  # Same as above - equate Tour Guide (39-7000) to Baggage Porters, Bellhops, and Concierges (39-6000)
  mutate(SOC_minor_temp = ifelse(SOC_minor == "39-7000", "39-6000", NA)) %>% 
  left_join(., dat_minor, by = c("SOC_minor_temp" = "SOC_minor")) %>% 
  select(-SOC_minor_temp)

test_missing(second_matches)

total_df <- bind_rows(first_matches,
                      second_matches) %>%
  filter(!is.na(ed_req)) %>%
  select(SOC_minor, ed_req, experience_req, otj_training, entry_level, entry_age) %>%
  distinct(.)

assert_that(soc_all %>% filter(!SOC_minor %in% total_df$SOC_minor) %>% nrow(.) == 0)
assert_that(soc_all %>% group_by(SOC_minor) %>% n_groups(.) == nrow(soc_all))

final_exp <- total_df %>%
  mutate(experience_age = case_when(experience_req == "None" ~ entry_age + 0,
                                    experience_req == "Less than 5 years" ~ entry_age + 4,
                                    experience_req == "5 years or more" ~ entry_age + 7))

temp <- soc_all
assert_that(temp %>% filter(!(SOC_minor %in% unique(final_exp$SOC_minor))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(!(SOC_minor %in% unique(temp$SOC_minor))) %>% nrow(.) == 0)
assert_that(final_exp %>% filter(any(is.na(.))) %>% nrow(.) == 0)

temp %>%
  left_join(., final_exp, by = "SOC_minor") %>%
  mutate(entry_level = as.numeric(experience_req == "None")) -> tosave

stopifnot(identical(temp$SOC_minor, soc_all$SOC_minor))

tosave %>%
  write.csv(here("calibration_remote/dRC_Replication/data/ipums_variables_SOC_minor_w_exp.csv"))
