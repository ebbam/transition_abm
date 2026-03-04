### CLEANING EMPLOYMENT DATA
# Occupational Employment and Wage Statistics
# https://www.bls.gov/oes/

library(tidyverse)
library(here)
library(conflicted)
library(janitor)
library(readxl)
library(assertthat)
conflict_prefer_all("dplyr", quiet = TRUE)

df <- read_xls(here(paste0("data/occ_macro_vars/OEWS/national_M2010_dl.xls")))

clean_oews_emp <- function(df, yr){
  
  df <- df %>% 
    clean_names %>% 
    select(where(~ n_distinct(.) > 1))
  
  temp <- df %>% 
    select(occ_code, occ_title, contains("group"), tot_emp, emp_prse) %>% 
    mutate(across(tot_emp:emp_prse, ~as.numeric(.))) %>% 
    mutate(year = yr) %>% 
    rename_with(~ "occ_group", .cols = any_of(c("o_group", "group", "occ_group")))
  return(temp)
  
}


files <- list.files(here("data/occ_macro_vars/OEWS"))
files <- files[which(grepl("xls", files))]
occ_emp <- data.frame()
for(yr in 2000:2023){
  print(yr)
  temp_name <- files[which(grepl(paste0(as.character(yr), "_dl"), files))]
  if(length(temp_name) == 1){
    if(yr <= 2000){
      temp_df <- read_xls(here(paste0("data/occ_macro_vars/OEWS/", temp_name)), skip = 37) %>% 
        clean_oews_emp(., yr)
    }
    else if(yr > 2000 & yr < 2014){
      temp_df <- read_xls(here(paste0("data/occ_macro_vars/OEWS/", temp_name))) %>% 
        clean_oews_emp(., yr) 
    }else{
      temp_df <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/", temp_name))) %>% 
        clean_oews_emp(., yr) 
    }
  }else{
    print("One or multiple files found for that year.")
    break
  }
  occ_emp <- rbind(occ_emp, temp_df)
}

# Crosswalk from ABM data - written in occ_soc_cps_codes_crosswalk.R
abm_occs <- read.csv(here("data/crosswalk_occ_soc_cps_codes.csv")) %>% 
  tibble %>% 
  mutate(SOC2010 = gsub("X", "0", SOC2010))

# Confirm that all OCS codes in abm_vars are present in the crosswalk
abm_occs %>% filter(!(SOC2010 %in% unique(occ_emp$occ_code))) %>% nrow(.) == 0
abm_occs %>% filter(!(SOC2010 %in% unique(occ_emp$occ_code)))

occ_emp %>% 
  ggplot() + 
  geom_line(aes(x = year, y = tot_emp, group = occ_code))

# Convert all occupational categories to 2010 codes 
occs_00_10 <- read_xls(here("data/occ_macro_vars/OEWS/soc_2000_to_2010_crosswalk.xls"), skip = 6) %>% 
  clean_names() %>% 
  filter(!if_all(everything(), is.na)) 

#write.csv(occs_00_10, here("data/occ_macro_vars/OEWS/soc_2000_to_2010_crosswalk.csv"))

occs_00_10_short <- occs_00_10 %>% 
  # Retain only those codes that are not equivalent across the years - 79 in the 00-10 code
  filter(x2010_soc_code != x2000_soc_code)

occs_10_18 <- read_xlsx(here("data/occ_macro_vars/OEWS/soc_2010_to_2018_crosswalk.xlsx"), skip = 6) %>% 
  clean_names() %>% 
  filter(!if_all(everything(), is.na)) 

occs_10_18_short <- occs_10_18 %>%
  # Retain only those codes that are not equivalent across the years - 148 in the 10-18 code
  filter(x2010_soc_code != x2018_soc_code)

identical(sort(unique(occs_00_10$x2010_soc_code)), sort(unique(occs_10_18$x2010_soc_code)))

occ_emp %>% 
  filter(occ_code %in% c(abm_occs$SOC2010)) %>% 
  ggplot() + 
  geom_line(aes(x = year, y = tot_emp, group = occ_code))

occ_emp %>% write.csv(here("data/occ_macro_vars/OEWS/occ_employment_levels_oews.csv"))

# occ_emp %>% 
#   select(year, occ_code) %>% 
#   write.csv(here("data/occ_macro_vars/OEWS/occ_emp_short_oews.csv"))

##### SOC 3 CODES #########

setdiff(lightcast$soc_3, occ_emp$occ_code) # Military occupations and otherwise unclassified missing! This is fine :) 
occ_emp_cleaned <- occ_emp %>% 
  group_by(occ_code) %>% 
  mutate(occ_title = last(occ_title, na_rm = TRUE)) %>% 
  ungroup
# Save the years in which we have an exact match for a SOC3 code
complete <- occ_emp %>% 
  filter(occ_code %in% lightcast$soc_3) %>% 
    # Reconciling non-matching occupation titles for the same occ_code
    group_by(occ_code) %>% 
    complete(year = 2012:2023) %>% 
    mutate(occ_title = last(occ_title, na_rm = TRUE))  
#mutate(n = n(), min_year = min(year), max_year = max(year)) # NOTE THERE ARE A FEW CODES THAT DO NOT HAVE COMPLETE VALUES FOR ALL YEARS....
#filter(any(is.na(tot_emp))) %>% select(occ_code, occ_title) # NOTE THERE ARE 6 CODES THAT ARE NOT COMPLETE - LIKELY BECAUSE OF RECLASSIFICATION
#15-1200  Computer Occupations                                                                         
#19-5000  Occupational Health and Safety Specialists and Technicians                                   
#31-1100  Home Health and Personal Care Aides; and Nursing Assistants, Orderlies, and Psychiatric Aides
#39-4000  Funeral Service Workers                                                                      
#39-9000  Other Personal Care and Service Workers                                                      
#45-3000  Fishing and Hunting Workers     

# THE LAST CODE TO BE MATCHED IS COMPUTER AND SCIENCE OCCS!
completed <-  occ_emp %>% 
  mutate(match_code = paste0(substr(occ_code, 1, 4), "000")) %>% 
  filter(substr(occ_code, 4,8) != "0000") %>% 
  relocate(match_code) %>% 
  group_by(year) %>% 
  filter(all(match_code != occ_code) & year >= 2010) %>% 
  mutate(match_code = ifelse(match_code %in% c("15-1000"), "15-0000", match_code)) %>% 
    group_by(year, match_code) %>% 
    summarise(tot_emp = sum(tot_emp))

complete_emp <- complete %>% 
  bind_rows(completed) 

complete_emp %>% 
  group_by(match_code) %>% 
  filter(match_code == "15-2000") %>% 
  fill(occ_title, .direction = "updown") %>% ggplot(aes(x = year, y = tot_emp)) + geom_line()

lightcast %>% 
  rename(match_code = soc_3)  %>% 
  left_join(., complete_emp, by = c("match_code", "year")) %>% 
  group_by(year, match_code, occ_title) %>% 
  summarise(vacs_tot = sum(n_ft_vacancies), tot_emp = unique(tot_emp)) %>% 
  mutate(vac_rate = vacs_tot/tot_emp) %>% ggplot(aes(x = year, y = vac_rate, color = match_code)) + 
  geom_line() + theme(legend.position = "none")

################################################################################
# Load data
occ_emp <- read_csv(here("data/occ_macro_vars/OEWS/occ_emp_short_oews.csv"))
crosswalk_00_10 <- read_xls(here("data/occ_macro_vars/OEWS/soc_2000_to_2010_crosswalk.xls"), skip = 6) %>% 
  clean_names() %>% 
  filter(!if_all(everything(), is.na)) 
crosswalk_10_18 <- read_xlsx(here("data/occ_macro_vars/OEWS/soc_2010_to_2018_crosswalk.xlsx"), skip = 6) %>% 
  clean_names() %>% 
  filter(!if_all(everything(), is.na)) 

# Clean and prepare crosswalks
crosswalk_00_10 <- crosswalk_00_10 %>%
  select(from_code = x2000_soc_code, soc2010_code = x2010_soc_code) %>%
  mutate(version = "2000")

crosswalk_10_18 <- crosswalk_10_18 %>%
  rename(soc2010_code = x2010_soc_code, from_code = x2018_soc_code) %>%
  select(from_code, soc2010_code) %>%
  filter(!is.na(from_code), !is.na(soc2010_code)) %>%
  mutate(version = "2018")

# Assume codes not in either crosswalk are already SOC2010
existing_2010 <- occ_emp %>%
  distinct(occ_code) %>%
  rename(from_code = occ_code) %>%
  mutate(soc2010_code = from_code, version = "assumed_2010")

# Combine all mappings
harmonized_map <- bind_rows(crosswalk_00_10, crosswalk_10_18, existing_2010) %>%
  distinct(from_code, soc2010_code, .keep_all = TRUE)

harmonized_map <- harmonized_map %>%
  mutate(priority = case_when(
    version == "2000" ~ 1,
    version == "2018" ~ 2,
    version == "assumed_2010" ~ 3,
    TRUE ~ 4
  )) %>%
  group_by(from_code) %>%
  slice_min(priority, with_ties = FALSE) %>%
  ungroup()

# Merge harmonized SOC2010 codes into wage data
occ_emp_with_soc2010 <- occ_emp %>%
  left_join(harmonized_map, by = c("occ_code" = "from_code"))

# Save the result
# write_csv(occ_emp_with_soc2010 %>% select(year, occ_code, soc2010_code),
#           "output/occ_emp_with_soc2010.csv")

################################################################################
occ_emp %>% filter(occ_code %in% abm_occs$SOC2010) %>% 
  complete(occ_code, year) %>% 
  group_by(occ_code) %>% 
  filter(any(is.na(h_mean))) %>% 
  ungroup %>% filter(occ_code %in% c(unique(occs_00_10_short$x2000_soc_code), unique(occs_10_18_short$x2018_soc_code)))

