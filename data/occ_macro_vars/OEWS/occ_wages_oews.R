### CLEANING OEWS DATA
# Occupational Employment and Wage Statistics
# https://www.bls.gov/oes/

library(tidyverse)
library(here)
library(conflicted)
library(janitor)
library(readxl)
library(assertthat)
conflict_prefer_all("dplyr", quiet = TRUE)

df <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/national_M2023_dl.xlsx")))

clean_oews <- function(df, yr){
  
  df <- df %>% 
    clean_names %>% 
  select(where(~ n_distinct(.) > 1))
  
  # If 'hourly' doesn't exist, create it based on logic (example logic shown here)
  if (!"hourly" %in% names(df)) {
    print("entered fixing step")
    df <- df %>%
      mutate(hourly = is.na(a_mean) & !is.na(h_mean))
  }
  
  temp <- df %>% 
  select(occ_code, occ_title, contains("group"), tot_emp, h_mean, a_mean, h_median, a_median, annual, hourly) %>% 
  mutate(across(tot_emp:a_median, ~as.numeric(.)),
         across(annual:hourly, ~as.logical(.))) %>%
  # From documentation: 
  # annual: Contains "TRUE" if only the annual wages are released. 
            # The OES program releases only annual wages for some occupations 
            # that typically work fewer than 2,080 hours per year but are 
            # paid on an annual basis, such as teachers, pilots, and athletes.
  # hourly: Contains "TRUE" if only the hourly wages are released. 
            # Some occupations, such as actors, dancers, and musicians and singers, 
            # are paid hourly and generally don't work a standard 2,080 hour work year. 
  
         # In the case where annual wages are reported only we divide the a_mean by 2080 hours
  mutate(h_mean = case_when(annual & is.na(h_mean) ~ a_mean/2080, 
                            TRUE ~ h_mean),
         h_median = case_when(annual & is.na(h_median) ~ a_median/2080, 
                            TRUE ~ h_median),
         # In the case where hourly wages are reported only we multiply the h_mean by 2080 hours
         a_mean = case_when(hourly & is.na(a_mean) ~ h_mean*2080, 
                            TRUE ~ a_mean),
         a_median = case_when(hourly & is.na(a_median) ~ h_median*2080, 
                            TRUE ~ a_median)) %>% 
    # there are several median values that are still NA values because they report wages greater than $100/hour or $208,000/year
  select(-c(a_median, h_median, hourly, annual)) %>% 
    mutate(year = yr) %>% 
    rename_with(~ "occ_group", .cols = any_of(c("o_group", "group", "occ_group")))
  
  #assert_that(nrow(temp) == nrow(drop_na(temp)))
  
  return(temp)

  }

         
files <- list.files(here("data/occ_macro_vars/OEWS"))
files <- files[which(grepl("xls", files))]
occ_wages <- data.frame()
for(yr in 2000:2023){
  print(yr)
  temp_name <- files[which(grepl(paste0(as.character(yr), "_dl"), files))]
  if(length(temp_name) == 1){
    if(yr <= 2000){
      temp_df <- read_xls(here(paste0("data/occ_macro_vars/OEWS/", temp_name)), skip = 37) %>% 
        clean_oews(., yr)
    }
    else if(yr > 2000 & yr < 2014){
      temp_df <- read_xls(here(paste0("data/occ_macro_vars/OEWS/", temp_name))) %>% 
        clean_oews(., yr) 
      }else{
        temp_df <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/", temp_name))) %>% 
          clean_oews(., yr) 
    }
  }else{
    print("One or multiple files found for that year.")
    break
  }
  occ_wages <- rbind(occ_wages, temp_df)
}

# Crosswalk from ABM data - written in occ_soc_cps_codes_crosswalk.R
abm_occs <- read.csv(here("data/crosswalk_occ_soc_cps_codes.csv")) %>% 
  tibble %>% 
  mutate(SOC2010 = gsub("X", "0", SOC2010))

# Confirm that all OCS codes in abm_vars are present in the crosswalk
abm_occs %>% filter(!(SOC2010 %in% unique(occ_wages$occ_code))) %>% nrow(.) == 0
abm_occs %>% filter(!(SOC2010 %in% unique(occ_wages$occ_code)))

occ_wages %>% 
  ggplot() + 
  geom_line(aes(x = year, y = tot_emp, group = occ_code))

# Convert all occupational categories to 2010 codes 
occs_00_10 <- read_xls(here("data/occ_macro_vars/OEWS/soc_2000_to_2010_crosswalk.xls"), skip = 6) %>% 
  clean_names() %>% 
  filter(!if_all(everything(), is.na)) 

write.csv(occs_00_10, here("data/occ_macro_vars/OEWS/soc_2000_to_2010_crosswalk.csv"))

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

occ_wages %>% 
  filter(occ_code %in% c(abm_occs$SOC2010)) %>% 
  ggplot() + 
  geom_line(aes(x = year, y = tot_emp, group = occ_code))


occ_wages %>% 
  select(year, occ_code) %>% 
  write.csv(here("data/occ_macro_vars/OEWS/occ_wages_short_oews.csv"))




################################################################################
# Load data
occ_wages <- read_csv(here("data/occ_macro_vars/OEWS/occ_wages_short_oews.csv"))
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
existing_2010 <- occ_wages %>%
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
occ_wages_with_soc2010 <- occ_wages %>%
  left_join(harmonized_map, by = c("occ_code" = "from_code"))

# Save the result
# write_csv(occ_wages_with_soc2010 %>% select(year, occ_code, soc2010_code),
#           "output/occ_wages_with_soc2010.csv")

################################################################################
occ_wages %>% filter(occ_code %in% abm_occs$SOC2010) %>% 
  complete(occ_code, year) %>% 
  group_by(occ_code) %>% 
  filter(any(is.na(h_mean))) %>% 
  ungroup %>% filter(occ_code %in% c(unique(occs_00_10_short$x2000_soc_code), unique(occs_10_18_short$x2018_soc_code)))

