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

#df <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/national_M2023_dl.xlsx")))

# There are upper limits for the hourly and annual wages that are not reported in the data and are replaced by a # value. We extract these limits from the field description files
limits <- tibble()
folders <- list.files(here("data/occ_macro_vars/OEWS"))
folders <- folders[which(!grepl(".", folders, fixed = TRUE) & (!folders %in% c("data_crosswalk_documentation", "industry_specific")))]
for(yr in 2000:2023){
  print(yr)
  temp_name <- folders[which(grepl(as.character(substr(yr, 3,4)), folders))]
  print(temp_name)
  if(yr == 2000){
    tmp_lim <- read_xls(here("data/occ_macro_vars/OEWS/national_2000_dl.xls")) %>% rename(rel_col = 2) %>% 
      filter(grepl("#", rel_col)) %>% select(rel_col) %>% 
      separate(rel_col, into = c("prefix", "hourly_max", "annual_max"), sep = "\\$", remove = TRUE) %>% 
      mutate(year = yr)
    print(tmp_lim)
  }else if(yr %in% c(2003, 2004)){
    tmp_lim <- read_xls(here(paste0("data/occ_macro_vars/OEWS/oesm04nat/field_descriptions.xls"))) %>% rename(rel_col = 1) %>% 
      filter(grepl("#", rel_col)) %>% select(rel_col) %>% 
      separate(rel_col, into = c("prefix", "hourly_max", "annual_max"), sep = "\\$", remove = TRUE) %>% 
      mutate(year = yr)
    print(tmp_lim)
  }else if(length(temp_name) == 1){
    if(yr < 2014){
      tmp_lim <- read_xls(here(paste0("data/occ_macro_vars/OEWS/", temp_name, "/field_descriptions.xls"))) %>% 
        rename(rel_col = 1) %>% 
        filter(grepl("#", rel_col)) %>% select(rel_col) %>% 
        separate(rel_col, into = c("prefix", "hourly_max", "annual_max"), sep = "\\$", remove = TRUE)%>% 
        mutate(year = yr)
      print(tmp_lim)
    }else if(yr >= 2014 & yr < 2019){
      tmp_lim <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/", temp_name, "/field_descriptions.xlsx"))) %>% 
        rename(rel_col = 1) %>% 
        filter(grepl("#", rel_col)) %>% select(rel_col) %>% 
        separate(rel_col, into = c("prefix", "hourly_max", "annual_max"), sep = "\\$", remove = TRUE)%>% 
        mutate(year = yr)
      print(tmp_lim)
    }else if(yr >= 2019){
      tmp_lim <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/national_M", as.character(yr), "_dl.xlsx")), sheet = 2) %>% 
        rename(rel_col = 1) %>% 
        filter(grepl("#", rel_col)) %>% select(rel_col) %>% 
        separate(rel_col, into = c("prefix", "hourly_max", "annual_max"), sep = "\\$", remove = TRUE)%>% 
        mutate(year = yr)
      print(tmp_lim)
    }
      }else{
    print("One or multiple files found for that year.")
    break
  }
  limits <- bind_rows(limits, tmp_lim)
}

limits_df <- limits %>% 
  mutate(hourly_max = trimws(gsub(" per hour or ","",  hourly_max)),
         annual_max = trimws(gsub(",", "", gsub(" per year","",  annual_max), fixed = TRUE)))

clean_oews <- function(df, yr, limits){
  
  hourly_max <- limits %>% filter(year == yr) %>% pull(hourly_max)
  annual_max <- limits %>% filter(year == yr) %>% pull(annual_max)
  
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
    rename_with(~gsub("wpct", "pct", .), contains("wpct")) %>% 
    select(occ_code, occ_title, contains('group'), tot_emp, emp_prse, h_mean, a_mean, mean_prse, h_pct10, h_pct25, h_median, h_pct75,  
           h_pct90, a_pct10, a_pct25, a_median, a_pct75, a_pct90, annual, hourly) %>% 
    mutate(across(c(h_mean, h_pct10, h_pct25, h_median, h_pct75, h_pct90), ~ifelse(. == "#", hourly_max, .)),
           across(c(a_mean, a_pct10, a_pct25, a_median, a_pct75, a_pct90), ~ifelse(. == "#", annual_max, .)),
           across(tot_emp:a_pct90, ~as.numeric(.)), 
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
           h_pct10 = case_when(annual & is.na(h_pct10) ~ a_pct10/2080, 
                              TRUE ~ h_pct10),
           h_pct25 = case_when(annual & is.na(h_pct25) ~ a_pct25/2080, 
                              TRUE ~ h_pct25),
           h_pct75 = case_when(annual & is.na(h_pct75) ~ a_pct75/2080, 
                              TRUE ~ h_pct75),
           h_pct90 = case_when(annual & is.na(h_pct90) ~ a_pct90/2080, 
                               TRUE ~ h_pct90),
           # In the case where hourly wages are reported only we multiply the h_mean by 2080 hours
           a_mean = case_when(hourly & is.na(a_mean) ~ h_mean*2080, 
                              TRUE ~ a_mean),
           a_median = case_when(hourly & is.na(a_median) ~ h_median*2080, 
                                TRUE ~ a_median),
           a_pct10 = case_when(hourly & is.na(a_pct10) ~ h_pct10*2080, 
                               TRUE ~ a_pct10),
           a_pct25 = case_when(hourly & is.na(a_pct25) ~ h_pct25*2080, 
                               TRUE ~ a_pct25),
           a_pct75 = case_when(hourly & is.na(a_pct75) ~ h_pct75*2080, 
                               TRUE ~ a_pct75),
           a_pct90 = case_when(hourly & is.na(a_pct90) ~ h_pct90*2080, 
                               TRUE ~ a_pct90)) %>% 
    # there are several median values that are still NA values because they report wages greater than $100/hour or $208,000/year
    select(-c(hourly, annual)) %>% 
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
      temp_df <- read_xls(here(paste0("data/occ_macro_vars/OEWS/", temp_name)), skip = 37)
      print(names(temp_df))
      temp_df <- temp_df %>% clean_oews(., yr, limits_df)
    }
    else if(yr > 2000 & yr < 2014){
      temp_df <- read_xls(here(paste0("data/occ_macro_vars/OEWS/", temp_name))) 
      print(names(temp_df))
      temp_df <- temp_df %>% 
        clean_oews(., yr, limits_df) 
    }else{
      temp_df <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/", temp_name))) 
      print(names(temp_df))
      temp_df <- temp_df %>% 
        clean_oews(., yr, limits_df) 
    }
  }else{
    print("One or multiple files found for that year.")
    break
  }
  occ_wages <- rbind(occ_wages, temp_df)
}

occ_wages %>% group_by(occ_code) %>% select(-occ_group) %>% filter(if_any(everything(), is.na)) %>% arrange(occ_code, year)

################################################################################
################################################################################
########################### FULL OMN  ##########################################
################################################################################
################################################################################
abm_occs <- read.csv(here('data/crosswalk_occ_soc_cps_codes_full_omn.csv')) %>% 
  tibble %>% 
  mutate(SOC2010_cleaned = gsub("X", "0", SOC2010)) %>% 
  mutate(OCC2010_match_cps = as.character(OCC2010_cps))

ipums_vars <- read.csv(here("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/occ_names_employment_asec_occ_ipums_vars.csv")) %>% tibble %>%
  mutate(acs_occ_code = occ) %>%
  rename(id = `X.1`)

# # Crosswalk from ABM data - written in occ_soc_cps_codes_crosswalk.R
# abm_occs <- read.csv(here("data/crosswalk_occ_soc_cps_codes.csv")) %>% 
#   tibble %>% 
#   mutate(SOC2010 = gsub("X", "0", SOC2010))

# Confirm that all OCS codes in abm_vars are present in the crosswalk
abm_occs %>% filter(!(SOC2010_cleaned %in% unique(occ_wages$occ_code))) %>% nrow(.) == 0
abm_occs %>% filter(!(SOC2010_cleaned %in% unique(occ_wages$occ_code)))

occ_wages_mean <- occ_wages %>% 
  group_by(occ_code) %>% 
  summarise(across(c(h_mean, a_mean, h_pct10, h_pct25, h_median, h_pct75, h_pct90, a_pct10, a_pct25, a_median, a_pct75, a_pct90), ~mean(., na.rm = TRUE)))

wage_dist_temp <- abm_occs %>% 
  left_join(., occ_wages_mean, by = c("SOC2010_cleaned" = "occ_code")) %>% 
  select(-c(X, OCC2010_cps, acs_occ_code_label, major_cat, OCC2010_desc, OCC2010, SOC2010_cleaned, OCC2010_match_cps))

stopifnot(identical(wage_dist_temp$acs_occ_code, ipums_vars$acs_occ_code))

saveRDS(wage_dist_temp, here("data/occ_macro_vars/OEWS/wage_distributions_full_omn.csv"))


################################################################################
################################################################################
########################### ONET  ##############################################
################################################################################
################################################################################
abm_occs <- read.csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/acs_onet_2010_soc_cw.csv") %>% tibble %>% 
  rename(SOC2010 = soc_2010_code,
         acs_occ_code = acs_2010_code,
         label = title) %>% 
  # Hunters and trappers do not have a wage estimate in the network so we give them as we do in the occupational target demand setting process: In the case of the ONET Related Occupations Network both of the above are missing in addition to 45-3021 (hunters and trappers). We assign hunters and trappers the same occupational shocks as the only other occupation in the same Minor SOC Group (45-3011 - Fishers and Related Fishing Workers). They are both categorised within the same "Fishing and Hunting Workers" Minor SOC Group (45-3000).
  mutate(SOC2010 = case_when(SOC2010 == "45-3021" ~ "45-3011",
                   TRUE ~ SOC2010)) %>% 
  group_by(acs_occ_code, label) %>% 
  summarise(SOC2010 = first(SOC2010))  %>% 
  ungroup
  

ipums_vars <- read.csv(here("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/acs_onet_2010_ipums_vars.csv")) %>% tibble %>% 
  rename(acs_occ_code = occ)

# Confirm that all OCS codes in abm_vars are present in the crosswalk
stopifnot(abm_occs %>% filter(!(SOC2010 %in% unique(occ_wages_mean$occ_code))) %>% nrow(.) == 0)

wage_dist_temp <- abm_occs %>% 
  left_join(., occ_wages_mean, by = c("SOC2010" = "occ_code")) %>% 
  select(-c(label))

stopifnot(identical(wage_dist_temp$acs_occ_code, ipums_vars$acs_occ_code))

saveRDS(wage_dist_temp, here("data/occ_macro_vars/OEWS/wage_distributions_onet.csv"))

# # Convert all occupational categories to 2010 codes 
# occs_00_10 <- read_xls(here("data/occ_macro_vars/OEWS/soc_2000_to_2010_crosswalk.xls"), skip = 6) %>% 
#   clean_names() %>% 
#   filter(!if_all(everything(), is.na)) 
# 
# #write.csv(occs_00_10, here("data/occ_macro_vars/OEWS/soc_2000_to_2010_crosswalk.csv"))
# 
# occs_00_10_short <- occs_00_10 %>% 
#   # Retain only those codes that are not equivalent across the years - 79 in the 00-10 code
#   filter(x2010_soc_code != x2000_soc_code)

# occs_10_18 <- read_xlsx(here("data/occ_macro_vars/OEWS/soc_2010_to_2018_crosswalk.xlsx"), skip = 6) %>% 
#   clean_names() %>% 
#   filter(!if_all(everything(), is.na)) 
# 
# occs_10_18_short <- occs_10_18 %>%
#   # Retain only those codes that are not equivalent across the years - 148 in the 10-18 code
#   filter(x2010_soc_code != x2018_soc_code)
# 
# identical(sort(unique(occs_00_10$x2010_soc_code)), sort(unique(occs_10_18$x2010_soc_code)))
# 
# occ_wages %>% 
#   filter(occ_code %in% c(abm_occs$SOC2010)) %>% 
#   ggplot() + 
#   geom_line(aes(x = year, y = tot_emp, group = occ_code))


# occ_wages %>% 
#   select(year, occ_code) %>% 
#   write.csv(here("data/occ_macro_vars/OEWS/occ_wages_short_oews.csv"))

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






# # OEWS Wage Distributions
# 
# library(tidyverse)
# library(rvest)
# library(devtools)
# 
# install_github("mikeasilva/blsAPI")
# library(blsAPI)
# library(rjson)
# 
# ## One or More Series, Specifying Years 
# payload <- list('seriesid'=c('LAUCN040010000000005','LAUCN040010000000006'), 'startyear'='2010', 'endyear'='2012') 
# payload <- list(
#   'seriesid' = c('OEUM00000000000000000000000000000315'),
#   'startyear' = '2022',
#   'endyear' = '2022'
# )
# 
# # Call API
# response <- blsAPI(payload)
# json_data <- jsonlite::fromJSON(response, flatten = TRUE)
# 
# 
# # tidy the nested structure
# df <- json_data$Results$series %>%
#   select(seriesID, data) %>%
#   unnest_longer(data) %>%
#   unnest_wider(data)
# 
# # clean types
# df <- df %>%
#   mutate(
#     year   = as.integer(year),
#     value  = as.numeric(value)
#   )
# 
# df %>% select(-footnotes) %>% pivot_wider(id_cols = c(year, period, periodName), names_from = seriesID, values_from = value)
# 
# 
# oe_occupation <- read.delim("https://download.bls.gov/pub/time.series/oe/oe.occupation")
# 
# 
# library(httr)
# library(readr)
# # Variable codes
# data_type_url <- "https://download.bls.gov/pub/time.series/oe/oe.datatype"
# 
# # Request the file with a browser-like user-agent
# res <- GET(data_type_url, user_agent("Oxford PhD student ebba.mark@gmail.com"))
# stop_for_status(res)
# 
# # Read the content into R as a tibble
# oe_datatype <- read_tsv(content(res, as = "raw"))
# data_types <- oe_datatype %>% 
#   filter(grepl("Annual", datatype_name))
# 
# # Occupation Codes
# occ_url <- "https://download.bls.gov/pub/time.series/oe/oe.occupation"
# 
# # 1. Request the file with a browser-like user-agent
# res <- GET(data_type_url, user_agent("Oxford PhD student ebba.mark@gmail.com"))
# stop_for_status(res)
# 
# # 2. Read the content into R as a tibble
# oe_datatype <- read_tsv(content(res, as = "raw"))
# data_types <- oe_datatype %>% 
#   filter(grepl("Annual", datatype_name))
# 
# series_id <- "OEUM000040000000000000001"
# 
# nchar(series_id)
# 
# payload <- list(
#   'seriesid' = c(series_id),
#   'startyear' = '2019',
#   'endyear' = '2024'
# )
# 
# # Call API
# response <- blsAPI(payload)
# json_data <- jsonlite::fromJSON(response, flatten = TRUE)
# 
# json_data
# 
# # tidy the nested structure
# df <- json_data$Results$series %>%
#   select(seriesID, data) %>%
#   unnest_longer(data) %>%
#   unnest_wider(data)
# 
# # clean types
# df <- df %>%
#   mutate(
#     year   = as.integer(year),
#     value  = as.numeric(value)
#   )
# 
# df %>% select(-footnotes) %>% pivot_wider(id_cols = c(year, period, periodName), names_from = seriesID, values_from = value)
# 
# 
# url <- "https://download.bls.gov/pub/time.series/oe/oe.data.1.AllData"
# res <- GET(url, user_agent("Oxford PhD student ebba.mark@gmail.com"))
# 
# # Create a temporary connection
# con <- rawConnection(content(res, "raw"))
# 
# # Read only first 1000 rows
# oe_sample <- read_tsv(con, skip = 1000, n_max = 1000)
# 
# close(con)
# 
# # Inspect
# glimpse(oe_sample)
# head(oe_sample)
# 
# survey = "OE"
# seasonality = "U" 
# area_type_code = M 
# area_code = 0010180 
# industry = "000000" 
# occupation = "000000" 
# data_type = "01"
