### CLEANING OEWS DATA - industry-specific occupational employment
# Occupational Employment and Wage Statistics
# 
# https://www.bls.gov/oes/tables.htm
# Download the industry-specific table for each

library(tidyverse)
library(here)
library(conflicted)
library(janitor)
library(readxl)
library(assertthat)
conflict_prefer_all("dplyr", quiet = TRUE)


read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes00in3/nat2d_sic_2000_dl.xls")),
               skip = 33)

df <- tibble()
for(yr in 1999:2024){
  yr_tmp <- substr(yr, 3,4)
  print(yr_tmp)
  if(yr == 1999){
    tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in3/nat2d_sic_", as.character(yr), "_dl.xls")), skip = 35) 
  }else if(yr == 2000){
    tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in3/nat2d_sic_", as.character(yr), "_dl.xls")),  skip = 33)
  }else if(yr == 2001){
    tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in3/nat2d_sic_", as.character(yr), ".xls"))) 
  }else if(yr == 2002){
    tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in4/nat4d_", as.character(yr), "_dl.xls")))
  }else if(yr == 2003){
    tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/nat3d_may", as.character(yr), "_dl.xls"))) 
  }else if (yr > 2003 & yr < 2008){
    tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/natsector_may",as.character(yr), "_dl.xls"))) 
  }else if(yr >= 2008 & yr < 2014){
    tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/natsector_M",as.character(yr), "_dl.xls"))) 
  }else{
    tmp <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/natsector_M",as.character(yr), "_dl.xlsx")))  
  }
  
  df <- tmp %>% 
    clean_names %>% 
    mutate(year = yr) %>% 
    select(-contains("pct_rpt")) %>% 
    mutate(across(c(pct_total, a_mean, emp_prse, mean_prse), ~as.numeric(.)),
           annual = as.logical(annual)) %>% 
    bind_rows(df, .)
}

# Which occupations are in the crosswalk. 
abm <- read.csv(here("data/crosswalk_occ_soc_cps_codes.csv")) %>% 
  tibble %>% 
  mutate(SOC2010 = gsub("X", "0", SOC2010))

abm %>% filter(!(SOC2010 %in% df$occ_code)) %>% nrow(.) == 0

temp <- df %>% select(year, naics, naics_title, sic, sic_title, occ_code, occ_title, tot_emp, pct_total) %>% 
  filter(naics %in% c('11', '21', '22', '23', '31-33', '42', '44-45', '48-49', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81') & occ_code %in% abm$SOC2010) %>% 
  group_by(naics) %>% 
  # Make NAICS titles consistent with latest attributed title
  mutate(naics_title = last(naics_title)) %>% 
  ungroup %>% 
  select(year, naics, naics_title, occ_code, pct_total) %>% 
  distinct

  # group_by(occ_code, naics, naics_title) %>% 
  # fill(pct_total, .direction = "down") %>% 
  # ungroup %>% 
temp %>% 
  group_by(occ_code, naics, naics_title) %>% 
  fill(pct_total, .direction = "down") %>% 
  ungroup %>% 
  ggplot() + 
  geom_area(aes(x = year, y = pct_total, fill = occ_code)) +
  facet_wrap(~naics_title) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(x = "Year", y = "Reported Pct Shares", title = "Occupational Employment Shares by 2-digit NAICS")

temp %>% 
  group_by(occ_code, naics, naics_title) %>% 
  fill(pct_total, .direction = "down") %>% 
  ungroup %>% 
  group_by(naics_title, year) %>% 
  mutate(tot = sum(pct_total, na.rm = TRUE),
         pct_total_norm = pct_total/tot) %>% 
  ungroup %>% 
  ggplot() + 
  geom_area(aes(x = year, y = pct_total_norm, fill = occ_code), stat = "identity") +
  facet_wrap(~naics_title) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(x = "Year", title = "Occupational Employment Shares by 2-digit NAICS", y= "Shares of Total")
