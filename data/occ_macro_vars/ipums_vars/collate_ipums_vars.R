# Downloading occupation-level values from IPUMS
# Employment
# Unemployment
# Gender share
# Earnings
# Education
# Age

library(tidyverse)
library(here)
library(patchwork)
library(lubridate)
library(readxl)
library(ipumsr)
library(ggstats)
library(janitor)
source(here('code/formatting/plot_dicts.R'))

save_new = FALSE

ddi <- read_ipums_ddi(here("data/occ_macro_vars/ipums_vars/cps_00021.xml"))
data <- read_ipums_micro(ddi) %>% 
  clean_names

asec_data_raw <- data %>% 
  filter(!is.na(asecwt))

cps_data_raw <- data %>% 
  filter(!is.na(hwtfinl))


################################################################################
################################################################################
################################################################################
##### Crosswalking ASEC to SOC Codes #####
source(here('data/occ_soc_cw_2010.R'))

asec_data <- asec_data_raw %>% 
  left_join(., occ_soc_cw, by = c("occ2010" = "OCC"))

# Asserts that no OCC codes were not matched
stopifnot(asec_data %>% select(occ, occ2010, contains("soc")) %>%
            # Filter out military (9830) and NIU (9999) occ codes
            filter(!(occ2010 %in% c(9999, 9830))) %>% 
            summarise(across(everything(), ~sum(is.na(.)))) %>% rowSums(.) == 0)

# Asserts that asec_data is identical to asec_data_raw excluding the names in occ_soc_cw -convert occ2010 to integer to remove labels - otherwise does not pass identical test
stopifnot(asec_data %>% mutate(occ2010 = as.integer(occ2010)) %>% select(-any_of(names(occ_soc_cw))) %>% 
            identical(mutate(asec_data_raw, occ2010 = as.integer(occ2010))))

# Also passes if we remove occ2010 variable
stopifnot(asec_data %>% select(-any_of(names(occ_soc_cw)), -occ2010) %>%
            identical(select(asec_data_raw, -occ2010)))

ipums_conditions()

stopifnot(nrow(asec_data_raw) + nrow(cps_data_raw) - nrow(data) == 0)

# First, from ASEC data
stopifnot(asec_data_raw %>% 
  group_by(year, month, cpsidp) %>% 
  n_groups(.) == nrow(asec_data_raw))

print("Starting ASEC cleaning.")
for(occ_cat in c("occ2010", "occ", "soc_2010", "soc_2010_major", "soc_2010_minor", "soc_2010_broad")){
  print(occ_cat)
  occ_sym <- sym(occ_cat)
  # Annual dataset so can exclude month variable
  asec_data_annual <- asec_data %>% 
    filter(age >= 18 & !!occ_sym != 0) %>% 
    mutate(employed = empstat %in% c(10, 12),
           unemployed = empstat %in% c(21, 22)) %>%  
    select(-month) %>% 
    group_by(year, !!occ_sym) %>% 
    # Counts total labour force including At work (10), NILF (retired) (36), Has job (12), NILF Other (34), NILF Unable to work (32), Unemployed (21, 22) / Excludes Armed Forces (1) and NIU (0)
    summarise(lf_total = sum(asecwt*!(empstat %in% c(0,1))), # Excludes armed forces and NIU
              lf = sum(asecwt*empstat %in% c(10, 12, 21, 22)), # Excludes NILF (36, 34, 32) and armed forces (1) and NIU (0) as above
              emp = sum(asecwt*employed), 
              unemp = sum(asecwt*unemployed),
              male = sum(asecwt*(sex == 1 & (employed | unemployed))),
              female = sum(asecwt*(sex == 2 & (employed | unemployed))),
              female_share = female/(male + female),
              mean_weekly_earnings = weighted.mean(ifelse(round(earnweek) != 10000 & employed, earnweek, NA), na.rm = TRUE, weights = asecwt),
              median_weekly_earnings = ifelse(all(!(round(earnweek) != 10000 & employed)), NA, weighted.median(ifelse(round(earnweek) != 10000 & employed, earnweek, NA), na.rm = TRUE, w = asecwt)),
              mean_age = weighted.mean(ifelse(employed | unemployed, age, NA), na.rm = TRUE, weights = asecwt),
              median_age = ifelse(all(!(employed | unemployed)), NA, weighted.median(ifelse(employed | unemployed, age, NA), na.rm = TRUE, w = asecwt)),
              mean_educ = weighted.mean(ifelse(educ != 1 & (employed | unemployed), educ, NA), na.rm = TRUE, w = asecwt),
              median_educ = weighted.median(ifelse(educ != 1 & (employed | unemployed), educ, NA), na.rm = TRUE, w = asecwt),
              .groups = "drop") %>% 
    ungroup %>% 
    arrange(!!occ_sym, year) %>% 
    group_by(!!occ_sym) %>% 
    fill(median_weekly_earnings, mean_weekly_earnings,.direction = "downup") %>% 
      ungroup()
  
  if(save_new){
    write.csv(asec_data_annual, here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_asec_annual_", occ_cat, ".csv")))
    print("Saved annual ASEC file.")
  }else{
    if(occ_cat %in% c("occ", "occ2010")){
      tester <- mutate(asec_data_annual, !!occ_cat := as.integer(!!occ_sym))
    }else{tester <- asec_data_annual}
    print(paste0("Identical to previously saved file?: ", 
                 all.equal(mutate(tester, year = as.integer(year)), 
                           tibble(select(read.csv(here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_asec_annual_", occ_cat, ".csv"))), -X)))))
  }
  
  asec_data_mean <- asec_data_annual %>% 
    group_by(!!occ_sym) %>% 
    summarise(n_years = n(),
              n_years_explicit = n_distinct(year),
              min_year = min(year),
              max_year = max(year), 
              across(everything(), ~mean(., na.rm = TRUE)))
  
  stopifnot(asec_data_mean %>% filter(n_years != n_years_explicit) %>% nrow(.) == 0)
  
  if(save_new){
    write.csv(asec_data_mean, here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_asec_mean_", occ_cat, ".csv")))
    print("Saved mean ASEC file.")
  }else{
    if(occ_cat %in% c("occ", "occ2010")){
      tester <- mutate(asec_data_mean, !!occ_cat := as.integer(!!occ_sym))
    }else{tester <- asec_data_mean}
    print(paste0("Identical to previously saved file?: ", 
                 all.equal(mutate(tester), 
                           tibble(select(read.csv(here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_asec_mean_", occ_cat, ".csv"))), -X)))))
  }
  
  p1 <- asec_data_mean %>% 
    ggplot(aes(x = reorder(as.factor(!!occ_sym), n_years))) +
    geom_segment(aes(y = min_year, yend = max_year), color = "darkgreen", linewidth = 0.1) + 
    #geom_segment(aes(y = n_years, yend = min_year - 2000), color = "darkgreen") +
    geom_point(aes(y = n_years + 2000), color = "purple", size = 0.5) +
    coord_flip()
  
  print(p1)
}
  
gc()

################################################################################
print("Starting CPS cleaning.")
# CPS Data

ddi <- read_ipums_ddi(here("data/occ_macro_vars/ipums_vars/cps_00022.xml"))
data <- read_ipums_micro(ddi) %>% 
  clean_names

cps_data_raw <- data %>% 
  left_join(., occ_soc_cw, by = c("occ2010" = "OCC"))

# Asserts that no OCC codes were not matched
stopifnot(cps_data_raw %>% select(occ, occ2010, contains("soc")) %>%
            # Filter out military (9830) and NIU (9999) occ codes
            filter(!(occ2010 %in% c(9999, 9830))) %>% 
            summarise(across(everything(), ~sum(is.na(.)))) %>% rowSums(.) == 0)

# Asserts that asec_data is identical to asec_data_raw excluding the names in occ_soc_cw -convert occ2010 to integer to remove labels - otherwise does not pass identical test
stopifnot(cps_data_raw %>% mutate(occ2010 = as.integer(occ2010)) %>% select(-any_of(names(occ_soc_cw))) %>% 
            identical(mutate(data, occ2010 = as.integer(occ2010))))

# Also passes if we remove occ2010 variable
stopifnot(cps_data_raw %>% select(-any_of(names(occ_soc_cw)), -occ2010) %>%
            identical(select(data, -occ2010)))

# Check which variables are missing from the entire dataset
cps_data_raw %>% summarise(across(everything(), ~sum(!is.na(.))))

for(occ_cat in c("occ2010", "occ", "soc_2010", "soc_2010_major", "soc_2010_minor", "soc_2010_broad")){
  print(occ_cat)
  occ_sym <- sym(occ_cat)
  
  cps_monthly <- cps_data_raw %>% 
    # Removing variables where all values are NA from above line
    # select(-c(incwage, asecwt, asecwth, hflag)) %>% 
    filter(age >= 18 & !!occ_sym != 0) %>% 
    mutate(employed = empstat %in% c(10, 12),
           unemployed = empstat %in% c(21, 22)) %>% #select(hourwage2, qearnwee, hourwage, earnweek, qhourwag)
    group_by(year, month, !!occ_sym) %>% 
    # Counts total labour force including At work (10), NILF (retired) (36), Has job (12), NILF Other (34), NILF Unable to work (32), Unemployed (21, 22) / Excludes Armed Forces (1) and NIU (0)
    summarise(lf_total = sum(wtfinl*!(empstat %in% c(0,1))), # Excludes armed forces and NIU
              lf = sum(wtfinl*empstat %in% c(10, 12, 21, 22)), # Excludes NILF (36, 34, 32) and armed forces (1) and NIU (0) as above
              emp = sum(wtfinl*employed), 
              unemp = sum(wtfinl*unemployed),
              male = sum(wtfinl*(sex == 1 & (employed | unemployed))),
              female = sum(wtfinl*(sex == 2 & (employed | unemployed))),
              female_share = female/(male + female),
              mean_weekly_earnings = weighted.mean(ifelse(round(earnweek) != 10000 & employed, earnweek, NA), na.rm = TRUE, weights = wtfinl),
              median_weekly_earnings = ifelse(all(!(round(earnweek) != 10000 & employed)), NA, weighted.median(ifelse(round(earnweek) != 10000 & employed, earnweek, NA), na.rm = TRUE, w = wtfinl)),
              mean_age = weighted.mean(ifelse(employed | unemployed, age, NA), na.rm = TRUE, weights = wtfinl),
              median_age = ifelse(all(!(employed | unemployed)), NA, weighted.median(ifelse(employed | unemployed, age, NA), na.rm = TRUE, w = wtfinl)),
              mean_educ = weighted.mean(ifelse((employed | unemployed), educ, NA), na.rm = TRUE, w = wtfinl),
              median_educ = ifelse(all(!(employed | unemployed)), NA, weighted.median(ifelse((employed | unemployed), educ, NA), na.rm = TRUE, w = wtfinl)),
              .groups = "drop") %>% 
    ungroup %>% 
    arrange(!!occ_sym, year, month) %>% 
    group_by(!!occ_sym) %>% 
    fill(median_weekly_earnings, mean_weekly_earnings,.direction = "downup") %>% 
    ungroup()
  
  cps_annual <- cps_monthly %>% 
    group_by(year, !!occ_sym) %>% 
    summarise(n_months = n(),
              n_months_explicit = n_distinct(month),
              min_month = min(month),
              max_month = max(month), 
              across(everything(), ~mean(., na.rm = TRUE))) %>% 
    ungroup
  
  if(!(occ_cat %in% c("occ2010", "occ"))){
    write.csv(cps_annual, here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_cps_annual_", occ_cat, ".csv")))
    print("Saved annual CPS file.")
  }else{
    print(cps_annual %>% mutate(!!occ_cat := as.integer(!!occ_sym), year = as.integer(year)) %>% 
            all.equal(tibble(select(read.csv(here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_cps_annual_", occ_cat, ".csv"))), -X))))
    
  }
  
  cps_data_mean <- cps_annual %>% 
    group_by(!!occ_sym) %>% 
    summarise(n_years = n(),
              n_years_explicit = n_distinct(year),
              min_year = min(year),
              max_year = max(year), 
              across(everything(), ~mean(., na.rm = TRUE)))

  if(!(occ_cat %in% c("occ2010", "occ"))){
    write.csv(cps_data_mean, here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_cps_mean_", occ_cat, ".csv")))
    print("Saved mean CPS file.")
  }else{
    print(cps_data_mean %>% mutate(!!occ_cat := as.integer(!!occ_sym)) %>% 
            all.equal(tibble(select(read.csv(here(paste0("data/occ_macro_vars/ipums_vars/", "ipums_vars_cps_mean_", occ_cat, ".csv"))), -X))))
  }
  
  p1 <- cps_data_mean %>% 
    ggplot(aes(x = reorder(as.factor(!!occ_sym), n_years))) +
    geom_segment(aes(y = min_year, yend = max_year), color = "darkgreen", linewidth = 0.1) + 
    #geom_segment(aes(y = n_years, yend = min_year - 2000), color = "darkgreen") +
    geom_point(aes(y = n_years + 2000), color = "purple", size = 0.5) +
    coord_flip()
  
  print(p1)
}

# Testing completeness
for(occ_cat in c("occ2010", "occ", "soc_2010", "soc_2010_major", "soc_2010_minor", "soc_2010_broad")){
  print(occ_cat)
  test <- read.csv(here(paste0("data/occ_macro_vars/ipums_vars/ipums_vars_asec_mean_", occ_cat, ".csv")))
  test %>% summarise(across(everything(), ~sum(is.na(.)))) -> na_testing
  print("NA Testing Mean:")
  if(rowSums(na_testing) != 0){print(na_testing)}
  
  test %>% tibble %>% ggplot() + geom_col(aes(x = !!sym(occ_cat), y = female_share)) + labs(title = as.character(occ_cat))  + common_theme  + theme(legend.position = "none") -> p1
  
  old_test <- test
  
  test <- read.csv(here(paste0("data/occ_macro_vars/ipums_vars/ipums_vars_asec_annual_", occ_cat, ".csv")))
  test %>% summarise(across(everything(), ~sum(is.na(.)))) -> na_testing
  print("NA Testing Annual:")
  if(rowSums(na_testing) != 0){print(na_testing)}
  
  test %>% tibble %>% ggplot() + geom_line(aes(x = year, y = female_share, color = !!sym(occ_cat))) + labs(title = as.character(occ_cat))  + common_theme  + theme(legend.position = "none")-> p2
  
  test %>% left_join(., old_test, by = occ_cat) %>% 
    mutate(err_mean = female_share.x - female_share.y) %>% 
    ggplot() + geom_line(aes(x = year.x, y = err_mean, color = !!sym(occ_cat))) + labs(title = as.character(occ_cat)) + common_theme + theme(legend.position = "none") -> p3

  
  print((p1 + p2)/p3)
}


