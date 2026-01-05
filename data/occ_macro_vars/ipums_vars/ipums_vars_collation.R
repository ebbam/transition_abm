# Downloading occupation-level values from IPUMS
# Employment
# Unemployment
# Gender share
# Earnings
# Education
# Age

library(tidyverse)
library(here)
library(lubridate)
library(readxl)
library(ipumsr)
library(ggstats)
library(janitor)
library



ddi <- read_ipums_ddi(here("data/occ_macro_vars/ipums_vars/cps_00020.xml"))
data <- read_ipums_micro(ddi) %>% 
  clean_names

asec_data <- data %>% 
  filter(!is.na(asecwt))

cps_data <- data %>% 
  filter(!is.na(hwtfinl))

ipums_conditions()

stopifnot(nrow(asec_data) + nrow(cps_data) - nrow(data) == 0)

asec_data %>% 
  filter(age >= 18) %>% 
  # Employment status
  mutate(case_when(empstat %in% c(10, 12) ~ "Employed",
                   empstat %in% c(21, 22) ~ "Unemployed"))

# First, from ASEC data

stopifnot(asec_data %>% 
  group_by(year, month, cpsidp) %>% 
  n_groups(.) == nrow(asec_data))

# Annual dataset so can exclude month variable
asec_data_annual <- asec_data %>% 
  filter(age >= 18 & occ != 0) %>% 
  mutate(employed = empstat %in% c(10, 12),
         unemployed = empstat %in% c(21, 22)) %>%  
  select(-month) %>% 
  group_by(year, occ) %>% 
  # Counts total labour force including At work (10), NILF (retired) (36), Has job (12), NILF Other (34), NILF Unable to work (32), Unemployed (21, 22) / Excludes Armed Forces (1) and NIU (0)
  summarise(lf_total = sum(asecwt*!(empstat %in% c(0,1))), # Excludes armed forces and NIU
            lf = sum(asecwt*empstat %in% c(10, 12, 21, 22)), # Excludes NILF (36, 34, 32) and armed forces (1) and NIU (0) as above
            emp = sum(asecwt*employed), 
            unemp = sum(asecwt*unemployed),
            male = sum(asecwt*(sex == 1 & (employed | unemployed))),
            female = sum(asecwt*(sex == 2 & (employed | unemployed))),
            female_share = female/(male + female),
            mean_earnings = weighted.mean(ifelse(round(earnweek) != 10000 & employed, earnweek, NA), na.rm = TRUE, weights = asecwt),
            median_earnings = ifelse(all(!(round(earnweek) != 10000 & employed)), NA, weighted.median(ifelse(round(earnweek) != 10000 & employed, earnweek, NA), na.rm = TRUE, w = asecwt)),
            mean_age = weighted.mean(ifelse(employed | unemployed, age, NA), na.rm = TRUE, weights = asecwt),
            median_age = ifelse(all(!(employed | unemployed)), NA, weighted.median(ifelse(employed | unemployed, age, NA), na.rm = TRUE, w = asecwt)),
            mean_educ = weighted.mean(ifelse(educ != 1 & (employed | unemployed), educ, NA), na.rm = TRUE, w = asecwt),
            median_educ = weighted.median(ifelse(educ != 1 & (employed | unemployed), educ, NA), na.rm = TRUE, w = asecwt),
            .groups = "drop") %>% 
  ungroup %>% 
  arrange(occ, year) %>% 
  group_by(occ) %>% 
  fill(median_earnings, mean_earnings,.direction = "downup") %>% 
    ungroup()

asec_data_mean <- asec_data_annual %>% 
  group_by(occ) %>% 
  summarise(n_years = n(),
            n_years_explicit = n_distinct(year),
            min_year = min(year),
            max_year = max(year), 
            across(everything(), ~mean(., na.rm = TRUE)))

stopifnot(asec_data_mean %>% filter(n_years != n_years_explicit) %>% nrow(.) == 0)

asec_data_mean %>% 
  ggplot(aes(x = reorder(as.factor(occ), n_years))) +
  geom_segment(aes(y = min_year, yend = max_year), color = "darkgreen", linewidth = 0.1) + 
  #geom_segment(aes(y = n_years, yend = min_year - 2000), color = "darkgreen") +
  geom_point(aes(y = n_years + 2000), color = "purple", size = 0.5) +
  coord_flip()
  


