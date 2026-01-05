library(tidyverse)
library(readxl)
library(here)
library(openxlsx)

# Occupational similarity matrix from Mealy and dRC
temp2 <- read.csv(here("../dRC_Replication/data/occupational_mobility_network.csv")) %>% 
  tibble

acs_codes <- read.csv(here("../dRC_Replication/data/ipums_variables.csv")) %>% 
  tibble %>% 
  pull(acs_occ_code)


temp <- read_xlsx(here("../data/Related Occupations.xlsx")) %>% 
  rename(soc_code = 1, related_soc_code = 3, index = Index) %>% 
  select(soc_code, related_soc_code, index)

# Orhtoptists and Meter Readers, Utilities are the two occupations that are not listed as "related" to other occupations
temp %>% 
  pull(soc_code) %>% 
  unique %>% 
  setdiff(unique(temp$related_soc_code))

# SOC Code from 8-digit to 6-digit (same as excluding the last two values of the soc code)
# https://www.onetcenter.org/crosswalks.html
# test to see whether 6-digit and 8-digit soc codes are the same
read_xlsx(here("../data/2019_to_SOC_Crosswalk.xlsx"), skip = 2) %>%
    rename(soc_code_8 = 1, soc_code_6 = 3)  %>% 
    select(1,3) %>% 
    mutate(soc_code_8 = substr(soc_code_8,1,7),
           test = identical(soc_code_8, soc_code_6)) %>% 
    filter(!test) %>% nrow(.) == 0

# https://www.bls.gov/emp/documentation/crosswalks.htm
soc_acs <- read_xlsx(here("../data/nem-occcode-acs-crosswalk.xlsx"), skip = 4) %>%
  rename(soc_code_6 = 2, acs_code = 4) %>% 
  select(2,4)
  # Less values of ACS than SOC codes 
  #summarise(across(everything(), ~n_distinct(.)))
  # All observations are distinct pairs though
  # distinct

empir_trans_probs <- temp %>% 
  mutate(across(c(soc_code, related_soc_code), ~substr(., 1,7))) %>% 
  left_join(., soc_acs, join_by(soc_code == soc_code_6)) %>% 
  left_join(., soc_acs, join_by(related_soc_code == soc_code_6)) %>% 
  rename(acs_code = acs_code.x, related_acs_code = acs_code.y) %>% 
  filter(acs_code %in% acs_codes & related_acs_code %in% acs_codes) %>% 
  group_by(acs_code, related_acs_code) %>% 
  summarise(index = mean(index, na.rm = TRUE)) %>%  
  ungroup %>% 
  complete(acs_code, related_acs_code, fill = list(index = 0)) %>% 
  group_by(acs_code) %>% 
  mutate(nor_index = 1 - (index/max(index))) %>% 
  pivot_wider(id_cols = acs_code, names_from = related_acs_code, values_from = nor_index, values_fill = 0)

#write.xlsx(empir_trans_probs, here("../data/agnostic_occ_sim_matrix.xlsx"))
  



  
