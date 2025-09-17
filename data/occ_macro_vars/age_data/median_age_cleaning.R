###################
### Median occupation age
### Accessed here: https://www.bls.gov/cps/cps_aa2016.htm
### 11b. Employed persons by detailed occupation and age (HTML) (PDF) (XLSX)

library(tidyverse)
library(here)
library(readxl)
library(janitor)

age <- read_xlsx(here("data/occ_macro_vars/age_data/cpsaat11b.xlsx"), skip = 4) %>% 
  slice(-c(1,2,3)) %>% 
  clean_names %>% 
  rename("label" = x1) %>% 
  select(label, median_age) %>% 
  mutate(median_age = as.numeric(median_age),
         label = tolower(label))

ipums <- read.csv(here('dRC_Replication/data/ipums_variables.csv')) %>% 
  tibble %>% 
  mutate(label = trimws(label))

setdiff(age$label, ipums$label)
setdiff(ipums$label, age$label)


age <- age %>% 
  # No observation for "legislators"
  mutate(label = case_when(label == "chief executives" ~ "chief executives and legislators",
                           TRUE ~ label))

temp <- ipums %>% 
  select(label) %>% 
  left_join(age, by = 'label') %>% 
  filter(is.na(median_age))

temp 

age %>% 
  filter(!(label %in% ipums$label))

# Where possible, replace NA values with values from similar occupations
age %>% 
  mutate(median_age = case_when(is.na(median_age) & label == "compensation and benefits managers" ~ age$median_age[age$label %in% "human resources managers"],
         TRUE ~ median_age)) 









# All categories now either have a reported gender share OR another ipums label that is arguably similar
# This will need to be documented
#gend_shares %>% 
#  group_by(ipums_label) %>% 
  # Testing for line items that do not have a gender share NOR a similar category
  #filter(all(is.na(women_pct)) & all(is.na(ipums_similar)))

temp <- gend_shares %>% 
  # group by ipums_similar in the cases where they are a sum value of 
  group_by(ipums_label) %>% 
  summarise(women_pct = mean(women_pct, na.rm = TRUE),
            ipums_similar = unique(ipums_similar)) %>%
  ungroup


## Checking to see if all "similar" items have  a reported gender share
similars_comp <- temp %>% 
  filter(ipums_label %in% temp$ipums_similar & !is.na(women_pct)) %>% 
  select(ipums_label, women_pct)


# Replace women_pct where missing and a similar category has been identified
final_shares <- temp %>% 
  left_join(., similars_comp, by = c("ipums_similar" = "ipums_label")) %>% 
  mutate(women_pct = ifelse(is.na(women_pct.x),women_pct.y, women_pct.x)) %>% 
  select(ipums_label, women_pct) %>% 
  mutate(women_pct = women_pct/100)


setdiff(final_shares$ipums_label, ipums$label)
setdiff(ipums$label, final_shares$ipums_label)

#ipums %>% left_join(., final_shares, by = c("label" = "ipums_label")) %>% select(-12) %>% identical(ipums)
#ipums %>% left_join(., final_shares, by = c("label" = "ipums_label")) %>% write.csv(here("data/ipums_variables_w_gender.csv"))


