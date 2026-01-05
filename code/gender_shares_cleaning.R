###################
### Code to create gender shares of each occupation

library(tidyverse)
library(here)
library(readxl)
getwd()

gend_shares <- read_xlsx(here("data/occ_gender_shares.xlsx"), skip = 2, na = "-", 
                         col_types = c("text", "text", "text", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "text"),
                         col_names = c("occupation", "ipums_label", "ipums_similar", "total_employed", "women_pct", "white_pct", "black_afam_pct", "asian_pct", "hispanic_latinx_pct", "notes")) %>% 
  mutate(occupation = tolower(occupation)) %>% 
  filter(!is.na(ipums_label)) %>% 
  select(occupation, ipums_label, ipums_similar, women_pct)
         #ipums_label = trimws(ipums_label))

ipums <- read.csv(here('dRC_Replication/data/ipums_variables.csv')) %>% 
  tibble %>% 
  mutate(label = trimws(label))

#setdiff(gend_shares$ipums_label, ipums$label)
#setdiff(ipums$label, gend_shares$ipums_label)


#gend_shares %>% 
#  filter(is.na(women_pct)) 


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
ipums %>% left_join(., final_shares, by = c("label" = "ipums_label")) %>% write.csv(here("data/ipums_variables_w_gender.csv"))


