### Cleaning Unemployment Insurance Nonfilers ###
### BLS Supplement to the CPS 2018 and 2022 Annual Surveys ###
library(here)
library(tidyverse)
library(pdftools)
library(readr)

# 2022 files
bea_2022 <- read.csv(here('data/behav_params/CPS_BLS_Supplement_18_22/febmay22pub.csv')) 

# 
# PEA1 2
# Now, we also have a few questions about your
# experience looking for a new job over the last 2 months.
# How many jobs (have you/has name) applied for, if any,
# in the last 2 months?
#   1085 -1086
# EDITED UNIVERSE:
#   PELK = 1 or
# PEDWLK = 1 or
# PELAYLK = 1
# VALID ENTRIES:
#   0 0
# 1 1 to 10
# 2 11 to 20
# 3 21 to 80
# 4 81 or more
# -2 Don't Know
#  -3 Refused
#  -9 No Response
#  
#  PRUNEDUR 3 DURATION OF UNEMPLOYMENT FOR 407 - 409
#  LAYOFF AND LOOKING RECORDS
#  EDITED UNIVERSE: PEMLR = 3-4
# in weeks!!!
#  VALID ENTRIES
#  0 MIN VALUE
#  119 MAX VALUE
#  Topcoded consistent with PELAYDUR or PELKDUR,
#  as appropriate, starting April 2011.\

cps_codebook <- pdf_text(here('data/behav_params/CPS_BLS_Supplement_18_22/cpsmaysep18.pdf')) %>% 
  lapply(., function(x) str_split(x, "\n")) %>% 
  .[22:137] %>% 
  unlist

temp_book <- cps_codebook[cps_codebook!="" & !grepl("^NAME   ", cps_codebook)] %>% 
  tibble %>% 
  rename(code = ".") %>% 
  mutate(code =  trimws(str_sub(code, 1, 14))) %>% 
  filter(code != "") %>% 
  separate(code, sep = " +", into = c("code", "len")) %>% 
  filter(code != "NAME")

temp_book %>% 
  filter(is.na(len)) %>% pull(code) -> missing

result <- cps_codebook[sapply(cps_codebook, function(x) any(startsWith(x, missing)))]
missing_codes <- result %>% 
  tibble %>% 
  rename(code = ".") %>% 
  mutate(code = gsub(" +", " ", code)) %>% 
  separate(code, " ", into = c("code", "len_missing")) %>% filter(code != "PADDING")

temper <- temp_book %>% 
  left_join(., missing_codes, by = "code") %>% 
  mutate(len = ifelse(is.na(len) & !is.na(len_missing), len_missing, ifelse(is.na(len) & is.na(len_missing), 6, len))) %>% 
  select(-len_missing) %>% 
  mutate(len = as.numeric(len),
         len = ifelse(row_number() == 45, 4, 
                      ifelse(row_number() == 410, 47, len)))
  
temper$start <- c(1, head(cumsum(temper$len) + 1, -1))

final_codebook_2018 <- temper %>% 
  mutate(stop = start + len - 1)

# 2018 files
bea_2018 <- read_fwf(here('data/behav_params/CPS_BLS_Supplement_18_22/maysep18pub.dat'), fwf_widths(final_codebook_2018$len, final_codebook_2018$code))

