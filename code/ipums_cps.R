# Description: Extracting long-term unemployment rate by occupation
# Cite the following:
# Publications and research reports based on the IPUMS CPS database must cite it appropriately. The citation should include the following:
#   
# Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 11.0 [dataset]. Minneapolis, MN: IPUMS, 2023.
# https://doi.org/10.18128/D030.V11.0
# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

library(tidyverse)
library(here)
library(lubridate)

if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

# ddi2 <- read_ipums_ddi(here("data/macro_vars/CPS_LTUER/cps_00005.xml"))
# data2 <- read_ipums_micro(ddi)
# data %>% select(names(data2)) %>% identical(data2)

ddi <- read_ipums_ddi(here("data/macro_vars/CPS_LTUER/cps_00006.xml"))
data <- read_ipums_micro(ddi)

# Reference data from FRED
macros <- read.csv(here('data/macro_vars/CPS_LTUER/LNS13025703 (2).csv')) %>%
  tibble %>% 
  rename(LTUER = 2) %>% 
  mutate(YEAR = year(DATE),
         LTUER = as.numeric(LTUER)/100)

ipums_conditions()

test <- data %>% 
  filter(YEAR > 2003 & EMPSTAT %in% c(21, 22)) %>%  # Occupation classification scheme was redone in 2002, effective in 2003: https://cps.ipums.org/cps/occ_transition_2002_2010.shtml ; also filted to include only unemployed workers. 
  select(YEAR, MONTH, OCC2010, WKSUNEM1, WKSUNEM2) %>% # Select only relevant variables.
  mutate(across(c(WKSUNEM1), ~ifelse(. == 99, NA, .))) %>% # Replace "99" value with NA - 99 = NIU ie. "Not in universe" or simply not applicable to the correspondent. See explanation here: https://cps.ipums.org/cps-action/faq (Ctrl+F "NIU")
  mutate(LTUE1 = WKSUNEM1 >= 27,
         LTUE2 = WKSUNEM2 %in% c(5,6,7)) %>% # Counts the categories 5 (27-39 weeks), 6 (40+ weeks), and 7 (Over 26 weeks - period 1962-1967) as LTUE
  # LTUER by Occupation
  group_by(YEAR, OCC2010) %>% # Group by year, month, and occupation
  summarise(n_ue_occ = n(), # labour force in particular occupation
         n_ltue_occ = sum(LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ)) %>% # calculate LTUER as N_LTUE/N_LF 
  # Total LTUER - same calculation as above
  group_by(YEAR) %>% 
  mutate(n_ue = sum(n_ue_occ),
         n_ltue = sum(n_ltue_occ, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer = n_ltue/n_ue)

test %>% 
  select(YEAR, ltuer) %>% 
  distinct() %>% 
  ggplot() +
  geom_line(aes(x = YEAR, y = ltuer)) +
  geom_line(data = filter(macros, YEAR >= 2004), aes(x = YEAR, y = LTUER), linetype = "dashed")

test %>% 
  select(YEAR, OCC2010, ltuer_occ) %>% 
  # 25% of occupational ltuer rates missing....
  complete(YEAR, OCC2010) %>% 
  ggplot() +
  geom_line(aes(x = YEAR, y = ltuer_occ, color = as.factor(OCC2010))) +
  theme(legend.position = "none")


test %>% 
  select(YEAR, OCC2010, ltuer_occ) %>% 
  # 25% of occupational ltuer rates missing....
  complete(YEAR, OCC2010) %>% 
  ggplot() +
  geom_line(aes(x = YEAR, y = ltuer_occ, color = as.factor(OCC2010))) +
  theme(legend.position = "none")
  

