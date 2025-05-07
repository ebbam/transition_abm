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
library(readxl)

if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

cw <- read.csv(here("data/crosswalk_occ_soc_cps_codes.csv")) %>% 
  tibble

cw_short <- cw %>% 
  select(OCC2010_cps, SOC2010, SOC_major, SOC_minor, SOC_broad) %>% 
  group_by(OCC2010_cps) %>% 
  summarise(SOC2010 = first(SOC2010),
            SOC_major = first(SOC_major),
            SOC_minor = first(SOC_minor),
            SOC_broad = first(SOC_broad)) %>% 
  ungroup
# ddi2 <- read_ipums_ddi(here("data/macro_vars/CPS_LTUER/cps_00005.xml"))
# data2 <- read_ipums_micro(ddi)
# data %>% select(names(data2)) %>% identical(data2)

ddi <- read_ipums_ddi(here("data/occ_macro_vars/CPS_LTUER/cps_00016.xml"))
data <- read_ipums_micro(ddi)

ipums_conditions()

# Reference data from FRED
macros <- read.csv(here('data/macro_vars/LNS13025703.csv')) %>%
  tibble %>% 
  rename(LTUER = 2) %>% 
  mutate(YEAR = year(DATE),
         LTUER = as.numeric(LTUER)/100,
         DATE = as.Date(DATE))

filtered <- data %>% 
  left_join(., cw_short, by = join_by(OCC2010 == OCC2010_cps)) %>% 
  filter(YEAR > 2003 & EMPSTAT %in% c(21, 22)) %>%  # Occupation classification scheme was redone in 2002, effective in 2003: https://cps.ipums.org/cps/occ_transition_2002_2010.shtml ; also filtered to include only unemployed workers. 
  select(YEAR, MONTH, WTFINL, OCC2010, DURUNEMP, DURUNEM2, HWTFINL, names(cw_short[-1])) %>% # Select only relevant variables.
  mutate(DURUNEMP = ifelse(DURUNEMP == 999, NA, DURUNEMP),
         DURUNEM2 = ifelse(DURUNEM2 == 9, NA, DURUNEM2)) %>% # Replace "99" value with NA - 99 = NIU ie. "Not in universe" or simply not applicable to the correspondent. See explanation here: https://cps.ipums.org/cps-action/faq (Ctrl+F "NIU")
  mutate(LTUE1 = DURUNEMP >= 27,
         LTUE2 = DURUNEM2 >= 12) # Counts the categories 5 (27-39 weeks), 6 (40+ weeks), and 7 (Over 26 weeks - period 1962-1967) as LTUE
  
################################################################################
################### NATIONAL RATE ##############################################
################################################################################

ltuer_overall <- filtered %>% 
  # Total LTUER - same calculation as above
  group_by(YEAR, MONTH) %>% 
  summarise(n_ue = sum(WTFINL, na.rm = TRUE),
            n_ltue = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer = n_ltue/n_ue, 
         DATE = as.Date(paste0(YEAR, "-", MONTH, "-01")))
ggplot() +
  geom_line(data = ltuer_overall, aes(x = DATE, y = ltuer)) +
  geom_line(data = filter(macros, YEAR >= 2004), aes(x = DATE, y = LTUER), linetype = "dashed")

ltuer_occ <- filtered %>%   # LTUER by Occupation
  group_by(OCC2010) %>% # Group by year, and occupation
  summarise(n_ue_occ = sum(WTFINL, na.rm = TRUE), # labour force in particular occupation
         n_ltue_occ = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
         ltuer_occ_filled = ifelse(ltuer_occ %in% c(0, 1), NA, ltuer_occ),
         occ_cat = as.factor(zap_labels(OCC2010)),
         spec = as.character(nchar(OCC2010))) # calculate LTUER as N_LTUE/N_LF 

################################################################################
################### OCCUPATIONAL CATEGORIES ####################################
################################################################################
occ_cats <- names(cw_short[-1])
ltuer_list <- list()
for(cat in occ_cats){
  cat = sym(cat)
  temp <- filtered %>%   # LTUER by Occupation
    group_by(!!cat) %>% # Group by year, and occupation
    summarise(n_ue_occ = sum(WTFINL, na.rm = TRUE), # labour force in particular occupation
              n_ltue_occ = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
    ungroup %>% 
    mutate(ltuer_occ = (n_ltue_occ/n_ue_occ))
           #ltuer_occ_filled = ifelse(ltuer_occ %in% c(0, 1), NA, ltuer_occ))
  print(temp)
  
  p1 <- temp %>% 
    arrange(ltuer_occ) %>% 
    mutate(occupation = factor(!!cat, levels = !!cat)) %>% 
    ggplot() +
    geom_point(aes(x = occupation, y = ltuer_occ)) 
  
  print(p1)
  
}

ltuer_grouped <- filtered %>%
  group_by(OCC2010) %>% # LTUER by Occupation
  mutate(n_ue_occ = sum(WTFINL, na.rm = TRUE), # labour force in particular occupation
         n_ltue_occ = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ)) %>% 
  group_by(SOC2010) %>% # Group by year, and occupation
  mutate(n_ue_occ = sum(WTFINL, na.rm = TRUE), # labour force in particular occupation
            n_ltue_occ = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc2010 = (n_ltue_occ/n_ue_occ)) %>% 
  group_by(SOC_major) %>% 
  mutate(n_ue_occ = sum(WTFINL, na.rm = TRUE), # labour force in particular occupation
            n_ltue_occ = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_major = (n_ltue_occ/n_ue_occ)) %>% 
  group_by(SOC_minor) %>% 
  mutate(n_ue_occ = sum(WTFINL, na.rm = TRUE), # labour force in particular occupation
         n_ltue_occ = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_minor = (n_ltue_occ/n_ue_occ)) %>% 
  group_by(SOC_broad) %>% 
  mutate(n_ue_occ = sum(WTFINL, na.rm = TRUE), # labour force in particular occupation
         n_ltue_occ = sum(WTFINL*LTUE1, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_broad = (n_ltue_occ/n_ue_occ)) %>% 
  select(OCC2010, SOC2010, SOC_major, SOC_minor, SOC_broad, contains("ltuer")) %>% 
  distinct

ltuer_grouped %>% 
  arrange(ltuer_occ) %>% 
  mutate(occupation = factor(OCC2010, levels = OCC2010)) %>% 
  ggplot(aes(x = occupation)) +
  geom_point(aes(y = ltuer_occ), color = "black") + 
  geom_point(aes(y = ltuer_soc2010), color = "blue") +
  geom_point(aes(y = ltuer_soc_major), color = "purple") +
  geom_point(aes(y = ltuer_soc_minor), color = "darkorange") +
  geom_point(aes(y = ltuer_soc_broad), color = "darkgreen") +
  labs(x = "Occupational Category", y = "LTUER", title = "LTUER by Occupational Category")

# Create real-world observations to compare to in python code
cw %>% 
  left_join(ltuer_grouped, by = c("OCC2010_cps" = "OCC2010"), 
            suffix = c("_orig", "adj")) %>% select(contains("ltuer")) %>% 
  write.csv(here("