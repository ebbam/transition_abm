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
source(here("code/formatting/plot_dicts.R"))

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
  filter(YEAR > 2003 & EMPSTAT %in% c(10, 12, 21, 22)) %>%  # Occupation classification scheme was redone in 2002, effective in 2003: https://cps.ipums.org/cps/occ_transition_2002_2010.shtml ; also filtered to include only unemployed workers. 
  select(YEAR, MONTH, WTFINL, OCC2010, DURUNEMP, DURUNEM2, EMPSTAT, HWTFINL, names(cw_short[-1])) %>% # Select only relevant variables.
  mutate(DURUNEMP = ifelse(DURUNEMP == 999, NA, DURUNEMP),
         DURUNEM2 = ifelse(DURUNEM2 == 9, NA, DURUNEM2)) %>% # Replace "99" value with NA - 99 = NIU ie. "Not in universe" or simply not applicable to the correspondent. See explanation here: https://cps.ipums.org/cps-action/faq (Ctrl+F "NIU")
  mutate(UE = EMPSTAT %in% c(21,22),
         LF = EMPSTAT %in% c(10, 12, 21, 22),
         LTUE1 = DURUNEMP >= 27,
         LTUE2 = DURUNEM2 >= 12
         ) # Counts the categories 5 (27-39 weeks), 6 (40+ weeks), and 7 (Over 26 weeks - period 1962-1967) as LTUE
  

library(ggridges)
library(viridis)
ridges <- filtered %>% 
  mutate(DATE = as.factor(as.Date(paste0(YEAR, "-", MONTH, "-01")))) %>% 
  group_by(YEAR, DATE, DURUNEM2) %>% 
  summarise(n = n()) %>% 
  ungroup

ridges %>% 
  group_by(YEAR, DURUNEM2) %>% 
  summarise(n = mean(n, na.rm = TRUE)) %>% 
  ungroup %>% 
  filter(DURUNEM2 != 0) %>% 
  ggplot(., aes(x = DURUNEM2, y = YEAR, height = n, group = YEAR)) + 
  geom_density_ridges(stat = "identity", scale = 1) +
  theme(legend.position = "none") + coord_flip() + common_theme + theme(fill = "lightblue") +
  labs(y = "Year", x = "Unemployment Duration")

ridges2 <- filtered %>% 
  mutate(DATE = as.factor(as.Date(paste0(YEAR, "-", MONTH, "-01")))) %>% 
  group_by(YEAR, DATE, DURUNEMP) %>% 
  summarise(n = n()) %>% 
  ungroup

ridges2 %>% 
  group_by(YEAR, DURUNEMP) %>% 
  summarise(n = mean(n, na.rm = TRUE)) %>% 
  ungroup %>% 
  filter(DURUNEMP != 0) %>% 
  ggplot(., aes(x = DURUNEMP/4, y = YEAR, height = n, group = YEAR)) + 
  geom_density_ridges(stat = "identity", scale = 1) +
  theme(legend.position = "none") + 
  coord_flip() + common_theme + 
  labs(y = "Year", x = "Unemployment Duration (Months)", title = "Distribution of Unemployment Duration (Annual Average)") 

ggsave(here('data/occ_macro_vars/CPS_LTUER/unemp_distributions_cps.png'))

ridges2 %>% 
  group_by(YEAR, DURUNEMP) %>% 
  # summarise(n = mean(n, na.rm = TRUE)) %>% 
  # ungroup %>% 
  filter(DURUNEMP != 0) %>% 
  ggplot(., aes(x = DURUNEMP/4, y = YEAR, group = YEAR)) + 
  geom_violin(trim =FALSE) +
  theme(legend.position = "none") + 
  coord_flip() + common_theme + theme(fill = "lightblue") +
  labs(y = "Year", x = "Unemployment Duration (Months)")

ridges2 %>% 
  group_by(YEAR, DURUNEM) %>% 
  summarise(n = mean(n, na.rm = TRUE)) %>% 
  ungroup %>% 
  filter(DURUNEM2 != 0) %>% 
  ggplot(aes(x = DURUNEM2, y = YEAR, group = YEAR, height = n)) + 
  geom_density_ridges(stat = "binline", bins = 30, scale = 0.2, draw_baseline = TRUE) +
  theme(legend.position = "none") + coord_flip() + common_theme 

filtered %>% 
  mutate(DATE = as.factor(as.Date(paste0(YEAR, "-", MONTH, "-01")))) %>% 
  filter(DURUNEM2 != 0) %>% 
  ggplot(aes(x = DATE, y = DURUNEM2)) +
    geom_density() + 
  common_theme
################################################################################
################### NATIONAL RATE ##############################################
################################################################################

ltuer_overall <- filtered %>% 
  # Total LTUER - same calculation as above
  group_by(YEAR, MONTH) %>% 
  summarise(n_ue = sum(WTFINL*UE, na.rm = TRUE),
            n_lf = sum(WTFINL, na.rm = TRUE),
            n_ltue = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer = n_ltue/n_ue, 
         uer = n_ue/n_lf,
         DATE = as.Date(paste0(YEAR, "-", MONTH, "-01")))

ggplot() +
  geom_line(data = ltuer_overall, aes(x = DATE, y = ltuer)) +
  geom_line(data = filter(macros, YEAR >= 2004), aes(x = DATE, y = LTUER), linetype = "dashed")

ggplot() +
  geom_line(data = ltuer_overall, aes(x = DATE, y = uer)) +
  geom_line(data = filter(macros, YEAR >= 2004), aes(x = DATE, y = LTUER), linetype = "dashed")

ltuer_occ <- filtered %>%   # LTUER by Occupation
  group_by(OCC2010) %>% # Group by year, and occupation
  summarise(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
            n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
         uer_occ = n_ue_occ/n_lf_occ,
         ltuer_occ_filled = ifelse(ltuer_occ %in% c(0, 1), NA, ltuer_occ),
         uer_occ_filled = ifelse(uer_occ %in% c(0, 1), NA, ltuer_occ),
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
    summarise(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
              n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
              n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
    ungroup %>% 
    mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
           uer_occ = n_ue_occ/n_lf_occ)
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
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
            n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
            n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
         uer_occ = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC2010) %>% # Group by year, and occupation
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc2010 = (n_ltue_occ/n_ue_occ),
         uer_soc2010 = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_major) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_major = (n_ltue_occ/n_ue_occ),
         uer_soc_major = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_minor) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_minor = (n_ltue_occ/n_ue_occ),
         uer_soc_minor = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_broad) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_broad = (n_ltue_occ/n_ue_occ),
         uer_soc_broad = (n_ue_occ/n_lf_occ)) %>% 
  select(OCC2010, SOC2010, SOC_major, SOC_minor, SOC_broad, contains("uer")) %>% 
  distinct

LTUER <- ltuer_grouped %>% 
  arrange(ltuer_occ) %>% 
  mutate(occupation = factor(OCC2010, levels = OCC2010)) %>% 
  ggplot(aes(x = occupation)) +
  geom_point(aes(y = ltuer_occ), color = "black", alpha = 0.5) + 
  geom_point(aes(y = ltuer_soc2010), color = "blue", alpha = 0.5) +
  geom_point(aes(y = ltuer_soc_major), color = "purple", alpha = 0.5) +
  geom_point(aes(y = ltuer_soc_minor), color = "darkorange", alpha = 0.5) +
  geom_point(aes(y = ltuer_soc_broad), color = "darkgreen", alpha = 0.5) +
  labs(x = "Occupational Category", y = "LTUER", title = "LTUER")


UER <- ltuer_grouped %>% 
  filter(!is.na(SOC_broad)) %>% 
  arrange(uer_occ) %>% 
  mutate(occupation = factor(OCC2010, levels = OCC2010)) %>% 
  ggplot(aes(x = occupation)) +
  geom_point(aes(y = uer_occ), color = "black") + 
  geom_point(aes(y = uer_soc2010), color = "blue") +
  geom_point(aes(y = uer_soc_major), color = "purple") +
  geom_point(aes(y = uer_soc_minor), color = "darkorange") +
  geom_point(aes(y = uer_soc_broad), color = "darkgreen") +
  labs(x = "Occupational Category", y = "UER", title = "UER") +
  ylim(c(0,0.25))

library(patchwork)
LTUER + UER + plot_annotation(title = "UER & LTUER by Occupational Category (n = 464)")

# Create real-world observations to compare to in python code
cw %>% 
  left_join(ltuer_grouped, by = c("OCC2010_cps" = "OCC2010"), suffix = c("_orig", "_adj")) %>% 
  write.csv(here("data/occ_macro_vars/CPS_LTUER/occ_uer_ltuer_observed.csv"))
  
#ltuer_occ_filled = ifelse(ltuer_occ %in% c(0, 1), NA, ltuer_occ))
print(temp)

p1 <- temp %>% 
  arrange(ltuer_occ) %>% 
  mutate(occupation = factor(!!cat, levels = !!cat)) %>% 
  ggplot() +
  geom_point(aes(x = occupation, y = ltuer_occ)) 

print(p1)


