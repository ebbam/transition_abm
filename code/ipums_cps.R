# Description: Extracting long-term unemployment rate by occupation
# Cite the following:
# Publications and research reports based on the IPUMS CPS database must cite it appropriately. The citation should include the following:
#   
#   Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 11.0 [dataset]. Minneapolis, MN: IPUMS, 2023.
# https://doi.org/10.18128/D030.V11.0
# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

ddi <- read_ipums_ddi(here("data/macro_vars/CPS_LTUER/cps_00005.xml"))
data <- read_ipums_micro(ddi)


test <- data %>% 
  select(YEAR, MONTH, OCC, WKSUNEM1, WKSUNEM2) %>% 
  mutate(across(c(WKSUNEM1, WKSUNEM2), ~ifelse(. == 99, NA, .))) %>% 
  mutate(LTUE1 = WKSUNEM1 >= 27,
         LTUE2 = WKSUNEM2 %in% c(5,6)) %>% 
  group_by(YEAR, MONTH, OCC) %>% 
  mutate(n_lf_occ = n(),
         n_ltue_occ = sum(LTUE1, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer_occ = n_ltue_occ/n_lf_occ) %>% 
  group_by(YEAR, MONTH) %>% 
  mutate(n_lf = n(),
         n_ltue = sum(LTUE1, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer = n_ltue/n_lf) 

test %>% 
  select(YEAR, ltuer) %>% 
  distinct() %>% 
  ggplot() +
  geom_line(aes(x = YEAR, y = ltuer))

test %>% 
  select(YEAR, OCC, ltuer_occ) %>% 
  distinct() %>% 
  ggplot() +
  geom_line(aes(x = YEAR, y = ltuer_occ, color = OCC))
  

#########################################

# My title: Occupational mobility extract
# IPUMS CPS DATA
# Citation: Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, 
# J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, 
# Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 11.0 [dataset]. 
# Minneapolis, MN: IPUMS, 2023. https://doi.org/10.18128/D030.V11.0



library(tidyverse)
library(here)
read.delim(here("data/us_cps/cps_00001.dat"), sep=" ")
