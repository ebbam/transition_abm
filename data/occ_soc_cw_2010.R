# Script to be sourced for the asec-soc-cw
################################################################################
################################################################################
################################################################################
##### Crosswalking ASEC to SOC Codes #####
## Crosswalking to SOC2010 Codes
## Using crosswalk OCC2010-SOC Code Crosswalk
## Raw source: https://www.census.gov/topics/employment/industry-occupation/guidance/code-lists.html
### More specifically: 2010 Census Occupation Codes with Crosswalk
## I have cleaned this raw excel file in the following script:
### /Users/ebbamark/Library/CloudStorage/OneDrive-Nexus365/GenerateOccMobNets/scripts/make_occmob_asec_occ_em_normalised_2025.ipynb
### Creates occ_soc_2010_crosswalk_cleaned.csv
### Cleaning process under chunk labelled:## SOC Network for Higher-Level Aggregation
### OCC = occ2010 from above

occ_soc_cw <- read.csv("/Users/ebbamark/Library/CloudStorage/OneDrive-Nexus365/GenerateOccMobNets/data/occ_soc_2010_crosswalk_cleaned.csv") %>% 
  mutate(across(c(soc_2010, soc_2010_major, soc_2010_minor, soc_2010_broad)))


# Printing label structure
library(tidyverse)
library(here)
library(readxl)
soc_desc <- read_xls(here('data/soc_structure_2010.xls'), skip = 11) %>% 
  rename('soc_2010_major' = 1,
         'soc_2010_minor' = 2,
         'soc_2010_broad' = 3,
         'soc_2010' = 4,
         'description' = 5) 

get_soc_table <- function(code){
  soc_desc %>% filter(!is.na(!!sym(code))) %>% select(!!code, description) %>% return(.)
}

get_soc_table('soc_2010_minor') %>% View()
get_soc_table('soc_2010_broad') %>% View()
         
