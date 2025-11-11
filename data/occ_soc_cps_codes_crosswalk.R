# Create crosswalk from ABM occupational codes, OCC codes in CPS, and OCC-SOC Crosswalk codes
library(tidyverse)
library(here)
library(readxl)
library(ipumsr)
library(haven)

new_data = FALSE
# OCC2010-SOC Code Crosswalk
# https://www.census.gov/topics/employment/industry-occupation/guidance/code-lists.html
# More specifically: 2010 Census Occupation Codes with Crosswalk
cw <- read_xlsx(here('data/occ_macro_vars/CPS_LTUER/2010-occ-codes-with-crosswalk-from-2002-2011.xlsx'), sheet = 1, skip = 6) %>% 
  rename(major_cat = 1, OCC2010_desc = 2, OCC2010 = 3, SOC2010 = 4) %>% 
  filter(!grepl("-", OCC2010) & !is.na(OCC2010)) %>% 
  mutate(OCC2010 = as.numeric(OCC2010))


# Now, we need to figure out whether all the OCC codes in the ABM are present in the CPS data
# Surprise....they are not....so we need to do some fixing. 
ddi <- read_ipums_ddi(here("data/occ_macro_vars/CPS_LTUER/cps_00016.xml"))
data <- read_ipums_micro(ddi)
ipums_conditions()


filtered <- data %>% 
  filter(YEAR > 2003 & EMPSTAT %in% c(21, 22)) %>%  # Occupation classification scheme was redone in 2002, effective in 2003: https://cps.ipums.org/cps/occ_transition_2002_2010.shtml ; also filtered to include only unemployed workers. 
  select(OCC2010) # Counts the categories 


################################################################################
################################################################################
######## ORIGINAL OMN ##########################################################
################################################################################

abm_vars <- read.csv(here('calibration_remote/dRC_Replication/data/ipums_variables.csv')) %>% tibble

# Confirm that all OCS codes in abm_vars are present in the crosswalk
abm_vars %>% filter(!(acs_occ_code %in% unique(cw$OCC2010))) %>% nrow(.) == 0

# CREATE CROSSWALK TO BE ABLE TO MATCH IN CPS DATA
# ACS_OCC_CODE in the original ABM work IS the OCC2010 code. However, the CPS data does not have all OCC2010 codes. The below fills in in an intuitive way to be able to match the LTUERs. 
cross_walk <- dplyr::full_join(select(abm_vars, acs_occ_code, label), mutate(select(filtered, OCC2010), OCC2010copy = OCC2010), by = join_by(acs_occ_code == OCC2010copy)) %>% 
  distinct %>% mutate(acs_occ_code = ifelse(is.na(label), NA, acs_occ_code)) %>% 
  arrange(OCC2010) %>% mutate(common = ifelse(is.na(acs_occ_code), OCC2010, acs_occ_code)) %>% 
  arrange(common) %>% fill(OCC2010, .direction = "down") %>% 
  select(acs_occ_code, label, OCC2010) %>% 
  filter(!is.na(acs_occ_code)) %>% 
  rename(OCC2010_cps = OCC2010)

# Finally we wish to match against the crosswalk from above so we can group by higher-level soc codes. 
tmp <- cw %>% 
  mutate(OCC2010copy = OCC2010) %>% 
  left_join(cross_walk, ., by = join_by(acs_occ_code == OCC2010copy)) %>% 
  # Cut by major groups: https://www.bls.gov/soc/2010/2010_major_groups.htm
  # https://www.bls.gov/soc/soc_2010_user_guide.pdf
  mutate(SOC_major = substr(SOC2010, 1, 2), # 22/23
         SOC_minor = paste0(substr(SOC2010, 1,4), "000"), # 94/97
         SOC_broad = paste0(substr(SOC2010, 1,6), "0"), 
         major_cat = as.logical(major_cat),
         SOC_major = as.numeric(SOC_major)) %>% # 389/461
  mutate(
    # extract the numeric value (underlying code)
    OCC2010_cps_label = as.character(as_factor(OCC2010_cps)),
    OCC2010_cps = as.integer(OCC2010_cps)
    # extract the label (as string)
  ) %>% 
  select(-OCC2010_cps_label) 

if(new_data){
  tmp %>% all.equal(., select(tibble(read.csv(here("data/crosswalk_occ_soc_cps_codes.csv"))),-X))
}

rm(abm_vars, cross_walk, tmp)

################################################################################
################################################################################
######## FULL OMN ##############################################################
################################################################################

abm_vars <- read.csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/ONET/occ_names_employment_asec_occ_ipums_vars.csv") %>% 
  mutate(OCC2010_match_cps = as.character(occ),
         OCC2010_desc = OCC2010_match_cps) %>% 
  rename(acs_occ_code = occ) %>% 
  tibble

# Confirm that all OCS codes in abm_vars are present in the crosswalk
abm_vars %>% filter(!(acs_occ_code %in% unique(cw$OCC2010))) %>% nrow(.) == 0

# CREATE CROSSWALK TO BE ABLE TO MATCH IN CPS DATA
# ACS_OCC_CODE in the original ABM work IS the OCC2010 code. However, the CPS data does not have all OCC2010 codes. The below fills in in an intuitive way to be able to match the LTUERs. 
cross_walk <- dplyr::full_join(select(abm_vars, acs_occ_code), mutate(select(filtered, OCC2010), OCC2010copy = OCC2010), by = join_by(acs_occ_code == OCC2010copy)) %>% 
  distinct %>%
  arrange(OCC2010) %>% mutate(common = ifelse(is.na(acs_occ_code), OCC2010, acs_occ_code)) %>% 
  arrange(common) %>% fill(OCC2010, .direction = "down") %>% 
  select(acs_occ_code, OCC2010) %>% 
  filter(!is.na(acs_occ_code)) %>% 
  rename(OCC2010_cps = OCC2010) %>% 
  mutate(
    # extract the numeric value (underlying code)
    acs_occ_code_label = as.character(as_factor(acs_occ_code)),
    acs_occ_code = as.integer(acs_occ_code)
    # extract the label (as string)
  )

stopifnot(setdiff(abm_vars$acs_occ_code, cross_walk$acs_occ_code) == 0)
# Finally we wish to match against the crosswalk from above so we can group by higher-level soc codes. 
tmp <- cw %>% 
  mutate(OCC2010copy = OCC2010) %>% 
  left_join(cross_walk, ., by = join_by(acs_occ_code == OCC2010copy)) %>% 
  filter(acs_occ_code %in% abm_vars$acs_occ_code) %>% 
  # Cut by major groups: https://www.bls.gov/soc/2010/2010_major_groups.htm
  # https://www.bls.gov/soc/soc_2010_user_guide.pdf
  mutate(SOC_major = substr(SOC2010, 1, 2), # 22/23
         SOC_minor = paste0(substr(SOC2010, 1,4), "000"), # 94/97
         SOC_broad = paste0(substr(SOC2010, 1,6), "0"), 
         major_cat = as.logical(major_cat)) %>% # 389/461
  mutate(
    # extract the numeric value (underlying code)
    OCC2010_cps_label = as.character(as_factor(OCC2010_cps)),
    OCC2010_cps = as.integer(OCC2010_cps), 
    acs_occ_code = as.integer(acs_occ_code)
    # extract the label (as string)
  ) %>% 
  select(-OCC2010_cps_label)


stopifnot(identical(abm_vars$acs_occ_code, tmp$acs_occ_code))

stopifnot(tmp %>% 
  select(-major_cat) %>% 
  summarise(across(everything(), ~sum(is.na(.)))) %>% 
  rowSums() == 0)

if(new_data){
  tmp %>% write.csv(here('data/crosswalk_occ_soc_cps_codes_full_omn.csv'))
}


rm(abm_vars, cross_walk, tmp)

################################################################################
################################################################################
######## ONET ##################################################################
################################################################################

# # ONET
abm_vars <- read.csv("/Users/ebbamark/Dropbox/GenerateOccMobNets/ONET/acs_onet_2010_soc_cw.csv") %>% tibble %>%
  mutate(acs_occ_code = acs_2010_code,
         OCC2010_match_cps = as.character(acs_occ_code),
         label = title)

# Confirm that all OCS codes in abm_vars are present in the crosswalk
abm_vars %>% filter(!(acs_occ_code %in% unique(cw$OCC2010))) %>% nrow(.) == 0

# CREATE CROSSWALK TO BE ABLE TO MATCH IN CPS DATA
# ACS_OCC_CODE in the original ABM work IS the OCC2010 code. However, the CPS data does not have all OCC2010 codes. The below fills in in an intuitive way to be able to match the LTUERs. 
cross_walk <- dplyr::full_join(distinct(select(abm_vars, acs_occ_code)), mutate(select(filtered, OCC2010), OCC2010copy = OCC2010), by = join_by(acs_occ_code == OCC2010copy)) %>% 
  distinct %>%
  arrange(OCC2010) %>% mutate(common = ifelse(is.na(acs_occ_code), OCC2010, acs_occ_code)) %>% 
  arrange(common) %>% fill(OCC2010, .direction = "down") %>% 
  select(acs_occ_code, OCC2010) %>% 
  filter(!is.na(acs_occ_code)) %>% 
  rename(OCC2010_cps = OCC2010) %>% 
  mutate(
    # extract the numeric value (underlying code)
    acs_occ_code_label = as.character(as_factor(acs_occ_code)),
    acs_occ_code = as.integer(acs_occ_code)
    # extract the label (as string)
  )

stopifnot(setdiff(abm_vars$acs_occ_code, cross_walk$acs_occ_code) == 0)
# Finally we wish to match against the crosswalk from above so we can group by higher-level soc codes. 
tmp <- cw %>% 
  mutate(OCC2010copy = OCC2010) %>% 
  left_join(cross_walk, ., by = join_by(acs_occ_code == OCC2010copy)) %>% 
  filter(acs_occ_code %in% abm_vars$acs_occ_code) %>% 
  # Cut by major groups: https://www.bls.gov/soc/2010/2010_major_groups.htm
  # https://www.bls.gov/soc/soc_2010_user_guide.pdf
  mutate(SOC_major = substr(SOC2010, 1, 2), # 22/23
         SOC_minor = paste0(substr(SOC2010, 1,4), "000"), # 94/97
         SOC_broad = paste0(substr(SOC2010, 1,6), "0"), 
         major_cat = as.logical(major_cat)) %>% # 389/461
  mutate(
    # extract the numeric value (underlying code)
    OCC2010_cps_label = as.character(as_factor(OCC2010_cps)),
    OCC2010_cps = as.integer(OCC2010_cps), 
    acs_occ_code = as.integer(acs_occ_code)
    # extract the label (as string)
  ) %>% 
  select(-OCC2010_cps_label)


stopifnot(identical(unique(abm_vars$acs_occ_code), tmp$acs_occ_code))

stopifnot(tmp %>% 
            select(-major_cat) %>% 
            summarise(across(everything(), ~sum(is.na(.)))) %>% 
            rowSums() == 0)

if(new_data){
  tmp %>% write.csv(here('data/crosswalk_occ_soc_cps_codes_onet.csv'))
}
