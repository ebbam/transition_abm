# Load necessary libraries
library(dplyr)
library(tidyr)
library(haven)
library(stringr)
library(here)

# Set working directory to raw ATUS data
#setwd("path_to_raw_ATUS") # Replace with your actual path
base <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/")

# Load datasets
# atus_resp <- read_dta(paste0(base, "ATUS/atusresp_0314.dta"))
# atus_sum <- read_dta(paste0(base, "ATUS/atussum_0314.dta"))
# atus_rost <- read_dta(paste0(base, "ATUS/atusrost_0314.dta"))
# atus_cps <- read_dta(paste0(base, "ATUS/atuscps_0314.dta"))
# intvw_travel <- read_dta(paste0(base, "ATUS/intvwtravel_0314.dta"))

atus_resp <- readRDS(paste0(base, "ATUS/atusresp_0323.rds"))
atus_sum <- readRDS(paste0(base, "ATUS/atussum_0323.rds"))
atus_rost <- readRDS(paste0(base, "ATUS/atusrost_0323.rds"))
atus_cps <- readRDS(paste0(base, "ATUS/atuscps_0323.rds"))
intvw_travel <- readRDS(paste0(base, "ATUS/intvwtravel_0323.rds"))

ref_file <- read_dta(paste0(base, "ATUS/merged_ATUS_2014.dta")) %>%
  data.frame()

# Merge datasets
data <- atus_resp %>%
  left_join(atus_sum, by = c("tucaseid", "tuyear", "telfs", "tufnwgtp", "tudiaryday")) %>% 
  left_join(atus_rost, by = c("tucaseid", "tulineno", "teage", "tesex")) %>%
  filter(!is.na(tulineno)) %>% # Drop unmatched records
  left_join(atus_cps, by = c("tucaseid", "tulineno", "ptdtrace", "peeduca")) %>%
  filter(!is.na(tulineno)) %>%
  left_join(intvw_travel, by = "tucaseid")


data <- data %>% 
  rename_with(~gsub("tu",  "", .), contains("tulkdk")) %>%
  rename_with(~gsub("tu",  "", .), contains("tulkps")) %>% 
  rename_with(~gsub("tu",  "", .), contains("tulkm")) %>% 
  rename(lkm1 = telkm1)

data <- data %>%
  mutate(
    empldir = ifelse(lkm1==1 | lkm2==1 | lkm3==1 | lkm4==1 | lkm5==1 | lkm6==1 | lkps1==1 | lkps2==1 |  lkps3==1 |  lkps4==1 |   lkdk1==1 | lkdk2==1 |  lkdk3==1 |  lkdk4==1, 1, 0),
    pubemkag  = ifelse(lkm1==2 | lkm2==2 | lkm3==2 | lkm4==2 | lkm5==2 | lkm6==2 | lkps1==2 | lkps2==2 |  lkps3==2 |  lkps4==2 | lkdk1==2 | lkdk2==2 |  lkdk3==2 |  lkdk4==2, 1, 0),
    PriEmpAg = ifelse( lkm1==3 | lkm2==3 | lkm3==3 | lkm4==3 | lkm5==3 | lkm6==3 | lkps1==3 | lkps2==3 |  lkps3==3 |  lkps4==3 | lkdk1==3 | lkdk2==3 |  lkdk3==3 |  lkdk4==3, 1, 0),
    FrendRel = ifelse( lkm1==4 | lkm2==4 | lkm3==4 | lkm4==4 | lkm5==4 | lkm6==4 | lkps1==4 | lkps2==4 |  lkps3==4 |  lkps4==4 | lkdk1==4 | lkdk2==4 |  lkdk3==4 |  lkdk4==4, 1, 0),
    SchEmpCt = ifelse( lkm1==5 | lkm2==5 | lkm3==5 | lkm4==5 | lkm5==5 | lkm6==5 | lkps1==5 | lkps2==5 |  lkps3==5 |  lkps4==5 | lkdk1==5 | lkdk2==5 |  lkdk3==5 |  lkdk4==5, 1, 0),
    Resumes = ifelse( lkm1==6 | lkm2==6 | lkm3==6 | lkm4==6 | lkm5==6 | lkm6==6 | lkps1==6 | lkps2==6 |  lkps3==6 |  lkps4==6 |  lkdk1==6 | lkdk2==6 |  lkdk3==6 |  lkdk4==6, 1, 0),
    Unionpro = ifelse( lkm1==7 | lkm2==7 | lkm3==7 | lkm4==7 | lkm5==7 | lkm6==7 | lkps1==7 | lkps2==7 |  lkps3==7 |  lkps4==7 | lkdk1==7 | lkdk2==7 |  lkdk3==7 |  lkdk4==7, 1, 0),
    Plcdads = ifelse( lkm1==8 | lkm2==8 | lkm3==8 | lkm4==8 | lkm5==8 | lkm6==8 | lkps1==8 | lkps2==8 |  lkps3==8 |  lkps4==8 |  lkdk1==8 | lkdk2==8 |  lkdk3==8 |  lkdk4==8, 1, 0),
    Otheractve = ifelse( lkm1==9 | lkm2==9 | lkm3==9 | lkm4==9 | lkm5==9 | lkm6==9 | lkps1==9 | lkps2==9 |  lkps3==9 |  lkps4==9 | lkdk1==9 | lkdk2==9 |  lkdk3==9 |  lkdk4==9, 1, 0),
    lkatads = ifelse( lkm1==10 | lkm2==10 | lkm3==10 | lkm4==10 | lkm5==10 | lkm6==10 | lkps1==10 | lkps2==10 |  lkps3==10 |  lkps4==10 | lkdk1==10 | lkdk2==10 |  lkdk3==10 |  lkdk4==10 , 1, 0),
    Jbtrnprg = ifelse( lkm1==11 | lkm2==11 | lkm3==11 | lkm4==11 | lkm5==11 | lkm6==11 | lkps1==11 | lkps2==11 |  lkps3==11 |  lkps4==11 |  lkdk1==11 | lkdk2==11 |  lkdk3==11 |  lkdk4==11 , 1, 0),
    otherpas = ifelse( lkm1==13 | lkm2==13 | lkm3==13 | lkm4==13 | lkm5==13 | lkm6==13 | lkps1==13 | lkps2==13 |  lkps3==13 |  lkps4==13 |  lkdk1==13 | lkdk2==13 |  lkdk3==13 |  lkdk4==13 , 1, 0)
  ) %>% 
  rename(mlr = telfs) %>% 
  mutate(across(c(empldir, pubemkag, PriEmpAg, FrendRel, SchEmpCt, Resumes, Unionpro, Plcdads, Otheractve, lkatads,  Jbtrnprg, otherpas), ~ifelse(mlr != 4, 0, .)))

# Generate numsearch variable
data <- data %>%
  mutate(numsearch = rowSums(select(., empldir:otherpas), na.rm = TRUE))

# Time spent on job search
data <- data %>%
  rowwise() %>% 
  mutate(
    #timesearch_travel = if_else(tuyear != 2003, t050481 + t050405 + t050404 + t050403 + t050499 + intvwtravel, NA_real_),
    #timesearch = t050481 + t050405 + t050404 + t050403 + t050499,
    #timesearch_old = timesearch
    
    timesearch_travel = if_else(tuyear != 2003, sum(t050401, t050402, t050405, t050404, t050403, t050499, intvwtravel, na.rm = TRUE), NA_real_),
    timesearch = sum(t050401, t050402, t050405, t050404, t050403, t050499, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(
    timesearch_old = timesearch
  )

# Generate and clean date variables
data <- data %>%
  rename(wgt = tufnwgtp,
         age = teage,
         sex = tesex,
         race = ptdtrace,
         untype = pruntype) %>% 
  mutate(
    year = tuyear,
    month = str_pad(as.character(tumonth), width = 2, pad = "0"),
    date2 = paste0(year, month),
    quarter = case_when(
      month %in% c("01", "02", "03") ~ 1,
      month %in% c("04", "05", "06") ~ 2,
      month %in% c("07", "08", "09") ~ 3,
      TRUE ~ 4
    )
  ) %>% 
  mutate(year = as.numeric(year),
         month = as.numeric(month))

# Define demographic and other variables
data <- data %>%
  mutate(
    attached = if_else(mlr %in% c(3, 4) | (mlr == 5 & prwntjob == 1), 1, NA_real_),
    black = if_else(race %in% c(2, 6, 10, 11, 12, 15, 19), 1, 0),
    married = if_else(prmarsta %in% 1:3, 1, 0),
    educ = case_when(
      peeduca <= 38 ~ 1,
      peeduca == 39 ~ 2,
      peeduca >= 40 & peeduca < 43 ~ 3,
      peeduca >= 43 ~ 4,
      TRUE ~ NA_real_
    )
  ) %>% 
  rename(ind = peio1icd, 
         undur = prunedur,
         state = gestfips) %>% 
  rename(occ = peio1ocd,
         fips = state)

# Merge state FIPS codes
state_fips <- read_dta(paste0(base, "maps/state_fips.dta"))
data <- data %>%
  left_join(state_fips, by = c("fips" = "fips")) %>%
  filter(!is.na(fips)) # Drop unmatched records

# // searchers are people actively looking for a job (all unemployed who are looking)
data <- data %>% 
  mutate(searchers = ifelse(mlr == 4, 1, 0),
         nonsearchers = ifelse(mlr==5 & prwntjob==1, 1, 0))

# Cleaning the occupation variable
data <- data %>%
  mutate(
    occ_2011 = if_else(year %in% 2011:2014, occ, NA_integer_),
    occ = if_else(year %in% 2011:2014, NA_integer_, occ)
  )

# Merge OCC-SOC conversion (2010 codes)
occ_soc_conversion <- read_dta(paste0(base, "maps/OCC-SOC_conversion.dta"))
data <- data %>%
  left_join(occ_soc_conversion, by = c("occ")) %>%
  mutate(
    soc = case_when(
      occ == 400 ~ 11,
      occ == 9840 ~ 55, # Military
      TRUE ~ soc
    )
  ) %>%
  rename(soc_2010 = soc) %>%
  select(-occupationname, -soc6, -soc3, -occ) %>% 
  rename(occ = occ_2011)

# Merge OCC-SOC conversion (2011 codes)
occ_soc_conversion_2011 <- read_dta(paste0(base, "maps/OCC-SOC_conversion_2011.dta"))
data <- data %>%
  left_join(occ_soc_conversion_2011, by = c("occ" = "occ")) %>%
  mutate(
    soc = case_when(
      year == 2011 & occ %in% 10:430 ~ 11,
      year == 2011 & occ %in% c(560, 620, 730) ~ 13,
      year == 2011 & occ == 9840 ~ 55,
      year == 2011 & occ == 1960 ~ 19,
      year == 2011 & occ == 2020 ~ 21,
      year == 2011 & occ %in% c(2140, 2150) ~ 23,
      year == 2011 & occ == 2820 ~ 27,
      year == 2011 & occ %in% 3000:3540 ~ 29,
      year == 2011 & occ == 3650 ~ 31,
      year == 2011 & occ %in% c(3920, 3950) ~ 33,
      year == 2011 & occ == 4550 ~ 39,
      year == 2011 & occ == 4960 ~ 41,
      year == 2011 & occ == 5930 ~ 43,
      year == 2011 & occ == 6000 ~ 45,
      year == 2011 & occ %in% 6200:6940 ~ 47,
      year == 2011 & occ %in% c(7310, 7620) ~ 49,
      year == 2011 & occ %in% 7700:8965 ~ 51,
      year <= 2010 ~ soc_2010,
      soc == 99 ~ NA_integer_,
      TRUE ~ soc
    )
  )

# Generate weekend indicator
data <- data %>%
  mutate(
    weekend = if_else(tudiaryday %in% c(1, 7), 1, 0)
  ) %>% 
  rename(day = tudiaryday)
        # missing = t500106)

# # Select relevant columns
data <- data %>%
  select(
    tucaseid, weekend, year, month, ind, soc, educ, married, searchers,
    nonsearchers, black, attached, quarter, race, sex, age, wgt, mlr, numsearch,
    timesearch, timesearch_travel, undur, gereg, state, empldir, pubemkag,
    PriEmpAg, FrendRel, SchEmpCt, Resumes, Unionpro, Plcdads, Otheractve,
    lkatads, Jbtrnprg, otherpas, untype, pelklwo, 
    #t050481, t050405, t050404, t050403, t050499, 
    t050401, t050402, t050405, t050404, t050403, t050499, #intvwtravel,
    day #, missing
  )

# Clean labor force variables
data <- data %>%
  mutate(
    layoff = if_else(mlr == 3, 1, 0),
    unemp = if_else(mlr %in% c(3, 4), 1, 0),
    np_other = if_else(mlr == 5 & nonsearchers != 1, 1, 0),
    nonemp = if_else(mlr %in% c(3, 4, 5), 1, NA_real_)
  )

# Cleaning demographics
data <- data %>%
  mutate(
    female = if_else(sex == 2, 1, 0),
    marriedfemale = married * female,
    age2 = age^2,
    age3 = age^3,
    age4 = age^4,
    hs = if_else(educ == 2, 1, 0),
    somecol = if_else(educ == 3, 1, 0),
    college = if_else(educ == 4, 1, 0)
  )

# Dropping outliers and cleaning age
data <- data %>%
  mutate(
    time_less8 = if_else(timesearch < 480, timesearch, NA_real_),
    time_less8_travel = if_else(timesearch_travel < 480, timesearch_travel, NA_real_)
  ) %>%
  filter(age >= 25 & age <= 70, !is.na(age))

data_short <- data %>% 
  filter(year <= 2014) %>% 
  select(-contains('t0504'))

ref_file_short <- ref_file %>% 
  select(-contains('t0504'), -missing)
# Same number of observations
nrow(data_short) == nrow(ref_file)

# All variable names are the same
if(length(setdiff(names(data_short), names(ref_file_short))) == 0 & length(setdiff(names(ref_file_short), names(data_short))) == 0){
  data_short <- data_short[, colnames(ref_file_short)]
}
# Columns are identical in names - ie all columns exist
all.equal(names(ref_file_short), names(data_short))

data_short %>% 
  arrange(tucaseid, year, month, day) %>% identical(data_short)

ref_file_short <- ref_file_short %>% 
  arrange(tucaseid, year, month, day) %>% 
  tibble

zap_labels(ref_file_short) %>% all.equal(zap_labels(data_short), check.attributes = FALSE, tolerance = 5e-7)



#saveRDS(data, paste0(base, "ATUS/merged_ATUS_2023.rds"))

# This has worked when importing the atussum, resp, etc. files with suffix in 0314.dta
#write_dta(data, "merged_ATUS_2014.dta")

