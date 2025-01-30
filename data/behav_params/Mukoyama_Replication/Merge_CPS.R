# Load necessary libraries
library(plyr) # Only need this for rbind.fill - load this first
library(tidyverse)
library(conflicted)
library(data.table)
library(haven)
library(stringr)
library(here)
library(readr)
library(rio)
library(assertthat)
library(stats) # for normal distribution functions
conflicted::conflict_prefer_all("dplyr", quiet = TRUE)
conflicts_prefer(here::here)

first = FALSE # set depending on whether the first time running script - if TRUE it will reconvert all dta files to csv files
new = FALSE
test_equal <- function(ref_df = ref_file, new_df = cps_data, verbose = FALSE, full = FALSE){
  if(full){
    res <- all.equal(ref_df, new_df, check.attributes = FALSE)
  }else{
    res <- ref_df %>% select(any_of(names(new_df))) %>% all.equal(select(new_df, names(.)), check.attributes = FALSE) 
  }
  
  if(verbose){
    return(res)
  }
  res %>% isTRUE(.) %>% assert_that(., msg = "Something happened here...")
}

base <- here("data/behav_params/Mukoyama_Replication/")

# RUN HERE STEP 1 FOR NUMSEARCH CORRECTION
if(new){
  # I need to save them in chunks because the files crash R...
  for(y in 2015:2024){
    print(y)
    base_new <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/CPS/R_int/")
    
    # Appending all files with "*intermediate*" pattern
    # filenames <- list.files(here(base_new), pattern = paste0("*intermediate_", as.character(y), "correct.rds")) %>% 
    #   paste0(base,"mukoyama_all/int_data/CPS/R_int/", .)
    
    filenames <- list.files(here(base_new), pattern = paste0("intermediate_", y, "[0-9]{2}correct\\.rds")) %>%
      paste0(base_new, .)
    print(filenames)
    
    cps_data_new <- tibble()
    for(i in 1:length(filenames)){
      print(i)
      cps_temp <- readRDS(filenames[i])
      # This will need to be incorporated into bind_rows if necessary
      if("marsta2" %in% names(cps_temp)){
        #print('Name of married variable changed.')
        cps_temp <- cps_temp %>%
          rename(married = marsta2)
      }else{}
      #print(paste0("Missing names in new file: ", setdiff(names(cps_data_new), names(cps_temp))))
      #print(paste0("New names in new file: ", setdiff(names(cps_temp), names(cps_data_new))))
      cps_data_new <- rbind.fill(cps_data_new, cps_temp)
    }
    saveRDS(cps_data_new, here(paste0("data/behav_params/Mukoyama_Replication/mukoyama_all/final_data/R_final/cps_data_no_time_temp_correct_", as.character(y), ".rds")))
    print(paste0("Saved: ", as.character(y)))}
}


################################################################################
# Appending all files with "*intermediate*" pattern
filenames <- list.files(here(paste0(base, "mukoyama_all/int_data/CPS/")), pattern = "*intermediate*") %>% 
  paste0(base,"mukoyama_all/int_data/CPS/", .)


# Convert files to be used to csv for easier/quicker loading
if(first){
  # Reference data
  # Create CSV file
  convert(here(paste0(base,"mukoyama_all/final_data/full_CPS_data.dta")), 
          here(paste0(base,"mukoyama_all/final_data/R_final/full_CPS_data.csv")))
  
  # Create RDS file
  convert(here(paste0(base,"mukoyama_all/final_data/full_CPS_data.dta")), 
          here(paste0(base,"mukoyama_all/final_data/R_final/full_CPS_data.RDS")))

 
  filenames <- list.files(here(paste0(base, "mukoyama_all/int_data/CPS/")), pattern = "*intermediate*")
  filenames_csv <- lapply(filenames, function(x) paste0(substr(x, 1, nchar(x) - 3), "csv"))
  for(i in 1:length(filenames)){
    convert(here(paste0(base,"mukoyama_all/int_data/CPS/", filenames[i])), here(paste0(base,"mukoyama_all/int_data/CPS/R_int/", filenames_csv[i])))
  }
  #cps_data <- read.csv(here(paste0(base, "mukoyama_all/int_data/CPS/intermediate_199401.csv"))) #lapply(filenames[143:145], read_dta) %>% bind_rows()
  # Appending all files with "*intermediate*" pattern in int_R folder so we only get our transformed CSV file
  filenames <- list.files(here(paste0(base, "mukoyama_all/int_data/CPS/R_int")), pattern = "*intermediate*") %>% 
    paste0(base,"mukoyama_all/int_data/CPS/R_int/", .)
  # Need a for loop for now to rename marsta2 at each stop conditional on whether it exists or not
  cps_data <- tibble()
  for(i in 1:length(filenames)){
    print(i)
    cps_temp <- read.csv(filenames[i])
    # This will need to be incorporated into bind_rows if necessary
    if("marsta2" %in% names(cps_temp)){
      print('Name of married variable changed.')
      cps_temp <- cps_temp %>% 
        rename(married = marsta2)
    }else{}
    print(paste0("Missing names in new file: ", setdiff(names(cps_data), names(cps_temp))))
    print(paste0("New names in new file: ", setdiff(names(cps_temp), names(cps_data))))
    cps_data <- rbind.fill(cps_data, cps_temp)
  }
}

if(first){
  ref_file <- read.csv(here("data/behav_params/Mukoyama_Replication/mukoyama_all/final_data/R_final/full_CPS_data.csv"))
}

# saving this file temporarily to avoid having to do the above ^^
#saveRDS(cps_data, paste0(base, "mukoyama_all/int_data/cps_data_temp.RDS"))
# ref_file <- read.csv(paste0(base, "mukoyama_all/final_data/R_final/full_CPS_data.RDS"))
# cps_data_full <- readRDS(paste0(base, "mukoyama_all/int_data/cps_data_temp.RDS"))
# cps_data_short <- cps_data_full %>% filter(year > 2003 & year <= 2005)
# ref_file_short <- ref_file %>% filter(year > 2003 & year <= 2005) 

# RUN HERE STEP 2 FOR NUMSEARCH CORRECTION
if(new){
  for(y in 2015:2024){
    base_new <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/final_data/R_final/")
    
    # Appending all files with "*intermediate*" pattern
    filenames <- list.files(here(base_new), pattern = paste0("*no_time_temp_correct_", as.character(y))) %>% 
      paste0(base_new, .)
    
    for(i in 1:length(filenames)){
      print(i)
      cps_data <- readRDS(filenames[i])
      
      # Rename and adjust weights
      cps_data <- cps_data %>%
        rename(newwgt = wgt) %>%
        mutate(newwgt = newwgt / 10000,
               newwgt = ifelse(year >= 2003 & year <= 2012, newwgt / 10000, newwgt))
      
      # Filter by age and labor force information
      cps_data <- cps_data %>%
        filter(age >= 25, age <= 70) %>%
        filter(!mlr %in% c(127, -1, -10, NA))
      
      # Create identifiers and demographic variables
      cps_data <- cps_data %>%
        mutate(date = year * 12 + month,
               firstInt = case_when(
                 mis == 1 ~ date,
                 mis == 2 ~ date - 1,
                 mis == 3 ~ date - 2,
                 mis == 4 ~ date - 3,
                 mis == 5 ~ date - 12,
                 mis == 6 ~ date - 13,
                 mis == 7 ~ date - 14,
                 mis == 8 ~ date - 15
               ),
               # I've modified the below becauase i think 1=1 and 2=2 no matter the year
               ethn = case_when(
                 race == 1 ~ 1, # & date >= 348 & date < 432
                 race == 2 ~ 2, # & date >= 348 & date < 432 
                 race %in% c(3, 4, 5) & date >= 348 & date < 432 ~ 3,
                 #race == 1 & date >= 432 & date < 516 ~ 1,
                 #race == 2 & date >= 432 & date < 516 ~ 2,
                 race %in% c(3, 4) & date >= 432 & date < 516 ~ 3,
                 #race == 1 & date >= 516 ~ 1,
                 #race == 2 & date >= 516 ~ 2,
                 race %in% c(3:21) & date >= 516 ~ 3
               ))
      
      
      # Handling serial and hhid2 for different years
      cps_data <- cps_data %>%
      # Serial is the 3rd and 4th characters of hhid2
      #   The component parts of this number are as follows:
      #   71-72	Numeric component of the sample number (HRSAMPLE)
      # 73-74	Serial suffix-converted to numerics (HRSERSUF)
      # 75		Household Number (HUHHNUM)
        mutate(serial = substr(hhid2, 3, 4),
          serial = ifelse(year <= 2002 & serial == "47", "", serial),
          serial = case_when(year == 1994 & month == 1 & mis > 1 ~ "",
                             year == 1994 & month == 2 & mis > 2 ~ "",
                             year == 1994 & month == 3 & mis > 3 ~ "",
                             year == 1994 & month >= 4 & mis > 4 ~ "",
                             year == 1995 & month == 1 & mis > 5 ~ "",
                             year == 1995 & month == 2 & mis > 6 ~ "",
                             year == 1995 & month == 3 & mis > 7 ~ "",
                             TRUE ~ serial),
               hhid2 = case_when(
                  year < 2004 ~ "",
                  year == 2004 & month %in% c(1,2,3,4) ~ "",
                  year == 2004 & month == 5 & mis > 1 ~ "",
                  year == 2004 & month == 6 & mis > 2 ~ "",
                  year == 2004 & month == 7 & mis > 3 ~ "",
                  year == 2004 & month %in% c(8, 9, 10, 11, 12) & mis > 4 ~ "",
                  year == 2005 & month %in% c(1,2,3,4) & mis > 4 ~ "",
                  year == 2005 & month == 5 & mis > 5 ~ "",
                  year == 2005 & month == 6 & mis > 6 ~ "",
                  year == 2005 & month == 7 & mis > 7 ~ "",
                  TRUE ~ NA),
          serial = ifelse(serial %in% c("", "-1"), "0", serial)) %>% 
        rename(state = gestfips) %>% 
        group_by(hhid, hhid2, lineno, state, serial, firstInt, sex, ethn, mis) %>% 
        mutate(dup = row_number() - 1) %>% 
        ungroup
      
      cps_data %>% pull(dup) %>% table
      
      # Construct unique ID
      cps_data <- cps_data %>%
        mutate(final_id_str = str_c(hhid, hhid2, lineno, serial, state, firstInt, sex, ethn, dup, sep = "_")) %>% 
        group_by(final_id_str) %>% 
               mutate(final_id = cur_group_id()) %>% 
        ungroup #%>% 
        #select(-c(lineno, state, dup, ethn, final_id_str, ethn, hhid2, hhid, serial, lineno))
      
      # Adjust demographic variables
      cps_data <- cps_data %>%
        mutate(black = ifelse((race %in% c(2, 6, 10, 11, 12) & year > 2002) | (race == 2 & year <= 2002), 1, 0),
               married = ifelse(married %in% c(1, 2), 1, 0),
               married = ifelse(married %in% c(3, 4, 5, 6, -1), 0, married),
               female = ifelse(sex == 2, 1, 0),
               marriedfemale = married * female)
      
      # Additional feature engineering (e.g., Census regions, education levels)
      cps_data <- cps_data %>%
        mutate(cen_region = case_when(
          gereg == 1 ~ 2,
          gereg == 2 ~ 1,
          gereg == 3 ~ 3,
          gereg == 4 ~ 4
        ),
        educ = case_when(
          grdatn <= 38 ~ 1,
          grdatn == 39 ~ 2,
          grdatn > 39 & grdatn <= 42 ~ 3,
          grdatn >= 43 & grdatn != NA ~ 4
        ),
        hs = ifelse(educ == 2, 1, 0),
        somecol = ifelse(educ == 3, 1, 0),
        college = ifelse(educ == 4, 1, 0))
      
      # Occupation adjustments
      cps_data <- cps_data %>%
        # do not know where occdt comes from....same values as PRDTOCC1 but in different years
        mutate(occdt = prdtocc1) %>% 
        mutate(occ_pre02 = ifelse(year <= 2002, prdtocc1, NA),
               soc = case_when(
                 occdt==1 & year >= 2003  ~ 11,
                 occdt==2 & year >= 2003 ~ 13,
                 occdt==3 & year>=2003 ~ 15, 
                 occdt==4 & year>=2003 ~ 17 , 
                 occdt==5 & year>=2003 ~ 19 ,
                 occdt==6 & year>=2003 ~ 21 ,
                 occdt==7 & year>=2003 ~ 23,
                 occdt==8 & year>=2003 ~ 25, 
                 occdt==9 & year>=2003 ~ 27, 
                 occdt==10 & year>=2003 ~ 29,
                 occdt==11 & year>=2003 ~ 31,
                 occdt==12 & year>=2003 ~ 33, 
                 occdt==13 & year>=2003 ~ 35,
                 occdt==14 & year>=2003 ~ 37,
                 occdt==15 & year>=2003 ~ 39,
                 occdt==16 & year>=2003 ~ 41,
                 occdt==17 & year>=2003 ~ 43,
                 occdt==18 & year>=2003 ~ 45,
                 occdt==19 & year>=2003 ~ 47,
                 occdt==20 & year>=2003 ~ 49,
                 occdt==21 & year>=2003 ~ 51,
                 occdt==22 & year>=2003 ~ 53),
               # possibly a typo and should be occ_pre02
          soc = case_when(
            occ_pre02 %in% c(1, 2, 3) ~ 11,
            occ_pre02 == 5 ~ 15,
            occ_pre02 %in% c(4, 14, 15) ~ 17,
            occ_pre02 == 6 ~ 19,
            occ_pre02 == 12 ~ 21,
            occ_pre02 == 11 ~ 23,
            occ_pre02 %in% c(9, 10) ~ 25,
            occ_pre02 %in% c(7, 8, 13) ~ 29,
            occ_pre02 == 30 ~ 31,
            occ_pre02 == 28 ~ 33,
            occ_pre02 == 29 ~ 35,
            occ_pre02 %in% c(31, 37) ~ 37,
            occ_pre02 %in% c(32, 27) ~ 39,
            occ_pre02 %in% c(16, 17, 18, 19, 20) ~ 41,
            occ_pre02 %in% c(21, 22, 23, 24, 25, 26) ~ 43,
            occ_pre02 %in% c(43, 44, 45) ~ 45,
            occ_pre02 %in% c(34, 40) ~ 47,
            occ_pre02 == 33 ~ 49,
            occ_pre02 %in% c(35, 36, 37) ~ 51,
            occ_pre02 %in% c(38, 39, 41, 42) ~ 53,
            TRUE ~ NA_real_),
          soc_RC = case_when(
            soc %in% c(11, 13, 15, 17, 19, 23, 27, 29, 21, 25) ~ 1,
            soc %in% c(39, 35, 33, 31, 37) ~ 2,
            soc %in% c(41, 43) ~ 3,
            soc %in% c(51, 47, 53, 45, 49) ~ 4,
            TRUE ~ NA_real_)) #%>%
        #select(-occ_pre, -occdt, -PRDTOCC1, -PRDTIND1)
      
      # Clean labor force status
      cps_data <- cps_data %>%
        mutate(searchers = as.numeric(mlr == 4),
               nonsearchers = as.numeric(lfs == "D"),
               np_other = as.numeric(((mlr==5 | mlr==6 | mlr==7) & lfs!="D")),
               unemp = as.numeric(mlr %in% c(3, 4)),
               nonpart = as.numeric(mlr %in% c(5, 6, 7)),
               emp = as.numeric(mlr %in% c(1, 2)),
               layoff = as.numeric(mlr == 3)) %>% 
        rename_with(toupper, c("prdtind1", "prdtocc1"))
      
    saveRDS(cps_data, here(paste0(base_new, "cps_data_no_time_correct_", as.character(y), ".rds")))
    print(paste0("Saved: ", as.character(y)))
    }
  }
}

cps_data <- tibble()
# Due to data size constraints the following process was run in two parts
# temp_full_CPS_2015_19_correct.RDS
# temp_full_CPS_2020_24_correct.RDS
for(y in 2015:2024){
  print(y)
  base_new <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/final_data/R_final/")
  
  filenames <- list.files(here(base_new), pattern = paste0("*cps_data_no_time_correct_", as.character(y))) %>% 
    paste0(base_new, .)
  cps_temp <- readRDS(filenames)
  cps_data <- cps_data %>% 
    bind_rows(cps_temp)
  saveRDS(cps_data, here(paste0(base_new, "temp_full_CPS_data_correct.RDS")))
}

#saveRDS(cps_data, here(base, "mukoyama_all/final_data/R_final/temp_full_CPS_data_15_17.rds"))
# Save final dataset
#saveRDS(cps_data, here(base, "mukoyama_all/final_data/R_final/temp_full_CPS_data_before_new_years.rds"))

#### Adding additional years

# STILL NEEDED
# ***************************************************
#   ****Merging in UI Benefit Information
# ***************************************************
#   
#   replace undur = . if undur==-10 | undur==-1
# decode state, gen(statename)
# rename state statenum
# rename statename state
# merge m:1 year month undur state using "$other/UI_benefits_by_duration.dta" // note that this only goes through 2011.
# drop if _m==2
# /*conditioning on eligibility for benefits */
#   replace wksleft_tot = . if (untype!=1 & untype!=2 & untype!=3)
#     replace wks_tot = . if (untype!=1 & untype!=2 & untype!=3)
#       replace wksused_tot = . if (untype!=1 & untype!=2 & untype!=3)
#         drop _m
# 
# bysort final_id: gen n = _n
# bys final_id: egen max = max(n)
# drop n 
# compress
# 
# saveold "$final_CPS/full_CPS_data_no_time.dta", replace 

#################################################################
####### Reweighting to construct Minutes series
#################################################################
base <- here("data/behav_params/Mukoyama_Replication/")
paths <- c(
 # paste0(base, "mukoyama_all/final_data/R_final/full_CPS_data"),
           #paste0(base, "mukoyama_all/final_data/R_final/temp_full_CPS_data_before_new_years"),
           paste0(base, "mukoyama_all/final_data/R_final/temp_full_CPS_2015_19_correct"),
           paste0(base, "mukoyama_all/final_data/R_final/temp_full_CPS_2020_24_correct"))

data_time <- read_dta(here(base, "mukoyama_all/int_data/ATUS/time_method_reweight.dta"))
data_time_2014_new <- read.csv(here(base, "mukoyama_all/int_data/ATUS/time_method_reweight_2014_new.csv")) %>% tibble
data_time_2023 <- read.csv(here(base, "mukoyama_all/int_data/ATUS/time_method_reweight_2023.csv")) %>% tibble

df_list <- list("orig" = data_time, "new_2014" = data_time_2014_new, "new_2023" = data_time_2023)

for(fpath in paths){
  print(fpath)
  # Load the data
  data <- readRDS(here(paste0(fpath, ".RDS")))
  print(data)
  
  # Generate higher-order age variables
  data <- data %>%
    mutate(
      age2 = age*age,
      age3 = age*age2,
      age4 = age*age3
    ) 

  for(k in names(df_list)){
    wgts <- df_list[[k]]
  # df_list <- ('orig' = data_time, '')
      data <- data %>% 
        bind_cols(., wgts)
      
      # Define the variable list
      variables <- c("empldir", "pubemkag", "PriEmpAg", "FrendRel", "SchEmpCt", "Unionpro",
                     "Resumes", "Plcdads", "Otheractve", "lkatads", "Jbtrnprg", "otherpas",
                     "age", "age2", "age3", "age4", "female", "hs", "somecol", "college",
                     "black", "married", "marriedfemale", "np_other", "layoff",
                     "nonsearchers", "constant")
      
      # Step 1: Calculate the Inverse Mills Ratio
      data <- data %>%
        mutate(
          pi = empldir * pweight_empldir + pubemkag * pweight_pubemkag + PriEmpAg * pweight_PriEmpAg +
            FrendRel * pweight_FrendRel + SchEmpCt * pweight_SchEmpCt + Unionpro * pweight_Unionpro +
            Resumes * pweight_Resumes + Plcdads * pweight_Plcdads + Otheractve * pweight_Otheractve +
            lkatads * pweight_lkatads + Jbtrnprg * pweight_Jbtrnprg + otherpas * pweight_otherpas +
            age * pweight_age + age2 * pweight_age2 + age3 * pweight_age3 + age4 * pweight_age4 +
            female * pweight_female + hs * pweight_hs + somecol * pweight_somecol +
            college * pweight_college + black * pweight_black + married * pweight_married +
            marriedfemale * pweight_marriedfemale + np_other * pweight_np_other +
            layoff * pweight_layoff + nonsearchers * pweight_nonsearchers + pweight_constant,
          invmills = dnorm(pi) / pnorm(pi),
          psearch = pnorm(pi)
        )
      
      label <- paste0("time_create_", k)
      print(label)
      # Generate the conditional search time
      data <- data %>%
        mutate(
          search_cond = exp(empldir * sweight_empldir + pubemkag * sweight_pubemkag + PriEmpAg * sweight_PriEmpAg +
                              FrendRel * sweight_FrendRel + SchEmpCt * sweight_SchEmpCt + Unionpro * sweight_Unionpro +
                              Resumes * sweight_Resumes + Plcdads * sweight_Plcdads + Otheractve * sweight_Otheractve +
                              lkatads * sweight_lkatads + Jbtrnprg * sweight_Jbtrnprg + otherpas * sweight_otherpas +
                              age * sweight_age + age2 * sweight_age2 + age3 * sweight_age3 + age4 * sweight_age4 +
                              female * sweight_female + hs * sweight_hs + somecol * sweight_somecol +
                              college * sweight_college + black * sweight_black + married * sweight_married +
                              marriedfemale * sweight_marriedfemale + np_other * sweight_np_other +
                              layoff * sweight_layoff + nonsearchers * sweight_nonsearchers +
                              sweight_constant + invmills * sweight_invmills + sigma_hat / 2),
          !!sym(label) := psearch * search_cond
        )
      
      # Replace employed searchers' search time with NA
      data <- data %>%
        mutate(!!sym(label) := ifelse(mlr == 1 | mlr == 2, NA, !!sym(label)))
      
      # Drop unnecessary variables
      data <- data %>%
        select(-matches("weight"), -sigma_hat, -psearch, -search_cond, -pi, -invmills) %>%
        filter(!is.na(final_id)) 
  }
  saveRDS(data, here(paste0(fpath, "_all_weights.RDS")))
  rm(data)
}
      
    
      #saveRDS(data, here(paste0(base_new, "cps_data_full_", as.character(y), ".rds")))
      #print(paste0("Saved: ", as.character(y)))

#saveRDS(data, here(paste0(base, "mukoyama_all/final_data/R_final/cps_data_full_2014_all_weights.rds")))

# Save the cleaned data
#write_dta(data, "$final_CPS/full_CPS_data.dta")

