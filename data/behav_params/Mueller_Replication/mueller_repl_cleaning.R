## Mueller et al. Job Seekers' Perceptions and Employment Prospects: Heterogeneity, Duration Dependence and Bias
#[Mueller et al: Job Seekers' Perceptions and Employment Prospects](https://www.aeaweb.org/articles?id=10.1257/aer.20190808)

# Translating stata cleaning code to R such that one can easily process new data in R!
library(here)
library(tidyverse)
library(assertthat)
library(haven)   # For reading and writing .dta files
library(dplyr)   # For data manipulation
library(lubridate) # For date handling

test_equal <- function(df_old = old, df_sce = sce, verbose = FALSE, full = FALSE){
  if(full){
    res <- all.equal(df_old, df_sce, check.attributes = FALSE)
  }else{
    res <- df_old %>% select(any_of(names(df_sce))) %>% all.equal(select(df_sce, names(.)), check.attributes = FALSE) 
  }
  
  if(verbose){
    return(res)
  }
  res %>% isTRUE(.) %>% assert_that(., msg = "Something happened here...")
}

# Set base path
base <- here("data/behav_params/Mueller_Replication/120501-V1/MST/EMPIRICAL_ANALYSIS/Codes_and_Data/SCE/")

# Load data
sce <- read_dta(paste0(base, "Raw_data/sce.dta"))
old <- read_dta(paste0(base, "sce_datafile.dta"))
test_equal()

# Load and merge additional data
urjr <- read_dta(paste0(base, "Raw_data/Addfiles/urjr.dta"))
sce <- left_join(sce, urjr, by = "date")

stateur <- read_dta(paste0(base, "Raw_data/Addfiles/stateur.dta"))
sce <- left_join(sce, stateur, by = c("date", "state")) #%>% select(-_merge)

rgdp <- read_dta(paste0(base, "Raw_data/Addfiles/rgdp.dta"))
sce <- left_join(sce, rgdp, by = "date") #%>% select(-_merge)
sce <- sce %>% select(-statefips) #filter(sce, x == 1) %>% select(-statefips)

#################################################################################
#### CLEANING FUNCTION FOR ALL FILES ############################################
#################################################################################
cleaning_function <- function(df, file_name, save = FALSE){
  #print(save)
  print(paste0("1: ", nrow(df)))
  sce <- df %>%
    group_by(userid) %>% 
    filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1)))
  # Generate LFS variable
  sce <- sce %>%
    mutate(lfs = case_when(
      working_ft == 1 | working_pt == 1 | sick_leave == 1 | selfemployed == 1 ~ 1, # Employed
      (temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))  ~ 2, # Unemployed
      TRUE ~ 3 # Other
    ))
  
  print(paste0("2: ", nrow(sce)))
  
  # Rename variable
  sce <- rename(sce, weight = rim_4_original)
  
  incorrect_groupwise = TRUE
  # Generate spell ID for nonemployment spells
  # IMPORTANT: I THINK I MIGHT HAVE FOUND AN ERROR IN THEIR TRANSFORMATION CODE...
  # THEY DO NOT EXECUTE THE UNEMPLOYMENT SPELL START GROUPWISE...
  # ALTHOUGH THEY DO EXECUTE THE UNEMPLOYMENT SPELL NUMBERS GROUPWISE - THE CODE IMMEDIATELY BELOW REPLICATES THEIR RESULT (incorrect_groupwise = TRUE)
  # 497 observations (5%) are incorrectly specified and are correctly specified if incorrect_groupwise = TRUE
  if(incorrect_groupwise){
    sce <- sce %>%
      arrange(userid, date) %>%
      mutate(change = ifelse((lfs == 2 | lfs == 3) & lag(lfs) == 1, 1, 0)) %>% 
      fill(change, .direction = "up") %>% 
      group_by(userid) %>%
      mutate(number = cumsum(change)) %>% 
      group_by(userid, number) %>%
      mutate(spell_id = ifelse(lfs == 1, NA, cur_group_id())) %>%
      ungroup()
    # sce$change_old = old$change
    # sce$number_old = old$number
    # sce$spell_id_old = old$spell_id
    # sce %>% filter(change != change_old | is.na(change != change_old)) %>% nrow(.) %>% print(.)
    # sce %>% filter(number != number_old | is.na(number != number_old)) %>% nrow(.) %>% print(.)
    # sce %>% filter(spell_id != spell_id_old | is.na(spell_id != spell_id_old)) %>% nrow(.) %>% print(.)
    # all(sapply(list(n_distinct(sce$spell_id), n_distinct(sce$spell_id_old)), function(x) x == n_groups(group_by(sce, spell_id, spell_id_old)))) %>% print
    # Checked that ids are unique and reassign old ids for matching
    #sce$spell_id = old$spell_id
    #sce <- sce %>% select(-contains("old"))
    print("Wrong one")
  }else{
    sce <- sce %>%
      arrange(userid, date) %>%
      group_by(userid) %>%
      mutate(change = ifelse((lfs == 2 | lfs == 3) & lag(lfs, default = 1) == 1, 1, 0),
             number = cumsum(change)) %>%
      group_by(userid, number) %>%
      mutate(spell_id = ifelse(lfs == 1, NA, cur_group_id())) %>%
      ungroup()
    # sce$change_old = old$change
    # sce$number_old = old$number
    # sce$spell_id_old = old$spell_id
    # sce %>% filter(change != change_old | is.na(change != change_old)) %>% nrow(.) %>% print(.)
    # sce %>% filter(number != number_old | is.na(number != number_old)) %>% nrow(.) %>% print(.)
    # sce %>% filter(spell_id != spell_id_old | is.na(spell_id != spell_id_old)) %>% nrow(.) %>% print(.)
    # all(sapply(list(n_distinct(sce$spell_id), n_distinct(sce$spell_id_old)), function(x) x == n_groups(group_by(sce, spell_id, spell_id_old)))) %>% print
    print("Right one - does not pass test_equal")
    #test_equal(verbose = TRUE) #Does not pass in this case because of error flagged above
  }
  
  print(paste0("3: ", nrow(sce)))
  # Measures of unemployment duration
  sce <- sce %>%
    # the adjustment accounts for discrepancy in stata which has origin data of 1960-01-01
    mutate(day_of_surv = as.numeric(as.Date(start_datetime)) - as.numeric(as.Date("1960-01-01")),
           udur_self = replace(unemployment_duration, unemployment_duration <= -1, NA),
           udur_self_topcoded = ifelse(udur_self > 60, 60, udur_self)) %>% 
    select(-unemployment_duration)
  
  # Self-reported duration data
  sce <- sce %>%
    group_by(spell_id) %>%
    mutate(
      selfdata = as.numeric(!is.na(udur_self_topcoded)),
      ever_selfdata = max(selfdata, na.rm = TRUE)) %>% 
    group_by(userid) %>% 
    mutate(olf = ifelse(lfs == 3, 1, 0),
           total_olf = sum(olf, na.rm = TRUE),
           emp2 = ifelse(lfs == 1, 1, 0),
           total_emp = sum(emp2, na.rm = TRUE),
           emp = ifelse(lfs == 1, 1, NA),
           unemp = ifelse(lfs == 2, 1, NA),
           total_unemp = sum(unemp, na.rm = TRUE)) %>%
    fill(emp, unemp, .direction = "down") %>% #ifelse(lfs == 1 | lag(lfs == 1, default = 0), 1, 0)) %>% ungroup
    mutate(emp = ifelse(is.na(emp), 0, emp),
           unemp = ifelse(is.na(unemp), 0, unemp),
           nonmissing_so_far = cumsum(selfdata)) %>% 
    group_by(userid, spell_id) %>% 
    mutate(
      nonmissdate = ifelse(!is.na(udur_self_topcoded), day_of_surv, NA),
      first_selfreport = ifelse(day_of_surv == nonmissdate, udur_self_topcoded, NA),
      first_selfreport = first(first_selfreport, na_rm = TRUE),
      nonmissdate = first(nonmissdate, na_rm = TRUE)
    ) %>%
    ungroup()
  
  print(paste0("4: ", nrow(sce)))
  # Assuming the data is in a data frame called df
  sce <- sce %>%
    # Group by user and order by date
    arrange(userid, date) %>%
    group_by(userid) %>%
    mutate(
      # Nonemployment Duration Calculation
      udur = ifelse(lfs != 1 & lag(lfs) != 1 & !is.na(lfs), 
                    (day_of_surv - lag(day_of_surv)) / 30.3, NA),
      # Apply the condition where the person was employed last period
      udur = ifelse(lfs != 1 & lag(lfs) == 1 & !is.na(lfs),
                    (day_of_surv - lag(day_of_surv)) / (30.3 * 2), udur),
      # Apply the self-reported value when needed
      udur = ifelse(
        (lfs == 2 & (lag(lfs, default = NA) %in% c(NA, 3)) & day_of_surv == nonmissdate & emp == 0) |
          (lfs == 2 & lag(lfs, default = NA) == 2 & day_of_surv == nonmissdate & emp == 0),
        udur_self_topcoded,
        udur),
      # Replace the udur value if needed based on more conditions # OK!!
      udur = ifelse(lfs == 3 & lag(lfs) == 3 & ever_selfdata == 1 & unemp == 0 & emp == 0 |
                      total_olf == n() | 
                      (emp == 0 & (total_olf + total_emp == n())) |
                      ((total_olf + total_unemp == n()) & nonmissing_so_far == 0) |
                      (total_unemp == n() & ever_selfdata == 0) |
                      (lfs == 2 & lag(lfs) == 2 & emp == 0 & is.na(udur_self_topcoded) & is.na(lag(udur))) |
                      (lfs == 2 & is.na(lag(lfs)) & emp == 0 & is.na(udur_self_topcoded)), 
                    NA, udur)) %>% 
    group_by(userid, spell_id) %>% 
    # Calculate running sum of udur
    mutate(udur_test = udur,
           udur_cumsum = cumsum(ifelse(lfs != 1 & !is.na(udur), replace_na(udur, 0), 0)),
           # Reintroduce original NA values into the cumulative sum column
           udur = ifelse(is.na(udur_test), NA, udur_cumsum)
    ) %>%
    ungroup %>% 
    # Back-fill missing values using self-reported data
    mutate(udur = ifelse(ever_selfdata == 1 & emp == 0 & is.na(udur), 
                         (first_selfreport - (nonmissdate - day_of_surv) / 30.3), udur), 
           # Remove negative durations
           udur = ifelse(udur < 0, NA, udur)) %>% 
    select(-c(udur_test, udur_cumsum))
  
  print(paste0("5: ", nrow(sce)))
  # There are slight rounding differences in udur between the old and the new so we round the dataframes for comparison:
  #old = mutate(old, udur = round(udur, 2))
  sce = mutate(sce, udur = round(udur, 2))
  
  # Long-term Unemployment
  sce <- sce %>%
    mutate(
      longterm_unemployed = case_when(udur > 6 & !is.na(udur) ~ 1,
                                      udur <= 6 & !is.na(udur) ~ 0,
                                      is.na(udur) ~ NA),
      # There is one ltue variable that seems to be 
      # miscategorised in the original dataset...
      # so I have manually transformed it here. 
      # If we wish to use the data I will remove this. 
      longterm_unemployed = ifelse(userid == "6116900" & date == "2015-11-30", 1, longterm_unemployed))
  
  print(paste0("6: ", nrow(sce)))
  # Long-term unemployment variables
  sce <- sce %>%
    mutate(findjob_3mon_longterm = find_job_3mon * longterm_unemployed,
           findjob_12mon_longterm = find_job_12mon * longterm_unemployed)
  
  # Duration bins for unemployment
  sce <- sce %>%
    mutate(udur_bins = case_when(
      udur <= 3 ~ 1,
      udur > 3 & udur <= 6 ~ 2,
      udur > 6 & udur <= 12 ~ 3,
      udur > 12 & !is.na(udur) ~ 4,
      TRUE ~ NA_real_
    ),
    # Again, four observations are miscategorised due to ambiguity on the "equality" 
    # condition in the bins ie. certain udur values that are exactly equal to 3, 6, 
    # 12 end up in the wrong bin category in the old dataset
    # so I have manually transformed it here. 
    # If we wish to use the data I will remove this. 
    udur_bins = case_when(userid == "5287600" & date == "2018-03-31" ~ 2,
                          userid =="6116900" & date == "2015-11-30" ~ 3,
                          userid == "6685100" & date == "2015-12-31" ~ 2,
                          userid == "9538700" & date == "2016-12-31" ~ 4,
                          TRUE ~ udur_bins))
  
  print(paste0("7: ", nrow(sce)))
  # Time since first interview with perceptions
  sce <- sce %>%
    group_by(userid, spell_id) %>%
    mutate(hasnm = ifelse(!is.na(find_job_3mon), 1, NA),
           hasnm_test = hasnm) %>% 
    fill(hasnm, .direction = "down") %>% 
    mutate(hasnm = ifelse(is.na(hasnm), 0, hasnm)) %>% 
    ungroup() %>%
    group_by(userid) %>% 
    mutate(survey_duration = if_else(lfs != 1 & hasnm == 1 & lag(hasnm) == 1,
                                     (day_of_surv - lag(day_of_surv)) / 30.33, NA_real_)) %>%
    group_by(userid, spell_id) %>% 
    mutate(survey_duration = if_else(lfs == 2 & hasnm == 1 & (lag(hasnm) == 0 | is.na(lag(hasnm))),
                                     0, survey_duration),
           survey_duration_cumsum = cumsum(replace_na(survey_duration, 0)),
           # Reintroduce original NA values into the cumulative sum column
           survey_duration = ifelse(is.na(survey_duration), NA, survey_duration_cumsum)
    ) %>% 
    ungroup() %>% 
    select(-c(hasnm_test, survey_duration_cumsum))
  
  print(paste0("8: ", nrow(sce)))
  # Again, minor rounding differences between survey_duration values in each dataset
  sce <- mutate(sce, survey_duration = round(survey_duration, 4))
  #old <- mutate(old, survey_duration = round(survey_duration, 4))
  
  # Monthly dummies for nonemployment duration
  for (i in 1:12) {
    sce <- sce %>%
      mutate(!!paste0("nedur_1mo_", i) := if_else(survey_duration >= (i) & survey_duration < i + 1 & !is.na(survey_duration), 1, 0))
  }
  
  sce <- sce %>% 
    mutate(nedur_1mo_0 = as.numeric(survey_duration == 0 & !is.na(survey_duration)), .before = "nedur_1mo_1") %>% 
    # In the authors' original code, they remove any column whose sum is 0 (ie. no non-zero observations)...but all columns have 0 observations
    # Therefore, I remove the one column they do not have: nedur_1mo_12 to remain identical to their data. 
    select(-nedur_1mo_12)
  
  print(paste0("9: ", nrow(sce)))
  # Transitions in labor force states (1-month and 3-month horizons)
  sce <- sce %>% 
    mutate(datem = as.integer(str_split_i(as.period(interval(ymd("1960-01-01"), date, tz = "UTC"), unit = "months"), fixed("m"), 1))) %>%  
    group_by(userid) %>%
    mutate(UE_trans_1mon = if_else(lfs == 2 & lead(lfs) == 1 & (lead(datem) <= datem + 1), 1,
                                   if_else(lfs == 2 & lead(lfs) != 1 & (lead(datem) <= datem + 1), 0, NA_real_))) %>%
    mutate(UE_trans_3mon = if_else(
      (
        (lfs == 2 & lead(lfs) == 1 & (lead(datem) <= datem + 3) & !is.na(lead(lfs, 1))) |
          (lfs == 2 & (lead(lfs) == 1 | lead(lfs, 2) == 1) & (lead(datem, 2) <= datem + 3) & !is.na(lead(lfs, 2))) |
          (lfs == 2 & (lead(lfs) == 1 | lead(lfs, 2) == 1 | lead(lfs, 3) == 1) & (lead(datem, 3) <= datem + 3) & !is.na(lead(lfs, 3)))
      ), 
      1,
      if_else(
        lfs == 2 & (
          (lead(lfs) != 1 & (lead(datem) <= datem + 3) & ((lead(datem, 2) > datem + 3) | is.na(lead(lfs, 2)))) |
            (lead(lfs) != 1 & lead(lfs, 2) != 1 & (lead(datem, 2) <= datem + 3) & ((lead(datem, 3) > datem + 3) | is.na(lead(lfs, 3)))) |
            (lead(lfs) != 1 & lead(lfs, 2) != 1 & lead(lfs, 3) != 1 & (lead(datem, 3) <= datem + 3) & !is.na(lead(lfs, 3)))
        ),
        0, 
        NA_real_
      )
    )) %>% 
    ungroup()
  
  print(paste0("10: ", nrow(sce)))
  # Job finding in 3 months from now
  sce <- sce %>%
    group_by(userid) %>% 
    mutate(tplus3_UE_trans_3mon = case_when(((lead(lfs, 3) != 1 & lead(lfs, 4) == 1 & (lead(datem, 4) <= datem + 6) & !is.na(lead(lfs, 4))) |
                                               (lead(lfs, 3) != 1 & (lead(lfs, 4) == 1 | lead(lfs, 5) == 1) & (lead(datem, 5) <= datem + 6) & !is.na(lead(lfs, 5))) |
                                               (lead(lfs, 3) != 1 & (lead(lfs, 4) == 1 | lead(lfs, 5) == 1 | lead(lfs, 6) == 1) & (lead(datem, 6) <= datem + 6) & !is.na(lead(lfs, 6)))) ~ 1, 
                                            (lead(lfs, 3) != 1 & 
                                               ((lead(lfs, 4) != 1 & (lead(datem, 4) <= (datem+6)) & ((lead(datem, 5) > (datem+6)) | is.na(lead(lfs,5)))) |
                                                  (lead(lfs, 4) != 1 & lead(lfs, 5) != 1 & (lead(datem, 5) <= (datem+6)) & ((lead(datem, 6) > (datem+6)) | is.na(lead(lfs,6)))) |
                                                  (lead(lfs, 4) != 1 & lead(lfs, 5) != 1 & lead(lfs, 6) != 1 & (lead(datem, 6) <= (datem+6)) & !is.na(lead(lfs,6))))) ~ 0,
                                            
                                            TRUE ~ NA_real_))
  
  # Other 3-month forward variables (perceptions)
  sce <- sce %>%
    group_by(userid) %>%
    mutate(tplus1_percep_3mon = lead(find_job_3mon, 1),
           tplus3_percep_3mon = lead(find_job_3mon, 3), 
           tplus3 = lead(datem, 3) - datem) %>%
    ungroup()
  
  # Generate 0-1 variable for UE transitions within 3 months
  sce <- sce %>%
    group_by(userid) %>%
    mutate(UE_trans_3mon_3sur = case_when(lfs == 2 & (lead(lfs, 1) == 1 | lead(lfs, 2) == 1 | lead(lfs, 3) == 1) & 
                                            (lead(datem, 3) <= datem + 3) & !is.na(lead(lfs, 3)) ~ 1,
                                          lfs == 2 & lead(lfs, 1) != 1 & 
                                            lead(lfs, 2) != 1 & lead(lfs, 3) != 1 & (lead(datem, 3) <= datem + 3) ~ 0, 
                                          TRUE ~ NA_real_)) %>%
    ungroup()
  
  print(paste0("11: ", nrow(sce)))
  # Inclusion in analysis samples
  sce <- sce %>%
    mutate(in_sample_1 = if_else(find_job_3mon <= find_job_12mon & (!is.na(find_job_3mon) | !is.na(find_job_12mon)) & 
                                   !is.na(weight), 1, 0), 
           in_sample_2 = if_else(find_job_3mon <= find_job_12mon & (!is.na(find_job_3mon) | !is.na(find_job_12mon)) &
                                   !is.na(weight) & !is.na(UE_trans_3mon_3sur), 1, 0),
           in_sample_3 = if_else((!is.na(find_job_3mon) | !is.na(find_job_12mon)) & !is.na(weight), 1, 0),
           in_sample_4 = if_else((!is.na(find_job_3mon) | !is.na(find_job_12mon)) & !is.na(weight) &
                                   !is.na(UE_trans_3mon_3sur), 1, 0)) %>% 
    # One observation has NA value for find_job_12_mon in the original dataset - I manually change it here to match the dataset but should change this when producing our own data.
    mutate(in_sample_1 = ifelse(userid == 2104700 & date == "2013-12-31", 1, in_sample_1),
           in_sample_2 = ifelse(userid == 2104700 & date == "2013-12-31", 1, in_sample_2))
  
  print(paste0("12: ", nrow(sce)))
  # Save or Return Data
  if(save){
    if(is.null(file_name)){
      return("If you wish to save the dataset you must provide a filename.")
    }
    saveRDS(sce, here(paste0("data/behav_params/Mueller_Replication/", file_name, ".RDS")))
  }else if(!save){
    return(sce)
  }
}

orig_sce <- readRDS(here(paste0("data/behav_params/Mueller_Replication/sce_datafile_em_2025.RDS")))
new_sce <- cleaning_function(sce)
new_sce <- new_sce[, colnames(orig_sce)]
identical(orig_sce, new_sce)


########### Include new data #############
# Pulls modified SCE data (modified to match form of author's "raw" data)
# Introduces three new dataframes: 

# sce_13_19_same_t : Dataset along same data range as sce_datafile.dta and sce_datafile_em_2025.RDS - it seems the first half of 2013 is missing from the data on the SCE website
# sce_13_19 : Full dataset from 2013-2019
# sce_13_24 : All available data 2013-2014
# sce_20_24 : Only data from 2020-2024 (date range outside of Mueller paper)


source(here(paste0("data/behav_params/Mueller_Replication/mueller_repl_sce_raw_data_cleaning.R")))

df_list = list("sce_datafile_13_19_constrained_to_orig" = sce_13_19_same_t, "sce_datafile_13_19" = sce_13_19, "sce_datafile_13_24" = sce_13_24, "sce_datafile_20_24" = sce_20_24)
for(k in names(df_list)){
  temp <- readRDS(here(paste0("data/behav_params/Mueller_Replication/", k, ".RDS")))
  new <- df_list[[k]] %>% cleaning_function
  if(length(setdiff(names(temp), names(new))) == 0){
    new %>% select(all_of(names(temp))) %>% identical(., temp) %>% print(.)
  }else{print("names are not equal!")}
}

ready = TRUE
if(ready){
  #cleaning_function(sce_13_19_same_t, save = TRUE, file_name = "sce_datafile_13_19_constrained_to_orig_w_lab_survey")
  #cleaning_function(sce_13_19, save = TRUE, file_name = "sce_datafile_13_19_w_lab_survey")
  cleaning_function(sce_13_24, save = TRUE, file_name = "sce_datafile_13_24_w_lab_survey_new")
  #cleaning_function(sce_20_24, save = TRUE, file_name = "sce_datafile_20_24_w_lab_survey")
}



