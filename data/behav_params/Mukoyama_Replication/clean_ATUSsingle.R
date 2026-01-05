# Load necessary libraries
library(here)
library(tidyverse)
library(data.table)
library(haven)
library(janitor)
library(assertthat)

checking_0314 = FALSE
# Set the working directory (assuming the paths are correctly set in variables)
base <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/ATUS")

if(checking_0314){
  # Function to read and combine single-year ATUS respondent files (2003-2014)
  resp_single0314 <- list()
  for (i in 2003:2014) {
    file_path <- file.path(base, as.character(i), paste0("atusresp_", i), paste0("atusresp_", i, ".dta"))
    resp_single0314[[i]] <- read_dta(file_path)
  }
  resp_single0314 <- bind_rows(resp_single0314)
  
  # Validate unique identifier
  if (anyDuplicated(resp_single0314$tucaseid)) {
    stop("tucaseid is not unique in respondent data.")
  }
  
  # Save intermediate data
  # saveRDS(resp_single0314, file = here(base, "R_int/resp_single0314.rds"))
  
  # Function to read and combine single-year ATUS summary files (2003-2014)
  sum_single0314 <- list()
  for (i in 2003:2014) {
    file_path <- file.path(base, as.character(i), paste0("atussum_", i), paste0("atussum_", i, ".dta"))
    year_data <- read_dta(file_path)
    year_data <- mutate(year_data, year = i)
    sum_single0314[[i]] <- year_data
  }
  sum_single0314 <- bind_rows(sum_single0314)
  
  # Validate unique identifier
  if (anyDuplicated(sum_single0314$tucaseid)) {
    stop("tucaseid is not unique in summary data.")
  }
  
  # Save intermediate data
  # saveRDS(sum_single0314, file = here(base, "R_int/sum_single0314.rds"))
  
  # Merge respondent and summary files on `tucaseid`
  combined_data <- merge(
    x = readRDS(here(base, "R_int/resp_single0314.rds")),
    y = readRDS(here(base, "R_int/sum_single0314.rds")),
    by = "tucaseid",
    all.x = TRUE
  )
  
  # Validate merge
  table(combined_data$year, combined_data$tuyear, useNA = "ifany")
  
  # Keep only interview travel time variables for relevant years
  intvw_data <- combined_data %>%
    select(tucaseid, tuyear, t170504, t180504) %>%
    mutate(
      intvwtravel = ifelse(tuyear == 2004, t170504, 
                           ifelse(tuyear != 2003 & tuyear != 2004, t180504, NA))
    )
  
  # Validate missing values for intvwtravel
  cat("Missing values for intvwtravel: ", sum(is.na(intvw_data$intvwtravel)), "\n")
  cat("Number of rows for year 2003: ", nrow(intvw_data %>% filter(tuyear == 2003)), "\n")
  
  # Drop tuyear and sort by `tucaseid`
  intvw_data <- intvw_data %>%
    select(-tuyear) %>%
    arrange(tucaseid)
  
  # Checking equality with reference file
  if (!all.equal(read_dta(here(base, "intvwtravel_0314.dta")), intvw_data, check.attributes = FALSE)) {
    stop("not equal to original file intvwtravel_0314.dta.")
  }
  
  intvw_data_ref <- read_dta(here(base, "intvwtravel_0314.dta"))
  
  if(all.equal(intvw_data, intvw_data_ref, check.attributes = FALSE)){
    # Save final dataset
    saveRDS(intvw_data, file = file.path(base, "R_int/intvwtravel_0314_new.rds"))
  }else{print("datasets not equivalent.")}

}

###### Creating rest

test_equal <- function(df1, df2){
  names_common <- intersect(names(df1), names(df2))
  df1_arranged <- df1 %>% arrange(tucaseid, tudiaryday)
  df2_arranged <- df2 %>% arrange(tucaseid, tudiaryday)
  stopifnot(
    identical(df1, df1_arranged),
    identical(df2, df2_arranged),
    all.equal(df1_arranged$tucaseid, df2_arranged$tucaseid, check.attributes = FALSE),
    all.equal(df1_arranged$tudiaryday, df2_arranged$tudiaryday, check.attributes = FALSE),
    
    df1 %>% group_by(tucaseid, tudiaryday) %>% n_groups(.) == nrow(df1),
    df2 %>% group_by(tucaseid, tudiaryday) %>% n_groups(.) == nrow(df2),
    all.equal(zap_labels(select(df1, all_of(names_common))), zap_labels(select(df2, all_of(names_common))), check.attributes = FALSE)
  )
}

################################################################################
# Load datasets
################################################################################
ref_atus_resp <- read_dta(here(base, "atusresp_0314.dta"))
ref_atus_sum <- read_dta(here(base, "atussum_0314.dta"))
ref_atus_rost <- read_dta(here(base, "atusrost_0314.dta"))
ref_atus_cps <- read_dta(here(base, "atuscps_0314.dta"))

################################################################################
# RESP
# Manage to replicate the data in 2003-2014 except trtfamily and trnumhou - 
# these vars are not used in any subsequent analysis so I remove the variables 
# from the dataset before saving to avoid erroneous use in the future
################################################################################

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_resp <- list()
for (i in 2003:2014) {
  file_path <- file.path(base, as.character(i), paste0("atusresp_", i), paste0("atusresp_", i, ".dat"))
  new_atus_resp[[i-2012]] <- read.delim(file_path, sep = ",")
}

new_atus_resp <- new_atus_resp %>% 
  bind_rows() %>% 
  tibble %>% 
  clean_names %>% 
  select(any_of(names(ref_atus_resp))) %>% 
  mutate(across(c("trtohh",
                  "trtonhh",
                  "trtnohh",
                  "trto",
                  "trthh",
                  "tremodr",
                  "trtalone_wk",
                  "trtccc_wk",
                  "trwbmodr",
                  "trlvmodr",
                  "trtec",
                  "tuecytd",
                  "tuelder",
                  "tuelfreq",
                  "tuelnum"), ~ifelse(is.na(.), -1, .)))

# Validate unique identifier
if (anyDuplicated(new_atus_resp$tucaseid)) {
  stop("tucaseid is not unique in respondent data.")
}

names_common <- intersect(names(new_atus_resp), names(ref_atus_resp))
all.equal(select(new_atus_resp, all_of(names_common)), zap_labels(select(ref_atus_resp, all_of(names_common))), check.attributes = FALSE)

# trtfamily does not match for 39 cases....!
# 34,693 cases of trnumhou are NA
for(k in c("trtfamily", "trnumhou")){
  new_atus_resp %>% left_join(., ref_atus_resp, by = "tucaseid") %>% select(tucaseid, paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% select(-tucaseid) %>% print
}

ext_atus_resp <- list()
for (i in 2015:2023) {
  file_path <- file.path(base, "additional_years", as.character(i), paste0("atusresp_", i), paste0("atusresp_", i, ".dat"))
  ext_atus_resp[[i-2014]] <- read.delim(file_path, sep = ",")
}
ext_atus_resp <- bind_rows(ext_atus_resp) %>% 
  tibble %>% 
  clean_names() %>% 
  mutate(across(c("trtohh",
                  "trtonhh",
                  "trtnohh",
                  "trto",
                  "trthh",
                  "tremodr",
                  "trtalone_wk",
                  "trtccc_wk",
                  "trwbmodr",
                  "trlvmodr",
                  "trtec",
                  "tuecytd",
                  "tuelder",
                  "tuelfreq",
                  "tuelnum"), ~ifelse(is.na(.), -1, .)), 
         tufinlwgt = ifelse(tuyear == 2020 & is.na(tufinlwgt), tu20fwgt, tufinlwgt)) %>% 
  # I BELIEVE THIS IS THE CORRECT RECLASSIFICATION TO MATCH VARIABLE NAMES....!! NOT SURE THOUGH - IMPORTANT TO CHECK!
  # 2012 CODEBOOK: https://www.bls.gov/tus/dictionaries/atusintcodebk12.pdf
  # 2022
  # Using 2020 data, we need to use the to20fwgt instead of tufinlwgt
  rename(tufnwgtp = tufinlwgt) %>% 
  select(all_of(names(ref_atus_resp)))

# ref_atus_resp %>% 
#   rbind(ext_atus_resp) %>% 
#   saveRDS(here(base, "atusresp_0323.rds"))


################################################################################
# SUM
################################################################################
# The most important variables are any that start with t0504
# As in the original paper: t050481 + t050405 + t050404 + t050403 + t050499 (+ intvwtravel)
# I think t050481 is renamed to t050401 in the following files...

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_sum <- list()
for (i in 2003:2014) {
  file_path <- file.path(base, as.character(i), paste0("atussum_", i), paste0("atussum_", i, ".dta"))
  year_data <- read_dta(file_path)
  year_data <- mutate(year_data, tuyear = i)
  new_atus_sum[[i-2002]] <- year_data
}
new_atus_sum <- bind_rows(new_atus_sum) %>% 
  tibble %>% 
  clean_names()

# Validate unique identifier
if (anyDuplicated(new_atus_sum$tucaseid)) {
  stop("tucaseid is not unique in respondent data.")
}

issue_vars <- c("gemetsta",
"t010499",
"t010501",
"t019999",
"t020199",
"t020299",
"t020599",
"t020699",
"t020899",
"t030204",
"t040202",
"t040203" ,
"t040301" ,
"t040303" ,
"t040399" ,
"t040499" ,
"t049999" ,
"t050203" ,
"t050204" ,
"t050302" ,
"t050499" ,
"t059999" ,
"t060104" ,
"t060203" ,
"t060402" ,
"t060403" ,
"t070199" ,
"t070299" ,
"t079999" ,
"t080102" ,
"t080299" ,
"t080302" ,
"t080499" ,
"t080599" ,
"t080602" ,
"t090101" ,
"t090102" ,
"t090104" ,
"t090299" ,
"t090302" ,
"t090599" ,
"t100299" ,
"t100399" ,
"t119999" ,
"t120199" ,
"t120502" ,
"t120599" ,
"t129999" ,
"t130115" ,
"t130205" ,
"t130206" ,
"t130207" ,
"t130209" ,
"t130210" ,
"t130212" ,
"t130214" ,
"t130215" ,
"t130217" ,
"t130219" ,
"t130220" ,
"t130221" ,
"t130223" ,
"t130229" ,
"t130230" ,
"t130231" ,
"t130232" ,
"t130399" ,
"t130401" ,
"t130402" ,
"t149999" ,
"t150301" ,
"t150399" ,
"t150499" ,
"t150599",
"t150699",
"t160107",
"t500104",
"t070301",
"t080199",
"t090499",
"t100199",
"t130108",
"t130121",
"t130123",
"t130135",
"t130204",
"gtmetsta",
"t010199",
"t060303",
"t080799",
"t090399",
"t090402",
"t100401",
"t109999",
"t130111",
"t140104",
"t180101",
"t180199",
"t180501",
"t180502",
"t180601",
"t180699",
"t180701",
"t180801",
"t180802",
"t180803",
"t180804",
"t180805",
"t180806",
"t180807",
"t180899",
"t180901",
"t180902",
"t180903",
"t180904",
"t180905",
"t180999",
"t181002",
"t181099",
"t181101",
"t181201",
"t181202",
"t181204",
"t181299",
"t181301",
"t181302",
"t181399",
"t181401",
"t181499",
"t181501",
"t181599",
"t181601" , 
"t181801" , 
"t181899" ,
"t189999" , 
"t080699" , 
"t180499" , 
"t040204" , 
"t050405" , 
"t130201" , 
"t140105" , 
"t180399" , 
"t040299" , 
"t120405" , 
"t010599" , 
"t130211" , 
"t100499" , 
"t080399" , 
"t181699")

non_issue_vars <- c()
for(k in issue_vars){
  test <- new_atus_sum %>% left_join(., ref_atus_sum, by = c("tucaseid", "tuyear")) %>% select(tucaseid, paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% select(-tucaseid) %>% distinct
  if(nrow(test) == 1){
    non_issue_vars <- c(k, non_issue_vars)
    print(paste0(k, "added to nonissuevars"))
  }else{print(test)}
}


# Change all NA values for non_issue_vars to 0 if NA
new_atus_sum <- new_atus_sum %>% 
  mutate(across(all_of(non_issue_vars), ~ifelse(is.na(.), 0, .)))
# Remove non-issue vars from issue_vars
issue_vars <- setdiff(issue_vars, non_issue_vars)

non_issue_vars_min_2003 <- c()
for(k in issue_vars){
  test <- new_atus_sum %>% left_join(., ref_atus_sum, by = c("tucaseid", "tuyear")) %>% filter(tuyear > 2003) %>% select(tucaseid, paste0(k, ".x"), paste0(k, ".y"))  %>% 
                                       filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% select(-tucaseid) %>% distinct
  if(nrow(test) == 1){
    non_issue_vars_min_2003 <- c(k, non_issue_vars_min_2003)
    print(paste0(k, "added to nonissuevars"))
    print(test)
  }else{
      print(test)
    }
}

# In the case of each variable, there is only one or two  observation where NA does not equate to 0 so we set NA equal to zero in all cases
# Change all NA values for non_issue_vars_min_2003 to 0 if NA
new_atus_sum <- new_atus_sum %>% 
  mutate(across(all_of(non_issue_vars_min_2003), ~ifelse(is.na(.) & tuyear > 2003, 0, .)))

# Remove non-issue vars from issue_vars
issue_vars <- setdiff(issue_vars, non_issue_vars_min_2003)

non_issue_vars_min_2004 <- c()
for(k in issue_vars){
  test <- new_atus_sum %>% left_join(., ref_atus_sum, by = c("tucaseid", "tuyear")) %>% filter(tuyear > 2004) %>% select(tucaseid, paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% select(-tucaseid) %>% distinct
  if(nrow(test) == 1){
    non_issue_vars_min_2004 <- c(k, non_issue_vars_min_2004)
    print(paste0(k, "added to nonissuevars"))
    print(test)
  }else{
    print(test)
  }
}

# Remove non-issue vars from issue_vars
issue_vars <- setdiff(issue_vars, non_issue_vars_min_2004)

new_atus_sum <- new_atus_sum %>% 
  mutate(gemetsta = ifelse(is.na(gemetsta) & tuyear > 2004, -1, gemetsta),
         across(non_issue_vars_min_2004[non_issue_vars_min_2004 != "gemetsta"], ~ifelse(is.na(.) & tuyear > 2004, 0, .)))

for(k in issue_vars){
  test <- new_atus_sum %>% left_join(., ref_atus_sum, by = c("tucaseid", "tuyear")) %>% select(tuyear, paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% distinct
  
  assert_that(all.equal(unique(test$tuyear), c(2003, 2004)))
}


names_common <- intersect(names(new_atus_sum), names(ref_atus_sum)) 
all.equal(zap_labels(select(new_atus_sum, all_of(names_common))), 
          zap_labels(select(ref_atus_sum, all_of(names_common))), 
                     check.attributes = FALSE)

ext_atus_sum <- list()
for (i in 2015:2023) {
  file_path <- file.path(base, "additional_years", as.character(i), paste0("atussum_", i), paste0("atussum_", i, ".dat"))
  year_data <- read.delim(file_path, sep = ",")
  year_data <- mutate(year_data, tuyear = i)
  ext_atus_sum[[i-2014]] <- year_data
}
ext_atus_sum <- bind_rows(ext_atus_sum) %>% 
  tibble %>% 
  clean_names()

# There are several variables that do not exist across all datasets...they are displayed below. I maintain all 
#ext_atus_sum %>% select(all_of(names(new_atus_sum)))%>% names %>% length
setdiff(names(new_atus_sum), names(ext_atus_sum))
setdiff(names(ext_atus_sum), names(new_atus_sum))

atussum_0323 <- new_atus_sum %>% 
  bind_rows(ext_atus_sum) %>% 
  zap_labels %>% 
  select(contains("wgt"), "tuyear", "tucaseid", "tryhhchild","teage", #"tu04fwgt", "tu06fwgt",
         "tesex","peeduca","ptdtrace","pehspnon","gemetsta","telfs","temjot",
         "trdpftpt","teschenr","teschlvl","trsppres","tespempnot","trernwa",
         "trchildnum","trspftpt","tehruslt","tudiaryday","trholiday",
         # Search strategy variables
         contains("t0504"), 
         # Traveling for interview time
         any_of(c("t170504", "t180504", "t180589"))) %>% 
  mutate(tufnwgtp = case_when(tuyear < 2006 ~ tu06fwgt, 
                              tuyear == 2020 ~ tu20fwgt,
                              TRUE ~ tufinlwgt))

#saveRDS(atussum_0323, here(base, "atussum_0323.rds"))

atussum_0323_wref <- ref_atus_sum %>% 
  bind_rows(ext_atus_sum) %>% 
  zap_labels %>% 
  select(contains("wgt"), "tuyear","tucaseid", "tufnwgtp", "tufinlwgt", "tryhhchild","teage", #"tu04fwgt", "tu06fwgt",
         "tesex","peeduca","ptdtrace","pehspnon","gemetsta","telfs","temjot",
         "trdpftpt","teschenr","teschlvl","trsppres","tespempnot","trernwa",
         "trchildnum","trspftpt","tehruslt","tudiaryday","trholiday",
         contains("t0504")) %>% 
  mutate(tufnwgtp = case_when(tuyear == 2020 ~ tu20fwgt,
                              tuyear >= 2015 & tuyear != 2020 ~ tufinlwgt, 
                              TRUE ~ tufnwgtp))

#saveRDS(atussum_0323_wref, here(base, "atussum_0323_wref.rds"))

testref <- atussum_0323_wref %>% #readRDS(here(base, "atussum_0323_wref.rds")) %>% #filter(tuyear <= 2014) %>% 
  select(tuyear, tucaseid, tudiaryday, tufnwgtp, contains("t0504")) %>% 
  group_by(tucaseid) %>% mutate(time = sum(t050403, t050404, t050405, t050481, t050499, t050401, na.rm = TRUE))
test <- atussum_0323 %>% 
  #filter(tuyear <= 2014) %>% 
  select(tuyear, tucaseid, tudiaryday, tufnwgtp, contains("t0504")) %>% 
  group_by(tucaseid) %>% mutate(time = sum(t050401, t050402, t050403, t050404, t050499, t050405, na.rm = TRUE))

if(all.equal(select(testref, tuyear, tucaseid, tudiaryday, tufnwgtp, time), select(test, tuyear, tucaseid, tudiaryday, tufnwgtp, time))){
  saveRDS(atussum_0323, here(base, "atussum_0323.rds"))
}

################################################################################
# ROST
################################################################################

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_rost <- list()
for (i in 2003:2014) {
  file_path <- file.path(base, as.character(i), paste0("atusrost_", i), paste0("atusrost_", i, ".dat"))
  year_data <- read.delim(file_path, sep = ",")
  year_data <- mutate(year_data, tuyear = i)
  new_atus_rost[[i-2002]] <- year_data
}

new_atus_rost <- bind_rows(new_atus_rost) %>% 
  tibble %>% 
  clean_names() %>% 
  select(tuyear, tucaseid, tulineno, terrp, teage, tesex)

# # Validate unique identifier
# if (anyDuplicated(new_atus_rost$tucaseid)) {
#   stop("tucaseid is not unique in respondent data.")
# }
# 
all.equal(select(new_atus_rost, -tuyear), zap_labels(ref_atus_rost), check.attributes = FALSE)

ext_atus_rost <- list()
for (i in 2015:2023) {
  file_path <- file.path(base, "additional_years", as.character(i), paste0("atusrost_", i), paste0("atusrost_", i, ".dat"))
  year_data <- read.delim(file_path, sep = ",")
  year_data <- mutate(year_data, tuyear = i)
  ext_atus_rost[[i-2014]] <- year_data
}
ext_atus_rost <- bind_rows(ext_atus_rost) %>% 
  tibble %>% 
  clean_names() %>% 
  select(tuyear, tucaseid, tulineno, terrp, teage, tesex)

# ext_atus_rost %>% 
#   rbind(new_atus_rost, .) %>% 
#   saveRDS(here(base, "atusrost_0323.rds"))
  

test <- readRDS(here(base, "atusrost_0323.rds"))

################################################################################
# CPS
################################################################################

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_cps <- list()
for (i in 2003:2014) {
    file_path <- file.path(base, as.character(i), paste0("atuscps_", i), paste0("atuscps_", i, ".dat"))
    year_data <- read.delim(file_path, sep = ",")
    year_data <- mutate(year_data, tuyear = i)
    new_atus_cps[[i-2002]] <- year_data
}
new_atus_cps <- bind_rows(new_atus_cps) %>% 
  tibble %>% 
  clean_names()

# # Validate unique identifier
# if (anyDuplicated(new_atus_cps$tucaseid)) {
#   stop("tucaseid is not unique in respondent data.")
# }

issue_vars <- c(
  "huspnish",
  "hufaminc",
  "huhhnum",
  "puafever",
  "pupelig",
  "peafwhen",
  "peernh1o",
  "hrsample",
  "hrsersuf",
  "prernhly"  ,             
  "gemetsta",
  "gtmetsta",
  "hrhhid2",
  "hetenure",
  "peafever","peafwhn1",
  "peafwhn2","peafwhn3",
  "peafwhn4","tratusr",
  "pedadtyp","pecohab", 
  "pelndad", "pemomtyp",
  "pelnmom", "pedisdrs",
  "pedisear", "pediseye",
  "pedisout", "pedisphy",
  "pedisrem", "prdisflg",
  "hefaminc", "hxfaminc",
  "prdasian",
  "gediv",  
  "pepdemp1",
  "ptnmemp1",
  "pepdemp2",
  "ptnmemp2"
)

non_issue_vars <- c()
for(k in issue_vars){
  test <- new_atus_cps %>% left_join(., ref_atus_cps, by = c("tucaseid", "tulineno")) %>% select(paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% distinct
  if(nrow(test) == 1){
    non_issue_vars <- c(k, non_issue_vars)
    print(paste0(k, "added to nonissuevars"))
  }
  print(test)
}

# Change all NA values for non_issue_vars to 0 if NA
new_atus_cps <- new_atus_cps %>% 
  mutate(across(all_of(non_issue_vars), ~ifelse(is.na(.), -1, .)))
# Remove non-issue vars from issue_vars
issue_vars <- setdiff(issue_vars, non_issue_vars)

for(k in issue_vars){
  test <- new_atus_cps %>% left_join(., ref_atus_cps, by = c("tucaseid", "tulineno")) %>% select(paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% distinct
  print(test)
}

new_atus_cps <- new_atus_cps %>% 
  mutate(across(all_of(c("hrsersuf", "hrsample")), ~ifelse(is.na(.) | . == " ", -1, .)))

# Demonstrates that the differences between peernh1o and prernhly are rounding errors
new_atus_cps %>% left_join(., ref_atus_cps, by = c("tucaseid", "tulineno")) %>% select(tuyear, peernh1o.x, peernh1o.y)  %>% 
  mutate(test = round(peernh1o.x - peernh1o.y, 1)) %>% filter(test != 0| is.na(test))
new_atus_cps %>% left_join(., ref_atus_cps, by = c("tucaseid", "tulineno")) %>% select(tuyear, prernhly.x, prernhly.y)  %>% 
  mutate(test = round(prernhly.x - prernhly.y, 1)) %>% filter(test != 0 | is.na(test))


# Remove non-issue vars from issue_vars; peernh1o, prernhly are identical once you round to 0 decimal places
issue_vars <- setdiff(issue_vars, c("hrsersuf", "hrsample", "peernh1o", "prernhly"))

non_issue_vars <- c()
for(k in issue_vars){
  test <- new_atus_cps %>% left_join(., ref_atus_cps, by = c("tucaseid", "tulineno")) %>% select(tuyear, paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((round(get(paste0(k, ".x")), 0) != round(get(paste0(k, ".y")), 0)) %in% c(TRUE, NA)) #%>% distinct
  if(nrow(test) == 1){
    non_issue_vars <- c(k, non_issue_vars)
    print(paste0(k, "added to nonissuevars"))
  }
  print(test)
}

# NOTE: HETENURE AND TRATUSR ARE MISSING IN 2003, 2004, 2005 (only tratusr) - I DO NOT BELIEVE THEY ARE USED ANYWHERE ELSE SO WILL IGNORE FOR NOW
names_common <- intersect(names(new_atus_cps), names(ref_atus_cps))
setdiff(names(ref_atus_cps), names_common) # only one name missing
setdiff(names(new_atus_cps), names_common)
all.equal(arrange(select(new_atus_cps, all_of(names_common)), tucaseid, tulineno), zap_labels(arrange(select(ref_atus_cps, all_of(names_common)), tucaseid, tulineno)), check.attributes = FALSE)

ext_atus_cps <- list()
for (i in 2015:2023) {
  file_path <- file.path(base, "additional_years", as.character(i), paste0("atuscps_", i), paste0("atuscps_", i, ".dat"))
  year_data <- read.delim(file_path, sep = ",")
  year_data <- mutate(year_data, tuyear = i)
  ext_atus_cps[[i-2014]] <- year_data
}
ext_atus_cps <- bind_rows(ext_atus_cps) %>%
  tibble %>%
  clean_names()

na_cols <- ext_atus_cps %>% summarise(across(everything(), ~sum(is.na(.)))) %>% select(where(~ any(. != 0))) %>% names

ext_atus_cps %>% select(tuyear, all_of(na_cols))

non_issue_vars <- c()
for(k in na_cols){
  test <- new_atus_cps %>% select(tuyear, paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((round(get(paste0(k, ".x")), 0) != round(get(paste0(k, ".y")), 0)) %in% c(TRUE, NA)) #%>% distinct
  if(nrow(test) == 1){
    non_issue_vars <- c(k, non_issue_vars)
    print(paste0(k, "added to nonissuevars"))
  }
  print(test)
}


### Conclusion - variables that have missing values are of three types:
# Mom & dad variables changed to parent variables in 2020
# Covid variables (covid and covr) added in month 5 of 2020
# Education related professional license and certification questions
# All NA values are tehrefore justified. I will not manipulate them here or try to consolidate the parenting relationship variables as they are not useful yet!

# Resources: https://www.bls.gov/tus/atususersguide.pdf
# https://www.bls.gov/tus/dictionaries/atuscpscodebk20.pdf
# https://www.bls.gov/tus/dictionaries/atuscpscodebk0320.pdf


# This variable was added in 2007 January CPS.
#   Starting in January 2020, CPS added PEPAR1TYP and PEPAR2TYP to describe the
#   type of parent for PEPAR1 and PEPAR2. These variables replaced PEMOMTYP and
#   PEDADTYP. Cases with [2007 < HRYEAR4 < 2020] will have values for PEDADTYP if
#   the father is present. Cases with [HRYEAR4 = 2020 or HRYEAR4 < 2007] will have
#   missing values.
 c("pedadtyp",
# This question was dropped from the CPS in January 2015. Therefore cases with
# [HRYEAR4 > 2014] will have missing values for PEGR6COR.
  "pegr6cor",
# This question was dropped from the CPS in January 2015. Therefore cases with
# [HRYEAR4 > 2014] will have missing values for PEGRPROF. - this value is encompassed in PEEDUCA
"pegrprof",
# This variable was added in 2007 January CPS.
# Starting in January 2020, CPS added PEPAR1 and PEPAR2 to identify line number of
# parents. These variables replaced PELNMOM and PELNDAD. Cases with [2006 <
#                                                                      HRYEAR4 < 2020] will have values for PELNDAD if the father is present. Cases with
# [HRYEAR4 = 2020 or HRYEAR4 < 2007] will have missing values.
"pelndad", 
# This variable was added in 2007 January CPS.
# Starting in January 2020, CPS added PEPAR1 and PEPAR2 to identify line number of
# parents. These variables replaced PELNMOM and PELNDAD. Cases with [2006 <
#                                                                      HRYEAR4 < 2020] will have values for PELNMOM if the mother is present. Cases with
# [HRYEAR4 = 2020 or HRYEAR4 < 2007] will have missing values.
"pelnmom", 
# This variable was added in 2007 January CPS.
# Starting in January 2020, CPS added PEPAR1TYP and PEPAR2TYP to describe the
# type of parent for PEPAR1 and PEPAR2. These variables replaced PEMOMTYP and
# PEDADTYP. Cases with [2006 < HRYEAR4 < 2020] will have values for PEMOMTYP if
# the mother is present. Cases with [HRYEAR4 = 2020 or HRYEAR4 < 2007] will have
# missing values.
"pemomtyp",
# This question was dropped from the CPS in January 2015. Therefore cases with
# [HRYEAR4 > 2014] will have missing values for PEMS123. - encompassed in peeduca
"pems123", 
# Starting in January 2020, PEPARENT is no longer on the CPS files. See PEPAR1 and
# PEPAR2 for information about the line numbers of the parents. Cases with [HRYEAR4 
# = 2020] will have missing values for PEPARENT.
"peparent",
"pxdadtyp",
# This question was dropped from the CPS in January 2015. Therefore cases with
# [HRYEAR4 > 2014] will have missing values for PEGR6COR.
"pxgr6cor",
"pxgrprof",
"pxlndad", 
"pxlnmom", 
"pxmomtyp",
"pxms123", 
"pxparent",
# Specific metropolitan core based statistical area (CBSA) code
# Not all CBSAs are identified. Values of 00000 are assigned for cases whose CBSA is
# not identified.
# In May 2004, CPS added GTCBSA to the monthly CPS files. Therefore, cases with
# [HRYEAR4 < 2004] or [HRYEAR4 = 2004 and HRMONTH < 5] will have missing values
# for GTCBSA.
# Values and codes were updated in August 2005, May 2014, and August 2015.
# See Appendix A for additional information about GTCBSA codes.
"gtcbsa",
# Federal Processing Standards (FIPS) county code 
# This code must be used in combination with a state code (GESTFIPS) in order to
# uniquely identify a county.
# In May 2004, CPS added GTCO to the monthly CPS files. Therefore, cases with
# [HRYEAR4 < 2004] or [HRYEAR4 = 2004 and HRMONTH < 5] will have missing values
# for GTCO.
# Values and codes were updated in August 2005, May 2014, and August 2015.
# See Appendix A for additional information about GTCO codes.
# Also, most counties are not identified. Values of 000 are assigned for cases whose
# county code is not identified.
"gtco",
# Various answers about professional certifications and licences
# Beginning in January 2015, CPS added this question to MIS-1 and MIS-5. This variable
# was added to the CPS monthly public use files in January 2017. Therefore cases with
# (HRYEAR4=2016 and HRMONTH < 12) or HRYEAR4 < 2016, will have missing values
# for PECERT1.
# Does not include business licenses, such as a liquor license or vending license. 
"pecert1",
"pecert2",
"pecert3",
"pxcert1",
"pxcert2", 
"pxcert3", 
# Various info about parenthood that replaced pemom and pedad
"pepar1",
"pepar2",
"pepar1typ", 
"pepar2typ", 
"pxpar1",
"pxpar2",
"pxpar1typ",
"pxpar2typ",
# Teleworking questions since 2022-2023
# https://www2.census.gov/programs-surveys/cps/techdocs/Telework_Extract_File_Technical_Documentation_010424.pdf
"ptcovid1",
"ptcovid2",
"ptcovid3",
"ptcovid4",
"ptcovid5w",
"prernmin", 
# Teleworking questions since 2022-2023
# https://www2.census.gov/programs-surveys/cps/techdocs/Telework_Extract_File_Technical_Documentation_010424.pdf
"ptcovr1",
"ptcovr2",
"ptcovr3",
"ptcovr4",
"pxcovr1", 
"pxcovr2", 
"pxcovr3", 
"pxcovr4")
 
 
 
 # Conclusion - NA values are fine as these are simply changes made in 2020 to parenting, professional certification, and covid variables
 # Vars needed
 # 
 # "tucaseid", "tulineno", "ptdtrace", "peeduca"
 
full_cps <- ext_atus_cps %>%
   bind_rows(new_atus_cps, .)
 
test_na <- full_cps %>% 
  select("tuyear", "tucaseid", "tulineno", "ptdtrace", "peeduca") %>% 
  summarise(across(everything(), ~sum(is.na(.)))) %>% 
  rowSums() == 0

if(test_na){
  saveRDS(full_cps, here(base, "atuscps_0323.rds"))
}


################################################################################
# intvwtravel
################################################################################

########
# Merge respondent and summary files on `tucaseid`
combined_data <- merge(
  x = readRDS(here(base, "atusresp_0323.rds")),
  y = readRDS(here(base, "atussum_0323.rds")),
  by = c("tucaseid", "tuyear"),
  all.x = TRUE
)

# Validate merge
table(combined_data$tuyear, combined_data$tuyear, useNA = "ifany")

# Keep only interview travel time variables for relevant years
intvw_data <- combined_data %>%
  select(tucaseid, tuyear, t170504, t180504) %>%
  mutate(
    intvwtravel = ifelse(tuyear == 2004, t170504, 
                         ifelse(tuyear != 2003 & tuyear != 2004, t180504, NA))
  )

# Validate missing values for intvwtravel
cat("Missing values for intvwtravel: ", sum(is.na(intvw_data$intvwtravel)), "\n")
cat("Number of rows for year 2003: ", nrow(intvw_data %>% filter(tuyear == 2003)), "\n")

# Drop tuyear and sort by `tucaseid`
intvw_data_short <- intvw_data %>%
  filter(tuyear <= 2014) %>% 
  select(-tuyear) %>%
  arrange(tucaseid)

intvw_data_ref <- read_dta(here(base, "intvwtravel_0314.dta"))

# Drop tuyear and sort by `tucaseid`
intvw_data <- intvw_data %>%
  select(-tuyear) %>%
  arrange(tucaseid)

intvw_data_ref %>% all.equal(slice(intvw_data, 1:nrow(intvw_data_ref)), check.attributes = FALSE)

if(all.equal(intvw_data_short, intvw_data_ref, check.attributes = FALSE)){
  # Save final dataset
  saveRDS(intvw_data_short, file = file.path(base, "R_int/intvwtravel_0314_new.rds"))
  saveRDS(intvw_data, file = file.path(base, "intvwtravel_0323.rds"))
}else{print("datasets not equivalent.")}




