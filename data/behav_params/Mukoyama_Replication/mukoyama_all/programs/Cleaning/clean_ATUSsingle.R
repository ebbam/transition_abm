# Load necessary libraries
library(here)
library(tidyverse)
library(data.table)
library(haven)
library(janitor)

# Set the working directory (assuming the paths are correctly set in variables)
base <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/ATUS")

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


# Save final dataset
# saveRDS(intvw_data, file = file.path(base, "R_int/intvwtravel_0314_new.rds"))


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
################################################################################

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_resp <- list()
for (i in 2003:2014) {
  file_path <- file.path(base, as.character(i), paste0("atusresp_", i), paste0("atusresp_", i, ".dat"))
  new_atus_resp[[i]] <- read.delim(file_path, sep = ",")
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
  ext_atus_resp[[i]] <- read.delim(file_path, sep = ",")
}
ext_atus_resp <- bind_rows(ext_atus_resp) %>% 
  tibble %>% 
  clean_names()

################################################################################
# SUM
################################################################################

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_sum <- list()
for (i in 2003:2014) {
  file_path <- file.path(base, as.character(i), paste0("atussum_", i), paste0("atussum_", i, ".dta"))
  year_data <- read_dta(file_path)
  year_data <- mutate(year_data, tuyear = i)
  new_atus_sum[[i]] <- year_data
}
new_atus_sum <- bind_rows(new_atus_sum)

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
  mutate(across(non_issue_vars, ~ifelse(is.na(.), 0, .)))
# Remove non-issue vars from issue_vars
issue_vars <- setdiff(issue_vars, non_issue_vars)

non_issue_vars_min_2003 <- c()
for(k in issue_vars){
  test <- new_atus_sum %>% left_join(., ref_atus_sum, by = c("tucaseid", "tuyear")) %>% filter(tuyear > 2003) %>% select(tucaseid, paste0(k, ".x"), paste0(k, ".y"))  %>% 
                                       filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% select(-tucaseid) %>% distinct
  if(nrow(test) == 1){
    non_issue_vars_min_2003 <- c(k, non_issue_vars_min_2003)
    print(paste0(k, "added to nonissuevars"))
  }else{
      print(test)
    }
}

# In the case of each variable, there is only one or two  observation where NA does not equate to 0 so we set NA equal to zero in all cases
# Change all NA values for non_issue_vars_min_2003 to 0 if NA
new_atus_sum <- new_atus_sum %>% 
  mutate(across(non_issue_vars_min_2003, ~ifelse(is.na(.), 0, .)))

# Remove non-issue vars from issue_vars
issue_vars <- setdiff(issue_vars, non_issue_vars_min_2003)

non_issue_vars_min_2004 <- c()
for(k in issue_vars){
  test <- new_atus_sum %>% left_join(., ref_atus_sum, by = c("tucaseid", "tuyear")) %>% filter(tuyear > 2004) %>% select(tucaseid, paste0(k, ".x"), paste0(k, ".y"))  %>% 
    filter((get(paste0(k, ".x")) != get(paste0(k, ".y"))) %in% c(TRUE, NA)) %>% select(-tucaseid) %>% distinct
  if(nrow(test) == 1){
    non_issue_vars_min_2004 <- c(k, non_issue_vars_min_2004)
    print(paste0(k, "added to nonissuevars"))
  }else{
    print(test)
  }
}

# Remove non-issue vars from issue_vars
issue_vars <- setdiff(issue_vars, non_issue_vars_min_2004)

new_atus_sum <- new_atus_sum %>% 
  mutate(across(non_issue_vars_min_2004, ~ifelse(is.na(.) & tuyear > 2004, 0, .)))


names_common <- intersect(names(new_atus_sum), names(ref_atus_sum)) 
all.equal(zap_labels(select(new_atus_sum, all_of(names_common))), 
          zap_labels(select(ref_atus_sum, all_of(names_common))), 
                     check.attributes = FALSE)

ext_atus_sum <- list()
for (i in 2015:2023) {
  file_path <- file.path(base, "additional_years", as.character(i), paste0("atussum_", i), paste0("atussum_", i, ".dat"))
  ext_atus_sum[[i]] <- read.delim(file_path, sep = ",")
}
ext_atus_sum <- bind_rows(ext_atus_sum) %>% 
  tibble %>% 
  clean_names()

################################################################################
# ROST
################################################################################

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_rost <- list()
for (i in 2003:2014) {
  file_path <- file.path(base, as.character(i), paste0("atusrost_", i), paste0("atusrost_", i, ".dat"))
  new_atus_rost[[i]] <- read.delim(file_path, sep = ",")
}

new_atus_rost <- bind_rows(new_atus_rost) %>% 
  tibble %>% 
  clean_names() %>% 
  select(tucaseid, tulineno, terrp, teage, tesex)

# # Validate unique identifier
# if (anyDuplicated(new_atus_rost$tucaseid)) {
#   stop("tucaseid is not unique in respondent data.")
# }
# 
# all.equal(new_atus_rost, zap_labels(ref_atus_rost), check.attributes = FALSE)

ext_atus_rost <- list()
for (i in 2015:2023) {
  file_path <- file.path(base, "additional_years", as.character(i), paste0("atusrost_", i), paste0("atusrost_", i, ".dat"))
  ext_atus_rost[[i]] <- read.delim(file_path, sep = ",")
}
ext_atus_rost <- bind_rows(ext_atus_rost) %>% 
  tibble %>% 
  clean_names() %>% 
  select(tucaseid, tulineno, terrp, teage, tesex)

################################################################################
# CPS
################################################################################

# Function to read and combine single-year ATUS respondent files (2003-2014)
new_atus_cps <- list()
for (i in 2003:2014) {
    file_path <- file.path(base, as.character(i), paste0("atuscps_", i), paste0("atuscps_", i, ".dat"))
    new_atus_cps[[i]] <- read.delim(file_path, sep = ",")
}
new_atus_cps <- bind_rows(new_atus_cps) %>% 
  tibble %>% 
  clean_names()

# # Validate unique identifier
# if (anyDuplicated(new_atus_cps$tucaseid)) {
#   stop("tucaseid is not unique in respondent data.")
# }
# 
# test_equal(new_atus_cps, ref_atus_cps)
# names_common <- intersect(names(new_atus_cps), names(ref_atus_cps)) 
# setdiff(names(ref_atus_cps), names_common)
# all.equal(select(new_atus_cps, all_of(names_common)), zap_labels(select(ref_atus_cps, all_of(names_common))), check.attributes = FALSE)

ext_atus_cps <- list()
for (i in 2015:2023) {
  file_path <- file.path(base, "additional_years", as.character(i), paste0("atuscps_", i), paste0("atuscps_", i, ".dat"))
  ext_atus_cps[[i]] <- read.delim(file_path, sep = ",")
}
ext_atus_cps <- bind_rows(ext_atus_cps) %>% 
  tibble %>% 
  clean_names() %>% 
  rbind(new_atus_cps, .)


