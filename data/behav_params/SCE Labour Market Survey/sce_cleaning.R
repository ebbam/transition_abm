# Cleaning the Survey on Consumer Expectations Labour Market Survey Supplement###
# Source: https://www.newyorkfed.org/microeconomics/sce/labor#/
# Codebook/questionnaire: https://www.newyorkfed.org/medialibrary/media/research/microeconomics/interactive/downloads/sce-labor-questionnaire.pdf?sc_lang=en
rm(list = ls())
library(here)
library(tidyverse)
library(readxl)
library(lubridate)
library(stargazer)
library(lfe)  # for regressions with clustering
library(weights)
library(diagis)
library(fixest)      # For regressions with clustering and fixed effects
library(patchwork)
library(broom) # For extracting model coefficients
library(modelsummary)
library(flextable)

var_names <- read_xlsx(here('data/behav_params/SCE Labour Market Survey/sce_labour_questionnaire_codebook.xlsx'))
sce_lab <- read_xlsx(here("data/behav_params/SCE Labour Market Survey/sce-labor-microdata-public.xlsx"), 
                     sheet = 3, skip = 1, col_types = var_names$type)
#sce_raw <- readRDS(paste0("data/behav_params/Mueller_Replication/sce_datafile_13_24_w_lab_survey.RDS")) 

# 1. Create glossary of the variable names
# sce_lab %>% 
#   names %>% 
#   data.frame("var" = .) %>% 
#  #write.xlsx(here('data/behav_params/SCE Labour Market Survey/sce_labour_questionnaire_codebook.xlsx'))

pull_name <- function(nm){
  var_names %>% 
    filter(var == nm) %>% 
    pull(short_var) %>% return(.)
}

sce_lab <- sce_lab %>% 
  rename_with(., .fn = pull_name) %>%
  mutate(sce_lab_survey = 1,
         date = ceiling_date(ym(date), 'month') - days(1)) 

#sce_full <- readRDS(here("data/behav_params/Mueller_Replication/sce_datafile_13_24.RDS")) 

# # 3191 cases are matched in the original file - ie. 3191 unemployed people!
# sce_13_24 %>% 
#   select(userid, date) %>% 
#   arrange(date, userid) %>% 
#   semi_join(select(arrange(sce_lab, date, userid), userid, date), ., by = c("userid", "date")) %>% nrow(.) == nrow(sce_lab)
# 
# sce_full %>% 
#   select(userid, date) %>% 
#   arrange(date, userid) %>% 
#   anti_join(sce_lab, ., by = c("userid", "date")) 

# Read in raw SCE files which now include all obsesrvations
#source(here(paste0("data/behav_params/Mueller_Replication/mueller_repl_sce_raw_data_cleaning.R")))
#rm(sce_13_19, sce_13_19_same_t, sce_20_24)
#saveRDS(sce_13_24, here("data/behav_params/SCE Labour Market Survey/sce_13_24_raw.rds"))
sce_13_24 <- readRDS(here("data/behav_params/SCE Labour Market Survey/sce_13_24_raw.rds"))
sce_13_24_raw <- sce_13_24
sce_13_24 <- sce_13_24_raw %>% 
   select(-names(sce_lab)[!(names(sce_lab) %in% c('userid', 'date'))])

sce_13_24 %>%
  select(userid, date) %>%
  arrange(date, userid) %>%
  semi_join(select(arrange(sce_lab, date, userid), userid, date), ., by = c("userid", "date")) %>% nrow(.) == nrow(sce_lab)


################################################################################
transform_fun <- function(df){
  df <- df %>% mutate(never_worked = ifelse(is.na(never_worked), 0 , 1),
                      agesq = age^2,
         self_employed = case_when(self_employed == 1 ~ 0,
                                   self_employed == 2 ~ 1, 
                                   TRUE ~ NA),
         looked_for_work_l4_wks = case_when(looked_for_work_l4_wks == 1 ~ 1,
                                            looked_for_work_l4_wks == 2 ~ 0, 
                                            TRUE ~ NA),
         across(c(reservation_wage_orig, wage_most_recent_job, current_wage_annual), ~as.integer(.)),
         reservation_wage_unit_scale = case_when(reservation_wage_unit == 1 & nchar(reservation_wage_orig) %in% c(2,3) ~ 2080, # Hourly - calculated 40 hours x 52 weeks
                                                 reservation_wage_unit == 2 & nchar(reservation_wage_orig) %in% c(3,4) ~ 52, # Weekly- 52 weeks
                                                 reservation_wage_unit == 3 & nchar(reservation_wage_orig) %in% c(4) ~ 26, # Bi-weekly ~ 52/2
                                                 reservation_wage_unit == 4 & nchar(reservation_wage_orig) %in% c(4,5)~ 12, # Monthly - 12
                                                 reservation_wage_unit == 5 & nchar(reservation_wage_orig) >= 5 ~ 1, # Annual - 1
                                                 TRUE ~ reservation_wage_unit),
         # Can somewhat safely assume that single-digit or two-digit reservation wage is a reported hourly wage
         reservation_wage_unit_scale = ifelse(nchar(reservation_wage_orig) %in% c(1,2), 2080, reservation_wage_unit_scale),
         reservation_wage = reservation_wage_orig * reservation_wage_unit_scale, 
         # Reservation wage is sometimes filled in as 0 - will replace with NA values
         reservation_wage = ifelse(reservation_wage == 0, NA, reservation_wage),
         current_wage_annual_cat = case_when(current_wage_annual < 10000 ~ 1, # tested: sce_lab %>% filter(!is.na(current_wage_annual) & !is.na(current_wage_annual_cat)) %>% nrow(.) == 0
                                             current_wage_annual  >= 10000 & current_wage_annual <= 19999 ~ 2,
                                             current_wage_annual  >= 20000 & current_wage_annual <= 29999 ~ 3,
                                             current_wage_annual  >= 30000 & current_wage_annual <= 39999 ~ 4,
                                             current_wage_annual  >= 40000 & current_wage_annual <= 49999 ~ 5,
                                             current_wage_annual  >= 50000 & current_wage_annual <= 59999 ~ 6,
                                             current_wage_annual  >= 60000 & current_wage_annual <= 74999 ~ 7,
                                             current_wage_annual  >= 75000 & current_wage_annual <= 99999 ~ 8,
                                             current_wage_annual  >= 100000 & current_wage_annual <= 149999 ~ 9,
                                             current_wage_annual  >= 150000 ~ 10,
                                             TRUE ~ current_wage_annual_cat),
         current_wage_annual = case_when(is.na(current_wage_annual) & current_wage_annual_cat == 1 ~ 10000, # If I only have the category (happens in 144 observations), I replace with the mean if a range is provided or the upper/lower range in the bounding categories, respectively.
                                         is.na(current_wage_annual) & current_wage_annual_cat == 2 ~ 15000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 3 ~ 25000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 4 ~ 35000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 5 ~ 45000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 6 ~ 55000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 7 ~ 67500,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 8 ~ 87500,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 9 ~ 125000,
                                         is.na(current_wage_annual) & current_wage_annual_cat == 10 ~ 150000,
                                         TRUE ~ current_wage_annual), 
         current_wage_annual = ifelse(current_wage_annual == 0, reservation_wage, current_wage_annual),
         res_wage_to_current = reservation_wage/current_wage_annual, 
         res_wage_to_latest = reservation_wage/wage_most_recent_job) %>% 
    filter(wage_most_recent_job < 1000000 | is.na(wage_most_recent_job)) %>% 
    filter(age >= 20 & age <= 65 & 
             (res_wage_to_current < 10 | is.na(res_wage_to_current)),
           (res_wage_to_latest < 10 | is.na(res_wage_to_latest))) %>% 
    mutate(accepted_salary_1 = job_offer_1_salary*(job_offer_1_accepted %in% c(1,2)),
           accepted_salary_2 = job_offer_2_salary*(job_offer_2_accepted %in% c(1,2)),
           accepted_salary_3 = job_offer_3_salary*(job_offer_3_accepted %in% c(1,2)),
           accepted_salary = max(accepted_salary_1, accepted_salary_2, accepted_salary_3, na.rm = TRUE),
           accepted_salary = ifelse(accepted_salary == -Inf, NA, accepted_salary),
           reservation_wage_latest = max(reservation_wage, na.rm = TRUE),
           reservation_wage_latest = ifelse(reservation_wage_latest == -Inf, NA, reservation_wage_latest),
           salary_prop_reswage = accepted_salary/reservation_wage_latest)
  return(df)
}

# Define base directory (replace this with the correct path later)
base <- here("data/behav_params/Mueller_Replication/")

# Load data file
data_13_24 <- #readRDS(paste0(base, "sce_datafile_13_24_w_lab_survey.RDS")) %>% 
  sce_lab %>% 
  left_join(., sce_13_24, by = c("userid", "date")) %>% 
  rename(reservation_wage_orig = reservation_wage,
         weight = rim_4_original) %>% 
  transform_fun(.)
  # filter((looked_for_work_l4_wks == 1 | looked_for_new_work_l4_wks %in% c(1,2)) & is.na(wks_looking_for_work)) %>% select(contains('look'))
  # #filter(is.na(reservation_wage_orig) | is.na(reservation_wage_unit) | is.na(reservation_wage)) %>% 
  # #filter(is.na(current_wage_annual) & is.na(wage_most_recent_job) & is.na(wage_most_recent_job_cat) & never_worked == 0) %>% 
  # select(userid, date, contains("current_emp"), never_worked) %>% filter(never_worked == 0) %>% 
  # mutate(NA_Count = rowSums(is.na(across(-c(userid, date))))) %>% arrange(-NA_Count) %>% filter(NA_Count != 16)
# 10,629 people are interviewed more than once!
panel_ids <- sce_lab %>% select(userid) %>% group_by(userid) %>% mutate(n = n()) %>% filter(n != 1) %>% pull(userid) %>% unique
# 1,119 observations are in the unemployed sample in sce
sce_13_24 %>% group_by(userid) %>% filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
  pull(userid) %>% unique %>% intersect(panel_ids, .) %>% length
# 9,520 observations are not in the unemployed sample in sce
sce_13_24 %>% group_by(userid) %>% filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
  pull(userid) %>% unique %>% setdiff(panel_ids, .) %>% length

unemp_only <- readRDS(here('data/behav_params/Mueller_Replication/sce_datafile_13_24_w_lab_survey_new.RDS')) %>% 
  rename(reservation_wage_orig = reservation_wage) %>% 
  transform_fun(.)
  
#### DEMONSTRATES THE RESERVATION WAGE TRAJECTORY ACROSS INDIVIDUALS ###
unemp_only %>% 
  filter(!is.na(reservation_wage)) %>% 
  group_by(userid) %>%
  arrange(userid, date) %>% 
  mutate(time_period = row_number()) %>% 
  relocate(time_period) %>% 
  select(time_period, userid, res_wage_to_latest, res_wage_to_current) %>% 
  pivot_longer(!c(time_period, userid)) %>% 
  ggplot() + 
  geom_point(aes(x = time_period, y = value, group = userid, color = name)) +
  geom_line(aes(x = time_period, y = value, group = userid, color = name)) +
  facet_wrap(~name, ncol = 1)

unemp_only %>% 
  filter(!is.na(reservation_wage)) %>% 
  group_by(userid) %>%
  arrange(userid, date) %>% 
  mutate(time_period = row_number()) %>% 
  relocate(time_period) %>% 
  select(time_period, userid, res_wage_to_latest, res_wage_to_current) %>% 
  lm(data = ., res_wage_to_latest ~ time_period)

################################################################################

controls <- c("female", "hispanic", "black", "r_asoth", "other_race", "age", "agesq", 
              "hhinc_2", "hhinc_3", "hhinc_4", "education_2", "education_3", "education_4", 
              "education_5", "education_6")

###################################################################
# 2. Histogram of elicited wage expectation conditional on pasts wage and job expectation (OO2new-OO2e2dk ~ L10-11)
###################################################################
# There are some issues with the histogram which I think has to do with the binning...left as is below for now

# Create fweight variable (rounded weights)
data_fig1 <- data_13_24 %>%
  mutate(fweight = round(weight, 1))

data_fig1 %>% 
  ggplot() +
  geom_histogram(
    aes(x = log(reservation_wage), weight = fweight),
  ) +
  labs(
    title = paste0("Figure 1. Histogram of Elicited Reservation Wage")
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

data_fig1 %>% 
  ggplot() +
  geom_histogram(
    aes(x = res_wage_to_current, weight = fweight, fill = "Reservation Wage to Current"),
  ) +
  geom_histogram(
    aes(x = res_wage_to_latest, weight = fweight, fill = "Reservation Wage to Latest")
  ) +
  labs(
    title = "Figure 1b. Reservation Wage as proportion of Current or Latest Held Wage"
  ) +
  theme_minimal() +
  theme(
   legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) 


data_fig1 %>% 
  ggplot() + 
  geom_point(aes(x = log(current_wage_annual), y = log(reservation_wage), color = "Reservation wage vs. Current")) +
  geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
  geom_abline(slope = 1, linetype = "dashed") +
  theme(legend.position = "bottom") +
  labs(title = "All workers")


unemp_only %>% 
  ggplot() + 
  geom_point(aes(x = log(current_wage_annual), y = log(reservation_wage), color = "Reservation wage vs. Current")) +
  geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
  #geom_abline(slope = 1, linetype = "dashed") +
  theme(legend.position = "bottom") +
  labs(title = "Unemployed only")

unemp_only %>% 
  ggplot() + 
  geom_point(aes(x = udur, y = reservation_wage)))+ #color = "Reservation wage vs. Current")) +
  #geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
  geom_abline(slope = 1, linetype = "dashed") +
  theme(legend.position = "bottom") +
  labs(title = "Unemployed only")


# 3. Averages of realised wage by bins of elicited wage expectation (NL1-NL4 ~ OO2new-OO2e2dk)

t1 <- unemp_only %>% 
  mutate(test1 = exp_salary_best_offer_4mos/reservation_wage, 
         test2 = exp_salary_best_offer_4mos/current_wage_annual, 
         test3 = exp_salary_best_offer_4mos/wage_most_recent_job,
         udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(test1)) %>% 
  ggplot(aes(x = as.factor(udur_bins), y = test1)) +
  geom_boxplot()+
  labs(title = "Ratio of expected best offer to reservation wage") +
  ylim(0,2)

t2 <- unemp_only %>% 
  mutate(test1 = exp_salary_best_offer_4mos/reservation_wage, 
         test2 = exp_salary_best_offer_4mos/current_wage_annual, 
         test3 = exp_salary_best_offer_4mos/wage_most_recent_job,
         udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(test2)) %>% 
  ggplot(aes(x = as.factor(udur_bins), y = test2)) +
  geom_boxplot()+
  labs(title = "Ratio of expected best offer to current wage") +
  ylim(0,2)

t3 <- unemp_only %>% 
  mutate(test1 = exp_salary_best_offer_4mos/reservation_wage, 
         test2 = exp_salary_best_offer_4mos/current_wage_annual, 
         test3 = exp_salary_best_offer_4mos/wage_most_recent_job,
         udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(test3)) %>% 
  ggplot(aes(x = as.factor(udur_bins), y = test3)) +
  geom_boxplot()+
  labs(title = "Ratio of expected best offer to most recent wage") +
  ylim(0,2)

t1 + t2 + t3

#contains("exp_salary"), exp_avg_salary_offer_4mos, exp_salary_best_offer_4mos)

unemp_only %>%
  mutate(accepted_salary_1 = job_offer_1_salary*(job_offer_1_accepted %in% c(1,2)),
            accepted_salary_2 = job_offer_2_salary*(job_offer_2_accepted %in% c(1,2)),
            accepted_salary_3 = job_offer_3_salary*(job_offer_3_accepted %in% c(1,2))) %>%
  group_by(userid) %>%
  summarise(accepted_salary = max(accepted_salary_1, accepted_salary_2, accepted_salary_3, na.rm = TRUE),
            accepted_salary = ifelse(accepted_salary == -Inf, NA, accepted_salary),
            reservation_wage = max(reservation_wage, na.rm = TRUE),
            reservation_wage = ifelse(reservation_wage == -Inf, NA, reservation_wage),
            salary_prop_reswage = accepted_salary/reservation_wage) %>%
  fill(reservation_wage, .direction = "down") %>% 
  ungroup %>% 
  ggplot() +
 # geom_jitter(aes(x = reservation_wage, y = accepted_salary)) +
  geom_histogram(aes(x = salary_prop_reswage))

# 6. changes in wage gap across and within spells

lm(data = unemp_only, res_wage_to_latest ~ udur) %>% summary()


####### REGRESSIONS ##########
#################################################################################################
### Realised wage as function of elicited wage expectation ###
# 4. regs of realised wage on expected wage (NL1-4 ~ OO2new-OO2e2dk)
#################################################################################################
NL1-NL4 ~ oo2new-oo2newdk

#################################################################################################
### Reservation wage as function of unemp duration (raw and as proportion of latest held or current wage) ###
#################################################################################################

reservation_wage ~ unempduration
reservation_wage/lastheldwage ~ unempduration
reservation_wage/currentwage ~ unempduration


#################################################################################################
### Accepted wage as function of  ###
#################################################################################################

accepted_wage ~ unempduration
accepted_wage/reservation_wage ~ unempduration
accepted_wage/lastheldwage ~ unempduration
accepted_wage/currentwage ~ unempduration


#################################################################################################
### Elicited versus realised wage by unemployment duration  ###
# 5. perceived versus realised wage by unemployment duration (JS7 - time spent searching)
#################################################################################################

elicited_wage/accepted_wage ~ unemp_duration





























# ######################################################################################
# ### Table 2: Regressions of Realized on Elicited 3-Month Job-Finding Probabilities (SCE) ###
# ######################################################################################
# 
# prod_tab2_data <- function(data){
#   # Keep only observations where age is between 20 and 65
#   data_tab2 <- data %>% 
#     filter(age >= 20 & age <= 65) %>% ungroup
#   
#   # Define cohort and time effects
#   data_tab2$year <- as.numeric(format(as.Date(as.character(data_tab2$date), "%Y%m%d"), "%Y"))
#   data_tab2$month <- data_tab2$date - data_tab2$year * 100
#   data_tab2 <- data_tab2 %>% group_by(userid) %>% mutate(cohorty = first(year))
#   
#   # Unemployed observation count
#   data_tab2 <- data_tab2 %>% group_by(userid, lfs) %>% mutate(numu = if_else(lfs == 2, row_number(), NA_integer_))
#   
#   # LFS at the beginning of survey
#   data_tab2 <- data_tab2 %>% group_by(userid) %>% mutate(firstlfs = first(lfs))
#   
#   # Defining labels (can be used for later reference)
#   # In R, we typically don't label variables in the same way as in Stata, but you can use attributes or documentation.
#   
#   # Define controls
#   data_tab2$agesq <- data_tab2$age^2
#   
#   # Sample 3-consecutive surveys only
#   data_tab2 <- data_tab2 %>% filter(in_sample_2 == 1)
#   
#   return(data_tab2)
# }
# 
# 
# # Table 2 - Panel A
# 
# # First regression: UE_trans_3mon on find_job_3mon
# # Clustering is still not correct - for all regressions below SE are off. Treatment effects are correct
# modelA_1 <- function(data_tab2){lfe::felm(UE_trans_3mon ~ find_job_3mon + (1 | userid), data = data_tab2, weights = data_tab2$weight) %>% return(.)}
# 
# # Second regression: UE_trans_3mon on controls
# modelA_2 <- function(data_tab2){lfe::felm(as.formula(paste0("UE_trans_3mon ~ ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2, weights = data_tab2$weight) %>% return(.)}
# 
# # Third regression: UE_trans_3mon on find_job_3mon + controls
# modelA_3 <- function(data_tab2){lfe::felm(as.formula(paste0("UE_trans_3mon ~ find_job_3mon + ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2, weights = data_tab2$weight)%>% return(.)}
# 
# # Fourth regression: UE_trans_3mon on find_job_3mon, findjob_3mon_longterm, controls, and longterm_unemployed
# modelA_4 <- function(data_tab2){lfe::felm(as.formula(paste0("UE_trans_3mon ~ find_job_3mon + findjob_3mon_longterm + ", paste0(controls, collapse = " + "), " + longterm_unemployed + (1 | userid)")), data = data_tab2, weights = data_tab2$weight)%>% return(.)}
# # Table 2 - Panel B
# 
# # Update variable labels for Panel B
# # data_tab2$find_job_3mon <- lag(data_tab2$find_job_3mon, 3)  # Lagged find_job_3mon by 3 months
# 
# # First regression: tplus3_UE_trans_3mon on tplus3_percep_3mon
# modelB_1 <- function(data_tab2){lfe::felm(tplus3_UE_trans_3mon ~ tplus3_percep_3mon + (1 | userid), data = data_tab2, weights = data_tab2$weight)%>% return(.)}
# 
# # # The next regressions use same sample as in the above regression which rules out any NA values for tplus3_UE_trans_3mon and tplus3_percep_3mon so we modify the dataset in each of the functions for Panel B
# 
# # Second regression: tplus3_UE_trans_3mon on controls
# modelB_2 <- function(data_tab2){  # The next regressions use same sample as in the above regression which rules out any NA values for tplus3_UE_trans_3mon and tplus3_percep_3mon
#   data_tab2_short <- data_tab2 %>% 
#     filter(!is.na(tplus3_UE_trans_3mon) & !is.na(tplus3_percep_3mon))
#   lfe::felm(as.formula(paste0("tplus3_UE_trans_3mon ~ tplus3_percep_3mon + ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2_short, weights = data_tab2_short$weight)%>% return(.)}
# 
# # Third regression: tplus3_UE_trans_3mon on find_job_3mon
# modelB_3 <- function(data_tab2){  # The next regressions use same sample as in the above regression which rules out any NA values for tplus3_UE_trans_3mon and tplus3_percep_3mon
#   data_tab2_short <- data_tab2 %>% 
#     filter(!is.na(tplus3_UE_trans_3mon) & !is.na(tplus3_percep_3mon))
#   lfe::felm(tplus3_UE_trans_3mon ~ find_job_3mon + (1 | userid), data = data_tab2_short, weights = data_tab2_short$weight)%>% return(.)}
# 
# # Fourth regression: tplus3_UE_trans_3mon on find_job_3mon + controls
# modelB_4 <- function(data_tab2){  # The next regressions use same sample as in the above regression which rules out any NA values for tplus3_UE_trans_3mon and tplus3_percep_3mon
#   data_tab2_short <- data_tab2 %>% 
#     filter(!is.na(tplus3_UE_trans_3mon) & !is.na(tplus3_percep_3mon))
#   lfe::felm(as.formula(paste0("tplus3_UE_trans_3mon ~ find_job_3mon + ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2_short, weights = data_tab2_short$weight)%>% return(.)}
# 
# # Clear models from memory
# mods <- list(#"modelA_1" = modelA_1, 
#   #"modelA_2" = modelA_2, 
#   "modelA_3" = modelA_3, 
#   "modelA_4" = modelA_4, 
#   #"modelB_1" = modelB_1, 
#   "modelB_2" = modelB_2, 
#   #"modelB_3" = modelB_3,
#   "modelB_4" = modelB_4)
# 
# 
# tab2_data <-lapply(df_list, prod_tab2_data)
# 
# print("Table 2—Regressions of Realized on Elicited 3-Month Job-Finding Probabilities (SCE)")
# for(panel in c("A", "B")){
#   if(panel == "A"){
#     print("Panel A. Contemporaneous elicitations")
#   }else{print("Panel B. Lagged elicitations")}
#   
#   
#   for(model in mods[grep(panel, names(mods))]){
#     res_list <- list()
#     i = 1
#     for(df in tab2_data){
#       res_list[i] <- list(model(df))
#       i = i+1
#     }
#     stargazer(res_list, type = "text",
#               star.cutoffs = c(0.1, 0.05, 0.01),
#               digits = 3,
#               column.labels = names(tab2_data), 
#               dep.var.labels = "T+3 UE Transitions (3-Months)",
#               #title = paste0()
#               #covariate.labels = c("Current Job-Finding Probability", "Lagged Job-Finding Probability"), # Switched order of these labels...still not quite sure about the lagging here
#               omit = controls) 
#   }
# }
# 
# 
# 
# ############################################################################################
# # Figure 3: Perceived vs. Realized Job Finding, by Duration of Unemployment (SCE)         #
# ############################################################################################
# # Combine dataframes in df_list with a source column
# combined_data_fig3 <- bind_rows(
#   lapply(seq_along(df_list), function(i) {
#     name <- names(df_list)[i]
#     df_list[[name]] %>%
#       filter(age >= 20 & age <= 65 & in_sample_2 == 1) %>%
#       select(udur_bins, UE_trans_3mon, find_job_3mon, weight) %>%
#       filter(!is.na(udur_bins) & !is.na(weight)) %>%
#       group_by(udur_bins) %>%
#       summarise(
#         rjob_find = weighted.mean(UE_trans_3mon, na.rm = TRUE, w = weight),
#         pjob_find = weighted.mean(find_job_3mon, na.rm = TRUE, w = weight),
#         se_rjob_find = weighted_se(UE_trans_3mon, na.rm = TRUE, w = weight),
#         se_pjob_find = weighted_se(find_job_3mon, na.rm = TRUE, w = weight),
#         nobs = n()
#       ) %>%
#       ungroup() %>%
#       rename(undur = udur_bins) %>%
#       mutate(
#         pjob_find_lower = pjob_find - qt(0.975, df = nobs - 1) * se_pjob_find,
#         pjob_find_upper = pjob_find + qt(0.975, df = nobs - 1) * se_pjob_find,
#         rjob_find_lower = rjob_find - qt(0.975, df = nobs - 1) * se_rjob_find,
#         rjob_find_upper = rjob_find + qt(0.975, df = nobs - 1) * se_rjob_find,
#         undur1 = undur + 0.1, # Offset for graphical purposes
#         undur_label = factor(
#           undur,
#           levels = 1:4,
#           labels = c("0-3 Months", "4-6 Months", "7-12 Months", "13 Months +")
#         ),
#         Source = name
#       )
#   })
# )
# 
# # Ensure Source follows the same order as df_list
# combined_data_fig3$Source <- factor(combined_data_fig3$Source, levels = names(df_list))
# 
# # Plot the data
# p <- ggplot(combined_data_fig3, aes(x = undur)) +
#   # Elicited probabilities with error bars
#   geom_line(aes(y = pjob_find, color = Source), size = 1) +
#   geom_point(aes(y = pjob_find, color = Source), size = 3, shape = 4) +
#   geom_errorbar(
#     aes(ymin = pjob_find_lower, ymax = pjob_find_upper, color = Source),
#     width = 0.1, size = 0.8
#   ) +
#   # Realized probabilities with error bars
#   geom_line(aes(x = undur1, y = rjob_find, color = Source), size = 1, linetype = "dashed") +
#   geom_point(aes(x = undur1, y = rjob_find, color = Source), size = 3, shape = 1) +
#   geom_errorbar(
#     aes(x = undur1, ymin = rjob_find_lower, ymax = rjob_find_upper, color = Source),
#     width = 0.1, size = 0.8, linetype = "dashed"
#   ) +
#   # Titles and axis labels
#   labs(
#     y = "3-Month Job-Finding Probability/Rate",
#     x = "Duration of Unemployment",
#     title = "Fig 3. Perceived vs. Realized Job Finding, by Duration of Unemployment",
#     subtitle = "Duration dependence is strongly negative across all samples. \nBias in beliefs of LTUE is also consistently high across samples."
#   ) +
#   scale_x_continuous(
#     breaks = 1:4,
#     labels = levels(combined_data_fig3$undur_label)
#   ) +
#   scale_color_manual(
#     values = c("Orig." = "black", setNames(hue_pal()(length(df_list) - 1), names(df_list)[-1]))
#   ) +
#   theme_minimal() +
#   theme(
#     legend.position = "none",
#     legend.title = element_blank()
#   ) +
#   facet_wrap(~Source, nrow = 2)
# 
# print(p)
# 
# ######################################################################################
# # Figure 3. Plotting by recession period
# ######################################################################################
# recs <- read.csv(here("data/macro_vars/collated_recessions.csv")) %>% 
#   tibble %>% 
#   mutate(date = ymd(DATE) - days(1))
# 
# combined_fig3_by_recession <- df_list$`2013-24` %>% 
#   left_join(., recs, by = "date") %>% 
#   filter(age >= 20 & age <= 65 & in_sample_2 == 1) %>%
#   select(udur_bins, USREC, UE_trans_3mon, find_job_3mon, weight) %>%
#   filter(!is.na(udur_bins) & !is.na(weight)) %>%
#   group_by(udur_bins, USREC) %>%
#   summarise(
#     rjob_find = weighted.mean(UE_trans_3mon, na.rm = TRUE, w = weight),
#     pjob_find = weighted.mean(find_job_3mon, na.rm = TRUE, w = weight),
#     se_rjob_find = weighted_se(UE_trans_3mon, na.rm = TRUE, w = weight),
#     se_pjob_find = weighted_se(find_job_3mon, na.rm = TRUE, w = weight),
#     nobs = n()
#   ) %>%
#   ungroup() %>%
#   rename(undur = udur_bins) %>%
#   mutate(
#     pjob_find_lower = pjob_find - qt(0.975, df = nobs - 1) * se_pjob_find,
#     pjob_find_upper = pjob_find + qt(0.975, df = nobs - 1) * se_pjob_find,
#     rjob_find_lower = rjob_find - qt(0.975, df = nobs - 1) * se_rjob_find,
#     rjob_find_upper = rjob_find + qt(0.975, df = nobs - 1) * se_rjob_find,
#     undur1 = undur + 0.1, # Offset for graphical purposes
#     undur_label = factor(
#       undur,
#       levels = 1:4,
#       labels = c("0-3 Months", "4-6 Months", "7-12 Months", "13 Months +")
#     ),
#     recession = case_when(USREC == 1 ~ "Recession",
#                           USREC == 0 ~"Non-recession",
#                           is.na(USREC) ~ "N/A"))
# 
# 
# # Ensure Source follows the same order as df_list
# combined_fig3_by_recession$recession <- factor(combined_fig3_by_recession$recession, levels = c("Recession", "Non-recession"))
# 
# # Plot the data
# p <- ggplot(combined_fig3_by_recession, aes(x = undur)) +
#   # Elicited probabilities with error bars
#   geom_line(aes(y = pjob_find, color = recession), size = 1) +
#   geom_point(aes(y = pjob_find, color = recession), size = 3, shape = 4) +
#   geom_errorbar(
#     aes(ymin = pjob_find_lower, ymax = pjob_find_upper, color = recession),
#     width = 0.1, size = 0.8
#   ) +
#   # Realized probabilities with error bars
#   geom_line(aes(x = undur1, y = rjob_find, color = recession), size = 1, linetype = "dashed") +
#   geom_point(aes(x = undur1, y = rjob_find, color = recession), size = 3, shape = 1) +
#   geom_errorbar(
#     aes(x = undur1, ymin = rjob_find_lower, ymax = rjob_find_upper, color = recession),
#     width = 0.1, size = 0.8, linetype = "dashed"
#   ) +
#   # Titles and axis labels
#   labs(
#     y = "3-Month Job-Finding Probability/Rate",
#     x = "Duration of Unemployment",
#     title = "Fig 3. Perceived vs. Realized Job Finding, by Duration of Unemployment",
#     subtitle = "Duration dependence is strongly negative across all samples. \nBias in beliefs of LTUE is also consistently high across samples."
#   ) +
#   scale_x_continuous(
#     breaks = 1:4,
#     labels = levels(combined_fig3_by_recession$undur_label)
#   ) +
#   #scale_color_manual(
#   #  values = c("Orig." = "black", setNames(hue_pal()(length(df_list) - 1), names(df_list)[-1]))
#   #) +
#   #ylim(0, 1)+
#   theme_minimal() +
#   theme(
#     legend.position = "none",
#     legend.title = element_blank()
#   ) +
#   facet_wrap(~recession, nrow = 1, scales = "free")
# 
# print(p)
# 
# 
# ##################################################################################################
# # Table 4 - Panel A: Linear Regressions of Elicited Job-Finding Probabilities on Duration of Unemployment (SCE)
# ##################################################################################################
# tab_4_fun <- function(data){  
#   # Load the dataset
#   data_tab4 <- data %>%
#     filter(age >= 20 & age <= 65)
#   
#   # # Time fixed effects
#   # data_tab4 <- data_tab4 %>%
#   #   mutate(across(starts_with("date"), ~ as.factor(.), .names = "dd_{col}")) %>%
#   #   select(-dd_1)
#   
#   # Generate indicators for labor force status
#   data_tab4 <- data_tab4 %>%
#     mutate(
#       olf2 = as.numeric(lfs == 3),   # Out of labor force indicator
#       emp3 = as.numeric(lfs == 1),  # Employed indicator
#       i3m = ifelse(lfs != 1, as.numeric(!is.na(find_job_3mon)), NA)  # Indicator for perception question
#     )
#   
#   # Indicator for labor force status in the next interview
#   data_tab4 <- data_tab4 %>%
#     arrange(userid, date) %>%
#     group_by(userid) %>%
#     mutate(next1lfs = lead(lfs))
#   
#   # Spell number and spell length
#   data_tab4 <- data_tab4 %>%
#     arrange(spell_id, date) %>%
#     group_by(spell_id) %>%
#     mutate(
#       spelln = row_number(),
#       spellN = n()
#     )
#   
#   # Number of observations with perception question in a spell
#   data_tab4 <- data_tab4 %>%
#     mutate(
#       n_f3m_spell = ifelse(lfs != 1, cumsum(i3m), NA),
#       N_f3m_spell = ifelse(lfs != 1, max(n_f3m_spell, na.rm = TRUE), NA)
#     )
#   
#   # Number of observations out of labor force in a spell
#   data_tab4 <- data_tab4 %>%
#     mutate(
#       n_olf_spell = ifelse(lfs != 1, cumsum(olf2), NA),
#       N_olf_spell = ifelse(lfs != 1, max(n_olf_spell, na.rm = TRUE), NA)
#     )
#   
#   # Label unemployment duration variable
#   data_tab4 <- data_tab4 %>%
#     #rename(udur = unemployment_duration) %>%  # Adjust the variable name to match your dataset
#     mutate(agesq = age^2)                     # Generate agesq
#   
#   # Keep if in the main sample
#   data_tab4 <- data_tab4 %>%
#     filter(in_sample_1 == 1)
#   
#   # Indicator for first survey
#   data_tab4 <- data_tab4 %>%
#     arrange(userid, date) %>%
#     group_by(userid) %>%
#     mutate(first_unemp_survey = row_number() == 1)
#   
#   # Run regressions
#   # Table 4.1: Simple regression with only udur and first_unemp_survey
#   model1 <- feols(
#     find_job_3mon ~ udur | 0,
#     data = filter(data_tab4, first_unemp_survey == 1),
#     weights = ~ weight,
#     cluster = ~ userid
#   )
#   
#   # Table 4.2: Regression with udur and weights
#   model2 <- feols(
#     find_job_3mon ~ udur | 0,
#     data = data_tab4,
#     weights = ~ weight,
#     cluster = ~ userid
#   )
#   
#   # Table 4.3: Regression with controls
#   model3 <- feols(
#     as.formula(paste("find_job_3mon ~ udur +", paste(controls, collapse = " + "))),
#     data = data_tab4,
#     weights = ~ weight,
#     cluster = ~ userid
#   )
#   
#   # Table 4.4: Regression with spell fixed effects
#   model4 <- feols(
#     find_job_3mon ~ udur | spell_id,
#     data = data_tab4,
#     weights = ~ weight,
#     cluster = ~ spell_id
#   )
#   
#   return(list("(1)" = summary(model1), "(2)" = summary(model2), "(3)" = summary(model3), "(4)" = summary(model4)))
# }
# 
# # Create Table 4 - Panel A
# tab4_all <- lapply(df_list, tab_4_fun) 
# names(tab4_all) <- names(df_list)
# 
# print("Table 4—Linear Regressions of Elicited Job-Finding Probabilities on Duration of Unemployment")
# tab4_all %>% 
#   modelsummary(.,
#                shape = "rbind",
#                #list(summary(model1), summary(model2), summary(model3), summary(model4)),
#                output = "markdown",
#                title = "Table 4 - Panel A: Linear Regressions of Elicited Job-Finding Probabilities on Duration of Unemployment (SCE)",
#                #dep.var.labels.include = FALSE,
#                #column.labels = c("(1)", "(2)", "(3)", "(4)"),
#                star.cutoffs = c(0.1, 0.05, 0.01),
#                notes.append = TRUE,
#                notes = "Standard errors are clustered at the user or spell level as indicated.",
#                gof_map = c("nobs", "r.squared"),
#                coef_map = c("udur"= "Unemployment Duration (Months)"),
#                fmt = 4
#   ) %>% print(.)
# #%>% modelsummary(., shape = "rbind")
# 
# 
# ##################################################################################################
# #### Figure 4 - Panel A: Elicited Job-Finding Probabilities, by Time since First Interview  (SCE) ###
# ##################################################################################################
# prod_fig4 <- function(data){
#   # Load data
#   data_fig4 <- data %>%
#     filter(age >= 20 & age <= 65, in_sample_1 == 1) %>%
#     mutate(
#       agesq = age^2
#     )
#   
#   # Regressions on monthly dummies for duration
#   # Create monthly dummies - weighting happens in individual function calls (LM() and FEOLS() below)
#   duration_vars <- grep("^nedur_1mo_", names(data_fig4), value = TRUE)
#   
#   # Regression 1: Simple OLS with weights and clustering
#   ols_model <- lm(as.formula(paste0("find_job_3mon ~ ", paste0(duration_vars, collapse = " + "))),
#                   data = data_fig4, weights = weight)
#   
#   # Extract coefficients and standard errors
#   ols_summary <- summary(ols_model)
#   ols_coeffs <- ols_summary$coefficients
#   ols_0_val <- ols_coeffs[rownames(ols_coeffs) == "nedur_1mo_0"][1]
#   
#   # Regression 2: Fixed-effects regression
#   fe_model <- feols(as.formula(paste0("find_job_3mon ~ ", paste0(duration_vars, collapse = " + "), " | spell_id")),
#                     data = data_fig4,
#                     weights = ~ weight,
#                     cluster = ~ spell_id)
#   
#   fe_summary <- summary(fe_model)
#   fe_coeffs <- tidy(fe_summary)
#   fe_0_val <- fe_coeffs %>% filter(term == "nedur_1mo_0") %>% pull(estimate) %>% as.numeric
#   
#   # Combine coefficients and standard errors for plotting
#   coeffs_data <- bind_rows(
#     ols_coeffs %>% as_tibble(rownames = "term") %>%
#       filter(term %in% duration_vars) %>%
#       rename(b = Estimate, se = `Std. Error`, statistic = `t value`, p.value =`Pr(>|t|)`) %>%
#       mutate(model = "OLS"),
#     
#     fe_coeffs %>%
#       filter(term %in% duration_vars) %>%
#       rename(b = estimate, se = std.error) %>%
#       mutate(model = "Fixed Effects")
#   )
#   
#   # Add confidence intervals
#   coeffs_data <- coeffs_data %>%
#     mutate(
#       high = b + 1.96 * se,
#       low = b - 1.96 * se
#     )
#   
#   # Generate scatter plots for each model
#   plot1 <- coeffs_data %>%
#     filter(model == "OLS" &
#              term %in% duration_vars[1:7]) %>%
#     ggplot() +
#     #geom_point(aes(x = term, y = b), color = "red") +
#     geom_point(aes(x = term, y = b-ols_0_val), color = "black") +
#     #geom_errorbar(aes(x = term, ymin = low, ymax = high), color = "red") +
#     geom_errorbar(aes(x = term, ymin = low-ols_0_val, ymax = high-ols_0_val), color = "black") +
#     geom_hline(yintercept = 0, color = "black", size = 1) +
#     xlab("Time Since First Interview, in Months") +
#     ylab("Elicited 3-Month Job-Finding Probability") +
#     ggtitle("Within and Across Spell Changes") +
#     ylim(-0.25, 0.15) +
#     theme_minimal()
#   
#   plot2 <- coeffs_data %>%
#     filter(model == "Fixed Effects" &
#              term %in% duration_vars[1:7]) %>%
#     ggplot() +
#     #geom_point(aes(x = term, y = b), color = "red") +
#     geom_point(aes(x = term, y = b - fe_0_val), color = "black") +
#     #geom_errorbar(aes(x = term, ymin = low, ymax = high), color = "red") +
#     geom_errorbar(aes(x = term, ymin = low - fe_0_val, ymax = high - fe_0_val), color = "black") +
#     geom_hline(yintercept = 0, color = "black", size = 1) +
#     ylim(-0.25, 0.15) +
#     xlab("Time Since First Interview, in Months") +
#     ylab("Elicited 3-Month Job-Finding Probability") +
#     ggtitle("Within Spell Changes Only") +
#     theme_minimal()
#   
#   return(plot1 + plot2)
# }
# 
# 
# prod_fig4_multi <- function(df_list) {
#   # Combine dataframes in df_list with a source column
#   combined_data <- bind_rows(
#     lapply(seq_along(df_list), function(i) {
#       name <- names(df_list)[i]
#       data <- df_list[[name]] %>%
#         filter(age >= 20 & age <= 65, in_sample_1 == 1) %>%
#         mutate(agesq = age^2)
#       
#       # Regressions on monthly dummies for duration
#       duration_vars <- grep("^nedur_1mo_", names(data), value = TRUE)
#       
#       # OLS model
#       ols_model <- lm(as.formula(paste0("find_job_3mon ~ ", paste0(duration_vars, collapse = " + "))),
#                       data = data, weights = weight)
#       ols_summary <- summary(ols_model)
#       ols_coeffs <- ols_summary$coefficients
#       ols_0_val <- ols_coeffs[rownames(ols_coeffs) == "nedur_1mo_0", 1]
#       
#       # Fixed Effects model
#       fe_model <- feols(as.formula(paste0("find_job_3mon ~ ", paste0(duration_vars, collapse = " + "), " | spell_id")),
#                         data = data, weights = ~ weight, cluster = ~ spell_id)
#       fe_summary <- summary(fe_model)
#       fe_coeffs <- tidy(fe_summary)
#       fe_0_val <- fe_coeffs %>% filter(term == "nedur_1mo_0") %>% pull(estimate) %>% as.numeric()
#       
#       # Combine coefficients and standard errors
#       coeffs_data <- bind_rows(
#         ols_coeffs %>% as_tibble(rownames = "term") %>%
#           filter(term %in% duration_vars) %>%
#           rename(b = Estimate, se = `Std. Error`) %>%
#           mutate(model = "OLS", ref_val = ols_0_val),
#         
#         fe_coeffs %>%
#           filter(term %in% duration_vars) %>%
#           rename(b = estimate, se = std.error) %>%
#           mutate(model = "Fixed Effects", ref_val = fe_0_val)
#       )
#       
#       # Add confidence intervals and source
#       coeffs_data %>%
#         mutate(
#           high = b + 1.96 * se,
#           low = b - 1.96 * se,
#           Source = name#,
#           #x_adjusted = term + (i - 1) * 0.01 # Adjust x positions
#         )
#     })
#   )
#   
#   # Ensure Source follows the order in df_list
#   combined_data$Source <- factor(combined_data$Source, levels = names(df_list))
#   combined_data$model <- factor(combined_data$model, levels = c("OLS", "Fixed Effects"))
#   # Offset adjustment: Convert `term` to numeric with an offset
#   term_levels <- unique(combined_data$term)
#   combined_data <- combined_data %>%
#     mutate(
#       term_num = as.numeric(factor(term, levels = term_levels)),
#       term_adjusted = term_num -1 + (as.numeric(Source) - 1) * 0.1  # Offset by Source index
#     )
#   
#   
#   # Plot the data
#   plot <- combined_data %>%
#     filter(term %in% grep("^nedur_1mo_", names(df_list[[1]]), value = TRUE)[1:12]) %>%
#     ggplot(aes(x = term_adjusted - 0.25, y = b - ref_val, color = Source)) +
#     geom_point(size = 3) +
#     geom_errorbar(aes(ymin = low - ref_val, ymax = high - ref_val), width = 0.05) +
#     geom_hline(yintercept = 0, color = "black", size = 1) +
#     facet_wrap(~model, scales = "free") +
#     scale_color_manual(
#       values = c("Orig." = "black", setNames(hue_pal()(length(df_list) - 1), names(df_list)[-1]))
#     ) +
#     scale_x_continuous(
#       breaks = 0:(length(term_levels)-1),
#       labels = c("0mo", "1mo", "2mo", "3mo", "4mo", "5mo", "6mo", "7mo", "8mo", "9mo", "10mo", "11mo"),
#       name = "Time Since First Interview, in Months"
#     ) +
#     geom_vline(xintercept = seq(-0.5, 6.5, by = 1), color = "gray70", linetype = "dashed") + # Vertical lines
#     labs(color = "Sample",
#          x = "Time Since First Interview, in Months",
#          y = "Elicited 3-Month Job-Finding Probability",
#          title = "Fig 4. Changes in Job-Finding Probability Across and Within Spells",
#          subtitle = "Figure 4 illustrates the difference between the observed (cross-sectional - left panel) dura-
# tion dependence and the true (individual-level - right panel) duration dependence in the reported
# beliefs graphically.") +
#     theme_minimal() +
#     theme(legend.position = "bottom", legend.title = element_blank(), panel.grid.major.x = element_blank())
#   
#   return(plot)
# }
# 
# # Call the function with the list of dataframes
# prod_fig4_plot <- prod_fig4_multi(df_list)
# print(prod_fig4_plot)
# 
# 
# ##################################################################################################
# #### Elicited Job-Finding Probabilities and Job-finding Rate by year###
# ##################################################################################################
# 
# combined_jf_uetr_year <- df_list$`2013-24` %>%
#   #select(year = year(date)) %>% 
#   select(date, find_job_3mon, UE_trans_3mon, age, weight, udur_bins, in_sample_2) %>% 
#   mutate(year = year(date)) %>% 
#   filter(age >= 20 & age <= 65 & in_sample_2 == 1) %>%
#   select(year, udur_bins, UE_trans_3mon, find_job_3mon, weight) %>%
#   filter(!is.na(udur_bins) & !is.na(weight)) %>%
#   group_by(year) %>%
#   summarise(
#     rjob_find = weighted.mean(UE_trans_3mon, na.rm = TRUE, w = weight),
#     pjob_find = weighted.mean(find_job_3mon, na.rm = TRUE, w = weight),
#     se_rjob_find = weighted_se(UE_trans_3mon, na.rm = TRUE, w = weight),
#     se_pjob_find = weighted_se(find_job_3mon, na.rm = TRUE, w = weight),
#     nobs = n()
#   ) %>%
#   ungroup() %>%
#   #rename(undur = udur_bins) %>%
#   mutate(
#     pjob_find_lower = pjob_find - qt(0.975, df = nobs - 1) * se_pjob_find,
#     pjob_find_upper = pjob_find + qt(0.975, df = nobs - 1) * se_pjob_find,
#     rjob_find_lower = rjob_find - qt(0.975, df = nobs - 1) * se_rjob_find,
#     rjob_find_upper = rjob_find + qt(0.975, df = nobs - 1) * se_rjob_find,
#     year1 = year + 0.25) # Offset for graphical purposes
# # undur_label = factor(
# #   undur,
# #   levels = 1:4,
# #   labels = c("0-3 Months", "4-6 Months", "7-12 Months", "13 Months +")
# # ))
# 
# 
# # Plot the data
# p <- ggplot(combined_jf_uetr_year, aes(x = year)) +
#   # Elicited probabilities with error bars
#   geom_line(aes(y = pjob_find), size = 1) +
#   geom_point(aes(y = pjob_find), size = 3, shape = 4) +
#   geom_errorbar(
#     aes(ymin = pjob_find_lower, ymax = pjob_find_upper),
#     width = 0.1, size = 0.8
#   ) +
#   # Realized probabilities with error bars
#   geom_line(aes(x = year1, y = rjob_find), size = 1, linetype = "dashed") +
#   geom_point(aes(x = year1, y = rjob_find), size = 3, shape = 1) +
#   geom_errorbar(
#     aes(x = year1, ymin = rjob_find_lower, ymax = rjob_find_upper),
#     width = 0.1, size = 0.8, linetype = "dashed"
#   ) +
#   # Titles and axis labels
#   labs(
#     y = "3-Month Job-Finding Probability/Rate",
#     x = "Duration of Unemployment",
#     title = "Fig 3. Perceived vs. Realized Job Finding, by Duration of Unemployment",
#     subtitle = "Duration dependence is strongly negative across all samples. \nBias in beliefs of LTUE is also consistently high across samples."
#   ) +
#   # scale_x_continuous(
#   #   breaks = 1:4,
#   #   labels = levels(combined_fig3_by_recession$undur_label)
#   # ) +
#   #scale_color_manual(
#   #  values = c("Orig." = "black", setNames(hue_pal()(length(df_list) - 1), names(df_list)[-1]))
#   #) +
#   #ylim(0, 1)+
#   theme_minimal() +
#   theme(
#     legend.position = "none",
#     legend.title = element_blank()
#   )
# 
# print(p)
# 
# combined_jf_uetr_year_by_udur_bin <- df_list$`2013-24` %>%
#   #select(year = year(date)) %>% 
#   select(date, find_job_3mon, UE_trans_3mon, age, weight, udur_bins, in_sample_2) %>% 
#   mutate(year = year(date)) %>% 
#   filter(age >= 20 & age <= 65 & in_sample_2 == 1) %>%
#   select(year, udur_bins, UE_trans_3mon, find_job_3mon, weight) %>%
#   filter(!is.na(udur_bins) & !is.na(weight)) %>%
#   group_by(year, udur_bins) %>%
#   summarise(
#     rjob_find = weighted.mean(UE_trans_3mon, na.rm = TRUE, w = weight),
#     pjob_find = weighted.mean(find_job_3mon, na.rm = TRUE, w = weight),
#     se_rjob_find = weighted_se(UE_trans_3mon, na.rm = TRUE, w = weight),
#     se_pjob_find = weighted_se(find_job_3mon, na.rm = TRUE, w = weight),
#     nobs = n()
#   ) %>%
#   ungroup() %>%
#   #rename(undur = udur_bins) %>%
#   mutate(
#     pjob_find_lower = pjob_find - qt(0.975, df = nobs - 1) * se_pjob_find,
#     pjob_find_upper = pjob_find + qt(0.975, df = nobs - 1) * se_pjob_find,
#     rjob_find_lower = rjob_find - qt(0.975, df = nobs - 1) * se_rjob_find,
#     rjob_find_upper = rjob_find + qt(0.975, df = nobs - 1) * se_rjob_find,
#     year1 = year + 0.25, # Offset for graphical purposes
#     udur_label = factor(
#       udur_bins,
#       levels = 1:4,
#       labels = c("0-3 Months", "4-6 Months", "7-12 Months", "13 Months +")
#     ))
# 
# 
# # Plot the data
# p <- ggplot(combined_jf_uetr_year_by_udur_bin, aes(x = year)) +
#   # Elicited probabilities with error bars
#   geom_line(aes(y = pjob_find, color = udur_label), size = 1) +
#   geom_point(aes(y = pjob_find, color = udur_label), size = 3, shape = 4) +
#   geom_errorbar(
#     aes(ymin = pjob_find_lower, ymax = pjob_find_upper, color = udur_label),
#     width = 0.1, size = 0.8
#   ) +
#   # Realized probabilities with error bars
#   geom_line(aes(x = year1, y = rjob_find, color = udur_label), size = 1, linetype = "dashed") +
#   geom_point(aes(x = year1, y = rjob_find, color = udur_label), size = 3, shape = 1) +
#   geom_errorbar(
#     aes(x = year1, ymin = rjob_find_lower, ymax = rjob_find_upper, color = udur_label),
#     width = 0.1, size = 0.8, linetype = "dashed"
#   ) +
#   # Titles and axis labels
#   labs(
#     y = "3-Month Job-Finding Probability/Rate",
#     x = "Year",
#     title = "Fig 3. Perceived and Realized Job Finding, by Year",
#     #subtitle = "Duration dependence is strongly negative across all samples. \nBias in beliefs of LTUE is also consistently high across samples."
#   ) +
#   # scale_x_continuous(
#   #   breaks = 1:4,
#   #   labels = levels(combined_fig3_by_recession$undur_label)
#   # ) +
#   #scale_color_manual(
#   #  values = c("Orig." = "black", setNames(hue_pal()(length(df_list) - 1), names(df_list)[-1]))
#   #) +
#   #ylim(0, 1)+
#   theme_minimal() +
#   theme(
#     legend.position = "none",
#     legend.title = element_blank()
#   ) + facet_wrap(~udur_label)
# 
# print(p)


