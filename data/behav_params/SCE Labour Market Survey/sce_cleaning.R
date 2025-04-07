# Cleaning the Survey on Consumer Expectations Labour Market Survey Supplement###
# Source: https://www.newyorkfed.org/microeconomics/sce/labor#/
# Codebook/questionnaire: https://www.newyorkfed.org/medialibrary/media/research/microeconomics/interactive/downloads/sce-labor-questionnaire.pdf?sc_lang=en
# rm(list = ls())
library(here)
library(tidyverse)
library(readxl)
library(lubridate)
library(stargazer)
library(lfe) # for regressions with clustering
library(weights)
library(diagis)
library(fixest) # For regressions with clustering and fixed effects
library(patchwork)
library(broom) # For extracting model coefficients
library(modelsummary)
library(flextable)
final = TRUE

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
  semi_join(select(arrange(sce_lab, date, userid), userid, date), ., 
            by = c("userid", "date")) %>% nrow(.) == nrow(sce_lab)


################################################################################
transform_fun <- function(df){
 df <- df  %>% 
  rename(reservation_wage_orig = reservation_wage) %>% 
  mutate(year = year(date), 
         month = month(date),
         never_worked = ifelse(is.na(never_worked), 0 , 1),
                   #   agesq = age^2,
         self_employed = case_when(self_employed == 1 ~ 0,
                                   self_employed == 2 ~ 1, 
                                   TRUE ~ NA),
         looked_for_work_l4_wks = case_when(looked_for_work_l4_wks == 1 ~ 1,
                                            looked_for_work_l4_wks == 2 ~ 0, 
                                            TRUE ~ NA),
         across(c(reservation_wage_orig, wage_most_recent_job, current_wage_annual), ~as.integer(.)),
         reservation_wage_unit_scale = case_when(reservation_wage_unit == 1 ~ 2080, # Hourly - calculated 40 hours x 52 weeks # & nchar(reservation_wage_orig) %in% c(2,3) 
                                                 reservation_wage_unit == 2 ~ 52, # Weekly- 52 weeks # & nchar(reservation_wage_orig) %in% c(3,4)
                                                 reservation_wage_unit == 3 ~ 26, # Bi-weekly ~ 52/2 # & nchar(reservation_wage_orig) %in% c(4)
                                                 reservation_wage_unit == 4 ~ 12, # Monthly - 12 # & nchar(reservation_wage_orig) %in% c(4,5)
                                                 reservation_wage_unit == 5  ~ 1, # Annual - 1 #& nchar(reservation_wage_orig) >= 5
                                                 date >= "2017-03-01" ~ 1),  
         # Can somewhat safely assume that single-digit or two-digit reservation wage is a reported hourly wage
         reservation_wage_unit_scale = ifelse(is.na(reservation_wage_unit) & nchar(reservation_wage_orig) %in% c(1,2), 2080, reservation_wage_unit_scale),
         reservation_wage_unit_scale = ifelse(nchar(reservation_wage_orig) >= 5, 1, reservation_wage_unit_scale),
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
         wage_most_recent_job = ifelse(wage_most_recent_job == 0, NA, wage_most_recent_job),
         current_wage_annual = ifelse(current_wage_annual == 0, NA, current_wage_annual),
         expbest4mos_rel_res = exp_salary_best_offer_4mos/reservation_wage, 
         expbest4mos_rel_current = exp_salary_best_offer_4mos/current_wage_annual, 
         expbest4mos_rel_most_recent = exp_salary_best_offer_4mos/wage_most_recent_job,
         #select(year, contains("reservation_wage"))
  # #filter(reservation_wage <= 10000000) %>%
  # group_by(year, month) %>%
  # summarise(res_wage = mean(reservation_wage, na.rm = TRUE)) %>%
  # ggplot() +
  # geom_line(aes(x = year + month/12,  y= res_wage))
         #current_wage_annual = ifelse(current_wage_annual == 0, reservation_wage, current_wage_annual),
         res_wage_to_current = reservation_wage/current_wage_annual,
         res_wage_to_latest = reservation_wage/wage_most_recent_job) %>%
    # filter(wage_most_recent_job < 1000000 | is.na(wage_most_recent_job)) %>%
    filter(age >= 20 & age <= 65) %>% 
    #          (res_wage_to_current < 10 | is.na(res_wage_to_current)),
    #        (res_wage_to_latest < 10 | is.na(res_wage_to_latest))) %>%
    mutate(accepted_salary_1 = job_offer_1_salary*(job_offer_1_accepted %in% c(1,2)),
           accepted_salary_2 = job_offer_2_salary*(job_offer_2_accepted %in% c(1,2)),
           accepted_salary_3 = job_offer_3_salary*(job_offer_3_accepted %in% c(1,2))) %>%
   rowwise %>%
   mutate(accepted_salary = max(accepted_salary_1, accepted_salary_2, accepted_salary_3, na.rm = TRUE),
          reservation_wage_latest = max(reservation_wage, na.rm = TRUE)) %>%
   ungroup %>%
           mutate(accepted_salary = ifelse(accepted_salary == -Inf, NA, accepted_salary),
                  accepted_salary = ifelse(accepted_salary == 0, NA, accepted_salary),
           reservation_wage_latest = ifelse(reservation_wage_latest == -Inf, NA, reservation_wage_latest),
           salary_prop_reswage = accepted_salary/reservation_wage_latest) %>%
 # Exclude anyone who sets a reservation wage below the minimum annual salary or above 1million USD
  filter(reservation_wage >= 14000 & reservation_wage < 1000000) %>% 
   filter(!(res_wage_to_latest > 2 & is.na(res_wage_to_current))) %>% 
   filter(!(res_wage_to_current > 2 & is.na(res_wage_to_latest)))
 return(df)
}

# Define base directory (replace this with the correct path later)
base <- here("data/behav_params/Mueller_Replication/")

# Load data file
data_13_24 <- #readRDS(paste0(base, "sce_datafile_13_24_w_lab_survey.RDS")) %>% 
  sce_lab %>% 
  left_join(., sce_13_24, by = c("userid", "date")) %>% 
  rename(weight = rim_4_original) %>% 
  transform_fun(.)

if(!final){
  # 10,629 people are interviewed more than once!
  panel_ids <- sce_lab %>% select(userid) %>% group_by(userid) %>% mutate(n = n()) %>% filter(n != 1) %>% pull(userid) %>% unique
  # 1,119 observations are in the unemployed sample in sce
  sce_13_24 %>% group_by(userid) %>% filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
    pull(userid) %>% unique %>% intersect(panel_ids, .) %>% length
  # 9,520 observations are not in the unemployed sample in sce
  sce_13_24 %>% group_by(userid) %>% filter(any(temp_laid_off == 1 | (not_working_wouldlike == 1 & looking_for_job == 1))) %>% 
    pull(userid) %>% unique %>% setdiff(panel_ids, .) %>% length
}

unemp_only <- readRDS(here('data/behav_params/Mueller_Replication/sce_datafile_13_24_w_lab_survey_new.RDS')) %>% 
  transform_fun(.)
  
# ######### TESTING SCE LAB! #
# sce_lab %>% 
#   transform_fun(.) %>% 
#   mutate(year = year(date)) %>% 
#   group_by(year) %>% 
#   summarise(mean_res_wage = mean(reservation_wage, na.rm = TRUE)) %>% 
#   ggplot(aes(x = year, y = mean_res_wage)) +
#   geom_line()


#### DEMONSTRATES THE RESERVATION WAGE TRAJECTORY ACROSS INDIVIDUALS ###
if(!final){
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
}

################################################################################

controls <- c("female", "hispanic", "black", "r_asoth", "other_race", "age", #"agesq", 
              "hhinc_2", "hhinc_3", "hhinc_4", "education_2", "education_3", "education_4", 
              "education_5", "education_6")

###################################################################
# 2. Histogram of elicited wage expectation conditional on pasts wage and job expectation (OO2new-OO2e2dk ~ L10-11)
###################################################################
# There are some issues with the histogram which I think has to do with the binning...left as is below for now

print("Plots of RESERVATION WAGE versus latest, current wage")

# Create fweight variable (rounded weights)
data_fig1 <- data_13_24 %>%
  mutate(fweight = round(weight, 1))

if(!final){
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
}
  
res_wage_prop1 <- data_fig1 %>% 
  ggplot() +
  geom_histogram(
    aes(x = res_wage_to_current, weight = fweight, fill = "Reservation Wage to Current (Employed: n = 8744)"), fill = "blue",
  ) +
  labs(
    title = "Reservation Wage as proportion of Current Wage (Employed: n = 8744)",
    x =  "Reservation Wage / Latest Held Wage"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) + 
  xlim(0.25, 2)

res_wage_prop2 <- data_fig1 %>% 
  ggplot() +
  geom_histogram(
    aes(x = res_wage_to_latest, weight = fweight, fill = "Reservation Wage to Latest"), fill = "red"
  ) +
  labs(
    title = "Reservation Wage as proportion of Latest Held Wage (Unemployed: n = 3145)",
    x = "Reservation Wage / Latest Held Wage"
  ) +
  theme_minimal() +
  theme(
   legend.position = "bottom",
    panel.grid.minor = element_blank()
  )  +
  xlim(0.25, 2)

print(res_wage_prop1 / res_wage_prop2)

# data_fig1 %>% 
#   ggplot() + 
#   geom_point(aes(x = log(current_wage_annual), y = log(reservation_wage), color = "Reservation wage vs. Current")) +
#   geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
#   geom_abline(slope = 1, linetype = "dashed") +
#   theme(legend.position = "bottom") +
#   labs(title = "All workers")

t1 <- unemp_only %>% 
  filter(!is.na(udur_bins)) %>% 
  ggplot(aes(x = udur_bins, y = log(reservation_wage), size = weight)) + 
  geom_point()+ #color = "Reservation wage vs. Current")) +
  geom_smooth(method = "lm", formula = y~x, mapping = aes(weight = weight), show.legend = FALSE) +
  #geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
  theme(legend.position = "none") +
  labs(title = "Reservation Wage (log) by Unemp. Dur.",
       x = "Unemployment Duration", 
       y = "(log) Reservation Wage") +
  scale_x_continuous(breaks = 1:4, labels = c("<4 mos", "4-6 mos",  "7-12 mos", ">12 mos"))

t2 <- unemp_only %>% 
  ggplot(aes(x = udur, y = log(reservation_wage), size = weight)) + 
  geom_point()+ #color = "Reservation wage vs. Current")) +
  geom_smooth(method = "lm", formula = y~x, mapping = aes(weight = weight), show.legend = FALSE) +
  #geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
  theme(legend.position = "none") +
  labs(title = "Reservation Wage (log) by Unemp. Dur.",
       x = "Unemployment Duration", 
       y = "(log) Reservation Wage")

t3 <- unemp_only %>% 
  filter(!is.na(udur_bins)) %>% 
  ggplot(aes(x = udur_bins, y = res_wage_to_latest, size = weight)) + 
  geom_point()+ #color = "Reservation wage vs. Current")) +
  geom_smooth(method = "lm", formula = y~x, mapping = aes(weight = weight), show.legend = FALSE) +
  #geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
  theme(legend.position = "none") +
  labs(title = "ResWage:Latest by Unemp. Dur.",
       x = "Unemployment Duration", 
       y = "Reservation Wage:Latest Held Wage") +
  scale_x_continuous(breaks = 1:4, labels = c("<4 mos", "4-6 mos",  "7-12 mos", ">12 mos"))

t4 <- unemp_only %>% 
  ggplot(aes(x = udur, y = res_wage_to_latest, size = weight)) + 
  geom_point()+ #color = "Reservation wage vs. Current")) +
  geom_smooth(method = "lm", formula = y~x, mapping = aes(weight = weight), show.legend = FALSE) +
  #geom_point(aes(x = log(wage_most_recent_job), y = log(reservation_wage), color = "Reservation wage vs. Last Held")) +
  theme(legend.position = "none") +
  labs(title = "ResWage:Latest by Unemp. Dur.",
       x = "Unemployment Duration", 
       y = "Reservation Wage:Latest Held Wage")

print((t1 + t3) / (t2 + t4))

##########################################################################################
####################### Plots of EXPECTATION versus latest, current, reservation wage ####

print("Plots of EXPECTED OFFER versus latest, current, reservation wage")

t1a <- data_fig1 %>% 
  filter(!is.na(expbest4mos_rel_current) & expbest4mos_rel_current < 2.5 & expbest4mos_rel_current > 0.25) %>% 
  ggplot(aes(y = expbest4mos_rel_current)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Exp. offer:current wage") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5),
        axis.title.x = element_blank(),  # Remove x-axis title
        axis.text.x = element_blank(),   # Remove x-axis labels
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank(),  # Remove major vertical grid lines
        panel.grid.minor.x = element_blank())   # Remove x-axis ticks)

t1b <- data_fig1 %>% 
  filter(!is.na(expbest4mos_rel_most_recent) & expbest4mos_rel_most_recent < 2.5 & expbest4mos_rel_most_recent > 0.25) %>% 
  ggplot(aes(y = expbest4mos_rel_most_recent)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Exp. offer:latest wage") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5),
        axis.title.x = element_blank(),  # Remove x-axis title
        axis.text.x = element_blank(),   # Remove x-axis labels
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank(),  # Remove major vertical grid lines
        panel.grid.minor.x = element_blank())   # Remove x-axis ticks)

t2 <- unemp_only %>% 
  filter(!is.na(udur_bins)) %>% 
  #mutate(udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(expbest4mos_rel_res) & expbest4mos_rel_res < 2.5 & expbest4mos_rel_res > 0.25) %>% 
  ggplot(aes(x = as.factor(udur_bins), y = expbest4mos_rel_res)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Exp. best offer:res. wage") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5)) +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos",  "7-12 mos", ">12 mos"))


t3 <- unemp_only %>% 
  filter(!is.na(udur_bins)) %>% 
  #mutate(udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(expbest4mos_rel_most_recent) & expbest4mos_rel_most_recent < 2.5 & expbest4mos_rel_most_recent > 0.25) %>% 
  ggplot(aes(x = as.factor(udur_bins), y = expbest4mos_rel_most_recent)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Exp. best offer:latest wage") +
  theme_minimal() +
  ylim(0,2.5)  +
  theme(plot.title=element_text(hjust=0.5)) +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos",  "7-12 mos", ">12 mos"))


t4 <- data_fig1 %>% 
  #mutate(udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(expbest4mos_rel_res) & expbest4mos_rel_res < 2.5 & expbest4mos_rel_res > 0.25) %>% 
  mutate(group = case_when(!is.na(expbest4mos_rel_current) ~ "Emp",
                   !is.na(expbest4mos_rel_most_recent) ~ "Unemp",
                   TRUE ~ NA)) %>% 
  ggplot(aes(x = as.factor(group), y = expbest4mos_rel_res)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Exp. best offer:res wage") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5))

print((t1a + t1b + t4) / (t2 + t3) + 
  plot_annotation("Ratio of Expected Best Offer to Various Benchmarks (Reservation, Current, Latest Held Wage",
                                               caption = "Notes: Regressions are estimated in the Survey of Consumer Expectations between 2014-2022. \nObservations are weighted by their SCE sample weight.",
                                               theme=theme(plot.title=element_text(hjust=0.5))))


##############################################################################################
####################### Plots of ACCEPTED SALARY versus latest, current, reservation wage ####
print("Plots of ACCEPTED SALARY versus latest, current, reservation wage")

t1 <- unemp_only %>%
  group_by(userid) %>% 
  fill(udur_bins, .direction = "down") %>%ungroup %>% 
  filter(salary_prop_reswage > 0.25 & salary_prop_reswage < 2.5) %>% 
  ggplot() +
  geom_histogram(aes(x = salary_prop_reswage)) +
  labs(
    title = "Unemployed Only",
    x = "Accepted Wage / Reservation Wage"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) 

t2 <- unemp_only %>%
  group_by(userid) %>% 
  fill(udur_bins, .direction = "down") %>% ungroup %>% 
  filter(salary_prop_reswage > 0.25 & salary_prop_reswage < 2.5) %>% 
  ggplot() +
  geom_jitter(aes(x = log(reservation_wage), y = log(accepted_salary))) +
  geom_abline(slope = 1) +
  xlim(9, 13) +  # Adjust limits as needed
  ylim(8.5, 13) +
  labs(
    title = "Unemployed Only",
    x = "(log) Reservation Wage",
    y = "(log) Accepted Wage"
  ) 

t3 <- data_fig1 %>%
  filter(salary_prop_reswage > 0.25 & salary_prop_reswage < 2.5) %>% 
  ggplot() +
  geom_histogram(aes(x = salary_prop_reswage)) +
  labs(
    title = "All Respondents",
    x = "Accepted Wage / Reservation Wage"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) 

t4 <- data_fig1 %>%
  filter(salary_prop_reswage > 0.5 & salary_prop_reswage < 2) %>% 
  ggplot() +
  geom_jitter(aes(x = log(reservation_wage), y = log(accepted_salary))) +
  geom_abline(slope = 1) +
  labs(
    title = "All Respondents",
    x = "(log) Reservation Wage",
    y = "(log) Accepted Wage"
  ) +
  xlim(9, 13) +  # Adjust limits as needed
  ylim(8.5, 13)

print((t1 + t2) / (t3 + t4))

t1 <- data_fig1 %>% 
  filter(!is.na(salary_prop_reswage) & salary_prop_reswage < 2.5 & salary_prop_reswage > 0.25) %>% 
  ggplot(aes(y = salary_prop_reswage)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Total (n = 735)") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5),
        axis.title.x = element_blank(),  # Remove x-axis title
        axis.text.x = element_blank(),   # Remove x-axis labels
        axis.ticks.x = element_blank(),
        panel.grid.major.x = element_blank(),  # Remove major vertical grid lines
        panel.grid.minor.x = element_blank())   # Remove x-axis ticks

t2 <- unemp_only %>% 
  group_by(userid) %>% 
  fill(udur_bins, .direction = "down") %>% 
  ungroup %>% 
  #mutate(udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(salary_prop_reswage) & salary_prop_reswage < 2.5 & salary_prop_reswage > 0.25 & !is.na(udur_bins)) %>% 
  ggplot(aes(x = as.factor(udur_bins), y = salary_prop_reswage)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Unemployed by Duration (n = 196)", x = "Unemployment Duration") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5)) +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos",  "7-12 mos", ">12 mos"))

t3 <- data_fig1 %>% 
  #mutate(udur_bins = ifelse(is.na(udur_bins), 0, udur_bins)) %>% 
  filter(!is.na(salary_prop_reswage) & salary_prop_reswage < 2.5 & salary_prop_reswage > 0.25) %>% 
  mutate(group = case_when(!is.na(expbest4mos_rel_current) ~ "Emp",
                           !is.na(expbest4mos_rel_most_recent) ~ "Unemp",
                           # THIS IS LIKELY THE WRONG CONVERSION
                           TRUE ~ "Unemp")) %>%
  ggplot(aes(x = as.factor(group), y = salary_prop_reswage)) +
  geom_boxplot(aes(weight = weight))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Employed (n = 532) vs. Unemployed (n = 203)", x = "Employment Status") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5))

print((t1 + t3 + t2) + 
  plot_annotation("Ratio of Accepted Salary to Reservation Wage",
                  caption = "Notes: Regressions are estimated in the Survey of Consumer Expectations between 2014-2022. \nObservations are weighted by their SCE sample weight.",
                  theme=theme(plot.title=element_text(hjust=0.5))))


# ####### REGRESSIONS ##########
# #################################################################################################
# ### Reservation wage as function of unemp duration (raw and as proportion of latest held or current wage) ###
# #################################################################################################
# 
# reservation_wage ~ unempduration
mod1a <- lm(data = filter(unemp_only, !is.na(reservation_wage)), log(reservation_wage) ~ udur_bins, weights = weight)
mod1b <- lm(data = filter(unemp_only, !is.na(reservation_wage)), 
            as.formula(paste0("log(reservation_wage) ~ udur_bins + ", paste0(controls, collapse = "+"))), weights = weight)

# reservation_wage/lastheldwage ~ unempduration
mod2a <- lm(data = filter(unemp_only, res_wage_to_latest < 2), res_wage_to_latest ~ udur_bins, weights = weight)
mod2b <- lm(data = filter(unemp_only, res_wage_to_latest < 2), 
            as.formula(paste0("res_wage_to_latest ~ udur_bins + ", paste0(controls, collapse = "+"))), weights = weight)

# #################################################################################################
# ### Accepted wage as function of unemp duration ###
# #################################################################################################
accepted_temp <- unemp_only %>% 
  group_by(userid) %>% 
  fill(udur_bins, .direction = "down")
# accepted_wage ~ unempduration
mod3a <- lm(data = filter(accepted_temp, accepted_salary > 14000 & accepted_salary < 1000000), log(accepted_salary) ~ udur_bins, weights = weight)
mod3b <- lm(data = filter(accepted_temp, accepted_salary > 14000 & accepted_salary < 1000000), as.formula(paste0("log(accepted_salary) ~ udur_bins +", paste0(controls, collapse = "+"))), weights = weight)

# accepted_wage/reservation_wage ~ unempduration
mod4a <- lm(data = filter(accepted_temp, salary_prop_reswage < 2.5 & salary_prop_reswage > 0.25), salary_prop_reswage ~ udur_bins, weights = weight)
mod4b <- lm(data = filter(accepted_temp, salary_prop_reswage < 2.5 & salary_prop_reswage > 0.25), as.formula(paste0("salary_prop_reswage ~ udur_bins +", paste0(controls, collapse = "+"))), weights = weight)

# accepted_wage/lastheldwage ~ unempduration

# #################################################################################################
# ### Elicited versus realised wage by unemployment duration  ###
# #################################################################################################
# elicited_wage/accepted_wage ~ unemp_duration

mod5a <- lm(data = filter(unemp_only, !is.na(expbest4mos_rel_res) & expbest4mos_rel_res < 2.5 & expbest4mos_rel_res > 0.25), expbest4mos_rel_res ~ udur_bins, weights = weight)
mod5b <- lm(data = filter(unemp_only, !is.na(expbest4mos_rel_res) & expbest4mos_rel_res < 2.5 & expbest4mos_rel_res > 0.25), as.formula(paste0("expbest4mos_rel_res ~ udur_bins +", paste0(controls, collapse = "+"))), weights = weight)

mod6a <- lm(data = filter(unemp_only, !is.na(expbest4mos_rel_most_recent) & expbest4mos_rel_most_recent < 2.5 & expbest4mos_rel_most_recent > 0.25), expbest4mos_rel_most_recent ~ udur_bins, weights = weight)
mod6b <- lm(data = filter(unemp_only, !is.na(expbest4mos_rel_most_recent) & expbest4mos_rel_most_recent < 2.5 & expbest4mos_rel_most_recent > 0.25), as.formula(paste0("expbest4mos_rel_most_recent ~ udur_bins +", paste0(controls, collapse = "+"))), weights = weight)


modelsummary(list("ResWage" = mod1a,"ResWage w.c" = mod1b, 
                  "ResWage/LastWage" =mod2a, "ResWage/LastWage w.c" =mod2b), output = "markdown",
             stars = TRUE, coef_omit = c(3:16), title = "Reservation Wages and Unemployment Duration") %>% print(.)

modelsummary(list(
                  "AccptWage" =mod3a, "AccptWage w.c" =mod3b,
                  "AccptWage/ResWage" =mod4a, "AccptWage/ResWage w.c" =mod4b), output = "markdown",
             stars = TRUE, coef_omit = c(3:15), title = "Accepted Wages and Unemployment Duration") %>% print(.)

modelsummary(list(
                  "ExpWage/ResWage" =mod5a, "ExpWage/ResWage w.c" =mod5b,
                  "ExpWage/LastWage" =mod6a, "ExpWage/LastWage w.c" =mod6b), output = "markdown",
             stars = TRUE, coef_omit = c(3:16), title = "Expected Wages and Unemployment Duration") %>% print(.)

