library(here)
library(tidyverse)
library(readxl)

source(here("data/behav_params/Mueller_Replication/mueller_repl_sce_raw_data_cleaning.R"))

# From mueller_repl_sce_raw_data_cleaning.R
unemp_temp <- sce_lab %>% 
  left_join(., sce_13_24_no_lab, by = c("userid", "date")) %>% 
  mutate(date = ceiling_date(ym(date), 'month') - days(1))
# ################################################################################
# transform_fun <- function(df){
#   df <- df  %>% 
#     #readRDS(here('data/behav_params/Mueller_Replication/sce_datafile_13_24_w_lab_survey_new.RDS'))%>% 
#     rename(reservation_wage_orig = reservation_wage) %>% 
#     mutate(year = year(date), 
#            month = month(date),
#            # never_worked = ifelse(is.na(never_worked), 0 , 1),
#            # #   agesq = age^2,
#            # self_employed = case_when(self_employed == 1 ~ 0,
#            #                           self_employed == 2 ~ 1, 
#            #                           TRUE ~ NA),
#            # looked_for_work_l4_wks = case_when(looked_for_work_l4_wks == 1 ~ 1,
#            #                                    looked_for_work_l4_wks == 2 ~ 0, 
#            #                                    TRUE ~ NA),
#            across(c(reservation_wage_orig, wage_most_recent_job, current_wage_annual), ~as.integer(.)),
#            reservation_wage_unit_scale = case_when(reservation_wage_unit == 1 ~ 2080, # Hourly - calculated 40 hours x 52 weeks # & nchar(reservation_wage_orig) %in% c(2,3) 
#                                                    reservation_wage_unit == 2 ~ 52, # Weekly- 52 weeks # & nchar(reservation_wage_orig) %in% c(3,4)
#                                                    reservation_wage_unit == 3 ~ 26, # Bi-weekly ~ 52/2 # & nchar(reservation_wage_orig) %in% c(4)
#                                                    reservation_wage_unit == 4 ~ 12, # Monthly - 12 # & nchar(reservation_wage_orig) %in% c(4,5)
#                                                    reservation_wage_unit == 5  ~ 1, # Annual - 1 #& nchar(reservation_wage_orig) >= 5
#                                                    date >= "2017-03-01" ~ 1),  
#            # Can somewhat safely assume that single-digit or two-digit reservation wage is a reported hourly wage
#            reservation_wage_unit_scale = ifelse(is.na(reservation_wage_unit) & nchar(reservation_wage_orig) %in% c(1,2), 2080, reservation_wage_unit_scale),
#            reservation_wage_unit_scale = ifelse(nchar(reservation_wage_orig) >= 5, 1, reservation_wage_unit_scale),
#            reservation_wage = reservation_wage_orig * reservation_wage_unit_scale, 
#            # Reservation wage is sometimes filled in as 0 - will replace with NA values
#            reservation_wage = ifelse(reservation_wage == 0, NA, reservation_wage), 
#            current_wage_annual_cat = case_when(current_wage_annual < 10000 ~ 1, # tested: sce_lab %>% filter(!is.na(current_wage_annual) & !is.na(current_wage_annual_cat)) %>% nrow(.) == 0
#                                                current_wage_annual  >= 10000 & current_wage_annual <= 19999 ~ 2,
#                                                current_wage_annual  >= 20000 & current_wage_annual <= 29999 ~ 3,
#                                                current_wage_annual  >= 30000 & current_wage_annual <= 39999 ~ 4,
#                                                current_wage_annual  >= 40000 & current_wage_annual <= 49999 ~ 5,
#                                                current_wage_annual  >= 50000 & current_wage_annual <= 59999 ~ 6,
#                                                current_wage_annual  >= 60000 & current_wage_annual <= 74999 ~ 7,
#                                                current_wage_annual  >= 75000 & current_wage_annual <= 99999 ~ 8,
#                                                current_wage_annual  >= 100000 & current_wage_annual <= 149999 ~ 9,
#                                                current_wage_annual  >= 150000 ~ 10,
#                                                TRUE ~ current_wage_annual_cat),
#            current_wage_annual = case_when(is.na(current_wage_annual) & current_wage_annual_cat == 1 ~ 10000, # If I only have the category (happens in 144 observations), I replace with the mean if a range is provided or the upper/lower range in the bounding categories, respectively.
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 2 ~ 15000,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 3 ~ 25000,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 4 ~ 35000,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 5 ~ 45000,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 6 ~ 55000,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 7 ~ 67500,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 8 ~ 87500,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 9 ~ 125000,
#                                            is.na(current_wage_annual) & current_wage_annual_cat == 10 ~ 150000,
#                                            TRUE ~ current_wage_annual),
#            wage_most_recent_job = ifelse(wage_most_recent_job == 0, NA, wage_most_recent_job),
#            current_wage_annual = ifelse(current_wage_annual == 0, NA, current_wage_annual)) %>% 
#     rename(wage_most_recent_job_orig = wage_most_recent_job,
#            current_wage_annual_orig = current_wage_annual) %>% 
#     mutate(wage_most_recent_job = case_when(nchar(wage_most_recent_job_orig) <= 2 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 2080,
#                                             nchar(wage_most_recent_job_orig) == 4 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 12, 
#                                             nchar(wage_most_recent_job_orig) == 3 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 52, 
#                                             TRUE ~ wage_most_recent_job_orig),
#            current_wage_annual = case_when(nchar(current_wage_annual_orig) <= 2 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 2080,
#                                            nchar(current_wage_annual_orig) == 4 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 12, 
#                                            nchar(current_wage_annual_orig) == 3 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 52, 
#                                            TRUE ~ current_wage_annual_orig),
#            expbest4mos_rel_res = exp_salary_best_offer_4mos/reservation_wage, 
#            expbest4mos_rel_current = exp_salary_best_offer_4mos/current_wage_annual, 
#            expbest4mos_rel_most_recent = exp_salary_best_offer_4mos/wage_most_recent_job,
#            res_wage_to_current = reservation_wage/current_wage_annual,
#            res_wage_to_latest = reservation_wage/wage_most_recent_job) %>%
#     filter(age >= 20 & age <= 65) %>% 
#     mutate(accepted_salary_1 = job_offer_1_salary*(job_offer_1_accepted %in% c(1,2)),
#            accepted_salary_2 = job_offer_2_salary*(job_offer_2_accepted %in% c(1,2)),
#            accepted_salary_3 = job_offer_3_salary*(job_offer_3_accepted %in% c(1,2))) %>%
#     rowwise %>%
#     mutate(accepted_salary = max(accepted_salary_1, accepted_salary_2, accepted_salary_3, na.rm = TRUE),
#            reservation_wage_latest = max(reservation_wage, na.rm = TRUE)) %>%
#     ungroup %>%
#     mutate(accepted_salary = ifelse(accepted_salary == -Inf, NA, accepted_salary),
#            accepted_salary = ifelse(accepted_salary == 0, NA, accepted_salary),
#            reservation_wage_latest = ifelse(reservation_wage_latest == -Inf, NA, reservation_wage_latest),
#            salary_prop_reswage = accepted_salary/reservation_wage_latest,
#            salary_to_latest = accepted_salary/wage_most_recent_job) %>%
#     # Exclude anyone who sets a reservation wage below the minimum annual salary or above 1million USD
#     filter(reservation_wage >= 14000 & reservation_wage < 1000000) %>% 
#     filter(!(res_wage_to_latest > 2 & is.na(res_wage_to_current))) %>% 
#     filter(!(res_wage_to_current > 2 & is.na(res_wage_to_latest))) %>% 
#     # exclude part-time workers
#     filter(working_pt == 0)
#   return(df)
# }

unemp_only <- unemp_temp %>% 
  mutate(date_last_worked_at_job = paste0(y_last_worked_at_job, ifelse(nchar(m_last_worked_at_job) == 1, paste0("0", m_last_worked_at_job), m_last_worked_at_job)), 
         date_last_worked_at_job = ceiling_date(ym(date_last_worked_at_job), 'month') - days(1),
         unemp_dur_test = interval(date_last_worked_at_job, date) %/% months(1), 
         udur = ifelse(is.na(unemp_dur_test), unemployment_duration, unemp_dur_test),
           udur_bins = case_when(
            udur <= 3 ~ 1,
            udur > 3 & udur <= 6 ~ 2,
            udur > 6 & udur <= 12 ~ 3,
            udur > 12 & !is.na(udur) ~ 4,
            TRUE ~ NA_real_
          )) %>% 
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
         current_wage_annual = ifelse(current_wage_annual == 0, NA, current_wage_annual)) %>% 
  rename(wage_most_recent_job_orig = wage_most_recent_job,
         current_wage_annual_orig = current_wage_annual) %>% 
  mutate(wage_most_recent_job = case_when(nchar(wage_most_recent_job_orig) <= 2 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 2080,
                                          nchar(wage_most_recent_job_orig) == 4 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 12, 
                                          nchar(wage_most_recent_job_orig) == 3 & is.na(current_wage_annual_orig) ~ wage_most_recent_job_orig * 52, 
                                          TRUE ~ wage_most_recent_job_orig),
         current_wage_annual = case_when(nchar(current_wage_annual_orig) <= 2 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 2080,
                                         nchar(current_wage_annual_orig) == 4 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 12, 
                                         nchar(current_wage_annual_orig) == 3 & is.na(current_wage_annual_orig) ~ current_wage_annual_orig * 52, 
                                         TRUE ~ current_wage_annual_orig),
         expbest4mos_rel_res = exp_salary_best_offer_4mos/reservation_wage, 
         expbest4mos_rel_current = exp_salary_best_offer_4mos/current_wage_annual, 
         expbest4mos_rel_most_recent = exp_salary_best_offer_4mos/wage_most_recent_job,
         res_wage_to_current = reservation_wage/current_wage_annual,
         res_wage_to_latest = reservation_wage/wage_most_recent_job) %>% 
  #filter(not_working_wouldlike == 1) %>% 
  rowwise %>% 
  mutate(effective_res_wage = max(reservation_wage, wage_most_recent_job, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(effective_res_wage = ifelse(effective_res_wage == -Inf, NA, effective_res_wage)) %>% 
  #filter(!is.na(effective_res_wage)) %>% 
  #filter(age >= 20 & age <= 65) %>% 
  mutate(accepted_salary_1 = job_offer_1_salary*(job_offer_1_accepted %in% c(1,2)),
         accepted_salary_2 = job_offer_2_salary*(job_offer_2_accepted %in% c(1,2)),
         accepted_salary_3 = job_offer_3_salary*(job_offer_3_accepted %in% c(1,2))) %>%
  rowwise %>%
  mutate(accepted_salary = max(accepted_salary_1, accepted_salary_2, accepted_salary_3, na.rm = TRUE),
         reservation_wage_latest = max(reservation_wage, na.rm = TRUE)) %>%
  ungroup %>%
  mutate(accepted_salary = ifelse(accepted_salary == -Inf, NA, accepted_salary),
         accepted_salary = ifelse(accepted_salary == 0, NA, accepted_salary),
         accepted_salary_trans = ifelse(nchar(accepted_salary) == 2, accepted_salary*2080, 
                                  ifelse(nchar(accepted_salary) %in% c(3, 4), accepted_salary*12, accepted_salary)),
         salary_prop_effreswage = accepted_salary_trans/effective_res_wage,
         reservation_wage_latest = ifelse(reservation_wage_latest == -Inf, NA, reservation_wage_latest),
         salary_prop_reswage = accepted_salary/reservation_wage_latest,
         salary_to_latest = accepted_salary/wage_most_recent_job) %>%
  rename(weight = rim_4_original)
  # Exclude anyone who sets a reservation wage below the minimum annual salary or above 1million USD
  # filter(reservation_wage >= 14000 & reservation_wage < 1000000) %>% 
  # filter(!(res_wage_to_latest > 2 & is.na(res_wage_to_current))) %>% 
  # filter(!(res_wage_to_current > 2 & is.na(res_wage_to_latest))) %>% 
  # #filter(!is.na(salary_prop_effreswage)) %>% 
  # #select(udur_bins, accepted_salary, accepted_salary_trans, effective_res_wage, salary_prop_effreswage) %>% 
  # filter(salary_prop_effreswage >= 0.25 & salary_prop_effreswage <= 2.5)
  # # Exclude anyone who sets a reservation wage below the minimum annual salary or above 1million USD
  # filter(reservation_wage >= 14000 & reservation_wage < 1000000) %>% 
  # filter(!(res_wage_to_latest > 2 & is.na(res_wage_to_current))) %>% 
  # filter(!(res_wage_to_current > 2 & is.na(res_wage_to_latest))) %>% 

  # # exclude part-time workers
  # filter(working_pt == 0)

#### DEMONSTRATES THE RESERVATION WAGE TRAJECTORY ACROSS INDIVIDUALS ###

unemp_only %>% 
    filter(!is.na(reservation_wage)) %>% 
    group_by(userid) %>%
    arrange(userid, date) %>% 
    mutate(time_period = row_number()) %>% 
    relocate(time_period) %>% 
    select(time_period, userid, reservation_wage) %>% 
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
    select(time_period, userid, res_wage_to_latest, salary_prop_effreswage) %>%
    lm(data = ., res_wage_to_latest ~ time_period)


################################################################################

controls <- c("female", #"hispanic", "black", "r_asoth", "other_race",
              "age", #"agesq", 
              "hhinc_2", "hhinc_3", "hhinc_4", 
              "education_2", "education_3", "education_4", 
              "education_5", "education_6")

###################################################################
# 2. Histogram of elicited wage expectation conditional on pasts wage and job expectation (OO2new-OO2e2dk ~ L10-11)
###################################################################
# There are some issues with the histogram which I think has to do with the binning...left as is below for now

print("Plots of RESERVATION WAGE versus latest, current wage")

# Create fweight variable (rounded weights)
data_fig1 <- data_13_24 %>%
  mutate(fweight = round(weight, 1))

res_wage_prop1 <- data_fig1 %>% 
  ggplot() +
  geom_histogram(
    aes(x = res_wage_to_current, weight = fweight, fill = "Reservation Wage to Current (Employed"), fill = "blue",
  ) +
  labs(
    title = "Reservation Wage as proportion of Current Wage",
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
    title = "Reservation Wage as proportion of Latest Held Wage",
    x = "Reservation Wage / Latest Held Wage"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )  +
  xlim(0.25, 2)

print(res_wage_prop1 / res_wage_prop2)


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
  #filter(!is.na(udur_bins)) %>% 
  filter(res_wage_to_latest >=0.25 & res_wage_to_latest <= 2.5 & udur <= 60) %>% 
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
  filter(res_wage_to_latest >=0.25 & res_wage_to_latest <=2.5 & udur <= 60) %>% 
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
  fill(udur_bins, .direction = "down") %>% ungroup %>% 
  filter(salary_prop_reswage >= 0.25 & salary_prop_reswage <= 2.5 & udur <= 60) %>% 
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
  filter(salary_prop_reswage >= 0.25 & salary_prop_reswage <= 2.5& udur <= 60) %>% 
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
  filter(salary_prop_reswage >= 0.25 & salary_prop_reswage <= 2.5& udur <= 60) %>% 
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
  filter(salary_prop_reswage >= 0.25 & salary_prop_reswage <= 2.5& udur <= 60) %>% 
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


####### ACCEPTED TO LATEST #########

data_cleaned <- unemp_only %>% 
  group_by(userid) %>% 
  #fill(udur_bins, .direction = "down") %>% 
  summarise(salary_prop_effreswage = last(na.omit(salary_prop_effreswage)),
            salary_to_latest = last(na.omit(salary_to_latest)),
            salary_prop_reswage = last(na.omit(salary_prop_reswage)),
            udur_bins = last(na.omit(udur_bins)),
            udur = last(na.omit(udur)),
            weight = last(na.omit(weight)),
            across(all_of(controls), ~last(na.omit(.)))) %>% 
  ungroup %>% 
  mutate(udur_bins = ifelse(is.na(udur_bins), 0, udur_bins))

# Base theme tweaks
pretty_theme <- theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title.x = element_text(margin = margin(t = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

t1 <- data_cleaned %>% 
  filter(!is.na(salary_to_latest) & between(salary_to_latest, 0.25, 2) & udur <= 60) %>%
  ggplot(aes(x = as.factor(udur_bins), y = salary_to_latest)) +
  geom_boxplot(aes(weight = weight), fill = "lavender", outlier.shape = NA) +
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Accepted : Latest Salary\n(n = 56)", y = NULL, x = NULL) +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos", "7-12 mos", ">12 mos")) +
  coord_cartesian(ylim = c(0, 2.5)) +
  pretty_theme

t2 <- data_cleaned %>% 
  filter(!is.na(salary_prop_reswage) & between(salary_prop_reswage, 0.25, 2) & udur <= 60) %>%
  ggplot(aes(x = as.factor(udur_bins), y = salary_prop_reswage)) +
  geom_boxplot(aes(weight = weight), fill = "skyblue", outlier.shape = NA, alpha = 0.4) +
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Accepted : Reservation\n(n = 160)", y = NULL, x = "Unemployment Duration Bin") +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos", "7-12 mos", ">12 mos")) +
  coord_cartesian(ylim = c(0, 2.5)) +
  pretty_theme

t3 <- data_cleaned %>% 
  filter(!is.na(salary_prop_effreswage) & between(salary_prop_effreswage, 0.25, 2) & udur <= 60) %>%
  ggplot(aes(x = as.factor(udur_bins), y = salary_prop_effreswage)) +
  geom_boxplot(aes(weight = weight), fill = "orange", outlier.shape = NA, alpha = 0.4) +
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Accepted : Effective Reservation\n(n = 184)", y = NULL, x = NULL) +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos", "7-12 mos", ">12 mos")) +
  coord_cartesian(ylim = c(0, 2.5)) +
  pretty_theme

# Combine plots with annotation
final_plot <- (t1 | t2 | t3) + 
  plot_annotation(
    title = "Ratio of Accepted Salary to Reservation Wages by Unemployment Duration",
    caption = "Notes: Data from the Survey of Consumer Expectations Labour Market Survey (2014–2022).\nOnly valid responses used. Boxplots are weighted by SCE sample weights.",
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      plot.caption = element_text(size = 10, hjust = 0.5, margin = margin(t = 10))
    )
  )

print(final_plot)

mod1a <- lm(data = filter(data_cleaned, !is.na(salary_to_latest) & salary_to_latest < 2 & salary_to_latest > 0.25 & udur <= 60), salary_to_latest ~ udur_bins, weights = weight)
mod2a <- lm(data = filter(data_cleaned, !is.na(salary_prop_reswage) & salary_prop_reswage < 2 & salary_prop_reswage > 0.25 & udur <= 60), salary_prop_reswage ~ udur_bins, weights = weight)
mod3a <- lm(data = filter(data_cleaned, !is.na(salary_prop_effreswage) & salary_prop_effreswage < 2 & salary_prop_effreswage > 0.25 & udur <= 60), salary_prop_effreswage ~ udur_bins, weights = weight)
mod1b <- lm(data = filter(data_cleaned,!is.na(salary_to_latest) & salary_to_latest < 2 & salary_to_latest > 0.25 & udur <= 60), paste0("salary_to_latest ~ udur_bins  +", paste0(controls, collapse = "+")), weights = weight)
mod2b <- lm(data = filter(data_cleaned,!is.na(salary_prop_reswage) & salary_prop_reswage < 2 & salary_prop_reswage > 0.25 & udur <= 60), paste0("salary_prop_reswage ~ udur_bins  +", paste0(controls, collapse = "+")), weights = weight)
mod3b <- lm(data = filter(data_cleaned,!is.na(salary_prop_effreswage) & salary_prop_effreswage < 2 & salary_prop_effreswage > 0.25 & udur <= 60), paste0("salary_prop_effreswage ~ udur_bins  +", paste0(controls, collapse = "+")), weights = weight)

modelsummary(list(
  "Accpt:Latest" =mod1a, "AccptWage w.c" =mod1b,
  "Accpt:ResWage" =mod2a, "AccptWage:ResWage w.c" =mod2b,
  "Accpt:EffResWage" =mod3a, "AccptWage:EffResWage w.c" =mod3b), output = "markdown",
  stars = TRUE, coef_omit = c(3:12), gof_omit = c("AIC|BIC|Log.Lik.|R2 Adj.|F"), title = "Accepted Wages and Unemployment Duration") %>% print(.)

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

