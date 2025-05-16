# Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 12.0 [dataset]. Minneapolis, MN: IPUMS, 2024.
# https://doi.org/10.18128/D030.V12.0
library(ipumsr)
library(tidyverse)
library(janitor)
library(here)
library(zoo)
library(haven)
library(assertthat)
library(patchwork)
#ddi <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00009.xml"))
ddi <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00014.xml"))
data1 <- read_ipums_micro(ddi) %>% 
  clean_names

data <- data1 %>% 
  zap_labels() %>% 
  mutate(age_bracket = case_when(age < 16 ~ NA, 
                                 age >= 16 & age <= 24 ~ "16-24",
                                 age >= 25 & age <= 54 ~ "25-54",
                                 age >= 55 ~ "55+"),
         age_bracket_det = case_when(age < 16 ~ NA, 
                                 age >= 16 & age <= 24 ~ "16-24",
                                 age >= 25 & age <= 39 ~ "25-39",
                                 age >= 40 & age <= 54 ~ "40-54",
                                 age >= 55 & age <= 65 ~ "55-65",
                                 age > 65 ~ "66+"),
         sex = as.character(sex)) %>% 
  mutate(occ = ifelse(occ == 0, NA, as.character(occ)),
         occ2010 = ifelse(occ2010 == 9999, NA, as.character(occ2010)))


ipums_conditions()

# Initialize an empty dataframe to store results
all_rates <- data.frame()
# For use when wanting to bind with data from 96-16
#all_rates <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds'))
# For use when the below loop requires to be broken up
#all_rates <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds'))

# Loop through years and months
for (i in 1996:2016) {
  print(i)
  data_short <- data %>% 
    filter(year == i | (year == i + 1 & month == 1))
  for (j in unique(data$month)) {
    print(j)
    # STEP 1.2: Filter data for the current year and two adjacent months
    # STEP 1.3: Recode month labels
    if(j >= 1 & j <= 11){
      df <- data_short %>%
        filter(year == i & (month == j | month == (j + 1))) %>% 
        mutate(month = ifelse(month == j, 1, ifelse(month == (j + 1), 2, NA)))
    }else if(j == 12){
      df <- data_short %>%
        filter((year == i & month == 12) | (year == i+1 & month == 1)) %>% 
        mutate(month = ifelse(month == 12, 1, ifelse(month == 1, 2, NA)))
    }
    
    # STEP 1.4: Sort data by cpsidp
    df <- df %>% arrange(cpsidp)
    
    # STEP 1.5: Drop observations not in the labor force
    df <- df %>% filter(labforce != 0)
    
    # STEP 1.6: Keep only individuals present in both months
    df <- df %>%
      group_by(cpsidp) %>%
      filter(n() == 2) %>%
      ungroup()
    
    #assert_that(df %>% select(cpsidp, wtfinl) %>% group_by(cpsidp, wtfinl) %>% n_groups == nrow(df))
    
    # SECTION 2: Calculate labor market flows
    df <- df %>%
      arrange(cpsidp, year, month) %>%
      group_by(cpsidp) %>% 
      summarise(
        first_emp = first(empstat),
        last_emp = last(empstat),
        last_same = last(empsame),
        first_lf = first(labforce),
        second_lf = last(labforce),
        wtfinl = first(wtfinl),
        sex = first(sex),
        age = first(age_bracket),
        #occ = first(occ),
        occ2010 = first(occ2010),
        .groups = 'drop'
      ) %>%
      mutate(
        diffEU = first_emp %in% c(10, 12) & last_emp %in% c(21, 22),
        diffUE = first_emp %in% c(21, 22) & last_emp %in% c(10, 12),
        diffEE = first_emp %in% c(10, 12) & last_emp %in% c(10, 12),
        diffEEm = diffEE & last_same == 1
      ) %>%
      ungroup()
    
    group_vars <- c("sex", "age", "occ2010")
    for (group_var in group_vars) {
      print(group_var)
      flows_by_group <- df %>%
        filter(!is.na(.data[[group_var]])) %>%
        group_by(.data[[group_var]])  %>%
        summarise(
          group_var = group_var,  # record which variable this is
          group_value = first(.data[[group_var]]),  # value within the group
          Etotal = sum(wtfinl[first_emp %in% c(10,12)], na.rm = TRUE),
          Utotal = sum(wtfinl[first_emp %in% c(21,22)], na.rm = TRUE),
          LFtotal = sum(wtfinl[first_lf == 2], na.rm = TRUE),
          NILFtotal = sum(wtfinl[first_lf == 1], na.rm = TRUE),
          EUtotal = sum(wtfinl[diffEU], na.rm = TRUE),
          UEtotal = sum(wtfinl[diffUE], na.rm = TRUE),
          EEtotal = sum(wtfinl[diffEEm], na.rm = TRUE),
          ENtotal = sum(wtfinl[first_emp %in% c(10,12) & second_lf == 1], na.rm = TRUE),
          UNtotal = sum(wtfinl[first_emp %in% c(21,22) & second_lf == 1], na.rm = TRUE),
          NEtotal = sum(wtfinl[first_lf == 1 & last_emp %in% c(10,12)], na.rm = TRUE),
          NUtotal = sum(wtfinl[first_lf == 1 & last_emp %in% c(21,22)], na.rm = TRUE),
          .groups = "drop"
        ) %>%
        mutate(
          ue = UEtotal / Utotal,
          eu = EUtotal / Etotal,
          ee = EEtotal / Etotal,
          en = ENtotal / Etotal,
          un = UNtotal / Utotal,
          ne = NEtotal / NILFtotal,
          nu = NUtotal / NILFtotal,
          u = Utotal / LFtotal,
          date = paste0(i, sprintf("%02d", j))  # e.g., "202304"
        )
      
      # Bind into your cumulative results
      all_rates <- bind_rows(all_rates, flows_by_group)
    }
    # Append new data using rbind
    #all_rates <- rbind(all_rates, new_row)
    #saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds'))
  }
  saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/grouped_transition_rates_99_16.RDS'))
  if(i == 2016 & j == 11){
    print("stopping...")
    break
  }
  gc()
}

#saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_23.rds'))

#saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/grouped_transition_rates_99_16.RDS'))
# 
# ######### Plotting
# # Generate time variable (monthly sequence from 1996m1 onwards)
# all_rates_new <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds')) %>%
#   mutate(time = as.yearmon("1996-01") + (row_number() - 1) / 12)
# 
# # Normalize flows by Labor Force
# all_rates_new <- all_rates_new %>%
#   mutate(
#     EE = EEtotal / LFtotal,
#     EU = EUtotal / LFtotal,
#     EN = ENtotal / LFtotal,
#     UE = UEtotal / LFtotal,
#     UN = UNtotal / LFtotal,
#     NE = NEtotal / LFtotal,
#     NU = NUtotal / LFtotal
#   )

#saveRDS(all_rates_new, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24_full.rds'))
# 
# # Plotting time series
# ggplot(all_rates_new, aes(x = time)) +
#   geom_line(aes(y = EU, color = "EU")) +
#   labs(title = "EU Flow Rates (1996m1 to 2016m9)", y = "EU Rate", x = "Time") +
#   theme_minimal()
# 
# ggplot(all_rates_new, aes(x = time)) +
#   geom_line(aes(y = UE, color = "UE")) +
#   labs(title = "UE Flow Rates (1996m1 to 2016m9)", y = "UE Rate", x = "Time") +
#   theme_minimal()
# 
# ggplot(all_rates_new, aes(x = time)) +
#   geom_line(aes(y = EE, color = "EE")) +
#   labs(title = "EE Flow Rates (1996m1 to 2016m9)", y = "EE Rate", x = "Time") +
#   theme_minimal()
# 
# ggplot(all_rates_new, aes(x = time)) +
#   geom_line(aes(y = EU, color = "EU")) +
#   geom_line(aes(y = EE, color = "EE")) +
#   labs(title = "EE and EU Flow Rates", y = "Rates", x = "Time", color = "Flow Type") +
#   theme_minimal()
# 
# test <- read_dta(here("data/behav_params/Eeckhout_Replication/116422-V1/data/All_rates_and_flows/allrates.dta")) %>% 
#   mutate(time = as.yearmon("1996-01") + (row_number() - 1) / 12) 

# # Plotting time series
# ggplot(all_rates_new, aes(x = time)) +
#   geom_line(aes(y = EU, color = "EU")) +
#   geom_line(aes(y = UE, color = "UE")) +
#   geom_line(aes(y = EE, color = "EE")) +
#   geom_line(data = test, aes(x = time, y = eu))+
#   labs(title = "EE Flow Rates (1996m1 to 2016m9)", y = "EE Rate", x = "Time") +
#   theme_minimal()
# 
# ggplot(all_rates_new, aes(x = time)) +
#   geom_line(aes(y = EU, color = "EU")) +
#   geom_line(aes(y = EE, color = "EE")) +
#   labs(title = "EE and EU Flow Rates", y = "Rates", x = "Time", color = "Flow Type") +
#   theme_minimal()
# 
all_rates <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/grouped_transition_rates_99_16.RDS')) %>% 
  rbind(readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/grouped_transition_rates_99_24.RDS')))


p1 <- all_rates %>%
  mutate(date = as.Date(paste0(date, "01"), format = "%Y%m%d")) %>% 
  filter(group_var == "age") %>%
  ggplot(aes(x = date, y = ee, group = group_value, color = group_value)) +
  geom_line() +
  labs(color = "Age")


p2 <- all_rates %>%
  mutate(date = as.Date(paste0(date, "01"), format = "%Y%m%d")) %>% 
  filter(group_var == "sex") %>%
  mutate(group_value = ifelse(group_value == 2, "Female", "Male")) %>% 
  ggplot(aes(x = date, y = ee, group = group_value, color = group_value)) +
  geom_line() + 
  labs(color = "Gender")

all_rates %>%
  filter(group_var == "occ2010") %>%
  ggplot(aes(x = date, y = ee, group = group_value, color = group_value)) +
  geom_line() +
  theme(legend.position = "below") + 
  labs(color = "Occupation")

p1 + p2

temp <- all_rates %>% 
  filter(group_var == "occ2010") %>% 
  mutate(year = substr(date, 1,4), 
         month = substr(date, 5, 6), 
         qtr = case_when(month >= 1 & month < 4 ~ 1, 
                         month >= 4 & month < 7 ~ 2,
                         month >= 7 & month < 10 ~ 3,
                         month >= 10 & month <= 12 ~ 4)) %>% 
  group_by(year, occ2010) %>% 
  summarise(EEtotal = sum(EEtotal, na.rm = TRUE),
            Etotal = sum(Etotal, na.rm = TRUE),
            UEtotal = sum(UEtotal, na.rm = TRUE),
            Utotal = sum(Utotal, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ee = EEtotal/Etotal,
         ue = UEtotal/Utotal) %>% 
  mutate(across(!c(year, occ2010), ~ifelse(. == 0, NA, .))) %>% 
  group_by(occ2010) %>%
  arrange(year, .by_group = TRUE) %>%
  mutate(
    ee = if (sum(!is.na(ee)) >= 2) approx(x = year, y = ee, xout = year, rule = 2)$y else ee,
    ue = if (sum(!is.na(ue)) >= 2) approx(x = year, y = ue, xout = year, rule = 2)$y else ue
  ) %>%
  ungroup()
  
temp %>% filter(nchar(occ2010) == 2) %>% 
  ggplot(aes(x = year, y = ee, group = occ2010, color = occ2010)) + 
  geom_line() + 
  theme(legend.position = "none") +
  ylim(0, 0.05)


# Total Economy Rates
all_rates_new <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds')) %>% 
  rbind(readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds'))) %>% 
      mutate(
        EE = EEtotal / LFtotal,
        EU = EUtotal / LFtotal,
        EN = ENtotal / LFtotal,
        UE = UEtotal / LFtotal,
        UN = UNtotal / LFtotal,
        NE = NEtotal / LFtotal,
        NU = NUtotal / LFtotal,
        date = ifelse(nchar(date) == 5, 
                     paste0(substr(date, 1, 4), "-0", substr(date, 5, 5), "-01"), 
                    paste0(substr(date, 1, 4), "-", substr(date, 5, 6), "-01"))
      ) %>% tibble

#write.csv(all_rates_new, here('data/behav_params/Eeckhout_Replication/cps_data/transition_rates_96_24.csv'))

# Plotting time series
ggplot(all_rates_new, aes(x = as.Date(date))) +
  geom_line(aes(y = EU, color = "EU")) +
  labs(title = "EU Flow Rates (1996m1 to 2016m9)", y = "EU Rate", x = "Time") +
  theme_minimal()

ggplot(all_rates_new, aes(x = as.Date(date))) +
  geom_line(aes(y = UE, color = "UE")) +
  labs(title = "UE Flow Rates (1996m1 to 2016m9)", y = "UE Rate", x = "Time") +
  theme_minimal()

ggplot(all_rates_new, aes(x = as.Date(date))) +
  geom_line(aes(y = EE, color = "EE")) +
  labs(title = "EE Flow Rates (1996m1 to 2016m9)", y = "EE Rate", x = "Time") +
  theme_minimal()

ggplot(all_rates_new, aes(x = as.Date(date))) +
  geom_line(aes(y = EU, color = "EU")) +
  geom_line(aes(y = EE, color = "EE")) +
  labs(title = "EE and EU Flow Rates", y = "Rates", x = "Time", color = "Flow Type") +
  theme_minimal()



