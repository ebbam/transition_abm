# Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 12.0 [dataset]. Minneapolis, MN: IPUMS, 2024.
# https://doi.org/10.18128/D030.V12.0
library(ipumsr)
library(tidyverse)
library(janitor)
library(here)
library(zoo)
library(haven)
library(assertthat)
#ddi <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00009.xml"))
ddi <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00013.xml"))
data <- read_ipums_micro(ddi) %>% 
  clean_names
ipums_conditions()

# Initialize an empty dataframe to store results
all_rates <- data.frame()
# For use when wanting to bind with data from 96-16
#all_rates <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds'))
# For use when the below loop requires to be broken up
#all_rates <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds'))

# Loop through years and months
for (i in 2017:2024) {
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
    
    assert_that(df %>% select(cpsidp, wtfinl) %>% group_by(cpsidp, wtfinl) %>% n_groups == nrow(df))
    
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
        .groups = 'drop'
      ) %>%
      mutate(
        diffEU = first_emp %in% c(10, 12) & last_emp %in% c(21, 22),
        diffUE = first_emp %in% c(21, 22) & last_emp %in% c(10, 12),
        diffEE = first_emp %in% c(10, 12) & last_emp %in% c(10, 12),
        diffEEm = diffEE & last_same == 1
      ) %>%
      ungroup()
    
    # Aggregate weighted totals
    Etotal <- df %>% 
      filter(first_emp %in% c(10,12)) %>% 
      summarise(Etotal = sum(wtfinl, na.rm = TRUE)) %>% pull()
      #sum(df$wtfinl[df$month == 1 & df$empstat %in% c(10, 12)], na.rm = TRUE)
    Utotal <- df %>% 
      filter(first_emp %in% c(21,22)) %>% 
      summarise(Utotal = sum(wtfinl, na.rm = TRUE))%>% pull()
      # sum(df$wtfinl[df$month == 1 & df$empstat %in% c(21, 22)], na.rm = TRUE)
    LFtotal <- df %>% 
      filter(first_lf == 2) %>% 
      summarise(LFtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
             #sum(df$wtfinl[df$month == 1 & df$labforce == 2], na.rm = TRUE)
    NILFtotal <- df %>% 
      filter(first_lf == 1) %>% 
      summarise(NILFtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
      #sum(df$wtfinl[df$month == 1 & df$labforce == 1], na.rm = TRUE)
    
    # Flow calculations
    EUtotal <- df %>% 
      filter(diffEU) %>% 
      summarise(EUtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
      #sum(df$wtfinl[df$diffEU & df$month == 1], na.rm = TRUE)df
    UEtotal <- df %>% 
      filter(diffUE) %>% 
      summarise(UEtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
      #sum(df$wtfinl[df$diffUE & df$month == 1], na.rm = TRUE)
    EEtotal <- df %>% 
      filter(diffEEm) %>% 
      summarise(EEtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
      #sum(df$wtfinl[df$diffEEm & df$month == 1], na.rm = TRUE)
    
    ENtotal <- df %>% 
      filter(first_emp %in% c(10,12) & second_lf == 1) %>% 
      summarise(ENtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
      #sum(df$wtfinl[df$month == 1 & df$empstat %in% c(10, 12) & df$labforce[2] == 1], na.rm = TRUE)
    UNtotal <- df %>% 
      filter(first_emp %in% c(21,22) & second_lf == 1) %>% 
      summarise(UNtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
      #sum(df$wtfinl[df$month == 1 & df$empstat %in% c(21, 22) & df$labforce[2] == 1], na.rm = TRUE)
    NEtotal <- df %>% 
      filter(first_lf == 1 & last_emp %in% c(10,12)) %>% 
      summarise(NEtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
       #sum(df$wtfinl[df$month == 1 & df$labforce[1] == 1 & df$empstat[2] %in% c(10, 12)], na.rm = TRUE)
    NUtotal <- df %>% 
      filter(first_lf == 1 & last_emp %in% c(21,22)) %>% 
      summarise(NUtotal = sum(wtfinl, na.rm = TRUE))%>% pull()
     #sum(df$wtfinl[df$month == 1 & df$labforce[1] == 1 & df$empstat[2] %in% c(21, 22)], na.rm = TRUE)
    
    # Generate rates
    eu <- EUtotal / Etotal
    ue <- UEtotal / Utotal
    ee <- EEtotal / Etotal
    en <- ENtotal / Etotal
    un <- UNtotal / Utotal
    ne <- NEtotal / NILFtotal
    nu <- NUtotal / NILFtotal
    u <- Utotal / LFtotal
    
    # Create a row for the current loop iteration
    new_row <- data.frame(
      date = paste0(i, j), ue, eu, ee, en, un, ne, nu, u,
      Etotal, Utotal, LFtotal, NILFtotal, EEtotal, ENtotal,
      EUtotal, UEtotal, UNtotal, NEtotal, NUtotal
    )
    
    # Append new data using rbind
    all_rates <- rbind(all_rates, new_row)
    #saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds'))
    if(i == 2024 & j == 11){
      print("stopping...")
      break
    }
  }
}

#saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_23.rds'))



######### Plotting
# Generate time variable (monthly sequence from 1996m1 onwards)
all_rates_new <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds')) %>%
  mutate(time = as.yearmon("1996-01") + (row_number() - 1) / 12)

# Normalize flows by Labor Force
all_rates_new <- all_rates_new %>%
  mutate(
    EE = EEtotal / LFtotal,
    EU = EUtotal / LFtotal,
    EN = ENtotal / LFtotal,
    UE = UEtotal / LFtotal,
    UN = UNtotal / LFtotal,
    NE = NEtotal / LFtotal,
    NU = NUtotal / LFtotal
  )

#saveRDS(all_rates_new, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24_full.rds'))

# Plotting time series
ggplot(all_rates_new, aes(x = time)) +
  geom_line(aes(y = EU, color = "EU")) +
  labs(title = "EU Flow Rates (1996m1 to 2016m9)", y = "EU Rate", x = "Time") +
  theme_minimal()

ggplot(all_rates_new, aes(x = time)) +
  geom_line(aes(y = UE, color = "UE")) +
  labs(title = "UE Flow Rates (1996m1 to 2016m9)", y = "UE Rate", x = "Time") +
  theme_minimal()

ggplot(all_rates_new, aes(x = time)) +
  geom_line(aes(y = EE, color = "EE")) +
  labs(title = "EE Flow Rates (1996m1 to 2016m9)", y = "EE Rate", x = "Time") +
  theme_minimal()

ggplot(all_rates_new, aes(x = time)) +
  geom_line(aes(y = EU, color = "EU")) +
  geom_line(aes(y = EE, color = "EE")) +
  labs(title = "EE and EU Flow Rates", y = "Rates", x = "Time", color = "Flow Type") +
  theme_minimal()

test <- read_dta(here("data/behav_params/Eeckhout_Replication/116422-V1/data/All_rates_and_flows/allrates.dta")) %>% 
  mutate(time = as.yearmon("1996-01") + (row_number() - 1) / 12) 

# Plotting time series
ggplot(all_rates_new, aes(x = time)) +
  geom_line(aes(y = EU, color = "EU")) +
  geom_line(aes(y = UE, color = "UE")) +
  geom_line(aes(y = EE, color = "EE")) +
  geom_line(data = test, aes(x = time, y = eu))+
  labs(title = "EE Flow Rates (1996m1 to 2016m9)", y = "EE Rate", x = "Time") +
  theme_minimal()

ggplot(all_rates_new, aes(x = time)) +
  geom_line(aes(y = EU, color = "EU")) +
  geom_line(aes(y = EE, color = "EE")) +
  labs(title = "EE and EU Flow Rates", y = "Rates", x = "Time", color = "Flow Type") +
  theme_minimal()


