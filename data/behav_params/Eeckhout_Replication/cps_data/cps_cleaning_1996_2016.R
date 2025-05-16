# Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 12.0 [dataset]. Minneapolis, MN: IPUMS, 2024.
# https://doi.org/10.18128/D030.V12.0

library(ipumsr)
library(tidyverse)
library(janitor)
library(here)
library(zoo)
library(haven)
ddi_17 <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00009.xml"))
data_17 <- read_ipums_micro(ddi_17) %>% 
  clean_names %>% filter(year == 2017 & month == 1)

ddi <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00010.xml"))
data <- read_ipums_micro(ddi) %>% 
  clean_names %>% 
  rbind(data_17)
ipums_conditions()
rm(ddi_17, data_17)

# Initialize an empty dataframe to store results
all_rates <- data.frame()

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
    
    # SECTION 2: Calculate labor market flows
    df <- df %>%
      group_by(cpsidp) %>%
      mutate(
        diffEU = (empstat[1] %in% c(10, 12) & empstat[2] %in% c(21, 22)),
        diffUE = (empstat[1] %in% c(21, 22) & empstat[2] %in% c(10, 12)),
        diffEE = (empstat[1] %in% c(10, 12) & empstat[2] %in% c(10, 12)),
        diffEEm = diffEE & empsame[2] == 1
      ) %>%
      ungroup()
    
    # Aggregate weighted totals
    Etotal <- sum(df$wtfinl[df$month == 1 & df$empstat %in% c(10, 12)], na.rm = TRUE)
    Utotal <- sum(df$wtfinl[df$month == 1 & df$empstat %in% c(21, 22)], na.rm = TRUE)
    LFtotal <- sum(df$wtfinl[df$month == 1 & df$labforce == 2], na.rm = TRUE)
    NILFtotal <- sum(df$wtfinl[df$month == 1 & df$labforce == 1], na.rm = TRUE)
    
    # Flow calculations
    EUtotal <- sum(df$wtfinl[df$diffEU & df$month == 1], na.rm = TRUE)
    UEtotal <- sum(df$wtfinl[df$diffUE & df$month == 1], na.rm = TRUE)
    EEtotal <- sum(df$wtfinl[df$diffEEm & df$month == 1], na.rm = TRUE)
    
    ENtotal <- sum(df$wtfinl[df$month == 1 & df$empstat %in% c(10, 12) & df$labforce[2] == 1], na.rm = TRUE)
    UNtotal <- sum(df$wtfinl[df$month == 1 & df$empstat %in% c(21, 22) & df$labforce[2] == 1], na.rm = TRUE)
    NEtotal <- sum(df$wtfinl[df$month == 1 & df$labforce[1] == 1 & df$empstat[2] %in% c(10, 12)], na.rm = TRUE)
    NUtotal <- sum(df$wtfinl[df$month == 1 & df$labforce[1] == 1 & df$empstat[2] %in% c(21, 22)], na.rm = TRUE)
    
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
    #saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds'))

  if(i == 2016 & j == 12){break
    }
  }
}
print(all_rates)

#saveRDS(all_rates, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds'))

######### Plotting
# Generate time variable (monthly sequence from 1996m1 onwards)
all_rates_new <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds')) %>%
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

#saveRDS(all_rates_new, here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16_full.rds'))

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


