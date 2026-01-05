## Eeckhout Replication
#rm(list = ls())
library(readxl)
library(tidyverse)
library(ggplot2)
library(mFilter)
library(zoo)
library(seasonal)
library(openxlsx)
library(haven)
library(here)
library(lubridate)
library(gridExtra)
library(modelsummary)
source(here('code/formatting/plot_dicts.R'))

integrate = TRUE
full = TRUE

################################################################################
################################################################################
##### drop_box_code.do
################################################################################
################################################################################
base <- here("data/behav_params/Eeckhout_Replication/116422-V1/data/All_rates_and_flows")


################################################################################
################################################################################
##### decom_EE_overlap_months_more_obs.do
################################################################################
################################################################################

# base <- here("data/behav_params/Eeckhout_Replication/116422-V1/data/EE_x")
# 
# # Load the data
# df <- read_dta("XXX.dta")  # Replace XXX with your actual file name
# 
# # Loop through years from 2006 to 2022
# for (i in 2006:2022) {
#   
#   # Filter the dataset for the given year
#   df_filtered <- df %>% filter(Year == i)
#   
#   # Loop through months from 1 to 12
#   for (j in 1:12) {
#     
#     # Adjust the Month column according to Stata's logic
#     df_adjusted <- df_filtered %>%
#       mutate(Month = case_when(
#         Month == j ~ 1,
#         Month == ((j + 1) %% 12) ~ 2,
#         Month == ((j + 2) %% 12) ~ 3,
#         Month == ((j + 3) %% 12) ~ 4,
#         Month == ((j + 4) %% 12) ~ 5,
#         Month == ((j + 5) %% 12) ~ 6,
#         Month == ((j + 6) %% 12) ~ 7,
#         Month == ((j + 7) %% 12) ~ 8,
#         Month == ((j + 8) %% 12) ~ 9,
#         Month == ((j + 9) %% 12) ~ 10,
#         Month == ((j + 10) %% 12) ~ 11,
#         Month == ((j + 11) %% 12) ~ 12,
#         TRUE ~ NA_real_  # To handle cases that don't match
#       )) %>%
#       filter(!is.na(Month))  # Remove rows where Month didn't match the criteria
#     
#     # Construct the file name
#     #file_name <- paste0("XXX_", i, "_", j, ".dta")  # Replace XXX with your desired output prefix
#     
#     # Save the modified dataset
#     #write_dta(df_adjusted, file_name)
#   }
# }
# 



################################################################################
################################################################################
##### quart_from_instantaneous.do
################################################################################
################################################################################

base <- here("data/behav_params/Eeckhout_Replication/116422-V1/data/")

# Load datasets
#allrates <- read_dta(here(paste0(base, "All_rates_and_flows/allrates.dta")))
allrates <- readRDS(here("data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16_full.rds"))
allrates_new <- readRDS(here("data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24_full.rds"))
if(full){
  allrates <- allrates %>% 
    rbind(allrates_new)
}

if(integrate){
  EE_x <- readRDS(here("data/behav_params/Eeckhout_Replication/EE_x_final.rds")) 
  # Drop unnecessary columns
  EE_x <- EE_x %>% 
    #select(-c(cpsidp, earnwt, previous_wage, new_wage, wtfinl_1)) %>% 
    filter(row_number() != 1) %>% 
    mutate(date = ifelse(substr(date, 5, 5) == 0, paste0(substr(date, 1, 4), substr(date, 6,6)), date))
}else{
  EE_x_old <- read_dta(here(paste0(base, "EE_x/EE_x.dta")))
  
  EE_x_old <- EE_x_old %>% 
  select(-c(cpsidp, earnwt, previous_wage, new_wage, wtfinl_1)) %>% 
  filter(row_number() != 1) %>% 
  mutate(date = as.character(date))
  }
  
# Merge datasets
final_data <- left_join(allrates, EE_x, by = "date")

# Sort and create new variables
final_data <- final_data %>% 
  mutate(date = as.Date(paste0(substr(date, 1, 4), "-", substr(date, 5, nchar(date)), "-01"))) %>% 
  arrange(time) %>%
  mutate(eu_ta_proxy = eu + 0.005)

# Filter based on time condition
final_data <- final_data %>% 
  #filter(date <= ymd("2016-09-01")) %>% 
  rename(time_x = time,
         time = date)

# Load additional datasets
#EE_wages <- read_dta(here(paste0(base, "Wages/EE/EE_wages.dta")))
#UE_wages <- read_dta(here(paste0(base, "Wages/UE/UE_wages.dta")))

#eu_ta <- read.xlsx(here(paste0(base, "Quarterly_rates_from_monthly/eu_ta.xlsx")), sheet = "Sheet1", colNames = FALSE) %>% 
#  rename(A = X1)

# Generate time variable
#EE_wages <- EE_wages %>% mutate(time = seq(ymd("1996-01-01"), by = "month", length.out = n())) %>% select(avg_wage_EE, time)
#UE_wages <- UE_wages %>% mutate(time = seq(ymd("1996-01-01"), by = "month", length.out = n())) %>% select(avg_wage_UE, time)
#eu_ta <- eu_ta %>% mutate(time = seq(ymd("1996-01-01"), by = "month", length.out = n())) %>% rename(eu_ta = A)

# Merge datasets
merged_data <- final_data #%>% 
  #left_join(EE_wages, by = "time") %>% 
  # left_join(UE_wages, by = "time") %>% 
  # left_join(eu_ta, by = "time")

# Compute instantaneous rates
merged_data <- merged_data %>% 
  mutate(
    UE_inst = -log(1 - UE),
    EE_inst = -log(1 - EE),
    EU_inst = -log(1 - EU),
    ue_inst = -log(1 - ue),
    ee_inst = -log(1 - ee),
    eu_inst = -log(1 - eu),
    #eu_ta_inst = -log(1 - eu_ta),
    UE_EE_total = UEtotal + EEtotal,
    ue_ee_s = UE_EE_total / LFtotal,
    ue_ee_inst = -log(1 - ue_ee_s)
  )

# Save instantaneous rates
data_to_save <- merged_data %>% select(UE_inst, EE_inst, EU_inst,# ue_inst, eu_inst, 
                                       ee_inst, per_more_x_5, u, time, 
                                       #avg_wage_UE, avg_wage_EE, 
                                       #eu_ta_inst, 
                                       ue_ee_inst)

# Compute quarterly rates
merged_data <- merged_data %>% 
  mutate(
    EU_quart = 1 - exp(-3 * EU_inst),
    EE_quart = 1 - exp(-3 * EE_inst),
    UE_quart = 1 - exp(-3 * UE_inst),
    ue_quart = 1 - exp(-3 * ue_inst),
    ee_quart = 1 - exp(-3 * ee_inst),
    eu_quart = 1 - exp(-3 * eu_inst),
    #eu_ta_quart = 1 - exp(-3 * eu_ta_inst),
    ue_ee_quart = 1 - exp(-3 * ue_ee_inst)
  )

# Generate quarterly time variable
merged_data <- merged_data %>% mutate(quarter = quarter(time), year = year(time), dq = paste0(year, "Q", quarter))

# Collapse by quarter
quarterly_rates <- merged_data %>% 
  group_by(dq) %>% 
  summarise(
    across(c(eu_quart, ee_quart, ue_quart, UE_quart, EE_quart, EU_quart, per_more_x_5, u, 
             #avg_wage_UE, avg_wage_EE, 
             #eu_ta_quart, 
             ue_ee_quart), mean, na.rm = TRUE)
  ) %>% 
  mutate(EE_x_5_quart = EE_quart * per_more_x_5)



################################################################################
################################################################################
##### raw_to_SA.m
################################################################################
################################################################################

base <- here("data/behav_params/Eeckhout_Replication/116422-V1/data/Seasonal_adjustment/")

if(integrate){
  raw_data <- quarterly_rates
  #print('Using calculated quarterly rates.')
}else{
  #raw_data <- read_excel(here(paste0(base, "Quarterly_raw_data.xls")), sheet = "Sheet1", range = "A2:N84", col_names = FALSE)
                         #print('Using saved rates.')
                         }

# Extract time column and numerical data
time <- raw_data[[1]]
data <- as.matrix(raw_data[, 2:11])

# Convert non-numeric values to NA
data[!sapply(data, is.numeric)] <- NA

# Assign column names
colnames(data) <- c("eu_quart", "ee_quart", "ue_quart", "UE_quart", "EE_quart", "EU_quart",
                    "per_more_x_5", "u_quart", #"avg_wage_UE", "avg_wage_EE",
                   # "eu_ta_quart", 
                   "ue_ee_quart", "EE_x_5_quart")

# Convert time to date format
date <- as.yearqtr(time)

# Adjust EE_x_5_quart by shifting values (corresponding to MATLAB indexing)
EE_x_5_quart <- c(rep(NA, 4), data[, "EE_x_5_quart"][-(1:4)])

# Define function for seasonal adjustment
seasonally_adjust <- function(series) {
  series_ts <- ts(series, start = c(as.numeric(format(date[1], "%Y")), as.numeric(format(date[1], "%q"))), frequency = 4)
  sa_result <- seas(series_ts, na.action = na.exclude)
  return(final(sa_result))
}

# Apply seasonal adjustment
flow_rates_sa <- data.frame(
  date = as.yearqtr(date),
  UE_quart_s = seasonally_adjust(data[, "UE_quart"]),
  EU_quart_s = seasonally_adjust(data[, "EU_quart"]),
  EE_quart_s = seasonally_adjust(data[, "EE_quart"]),
  ue_quart_s = seasonally_adjust(data[, "ue_quart"]),
  eu_quart_s = seasonally_adjust(data[, "eu_quart"]),
  ee_quart_s = seasonally_adjust(data[, "ee_quart"]),
  u_quart_s = seasonally_adjust(data[, "u_quart"]),
  #avg_wage_UE_s = seasonally_adjust(data[, "avg_wage_UE"]),
  #avg_wage_EE_s = seasonally_adjust(data[, "avg_wage_EE"]),
  #eu_ta_quart_s = seasonally_adjust(data[, "eu_ta_quart"]),
  ue_ee_quart_s = seasonally_adjust(data[, "ue_ee_quart"]),
  EE_x_5_quart_s = seasonally_adjust(EE_x_5_quart)
)


t1 <- flow_rates_sa %>% 
  select(date, eu_quart_s, ee_quart_s) %>% 
  pivot_longer(!date) %>% 
  ggplot(aes(x = date, y = value, color = name))+
  geom_line()

t2 <- flow_rates_sa %>% 
  select(date, EE_quart_s, EE_x_5_quart_s) %>% 
  pivot_longer(!date) %>% 
  ggplot(aes(x = date, y = value, color = name))+
  geom_line()

t2

################################################################################
################################################################################
##### calcgammaquarterlySA.m
################################################################################
################################################################################

base <- here("data/behav_params/Eeckhout_Replication/116422-V1/data/Gamma_lambda_Mm/")

if(integrate){
  data <- flow_rates_sa
  #print('Using calculated flow rates.')
}else{
  #data <- read_excel(here(paste0(base, "Quarterly_flow_rates_s.xlsx")), sheet = "Sheet1", range = "A2:M84", col_names = NA)
  #print('Using saved flow rates.')
}

# Rename columns for ease of use
colnames(data) <- c("time", "UE_quart_s", "EU_quart_s", "EE_quart_s", "ue_quart_s", "eu_quart_s", "ee_quart_s", "u_quart_s",  
                    #"avg_wage_UE_s", "avg_wage_EE_s", 
                    #"eu_ta_quart_s", 
                    "ue_ee_quart_s", "EE_x_5_quart_s")

# Define time range
startdate <- as.yearqtr("1997Q1")
enddate <- as.yearqtr("2016Q3")
enddatenew <- as.yearqtr("2024Q1")


p_list <- list()
vars <- list()
df_list <- list()
for(new in c(FALSE, TRUE)){
  data_temp <- data
  time_series <- data_temp$time
  time_index_start <- which(time_series == startdate)
  time_index_end <- which(time_series == enddate)
  if(new){
    time_index_end <- which(time_series == enddatenew)
  }
  diff_periods <- (time_index_end - time_index_start + 1)
  
  # Define gamma sequence
  sizegamma <- 20
  gamma_values <- seq(0.2, 0.8, length.out = sizegamma)
  
  grid2 <- matrix(0, nrow = diff_periods + 1, ncol = sizegamma)
  grid2[1, ] <- gamma_values
  
  # Forward iteration for gamma
  # if(new){
  #   data_temp <- data_temp %>%
  #     mutate(EE_x_5_quart_s = EE_quart_s)
  # }
  data_subset <- data_temp[time_index_start:time_index_end, ]
  for (i in 1:sizegamma) {
    for (t in 1:diff_periods) {
      grid2[t + 1, i] <- (max(0, grid2[t, i]) * ((1 - data_subset$u_quart_s[t]) - data_subset$EU_quart_s[t])) / (1 - data_subset$u_quart_s[t]) + data_subset$UE_quart_s[t] - data_subset$EE_x_5_quart_s[t]
      }
  }
  
  # Find the true gamma
  sum_gamma <- rowSums(grid2)
  avg_gamma <- sum_gamma / diff_periods
  diff_gamma <- abs(gamma_values - avg_gamma)
  true_gamma_index <- which.min(diff_gamma)
  true_gamma <- gamma_values[true_gamma_index]
  true_gamma_series <- grid2[-1, true_gamma_index]
  
  # Compute lambda
  lambda <- data_subset$EE_x_5_quart_s / ((data_subset$UE_quart_s / data_subset$u_quart_s) * true_gamma_series)
  
  # # Plot results
  # ggplot() +
  #   geom_line(aes(x = 1:diff_periods, y = lambda), color = "red") +
  #   labs(title = "Search Intensity Lambda", x = "Time", y = "Lambda") +
  #   theme_minimal()
  
  
  ################################################################################
  ################################################################################
  ##### plots.do
  ################################################################################
  ################################################################################
  
  # Load the first dataset
  if(integrate){
    final_gammas <- tibble("A" = true_gamma_series)
    #print('Using calculated gamma.')
  }else{
    #final_gammas <- read_xlsx(here(paste0(base, "final_gammas.xlsx")), sheet = "Sheet1", col_names = c("A"))
    #print('Using saved gamma.')
  }
  final_gammas <- final_gammas %>% mutate(time = as.yearqtr(1997 + (row_number() - 1) / 4))
  if(!new){
    final_gammas <- final_gammas %>% 
      filter(time <= as.yearqtr("2016Q3"))
  }
  
  # Apply HP filter
  final_gammas <- final_gammas %>%
    mutate(truegamma_s = A,
           log_gamma_s = log(A))
  valid_idx <- which(!is.na(final_gammas$truegamma_s))
  filtered_gamma <- hpfilter(final_gammas$truegamma_s[valid_idx], freq = 1600)
  final_gammas$truegamma_st <- NA
  final_gammas$truegamma_sc <- NA
  final_gammas$truegamma_st[valid_idx] = filtered_gamma$trend
  final_gammas$truegamma_sc[valid_idx] = filtered_gamma$cycle
  
  valid_idx <- which(!is.na(final_gammas$log_gamma_s))
  filtered_log_gamma <- hpfilter(final_gammas$log_gamma_s[valid_idx], freq = 1600)
  final_gammas$log_gamma_st <- NA
  final_gammas$log_gamma_sc <- NA
  final_gammas$log_gamma_st[valid_idx] = filtered_log_gamma$trend
  final_gammas$log_gamma_sc[valid_idx] = filtered_log_gamma$cycle
  
  # Load second dataset
  #quarterly_flow_rates <- read_excel(here(paste0(base, "Quarterly_flow_rates_s.xlsx")))
  quarterly_flow_rates <- flow_rates_sa %>%  mutate(time = as.yearqtr(1996 + (row_number() - 1) / 4))
  if(!new){
    quarterly_flow_rates <- quarterly_flow_rates %>% 
      filter(time <= as.yearqtr("2016Q3"))
  }
   
  # Merge datasets
  merged_data <- left_join(quarterly_flow_rates, final_gammas, by = "time")
  
  # Define recessions
  merged_data_old_p1 <- merged_data %>%
    mutate(recession1 = ifelse((time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4")) |
                                 (time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2")), 0.2, NA),
           recession2 = ifelse((time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4")) |
                                 (time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2")), -0.3, NA))
  
  # Plot
  p1_orig <- ggplot(merged_data_old_p1, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "gray80") +
    geom_bar(aes(y = recession2), stat = "identity", fill = "gray80") +
    geom_line(aes(y = log_gamma_sc, color = "Gamma")) +
    geom_line(aes(y = (u_quart_s*8) - 0.6, color = "Unemployment")) +
    scale_y_continuous(sec.axis = sec_axis(~ (./8)+0.6, name = "u (level)", breaks = c(0.04,0.08,0.1))) +
    labs(y = "Gamma (% deviation from trend)", x = "Time", title = paste0("New data: ", new)) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  #print(p1_orig)
  p_list <- append(p_list, list(p1_orig))
  
  # Load the dataset
  # final_SI <- read_excel(here(paste0(base, "final_SI.xlsx")), col_names = c("A"))
  if(integrate){
    final_SI <- tibble("A" = lambda)
    #print('Using calculated lambda.')
  }
  final_SI <- final_SI %>% mutate(time = as.yearqtr(1997 + (row_number() - 1) / 4))
  
  # Apply HP filter
  final_SI <- final_SI %>%
    mutate(SI_s = A,
           log_SI_s = log(A))
  
  valid_idx <- which(!is.na(final_SI$SI_s))
  filtered_SI <- hpfilter(final_SI$SI_s[valid_idx], freq = 1600)
  final_SI$SI_st <- NA
  final_SI$SI_sc <- NA
  final_SI$SI_st[valid_idx] = filtered_SI$trend
  final_SI$SI_sc[valid_idx] = filtered_SI$cycle
  
  valid_idx <- which(!is.na(final_SI$log_SI_s))
  log_filtered_SI <- hpfilter(final_SI$log_SI_s[valid_idx], freq = 1600)
  final_SI$log_SI_st <- NA
  final_SI$log_SI_sc <- NA
  final_SI$log_SI_st[valid_idx] = log_filtered_SI$trend
  final_SI$log_SI_sc[valid_idx] = log_filtered_SI$cycle
  
  # Merge datasets
  merged_data <- left_join(quarterly_flow_rates, final_SI, by = "time")
  
  # Define recessions
  merged_data_old_p2 <- merged_data %>%
    mutate(recession1 = ifelse(time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4") |
                                 time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2"), 0.3, NA),
           recession2 = ifelse(time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4") |
                                 time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2"), -0.24, NA))
  
  # # Plot
  p2_orig <- ggplot(merged_data_old_p2, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "gray80") +
    geom_bar(aes(y = recession2), stat = "identity", fill = "gray80") +
    geom_line(aes(y = log_SI_sc, color = "Lambda")) +
    geom_line(aes(y = (u_quart_s*8)-0.55, color = "Unemployment")) +
    scale_y_continuous(sec.axis = sec_axis(~ (./8)+0.55, name = "u (level)")) +
    labs(y = "Lambda (% deviation from trend)", x = "Time", title = paste0("New data: ", new)) +
    theme_minimal() + 
    theme(legend.position = "bottom")
  
  p_list <- append(p_list, list(p2_orig))
  
  #print(p2_orig)
  
  
  ################################################################################
  # Searcher Composition - taken from composition_analysis.do
  ################################################################################
  
  gamma_df <- #read_excel("final_gammas.xlsx") %>%
    final_gammas #%>% 
  
  if(!new){
    gamma_df <- gamma_df %>% 
      filter(time <= as.yearqtr("2016Q3"))
  }
    #rename(gamma_s = A) %>%
    #mutate(time = yearquarter("1997 Q1") + row_number() - 1)
  
  # Save gamma for later merge
  gamma_temp <- gamma_df
  
  # Load and prepare SI data
  SI_df <- final_SI #%>% #read_excel("final_SI.xlsx") %>%
    #rename(SI_s = A) %>%
    #mutate(time = yearquarter("1997 Q1") + row_number() - 1)
  
  if(!new){
    SI_df <- final_SI %>% 
      filter(time <= as.yearqtr("2016Q3"))
  }
  
  SI_temp <- SI_df
  
  # Load seasonal adjusted unemployment flow data
  #setwd(file.path("..", "Seasonal_adjustment"))
  flow_df <- quarterly_flow_rates #%>% #read_excel("Quarterly_flow_rates_s.xlsx", sheet = "Sheet1") %>%
    #select(-time) %>%
    #mutate(time = yearquarter("1996 Q1") + row_number() - 1)
  
  if(!new){
    flow_df <- flow_df %>% 
      filter(time <= as.yearqtr("2016Q3"))
  }
  
  # Merge datasets
  #setwd(file.path(destinationpath, "composition_analysis"))
  merged_df <- flow_df %>%
    left_join(gamma_temp, by = "time") %>%
    left_join(SI_temp, by = "time") %>%
    mutate(
      lambdagamma_s = truegamma_s * SI_s,
      effective_searchers_s = u_quart_s + lambdagamma_s,
      comp_searchers_s = lambdagamma_s / effective_searchers_s
    )
  
  # Compute deviation from trend (HP filter)
  merged_df <- merged_df %>%
    mutate(log_comp_searchers_s = log(comp_searchers_s))
  
  # Step 1: Identify non-NA indices
  valid_idx <- which(!is.na(merged_df$log_comp_searchers_s))
  
  # Step 2: Apply the HP filter only to non-NA values
  hp_result1 <- hpfilter(merged_df$log_comp_searchers_s[valid_idx], freq = 1600)
  
  # Step 3: Create full-length vectors initialized with NA
  merged_df$log_comp_searchers_sc <- NA
  merged_df$log_comp_searchers_st <- NA
  
  # Step 4: Fill in results at the correct positions
  merged_df$log_comp_searchers_sc[valid_idx] <- hp_result1$cycle
  merged_df$log_comp_searchers_st[valid_idx] <- hp_result1$trend
  
  # Recession indicators
  merged_df <- merged_df %>%
    mutate(
      recession1 = if_else(time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4") |
                             (time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2")),
                           0.2, NA_real_),
      recession2 = if_else(!is.na(recession1), -0.2, NA_real_)
    )
  
  rescale_for_secondary_axis <- function(primary, secondary, primary_limits = NULL) {
    # Remove NA values for calculations
    primary <- na.omit(primary)
    secondary <- na.omit(secondary)
    
    # Use full range if not provided
    if (is.null(primary_limits)) {
      primary_limits <- range(primary, na.rm = TRUE)
    }
    
    secondary_limits <- range(secondary, na.rm = TRUE)
    
    # Calculate scaling factor and offset
    scale_factor <- diff(primary_limits) / diff(secondary_limits)
    offset <- primary_limits[1] - secondary_limits[1] * scale_factor
    
    # Return scaled secondary series and inverse transform for labeling
    list(
      scaled_secondary = function(x) x * scale_factor + offset,
      inverse_transform = function(x) (x - offset) / scale_factor,
      scale_factor = scale_factor,
      offset = offset
    )
  }
  # First plot
  # Apply rescaling
  rescale <- rescale_for_secondary_axis(
    primary = merged_df$log_comp_searchers_sc,
    secondary = merged_df$u_quart_s,
    primary_limits = c(min(merged_df$log_comp_searchers_sc, na.rm = TRUE) -0.001, max(merged_df$log_comp_searchers_sc, na.rm = TRUE) +0.001)  # match Stata range
  )
  
  # Add rescaled column
  merged_df$u_quart_s_rescaled <- rescale$scaled_secondary(merged_df$u_quart_s)
  
  # Plot with independent visual scaling
  p1 <- ggplot(merged_df, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_bar(aes(y = recession2), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_line(aes(y = log_comp_searchers_sc), color = "red") +
    geom_line(aes(y = u_quart_s_rescaled), color = "red", linetype = "dashed") +
    scale_y_continuous(
      name = "% deviation from trend",
      limits = c(min(merged_df$log_comp_searchers_sc, na.rm = TRUE) -0.001, max(merged_df$log_comp_searchers_sc, na.rm = TRUE) +0.001),
      sec.axis = sec_axis(transform = rescale$inverse_transform, name = "u (level)")
    ) +
    theme_minimal() +
    ggtitle(paste0("New data: ", new)) +
    theme(plot.background = element_rect(fill = "white", color = NA), legend.position = "bottom")
  
  #print(p1)
  p_list <- append(p_list, list(p1))
  df_list <- append(df_list, list(merged_df))
  
  
  # First plot
  # Apply rescaling
  rescale <- rescale_for_secondary_axis(
    primary = merged_df$comp_searchers_s,
    secondary = merged_df$u_quart_s,
    primary_limits = c(0.2, 0.6)  # match Stata range
  )
  # Add rescaled column
  merged_df$u_quart_s_rescaled <- rescale$scaled_secondary(merged_df$u_quart_s)
  
  # Plot with independent visual scaling
  comp_searchers_plot <- ggplot(merged_df, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_bar(aes(y = recession2), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_line(aes(y = comp_searchers_s), color = "blue") +
    geom_hline(aes(yintercept = 0.32), color = "grey", linetype = "dashed") + 
    geom_hline(aes(yintercept = 0.48), color = "grey", linetype = "dashed") + 
    geom_line(aes(y = u_quart_s_rescaled), color = "red", linetype = "dashed") +
    scale_y_continuous(
      name = "% deviation from trend",
      limits = c(0.1, 0.6),
      sec.axis = sec_axis(transform = rescale$inverse_transform, name = "Unemployment Rate")
    ) +
    theme_minimal() +
    ggtitle(paste0("Employed as share of jobseekers (lamda*gamma/s) New data: ", new)) +
    theme(plot.background = element_rect(fill = "white", color = NA), legend.position = "bottom")
  
  save_for_plotting <- merged_df
  
  #print(p1)
  vars <- append(vars, list(comp_searchers_plot))
  
  # First plot
  # Apply rescaling
  rescale <- rescale_for_secondary_axis(
    primary = merged_df$truegamma_s,
    secondary = merged_df$u_quart_s,
    primary_limits = c(0,0.4)  # match Stata range
  )
  # Add rescaled column
  merged_df$u_quart_s_rescaled <- rescale$scaled_secondary(merged_df$u_quart_s)
  # Plot with independent visual scaling
  gamma_plot <- ggplot(merged_df, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_bar(aes(y = recession2), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_line(aes(y = truegamma_s), color = "blue") +
    geom_line(aes(y = u_quart_s_rescaled), color = "red", linetype = "dashed") +
    scale_y_continuous(
      name = "Raw",
      limits = c(0, 0.4),
      sec.axis = sec_axis(transform = rescale$inverse_transform, name = "u (level)")
    ) +
    theme_minimal() +
    ggtitle(paste0("Employed People (gamma): New data: ", new)) +
    theme(plot.background = element_rect(fill = "white", color = NA), legend.position = "bottom")
  
  #print(p1)
  vars <- append(vars, list(gamma_plot))
  
  # First plot
  # Apply rescaling
  rescale <- rescale_for_secondary_axis(
    primary = merged_df$lambdagamma_s,
    secondary = merged_df$u_quart_s,
    primary_limits = c(0.02,0.06)  # match Stata range
  )
  # Add rescaled column
  merged_df$u_quart_s_rescaled <- rescale$scaled_secondary(merged_df$u_quart_s)
  # Plot with independent visual scaling
  lambdagamma_plot <- ggplot(merged_df, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_bar(aes(y = recession2), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_line(aes(y = lambdagamma_s), color = "blue") +
    geom_line(aes(y = u_quart_s_rescaled), color = "red", linetype = "dashed") +
    scale_y_continuous(
      name = "Raw",
      limits = c(0.02, 0.06),
      sec.axis = sec_axis(transform = rescale$inverse_transform, name = "u (level)")
    ) +
    theme_minimal() +
    ggtitle(paste0("(Employed jobseekers - lambda*gamma) w. New data: ", new)) +
    theme(plot.background = element_rect(fill = "white", color = NA), legend.position = "bottom")
  
  #print(p1)
  vars <- append(vars, list(lambdagamma_plot))
  
  
  # First plot
  # Apply rescaling
  rescale <- rescale_for_secondary_axis(
    primary = merged_df$SI_s,
    secondary = merged_df$u_quart_s,
    primary_limits = c(0,0.67)  # match Stata range
  )
  # Add rescaled column
  merged_df$u_quart_s_rescaled <- rescale$scaled_secondary(merged_df$u_quart_s)
  lambda_plot <- ggplot(merged_df, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_bar(aes(y = recession2), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_line(aes(y = SI_s), color = "blue") +
    geom_line(aes(y = u_quart_s_rescaled), color = "red", linetype = "dashed") +
    scale_y_continuous(
      name = "Raw",
      limits = c(0, 0.67),
      sec.axis = sec_axis(transform = rescale$inverse_transform, name = "u (level)")
    ) +
    theme_minimal() +
    ggtitle(paste0("(Employed search intensity - lambda) w. New data: ", new)) +
    theme(plot.background = element_rect(fill = "white", color = NA), legend.position = "bottom")
  
  #print(p1)
  vars <- append(vars, list(lambda_plot))
  
  #ggsave(file.path(destinationpath, "Figures", "comp_crowding_out_perc_dev.pdf"), p1, width = 10, height = 6)
  
  # Compute second log measure and filter
  merged_df <- flow_df %>%
    left_join(gamma_temp, by = "time") %>%
    left_join(SI_temp, by = "time") %>%
    mutate(
      lambdagamma_s = truegamma_s * SI_s,
      effective_searchers_s = u_quart_s + lambdagamma_s
    ) %>%
    mutate(log_comp_emp_s = log(truegamma_s / (1 - u_quart_s)))
  
  
  # Step 1: Identify non-NA indices
  valid_idx <- which(!is.na(merged_df$log_comp_emp_s))
  
  # Step 2: Apply the HP filter only to non-NA values
  hp_result2 <- hpfilter(merged_df$log_comp_emp_s[valid_idx], freq = 1600)
  
  # Step 3: Create full-length vectors initialized with NA
  merged_df$log_comp_emp_sc <- NA
  merged_df$log_comp_emp_st <- NA
  
  # Step 4: Fill in results at the correct positions
  merged_df$log_comp_emp_sc[valid_idx] <- hp_result2$cycle
  merged_df$log_comp_emp_st[valid_idx] <- hp_result2$trend
  
  # Update recession indicators
  merged_df <- merged_df %>%
    mutate(
      recession1 = if_else(time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4") |
                             (time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2")),
                           1, NA_real_),
      recession2 = if_else(!is.na(recession1), -0.3, NA_real_)
    )
  
  # Second
  # Apply rescaling
  rescale <- rescale_for_secondary_axis(
    primary = merged_df$log_comp_emp_sc,
    secondary = merged_df$u_quart_s,
    primary_limits = c(-0.3, 0.2)  # match Stata range
  )
  
  # Add rescaled column
  merged_df$u_quart_s_rescaled <- rescale$scaled_secondary(merged_df$u_quart_s)
  
  # Second plot
  p2 <- ggplot(merged_df, aes(x = time)) +
    geom_bar(aes(y = recession1), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_bar(aes(y = recession2), stat = "identity", fill = "grey80", na.rm = TRUE) +
    geom_line(aes(y = log_comp_emp_sc), color = "red") +
    geom_line(aes(y = u_quart_s_rescaled), color = "red", linetype = "dashed") +
    scale_y_continuous(
      name = "% deviation from trend",
      limits = c(-0.3, 0.2),
      sec.axis = sec_axis(transform = rescale$inverse_transform, name = "u (level)")
    ) +
    theme_minimal() +
    ggtitle(paste0("New data: ", new)) +
    theme(plot.background = element_rect(fill = "white", color = NA), legend.position = "bottom")
  
  #print(p2)
  p_list <- append(p_list, list(p2))
}
# grid.arrange(grobs = p_list, ncol = 4)
# grid.arrange(grobs = p_list[1:4], ncol = 2)
grid.arrange(grobs = vars[1:4], ncol = 2, top = "Outcome Metrics from Eeckhout Replication without new data (1996Q1-2016Q3)")
grid.arrange(grobs = vars[5:8], ncol = 2, top = "Outcome Metrics from Eeckhout Replication without new data (1996Q1-2024Q4)")

#ggsave(file.path(destinationpath, "Figures", "comp_employed_workers_perc_dev.pdf"), p2, width = 10, height = 6)


##########################################
# Compute deviation from trend (HP filter)
# merged_temp <- df_list[[1]]%>%
#   mutate(effective_emp_searchers = log(lambdagamma_s))
# 
# # Step 1: Identify non-NA indices
# valid_idx <- which(!is.na(merged_temp$effective_emp_searchers))
# 
# # Step 2: Apply the HP filter only to non-NA values
# hp_result1 <- hpfilter(merged_temp$effective_emp_searchers[valid_idx], freq = 1600)
# 
# # Step 3: Create full-length vectors initialized with NA
# merged_temp$effective_emp_searchers_sc <- NA
# merged_temp$effective_emp_searchers_st <- NA
# 
# # Step 4: Fill in results at the correct positions
# merged_temp$effective_emp_searchers_sc[valid_idx] <- hp_result1$cycle
# merged_temp$effective_emp_searchers_st[valid_idx] <- hp_result1$trend
# 
# mean_searchers = mean(merged_temp$effective_emp_searchers, na.rm = TRUE)
# 
# merged_temp %>% 
#   ggplot(aes(x = date, y = effective_emp_searchers_sc + mean_searchers)) +
#   geom_line()

t2 <- flow_rates_sa %>% 
  select(date, EE_quart_s, EE_x_5_quart_s) %>% 
  pivot_longer(!date) %>% 
  ggplot(aes(x = date, y = value, color = name))+
  geom_line()

t2

filtered_gdp <-read.csv(here("calibration_remote/data/detrended_gdp_nocovid.csv")) %>% 
  mutate(date = as.yearqtr(as.Date(DATE))) %>% 
  mutate(log_Cycle = log_Cycle) %>% 
  left_join(flow_rates_sa, by = "date") %>% 
  #left_join(merged_temp, by = "date") %>% 
  #left_join(select(df_list[[1]], date, log_SI_sc, log_comp_searchers_sc, log_comp_searchers_st), by = "date") %>% 
  filter(!is.na(EE_quart_s))
  
plot_data <- tibble()
temp <- filtered_gdp
res_list <- list()
x <- temp$log_Cycle
y <- temp$EE_quart_s
y_hp <- hpfilter(y, freq = 1600)$cycle
trend <- index(x)
forms <- c("Linear" = "y~x", "Linear with Trend" = "y ~ x + trend", "HP Filter" = "y ~ x")
hp = FALSE
for(form in names(forms)){
  if(grepl("HP", form)){
    y <- y_hp
    hp = TRUE
  }
  # Run a linear regression to estimate the long-run relationship
  reg <- lm(as.formula(forms[which(names(forms) == form)]))
  #print(modelsummary(reg, output = "latex", stars = TRUE, title = form))
  res_list[[form]] <- reg
  
  # Get predictions with confidence intervals
  predictions <- predict(reg, interval = "confidence")
  plot_data <- data.frame(
    x = temp$date,
    actual = y,
    input_x = x,
    predicted = predictions[, "fit"],
    lower_ci = predictions[, "lwr"],
    upper_ci = predictions[, "upr"],
    group = form,
    hp = hp
  ) %>% 
    rbind(plot_data)
  # Plot actual vs predicted with confidence intervals
}

print(modelsummary(res_list, output = "latex", stars = TRUE))

ggplot(plot_data, aes(x = x)) +
  geom_line(aes(y = actual, color = "Actual"), size = 1, color = "black", linewidth = 0.75, alpha = 0.5) +  # Actual values
  geom_line(aes(y = predicted, color = group), size = 1, linetype = "dashed") +  # Predicted values
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci, fill = group), alpha = 0.2) +  # Confidence interval
  #geom_line(aes(y = -input_x*10, color = "Input X"), size = 1) +  # Input X on secondary axis
  scale_y_continuous(
    name = "EE Transition Rate",
    sec.axis = sec_axis(~ .*30+63, name = "De-trended Search Intensity")  # Secondary axis for input_x
  ) +
  labs(title = "Employed Search Intensity", 
       y = "Employment-Employment Transition Rate", 
       x = "Date",
       color = "Model",  # Unifies the legend title for both color & fill
       fill = "Model",
       caption = "The above are predicted values of EE transitions as a function of the HP-filtered GDP cycle.\nThe black line represents the real (left) and de-trended (right) EE transition rate (EE/employment)\nusing the Current Population Survey and tabulation method from Eeckhout et al.\nThe green and blue lines represent the fitted/predicted values using a linear predictar and linear predictor with a linear trend component, respectively.\nThe red line demonstrates the fitted values of the HP filtered EE series using a linear predictor on the HP-filtered GDP series.") +
  theme_minimal() +
  theme(
    title = element_text(size = 16),
    axis.title.y.left = element_text(size = 14),
    axis.title.y.right = element_text(size = 14)  # Different color for secondary axis
  ) +
  facet_wrap(~hp, scales = "free")

modelsummary(res_list, gof_omit = 'AIC|BIC|Lik|RMSE')


# SAVE RELEVANT PLOT

# Plot with independent visual scaling
comp_searchers_plot_save <- ggplot(save_for_plotting, aes(x = time)) +
  # recessions (kept out of legend)
  geom_bar(aes(y = recession1), stat = "identity", fill = "grey80", na.rm = TRUE, show.legend = FALSE) +
  geom_bar(aes(y = recession2), stat = "identity", fill = "grey80", na.rm = TRUE, show.legend = FALSE) +
  geom_line(aes(y = comp_searchers_s,
        color = "Employed / jobseekers",
        linetype = "Employed / jobseekers"),
    linewidth = 1) +
  geom_line(aes(y = u_quart_s_rescaled, color = "UER", linetype = "UER"), linewidth = 1) +
  geom_hline(aes(yintercept = 0.32), color = "grey", linetype = "dashed", show.legend = FALSE) +
  geom_hline(aes(yintercept = 0.48), color = "grey", linetype = "dashed", show.legend = FALSE) +
  scale_y_continuous(name = "Employed as share of jobseekers - % deviation from trend",
    limits = c(0.1, 0.6),
    sec.axis = sec_axis(transform = rescale$inverse_transform, name = "Unemployment Rate")) +
  scale_color_manual(name = "Series",
    values = c("Employed / jobseekers" = "darkorchid",
               "UER"                  = "darkorange")
  ) +
  scale_linetype_manual(
    name = "Series",
    values = c("Employed / jobseekers" = "solid",
               "UER"                  = "dashed")
  ) +
  
  theme_minimal() +
  ggtitle("Employed as share of jobseekers") +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "bottom"
  ) +
  common_theme

ggsave(
  filename = here("data/behav_params/Eeckhout_Replication/comp_searchers_plot.jpg"),    
  plot = comp_searchers_plot_save,         
  width = 10,                              
  height = 6,                               
  dpi = 300,                                
  device = "jpeg"                          
)

save_for_plotting %>% select(date, comp_searchers_s) %>% write.csv(here("data/behav_params/Eeckhout_Replication/comp_searchers_s_series_abm_validation.csv"))
