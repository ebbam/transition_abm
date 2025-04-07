## Eeckhout Replication
#rm(list = ls())
library(readxl)
library(tidyverse)
library(ggplot2)
library(mFilter)
library(zoo)
library(seasonal)
library(here)
library(openxlsx)
library(haven)
library(lubridate)
library(patchwork)

integrate = TRUE

################################################################################
################################################################################
##### drop_box_code.do
################################################################################
################################################################################
# This is written and cleaned in cps_data/cps_cleaning.R
all_rates_new <- readRDS(here('data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds')) %>% 
  rbind(readRDS(here("data/behav_params/Eeckhout_Replication/cps_data/new_allrates_17_24.rds"))) %>% 
  #rbind(readRDS(here("data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds")))
  # Create a time variable starting from 1996-01-01 and incrementing by 1 month for each row (similar to Stata's tm function)
  # mutate(time = as.Date(paste0(1996, "-", sprintf("%02d", 1:12), "-01"))) %>%
  #arrange(time) %>% 
  arrange(date) %>% 
  mutate(time = 432 + row_number(.),
    date = as.numeric(date),
    EE = EEtotal / LFtotal,
    EU = EUtotal / LFtotal,
    EN = ENtotal / LFtotal,
    UE = UEtotal / LFtotal,
    UN = UNtotal / LFtotal,
    NE = NEtotal / LFtotal,
    NU = NUtotal / LFtotal
  ) 

#allrates <- read_dta(here(paste0(base, "All_rates_and_flows/allrates.dta")))

# setdiff(names(allrates), names(all_rates_new))
# setdiff(names(all_rates_new), names(allrates))
# names <- names(allrates)
# 
# test <- allrates %>% 
#   left_join(., all_rates_new, by = "date")
# 
# for(k in names[names != "date"]){
#   print(k)
#   test %>% 
#     select(paste0(k, ".x"), paste0(k, ".y")) %>% 
#     rename("a" = 1, "b" = 2) %>% 
#     filter(round(a,2) != round(b,2)) %>% print
# }
# ggplot() +
#   geom_line(data = all_rates_new, aes(x = 1:nrow(all_rates_new), y = time)) +
#   geom_line(data = allrates, aes(x = 1:nrow(allrates), y = time), color = "red")



################################################################################
################################################################################
##### decom_EE_overlap_months_more_obs.do
################################################################################
################################################################################
# 
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


################################################################################
################################################################################
##### quart_from_instantaneous.do
################################################################################
################################################################################

base <- here("data/behav_params/Eeckhout_Replication/116422-V1/data/")

# Load datasets
if(integrate){
  #allrates <- readRDS(here("data/behav_params/Eeckhout_Replication/cps_data/new_allrates_96_16.rds"))
  allrates <- all_rates_new
}else{
  allrates <- read_dta(here(paste0(base, "All_rates_and_flows/allrates.dta"))) #%>% 
  # bind_rows(all_rates_new)
}

EE_x <- read_dta(here(paste0(base, "EE_x/EE_x.dta")))

# Drop unnecessary columns
EE_x <- EE_x %>% select(-c(cpsidp, earnwt, previous_wage, new_wage, wtfinl_1)) %>% filter(row_number() != 1)

# Merge datasets
final_data <- left_join(allrates, EE_x, by = "date")

# Sort and create new variables
final_data <- final_data %>% 
  mutate(date = ifelse(nchar(date) == 6, paste0(substr(date, 1, 4), "-", substr(date, 5, nchar(date)), "-01"), 
                       ifelse(nchar(date) == 5, paste0(substr(date, 1, 4), "-0", substr(date, 5, nchar(date)), "-01"), NA))) %>% 
  mutate(date = as.Date(date)) %>% 
  arrange(time) %>%
  mutate(eu_ta_proxy = eu + 0.005)

# Filter based on time condition
#final_data <- final_data %>% 
#filter(date <= ymd("2016-09-01")) %>% 
# rename(time_x = time,
#        time = date)

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
merged_data <- merged_data %>% 
  mutate(#time = as.yearmon("1996-01") + (row_number() - 1) / 12,
         quarter = quarter(date), 
         year = year(date), 
         dq = paste0(year, "Q", quarter))

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
  raw_data <- read_excel(here(paste0(base, "Quarterly_raw_data.xls")), sheet = "Sheet1", range = "A2:N84", col_names = FALSE)
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

# TEMPORARY!

flow_rates_sa <- flow_rates_sa %>% 
  fill(EE_x_5_quart_s, .direction = "down")
  #mutate(EE_x_5_quart_s = ifelse(is.na(EE_x_5_quart_s), 0, EE_x_5_quart_s))


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
  data <- read_excel(here(paste0(base, "Quarterly_flow_rates_s.xlsx")), sheet = "Sheet1", range = "A2:M84", col_names = NA)
  #print('Using saved flow rates.')
}

# Rename columns for ease of use
colnames(data) <- c("time", "UE_quart_s", "EU_quart_s", "EE_quart_s", "ue_quart_s", "eu_quart_s", "ee_quart_s", "u_quart_s",  
                    #"avg_wage_UE_s", "avg_wage_EE_s", 
                    #"eu_ta_quart_s", 
                    "ue_ee_quart_s", "EE_x_5_quart_s")

# Define time range
startdate <- as.yearqtr("1997Q1")
enddate <- as.yearqtr("2024Q3")

time_series <- data$time
time_index_start <- which(time_series == startdate)
time_index_end <- which(time_series == enddate)
diff_periods <- (time_index_end - time_index_start + 1)

# Define gamma sequence
sizegamma <- 28
gamma_values <- seq(0.2, 0.8, length.out = sizegamma)

grid2 <- matrix(0, nrow = diff_periods + 1, ncol = sizegamma)
grid2[1, ] <- gamma_values

# Forward iteration for gamma
data_subset <- data[time_index_start:time_index_end, ]
for (i in 1:sizegamma) {
  for (t in 1:diff_periods) {
    grid2[t + 1, i] <- (grid2[t, i] * ((1 - data_subset$u_quart_s[t]) - data_subset$EU_quart_s[t])) / (1 - data_subset$u_quart_s[t]) + data_subset$UE_quart_s[t] - data_subset$EE_x_5_quart_s[t]
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

# # # Plot results
# ggplot() +
#   geom_line(aes(x = 1:diff_periods, y = lambda), color = "red") +
#   labs(title = "Search Intensity Lambda", x = "Time", y = "Lambda") +
#   theme_minimal()
# 
# ggplot() +
#   geom_line(aes(x = 1:diff_periods, y = true_gamma_series), color = "red") +
#   labs(title = "Gamma Series", x = "Time", y = "Lambda") +
#   theme_minimal()


################################################################################
################################################################################
##### plots.do
################################################################################
################################################################################
base <- here("data/behav_params/Eeckhout_Replication/116422-V1/data/Gamma_lambda_Mm/")

# Load the first dataset
if(integrate){
  final_gammas <- tibble("A" = true_gamma_series)
  #print('Using calculated gamma.')
}else{
  final_gammas <- read_xlsx(here(paste0(base, "final_gammas.xlsx")), sheet = "Sheet1", col_names = c("A"))
  #print('Using saved gamma.')
}
final_gammas <- final_gammas %>% mutate(time = as.yearqtr(1997 + (row_number() - 1) / 4))

# Apply HP filter
final_gammas <- final_gammas %>%
  mutate(truegamma_s = A,
         log_gamma_s = log(A))
filtered_gamma <- hpfilter(final_gammas$truegamma_s, freq = 1600)
final_gammas <- final_gammas %>%
  mutate(truegamma_st = filtered_gamma$trend,
         truegamma_sc = filtered_gamma$cycle)
filtered_log_gamma <- hpfilter(final_gammas$log_gamma_s, freq = 1600)
final_gammas <- final_gammas %>%
  mutate(log_gamma_st = filtered_log_gamma$trend,
         log_gamma_sc = filtered_log_gamma$cycle)

# Load second dataset
#quarterly_flow_rates <- read_excel(here(paste0(base, "Quarterly_flow_rates_s.xlsx")))
quarterly_flow_rates <- flow_rates_sa
quarterly_flow_rates <- quarterly_flow_rates %>% mutate(time = as.yearqtr(1996 + (row_number() - 1) / 4))

# Merge datasets
merged_data <- left_join(quarterly_flow_rates, final_gammas, by = "time")

# Define recessions
merged_data1 <- merged_data %>%
  mutate(recession1 = ifelse((time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4")) |
                               (time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2")), 0.5, NA),
         recession2 = ifelse((time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4")) |
                               (time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2")), -0.3, NA))

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
filtered_SI <- hpfilter(final_SI$SI_s, freq = 1600)
final_SI <- final_SI %>%
  mutate(SI_st = filtered_SI$trend,
         SI_sc = filtered_SI$cycle)
filtered_log_SI <- hpfilter(final_SI$log_SI_s, freq = 1600)
final_SI <- final_SI %>%
  mutate(log_SI_st = filtered_log_SI$trend,
         log_SI_sc = filtered_log_SI$cycle)

# Merge datasets
merged_data <- left_join(quarterly_flow_rates, final_SI, by = "time")

# Define recessions
merged_data2 <- merged_data %>%
  mutate(recession1 = ifelse(time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4") |
                               time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2"), 0.5, NA),
         recession2 = ifelse(time >= as.yearqtr("2001 Q1") & time <= as.yearqtr("2001 Q4") |
                               time >= as.yearqtr("2007 Q4") & time <= as.yearqtr("2009 Q2"), -0.3, NA))

source(here('data/behav_params/Eeckhout_Replication/eeckhout_replication.R'))

# Plot
p1 <- ggplot() +
  geom_bar(data = merged_data1, aes(x = time, y = recession1), stat = "identity", fill = "gray80") +
  geom_bar(data = merged_data1, aes(x = time, y = recession2), stat = "identity", fill = "gray80") +
  geom_line(data = merged_data1, aes(x = time, y = log_gamma_sc, color = "Gamma")) +
  geom_line(data = merged_data1, aes(x = time, y = (u_quart_s*8) - 0.6, color = "Unemployment")) +
  geom_line(data = merged_data_old_p1, aes(x = time, y = log_gamma_sc, color = "Gamma Original")) +
  geom_line(data = merged_data_old_p1, aes(x = time, y = (u_quart_s*8) - 0.6, color = "Unemployment Original")) +
  scale_y_continuous(sec.axis = sec_axis(~ (./8)+0.6, name = "u (level)", breaks = c(0.04,0.08,0.1))) +
  labs(y = "Gamma (% deviation from trend)", x = "Time", 
  title = "On-the-job searchers γ (de-trended) and unemployment rate") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(ncol=2))

# Plot
p2 <- ggplot() +
  geom_bar(data = merged_data2, aes(x = time, y = recession1), stat = "identity", fill = "gray80") +
  geom_bar(data = merged_data2, aes(x = time, y = recession2), stat = "identity", fill = "gray80") +
  geom_line(data = merged_data2, aes(x = time, y = log_SI_sc, color = "Lambda")) +
  geom_line(data = merged_data2, aes(x = time, y = (u_quart_s*8)-0.55, color = "Unemployment")) +
  geom_line(data = merged_data_old_p2, aes(x = time, y = log_SI_sc, color = "Lambda Original")) +
  geom_line(data = merged_data_old_p2, aes(x = time, y = (u_quart_s*8)-0.55, color = "Unemployment Original")) +
  scale_y_continuous(sec.axis = sec_axis(~ (./8)+0.55, name = "u (level)")) +
  labs(y = "Lambda (% deviation from trend)", x = "Time", title = "Search intensity λ (de-trended) and unemployment rate") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(ncol=2))

#saveRDS(merged_data2, here('data/behav_params/Eeckhout_Replication/lambda_hat.RDS'))

print(p1 / p2)



