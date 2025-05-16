# Replicating figures and data in Mukoyama et al
# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(here)
library(haven) # To read .dta files
library(patchwork)
library(urca)
library(zoo)

# Set this to false or true depending on if you wish to recompile the input data to figures 3 and 4 based on updated data
first = FALSE

# Set working directory (equivalent to `cd "$raw_ATUS"`)
base <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/")

# ##################################################################****
#   ** Description: Creates Figure 1
# ##################################################################****
#   code adapted from Figure1.do

data <- read_dta(paste0(base, "raw_data/ATUS/merged_ATUS_2014.dta"))
data_2023 <- readRDS(paste0(base, "raw_data/ATUS/merged_ATUS_2023.rds"))

# Collapse data: calculate the mean of 'time_less8' weighted by 'wgt', grouped by 'numsearch'
collapsed_data_2014 <- data %>%
  group_by(numsearch) %>%
  summarise(time_less8 = weighted.mean(time_less8, wgt, na.rm = TRUE)) %>%
  ungroup()

collapsed_data_2023 <- data_2023 %>%
  group_by(numsearch) %>%
  summarise(time_less8 = weighted.mean(time_less8, wgt, na.rm = TRUE)) %>%
  ungroup()

collapsed_data_all <- data %>% 
  bind_rows(data_2023) %>%
  group_by(numsearch) %>%
  summarise(time_less8 = weighted.mean(time_less8, wgt, na.rm = TRUE)) %>%
  ungroup()

# Optional: Add descriptive labels for variables (no direct equivalent in R)
# You might use them as axis labels or in annotations later.

# Create the bar chart
p1 <- ggplot(collapsed_data_2014, aes(x = as.factor(numsearch), y = time_less8)) +
  geom_bar(stat = "identity", fill = "blue", color = "black") +
  labs(
    x = "Number of Search Methods",
    y = "Average Search Time Per Day",
    title = "Figure 1. The Average Minutes (per day) Spent on Job Search Activities by the Number of Search Methods",
    subtitle = "2003-2014",
    caption = "Notes: Each bin reflects the average search time in minutes per day\nby the number of search methods that the individual reports using in the previous month.\nData is pooled from 2003–2014 and observations are weighted by the individual sample weight."
  ) +
  theme_minimal(base_size = 15) +
  theme(
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "gray"),
    panel.grid.minor = element_blank()
  )

p2 <- ggplot(collapsed_data_2023, aes(x = as.factor(numsearch), y = time_less8)) +
  geom_bar(stat = "identity", fill = "purple", color = "black") +
  labs(
    x = "Number of Search Methods",
    y = "Average Search Time Per Day",
    title = "Figure 1. The Average Minutes (per day) Spent on Job Search Activities by the Number of Search Methods",
    subtitle = "2015-2023",
    caption = "Notes: Each bin reflects the average search time in minutes per day\nby the number of search methods that the individual reports using in the previous month.\nData is pooled from 2003–2014 and observations are weighted by the individual sample weight."
  ) +
  theme_minimal(base_size = 15) +
  theme(
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "gray"),
    panel.grid.minor = element_blank()
  )


p3 <- ggplot(collapsed_data_all, aes(x = as.factor(numsearch), y = time_less8)) +
  geom_bar(stat = "identity", fill = "coral", color = "black") +
  labs(
    x = "Number of Search Methods",
    y = "Average Search Time Per Day",
    title = "Figure 1. The Average Minutes (per day) Spent on Job Search Activities by the Number of Search Methods",
    subtitle = "2003-2023",
    caption = "Notes: Each bin reflects the average search time in minutes per day\nby the number of search methods that the individual reports using in the previous month.\nData is pooled from 2003–2014 and observations are weighted by the individual sample weight."
  ) +
  theme_minimal(base_size = 15) +
  theme(
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "gray"),
    panel.grid.minor = element_blank()
  )

p3 + p1 / p2

# # Save the plot to a PDF file
# output_dir <- Sys.getenv("figures") # Equivalent to `$figures`
# ggsave(filename = file.path(output_dir, "Figure1.pdf"), plot = plot, width = 8, height = 6)

##############################################################################
#   ** Description: Creates the data needed for Figure 3 
# adapted from Figure3.do
# 
################################################################################
#   ****Figure 3a: fraction of unemployed (U/U+N)
################################################################################
if(first){
  test_merged <- readRDS(here(paste0(base, "raw_data/ATUS/merged_ATUS_2023.rds")))
  pre_2015 <- readRDS(here(paste0(base, "final_data/R_final/full_CPS_data.RDS")))
  paths <- c(
    paste0(base, "final_data/R_final/full_CPS_data"),
    #paste0(base, "mukoyama_all/final_data/R_final/temp_full_CPS_data_before_new_years"),
    paste0(base, "final_data/R_final/temp_full_CPS_2015_19_correct"),
    paste0(base, "final_data/R_final/temp_full_CPS_2020_24_correct"))
  
  add_years <- tibble()
  add_years_unemp <- tibble()
  for(fpath in paths){
    print(fpath)
    data <- readRDS(here(paste0(fpath, "_all_weights_new.RDS")))
    
    data1 <- data %>% 
      group_by(year, month, searchers) %>%
      mutate(num_search = if_else(searchers == 1, sum(newwgt, na.rm = TRUE), NA_real_)) %>%
      ungroup() %>%
      group_by(year, month, nonsearchers) %>%
      mutate(num_nonsearch = if_else(nonsearchers == 1, sum(newwgt, na.rm = TRUE), NA_real_)) %>%
      ungroup() %>%
      group_by(year, month, unemp) %>%
      mutate(num_unemp = if_else(unemp == 1, sum(newwgt, na.rm = TRUE), NA_real_)) %>%
      ungroup() %>%
      group_by(year, month, emp) %>%
      mutate(num_emp = if_else(emp == 1, sum(newwgt, na.rm = TRUE), NA_real_)) %>%
      ungroup() %>%
      group_by(year, month, nonpart) %>%
      mutate(num_nonpart = if_else(nonpart == 1, sum(newwgt, na.rm = TRUE), NA_real_)) %>%
      ungroup() %>% 
      select(year, month, num_search, num_nonsearch, num_unemp, num_nonpart, num_emp, newwgt, contains("time_create"))
    
    add_years <- bind_rows(add_years, data1)
    
    data2 <- data %>% 
      filter(unemp == 1) %>%
      group_by(year, month) %>%
      summarise(
        numsearch = weighted.mean(numsearch, newwgt, na.rm = TRUE),
        across(contains("time_create"), ~weighted.mean(., newwgt, na.rm = TRUE))
      ) %>%
      ungroup()
    
    add_years_unemp <- bind_rows(add_years_unemp, data2)
    
  }
  # Load the CPS data
  #ORIGINAL:
  #data_fig3 <- readRDS(paste0(base, "final_data/R_final/full_CPS_data.RDS"))
  
  #data_fig3 <- readRDS(here(base, "final_data/R_final/temp_full_CPS_data_before_new_years.rds"))
  
  # Collapse the data by year and month
  collapsed_data_3a <- add_years %>%
    group_by(year, month) %>%
    summarise(across(c(num_search, num_nonsearch, num_unemp, num_nonpart, num_emp), ~mean(., na.rm = TRUE))) %>%
    ungroup() 
  
  # Seasonal adjustment using regression
  # Function to seasonally adjust a variable
  seasonal_adjust <- function(data, var) {
    # Perform regression
    fit <- lm(as.formula(paste(var, "~ factor(month)")), data = data)
    
    # Get the coefficients for each month
    coeffs <- coef(fit)
    
    # Subtract the monthly coefficients
    data <- data %>%
      rowwise() %>%
      mutate(!!var := !!sym(var) - (coeffs[paste0("factor(month)", month)] %>% coalesce(0))) %>%
      ungroup()
    
    return(data)
  }
  
  collapsed_data_3a_adj <- collapsed_data_3a
  # Apply the seasonal adjustment to each variable
  for (var in c("num_search", "num_nonsearch", "num_unemp", "num_nonpart", "num_emp")) {
    collapsed_data_3a_adj <- seasonal_adjust(collapsed_data_3a_adj, var)
  }
  
  #write.csv(collapsed_data_3a_adj, here(paste0(base, "final_data/R_final/Figure3a_data_extended_new_corrected.csv")))

# ##############################################################################
#   Figure 3b: Average search time (methods and Created time)
# ##############################################################################
# Unemployed data
data_unemp <- add_years_unemp

# Seasonal adjustment using regression
# Function to seasonally adjust a variable
seasonal_adjust <- function(data, var) {
  # Perform regression
  fit <- lm(as.formula(paste(var, "~ factor(month)")), data = data)
  
  # Get the coefficients for each month
  coeffs <- coef(fit)
  
  # Subtract the monthly coefficients
  data <- data %>%
    rowwise() %>%
    mutate(!!var := !!sym(var) - (coeffs[paste0("factor(month)", month)] %>% coalesce(0))) %>%
    ungroup()
  
  return(data)
}

collapsed_unemp_adj <- data_unemp
# Apply the seasonal adjustment to each variable
for (var in c("numsearch", "time_create")) {
  collapsed_unemp_adj <- seasonal_adjust(collapsed_unemp_adj, var)
}

#write.csv(collapsed_unemp_adj, here(paste0(base, "final_data/R_final/Figure3b_data_extended_new_corrected.csv")))
}

################################################################################
############ Figures 2-4 - Data prep

figure3a_data <-read.csv(paste0(base, "final_data/R_final/Figure3a_data.csv"))[-1] #read_csv(paste0(base, "int_data/CPS/Figure3a_data.csv")) # collapsed_data_3a_adj <- read.csv(here(paste0(base, "final_data/R_final/Figure3a_data.csv")))
figure3b_data <- read.csv(paste0(base, "final_data/R_final/Figure3b_data.csv"))[-1] #read.csv(paste0(base, "int_data/CPS/Figure3b_data.csv")) # read.csv(data_unemp_adj, here(paste0(base, "final_data/R_final/Figure3b_data.csv")))

# fig3a_new <- read.csv(paste0(base, "final_data/R_final/Figure3a_data_extended.csv"))[-1] #collapsed_data_3a_adj
# fig3b_new <-  read.csv(paste0(base, "final_data/R_final/Figure3b_data_extended.csv"))[-1]  #collapsed_unemp_adj

#fig3a_new_2023 <- read.csv(paste0(base, "final_data/R_final/Figure3a_data_extended_new.csv"))[-1] #collapsed_data_3a_adj
#fig3b_new_2023 <-  read.csv(paste0(base, "final_data/R_final/Figure3b_data_extended_new.csv"))[-1]  #collapsed_unemp_adj

fig3a_new_2023 <- read.csv(paste0(base, "final_data/R_final/Figure3a_data_extended_new_corrected.csv"))[-1] #collapsed_data_3a_adj
fig3b_new_2023 <-  read.csv(paste0(base, "final_data/R_final/Figure3b_data_extended_new_corrected.csv"))[-1]  #collapsed_unemp_adj
# Data preprocessing
  
date <- figure3a_data %>% mutate(date = year + (month/12)) %>% pull(date)
searchers <- figure3a_data[[3]]
nonsearchers <- figure3a_data[[4]]
unemp <- figure3a_data[[5]]
nonpart <- figure3a_data[[6]]
emp <- figure3a_data[[7]]

frac_unemp <- unemp / (unemp + nonpart)
time_unemp <- figure3b_data[[4]]

effort_unemp_UNE <- time_unemp * unemp / (unemp + nonpart + emp)
unemp_frac <- unemp / (unemp + nonpart + emp)

# date_new <- fig3a_new_2023 %>% mutate(date = year + (month/12)) %>% pull(date)
# searchers_new <- fig3a_new[[3]]
# nonsearchers_new <- fig3a_new[[4]]
# unemp_new <- fig3a_new[[5]]
# nonpart_new <- fig3a_new[[6]]
# emp_new <- fig3a_new[[7]]
# 
# frac_unemp_new <- unemp_new / (unemp_new + nonpart_new)
# time_unemp_new <- fig3b_new[[4]]
# 
# effort_unemp_UNE_new <- time_unemp_new * unemp_new / (unemp_new + nonpart_new + emp_new)
# unemp_frac_new <- unemp_new / (unemp_new + nonpart_new + emp_new)

date_new_2023 <- fig3a_new_2023 %>% mutate(date = year + (month/12)) %>% pull(date)
searchers_new_2023 <- fig3a_new_2023[[3]]
nonsearchers_new_2023 <- fig3a_new_2023[[4]]
unemp_new_2023 <- fig3a_new_2023[[5]]
nonpart_new_2023 <- fig3a_new_2023[[6]]
emp_new_2023 <- fig3a_new_2023[[7]]

frac_unemp_new_2023 <- unemp_new_2023 / (unemp_new_2023 + nonpart_new_2023)
time_unemp_new_2023 <- fig3b_new_2023[[7]]
time_unemp_2014_orig <- fig3b_new_2023[[5]]
time_unemp_2014_new <- fig3b_new_2023[[6]]

effort_unemp_UNE_new_2023 <- time_unemp_new_2023 * unemp_new_2023 / (unemp_new_2023 + nonpart_new_2023 + emp_new_2023)
unemp_frac_new_2023 <- unemp_new_2023 / (unemp_new_2023 + nonpart_new_2023 + emp_new_2023)

effort_unemp_UNE_2014_orig <- time_unemp_2014_orig * unemp_new_2023 / (unemp_new_2023 + nonpart_new_2023 + emp_new_2023)
unemp_frac_2014_orig <- unemp_new_2023 / (unemp_new_2023 + nonpart_new_2023 + emp_new_2023)

effort_unemp_UNE_2014_new <- time_unemp_2014_new * unemp_new_2023 / (unemp_new_2023 + nonpart_new_2023 + emp_new_2023)
unemp_frac_2014_new <- unemp_new_2023 / (unemp_new_2023 + nonpart_new_2023 + emp_new_2023)


# Helper function to add shaded recession areas
add_recession <- function(p) {
  p +
    annotate("rect", xmin = 2007 + 11/12, xmax = 2009.5, ymin = -Inf, ymax = Inf, alpha = 0.2) +
    annotate("rect", xmin = 2001 + 3/12, xmax = 2001 + 11/12, ymin = -Inf, ymax = Inf, alpha = 0.2) +
    annotate("rect", xmin = 2020 + 2/12, xmax = 2020 + 4/12, ymin = -Inf, ymax = Inf, alpha = 0.2)
}

##################################
############ Figures 2a-b ########
##################################

# Define file names
files <- c("Fig2a_data", "Fig2b_data", "Fig2a_data_new", "Fig2b_data_new", 
           "Fig2a_data2014_orig", "Fig2b_data2014_orig", "Fig2a_data2014_new", "Fig2b_data2014_new", 
           "Fig2a_data2023", "Fig2b_data2023")

# Load data dynamically
data_list <- lapply(files, function(f) read.csv(paste0(base, "int_data/ATUS/", f, ".csv")))
names(data_list) <- files

# Extract years
years <- lapply(data_list[grep("Fig2a", names(data_list))], function(df) df[[1]])

# Extract nonemployment and unemployment data
#nonemp_base <- lapply(data_list[grep("Fig2a", names(data_list))], function(df) df[2:3])
nonemp_base <- bind_rows(lapply(names(data_list[grep("Fig2a", names(data_list))]), function(name) {
  df <- data_list[[name]]  # Extract dataframe
  df$Dataset <- name         # Add a column with the original list element name
  return(df)
}), .id = NULL)

nonemp_base <- nonemp_base %>% mutate(label = case_when(Dataset == "Fig2a_data" ~ "From Paper",
                  Dataset == "Fig2a_data_new" ~ "Original Weights on Orig. TS new",
                  Dataset == "Fig2a_data2014_orig" ~ "Original Weights from Orig. TS",
                  Dataset == "Fig2a_data2014_new" ~ "Weighted",
                  Dataset == "Fig2a_data2023" ~ "Weighted Ext. TS")) %>% 
  filter(Dataset != "Fig2a_data2014_orig" & Dataset != "Fig2a_data_new")

unemp_base <- bind_rows(lapply(names(data_list[grep("Fig2b", names(data_list))]), function(name) {
  df <- data_list[[name]]  # Extract dataframe
  df$Dataset <- name         # Add a column with the original list element name
  return(df)
}), .id = NULL)
unemp_base <- unemp_base %>% mutate(label = case_when(Dataset == "Fig2b_data" ~ "From Paper",
                                                        Dataset == "Fig2b_data_new" ~ "Original Weights on Orig. TS new",
                                                        Dataset == "Fig2b_data2014_orig" ~ "Unweighted",
                                                        Dataset == "Fig2b_data2014_new" ~ "Weighted",
                                                        Dataset == "Fig2b_data2023" ~ "Weighted Ext. TS")) %>% 
filter(Dataset != "Fig2b_data2014_orig" & Dataset != "Fig2b_data_new")

# Define function to plot Figure 2
fig2a <- ggplot() +
  geom_line(data = nonemp_base, aes(x = year, y = time_create, color = label), size = 1) +
  geom_line(data = nonemp_base, aes(x = year, y = time_less8), color = "purple", size = 0.5, linetype = "dashed")+
  scale_x_continuous(breaks = 2003:2023) +
  scale_y_continuous(limits = c(0, 10), breaks = seq(0, 10, by = 2)) +
  theme_minimal() +
  theme(legend.position = "bottom")
fig2a <- add_recession(fig2a)

fig2b <- ggplot() +
  geom_line(data = unemp_base, aes(x = year, y = time_create, color = label), size = 1) +
  geom_line(data = unemp_base, aes(x = year, y = time_less8), color = "purple", size = 0.5, linetype = "dashed")+
  scale_x_continuous(breaks = 2003:2023) +
  scale_y_continuous(limits = c(15, 45), breaks = seq(0, 45, by = 5)) +
  theme_minimal() +
  theme(legend.position = "bottom")
fig2b <- add_recession(fig2b)

print(fig2a + fig2b + plot_annotation("Figure 2. Actual and Imputed Average Search Time (minutes per day) \nfor All Nonemployed Workers ( panel A) and Unemployed Workers ( panel B)",
                                caption = "Notes: Regressions are estimated in the ATUS from 2003–2014. \nWhile both panels A and B plot the fitted values from the sample regression, panel A plots the actual and imputed search time for all nonemployed, while panel B plots them for just the unemployed. \nObservations are weighted by their ATUS sample weight.",
                                theme=theme(plot.title=element_text(hjust=0.5))))

##################################
############ Figures 3a-b ########
##################################
# Figure 3a
# Define file names for Figure 3a and 3b
files_3a <- c("Figure3a_data",  "Figure3a_data_extended_new_corrected") # "Figure3a_data_extended",
files_3b <- c(#"Figure3b_data", "Figure3b_data_extended", 
  "Figure3b_data_extended_new_corrected")

# Load data dynamically
data_list_3a <- lapply(files_3a, function(f) read.csv(paste0(base, "final_data/R_final/", f, ".csv"))[-1])
data_list_3b <- lapply(files_3b, function(f) read.csv(paste0(base, "final_data/R_final/", f, ".csv"))[-1])

# Name the lists
names(data_list_3a) <- files_3a
names(data_list_3b) <- files_3b

# Merge Figure 3a data, adding a Dataset column
fig3a_base <- bind_rows(lapply(names(data_list_3a), function(name) {
  df <- data_list_3a[[name]]
  df$Dataset <- name  # Add dataset name as a column
  df$date <- df$year + (df$month / 12)  # Create date column
  return(df)
}), .id = NULL)

fig3b_base <- data_list_3b[[1]] %>% 
  mutate(date = year + month/12) %>% 
  #select(-year, -month) %>% 
  pivot_longer(!c(date, year, month, numsearch)) %>% 
  mutate(label = case_when(name == "time_create" ~ "Original Weights on Orig. TS",
                           name == "time_create_new_2014" ~ "New Weights from Orig. TS",
                           name == "time_create_new_2023" ~ "New Weights from Ext. TS",
                           name == "time_create_orig" ~ "Original Weights on Ext. TS"))

# Define function to plot Figure 3a
fig3a_plot <- ggplot() +
  geom_line(data = fig3a_base, aes(x = date, y = num_unemp / (num_unemp + num_nonpart), color = Dataset), size = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(2003, 2023, by = 2)) +
  theme_minimal() +
  labs(x = "Date", y = "Extensive Margin") +
  theme(legend.position = "none")

fig3a_plot <- add_recession(fig3a_plot)

# Define function to plot Figure 3b
fig3b_plot <- ggplot() +
  geom_line(data = filter(fig3b_base, date > 1999.9), aes(x = date, y = value, color = label), size = 0.5) +
  theme_minimal() +
  #scale_x_continuous(breaks = seq(2003, 2023, by = 2)) +
  scale_y_continuous(limits = c(10, 45), breaks = seq(0, 45, by = 5)) +
  theme_minimal() +
  labs(x = "Date", y = "Intensive Margin", title = "Period: 2000-2024", color = "Weights and Input Data") +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(ncol=2))

fig3bb_plot <- ggplot() +
  geom_line(data = filter(fig3b_base, date > 1999.9 & date < 2020), aes(x = date, y = value, color = label), size = 0.5) +
  theme_minimal() +
  #scale_x_continuous(breaks = seq(2003, 2023, by = 2)) +
  scale_y_continuous(limits = c(25, 45), breaks = seq(0, 45, by = 5)) +
  theme_minimal() +
  labs(x = "Date", y = "Intensive Margin", title = "Period: 2000-2019") +
  theme(legend.position = "none") +
  guides(color=guide_legend(ncol=2))

fig3b_plot <- add_recession(fig3b_plot)
fig3bb_plot <- add_recession(fig3bb_plot)

print(fig3a_plot + fig3b_plot + plot_annotation(
  "Figure 3. The Time Series of the Extensive Margin (U/(U + N )) ( panel A)\nand the Intensive Margin ( panel B), \nMeasured by the Average Minutes of Search per Day for Unemployed Workers",
  caption = "Red data is new data. Notes: Panel A plots the monthly ratio of the number of unemployed (U) to the total number of unemployed (U + N ) in the CPS from 1994–2014.", #\nPanel B plots the average minutes of search per day, constructed as described in the text. Each observation is weighted by its CPS sample weight.",
  theme=theme(plot.title=element_text(hjust=0.5))))

print(fig3b_plot + fig3bb_plot + plot_annotation(
  "Intensive Margin Measured by the Average Minutes of Search per Day for Unemployed Workers",
  caption = "Plots the average minutes of search per day,using the imputed minutes as a function of search methods used.\nEach observation is weighted by its CPS sample weight.",
  theme=theme(plot.title=element_text(hjust=0.5))))

ggplot() +
  geom_line(data = filter(fig3b_base, date > 1999.9 & name == "time_create_new_2023"), aes(x = date, y = value, color = label), size = 0.5) +
  theme_minimal() +
  #scale_x_continuous(breaks = seq(2003, 2023, by = 2)) +
  scale_y_continuous(limits = c(10, 45), breaks = seq(0, 45, by = 5)) +
  theme_minimal() +
  labs(x = "Date", y = "Intensive Margin", title = "Period: 2000-2024", color = "Weights and Input Data") +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(ncol=2))

library(slider)

quart_data <- fig3b_base %>%
  filter(year > 1999 & name == "time_create_new_2023") %>% 
  mutate(quarter = case_when(
    month <= 3 ~ 1,
    month > 3 & month <= 6 ~ 2,
    month > 6 & month <= 9 ~ 3, 
    month > 9 ~ 4
  )) %>%
  group_by(year, quarter) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  mutate(value_smooth = predict(loess(value ~ as.numeric(row_number()), span = 0.1)))
       #value_smooth = ifelse(year == 2020 & quarter == 2, 15, value_smooth)) 

monthly_data <- fig3b_base %>% 
  filter(year > 1999 & name == "time_create_new_2023") %>% 
  mutate(value_smooth = predict(loess(value ~ as.numeric(row_number()), span = 0.1))) %>% 
  select(year, month, value, value_smooth)
  

write.csv(quart_data, here("data/behav_params/Mukoyama_Replication/quarterly_search_ts.csv"))
write.csv(monthly_data, here("data/behav_params/Mukoyama_Replication/monthly_search_ts.csv"))

 
ggplot() +  
  geom_line(data = quart_data, aes(x = year + quarter/4, y = value, color = "Quarterly"), alpha = 0.5) +  # Original data
  geom_line(data = quart_data, aes(x = year + quarter/4, y = value_smooth, color = "Quarterly"),  size = 1) +  # Smoothed data
  geom_line(data = monthly_data, aes(x = year + month/12, y = value, color = "Monthly"), alpha = 0.5) +  # Original data
  geom_line(data = monthly_data, aes(x = year + month/12, y = value_smooth, color = "Monthly"), size = 1) +  # Smoothed data
  labs(title = "Smoothed Quarterly & Monthly Search Effort Data for Calibration", 
       subtitle = "Thin lines represent the observed data whereas the thicker lines represent a smoothing \nof each series using a LOESS fit with 0.1 span.", x = "Year", y = "Value (Minutes Searched per Day", color = "Frequency") +
  theme_minimal()
  #mutate(month = date%%round(date)) 

##################################
############ Figures 4a-b ########
##################################

# Figure 4a
fig4a <- ggplot() +
  #geom_line(aes(x = date_new, y = effort_unemp_UNE_new), color = "red", size = 0.5) +
  geom_line(aes(x = date_new_2023, y = effort_unemp_UNE_new_2023, color = "New Weights from Ext. TS"), size = 0.5) +
  geom_line(aes(x = date_new_2023, y = effort_unemp_UNE_2014_orig, color = "Original Weights on Ext. TS"), size = 0.5) +
  geom_line(aes(x = date_new_2023, y = effort_unemp_UNE_2014_new, color = "New Weights from Orig. TS"), size = 0.5) +
  geom_line(aes(x = date, y = effort_unemp_UNE, color = "Original Weights on Orig. TS"), size = 0.5) +
  scale_x_continuous(breaks = seq(1994, 2024, by = 2)) +
  scale_y_continuous(limits = c(0, 2.75), breaks = seq(0, 3, by = 0.5)) +
  #scale_color_manual(values = c("Original Weights on Orig. TS" = "blue", "New Weights from Orig. TS" = "green", "New Orig. TS" = "purple", "New Weights from Ext. TS" = "darkgreen")) + 
  labs(x = "Date", 
       y = "Total Search Effort (Extensive x Intensive Margin)") + 
       #title ="Panel A: Time Series of Total Search Effort") + 
  scale_color_manual(values = c("Original Weights on Orig. TS" = "blue", "New Weights from Orig. TS" = "green", "New Weights from Ext. TS" = "purple", "Original Weights on Ext. TS" = "darkgreen")) + 
  
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(ncol=1))
fig4a <- add_recession(fig4a)

# Figure 4b
fig4b <- ggplot() +
  geom_line(aes(x = date, y = effort_unemp_UNE / effort_unemp_UNE[1], color = "Original Weights on Orig. TS"), size = 0.5) +
  #geom_line(aes(x = date_new, y = effort_unemp_UNE_new / effort_unemp_UNE_new[1]), color = "red", size = 0.5) +
  geom_line(aes(x = date_new_2023, y = effort_unemp_UNE_2014_orig / effort_unemp_UNE_2014_orig[1], color = "Original Weights on Ext. TS"), size = 0.5) +
  geom_line(aes(x = date_new_2023, y = effort_unemp_UNE_2014_new / effort_unemp_UNE_2014_new[1], color = "New Weights from Orig. TS"), size = 0.5) +
  geom_line(aes(x = date_new_2023, y = effort_unemp_UNE_new_2023 / effort_unemp_UNE_new_2023[1], color = "New Weights from Ext. TS"), size = 0.5) +
  geom_line(aes(x = date, y = unemp_frac / unemp_frac[1], color = "Original Weights on Orig. TS"), linetype = "dashed", size = 0.5) +
  #geom_line(aes(x = date_new, y = unemp_frac_new / unemp_frac_new[1], color = "test remove"), linetype = "dashed", size = 0.5) +
  geom_line(aes(x = date_new_2023, y = unemp_frac_2014_orig / unemp_frac_2014_orig[1], color = "Original Weights on Ext. TS"), linetype = "dashed", size = 0.5) +
  geom_line(aes(x = date_new_2023, y = unemp_frac_2014_new / unemp_frac_2014_new[1], color = "New Weights from Orig. TS"),  linetype = "dashed", size = 0.5) +
  geom_line(aes(x = date_new_2023, y = unemp_frac_new_2023 / unemp_frac_new_2023[1], color = "New Weights from Ext. TS"),  linetype = "dashed", size = 0.5) +
  scale_x_continuous(breaks = seq(1994, 2024, by = 2)) +
  scale_y_continuous(limits = c(0, 2.75), breaks = seq(0, 2.5, by = 0.5)) +
  scale_color_manual(values = c("Original Weights on Orig. TS" = "blue", "New Weights from Orig. TS" = "green", "New Weights from Ext. TS" = "purple", "Original Weights on Ext. TS" = "darkgreen")) + 
  labs(x = "Date", 
       y = "Total Search Effort (Extensive x Intensive Margin)") + 
       #title ="Panel B: Time Series of Total Search Effort \n Using the Search Time of Unemployed Workers \n s*(U/(E + U + N)) (blue) \n vs. Using the Number of Unemployed Workers\nU/(E + U + N) (red)") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(ncol=1))
fig4b <- add_recession(fig4b)

print(fig4a + fig4b + plot_annotation('Figure 4. Time Series of (Panel A) Total Search Effort and \n(Panel B) Total Search Effort Using the Search Time of\nUnemployed Workers [solid: (s*(U/(E + U + N))] versus \nUsing the Number of Unemployed Workers [dashed: U/(E + U + N)) (panel B)',
                                theme=theme(plot.title=element_text(hjust=0.5))))

# # Save figures as PDFs
# ggsave("Figure2a.pdf", fig2a, width = 10.5, height = 8, units = "in")
# ggsave("Figure2b.pdf", fig2b, width = 10.5, height = 8, units = "in")
# ggsave("Figure3a.pdf", fig3a, width = 10.5, height = 8, units = "in")
# ggsave("Figure3b.pdf", fig3b, width = 10.5, height = 8, units = "in")
# ggsave("Figure4a.pdf", fig4a, width = 10.5, height = 8, units = "in")
# ggsave("Figure4b.pdf", fig4b, width = 10.5, height = 8, units = "in")

#lambda <- readRDS(here('data/behav_params/Eeckhout_Replication/lambda_hat.RDS'))

#############################################################################
###### Triangulating parameter value of search effort sensitivity to GDP ####

gdp_effort <- read.csv(here("data/macro_vars/detrended_gdp_nocovid.csv")) %>% 
  tibble %>% 
  select(DATE, log_Cycle) %>% 
  mutate(year = year(DATE),
         month = month(DATE),
         date = year + month/12) %>% 
  select(-DATE, -year, -month) %>% 
  left_join(filter(fig3b_base, label == "New Weights from Ext. TS"), ., by = "date") %>% 
  mutate(value_smooth = rollmean(value, k = 4, fill = NA, align = "right")) %>% 
  filter(!is.na(log_Cycle)) 

gdp_effort %>% 
  ggplot() +
  geom_line(aes(x = date, y = value)) +
  geom_line(aes(x = date, y = log_Cycle*30))

gdp_effort %>% 
  ggplot() +
  geom_line(aes(x = date, y = value_smooth)) +
  geom_line(aes(x = date, y = log_Cycle*30))


plot_data <- tibble()
form_list = list("Linear fit" = "y~x", "Linear fit with trend" = "y~x+trend")
for(form in names(form_list)){
  temp <- gdp_effort %>% 
    filter(date > 1999.9)
  formula = unlist(form_list[names(form_list) == form])
  
  # test for stationarity with augmented Dickey-Fuller Test
  x <- temp$log_Cycle
  y <- temp$value_smooth
  trend <- index(x)
  
  # Run a linear regression to estimate the long-run relationship
  reg <- lm(as.formula(formula))
  
  # # Get residuals from the regression
  # residuals <- reg$residuals
  # 
  # # Test if residuals are stationary (cointegration test)
  # coint_test <- ur.df(residuals, type = "none")
  # summary(coint_test)
  # 
  # summary(reg)  # Coefficients give the cointegration relationship
  # 
  # # Compute first differences
  # dy <- diff(temp$value_smooth)
  # dx <- diff(temp$log_Cycle)
  # lagged_residuals <- residuals[-1]  # Lag residuals (error correction term)
  # 
  # # ECM: Adjustments in dy based on dx and past equilibrium errors
  # ecm <- lm(dy ~ dx + lagged_residuals)
  # summary(ecm)
  
  # Get predictions with confidence intervals
  predictions <- predict(reg, interval = "confidence")
  plot_data <- data.frame(
    x = temp$date,
    actual = y,
    input_x = x,
    predicted = predictions[, "fit"],
    lower_ci = predictions[, "lwr"],
    upper_ci = predictions[, "upr"],
    group = form
  ) %>% 
    rbind(plot_data, .)
  # Plot actual vs predicted with confidence intervals
}

ggplot(plot_data, aes(x = x)) +
  geom_line(aes(y = actual, color = "Actual", color = group), size = 1, color = "blue", linewidth = 0.75) +  # Actual values
  geom_line(aes(y = predicted, color = group), size = 1, linetype = "dashed") +  # Predicted values
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci, fill = group), alpha = 0.2) +  # Confidence interval
  #geom_line(aes(y = -input_x*30+63, color = "Input X", group = group), size = 1) +  # Input X on secondary axis
  # scale_y_continuous(
  #   name = "Actual & Predicted Values",
  #   sec.axis = sec_axis(~ .*30+63, name = "Input X")  # Secondary axis for input_x
  # ) +
  labs(title = "Actual vs. Predicted Values of Search Effort", 
       y = "Minutes Searched", 
       x = "Date",
       color = "Functional Form", 
       fill = "Functional Form",
       caption = "The above are predicted values of search effort as a function of the HP-filtered GDP cycle.\nThe dark blue line represents the 'real' search effort in minutes per day calculated using the methodlogy from Mukoyama et al.\nThe red and green represent the fitted or predicted values using a linear predictar and linear predictor with a linear trend component, respectively.") +
  theme_minimal() +
  theme(
    axis.title.y.right = element_text(color = "red")  # Different color for secondary axis
  )

