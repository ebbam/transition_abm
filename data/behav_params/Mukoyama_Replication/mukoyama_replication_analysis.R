# Replicating figures and data in Mukoyama et al
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(here)
library(haven) # To read .dta files
library(patchwork)

# Set this to false or true depending on if you wish to recompile the input data to figures 3 and 4 based on updated data
first = FALSE

# Set working directory (equivalent to `cd "$raw_ATUS"`)
base <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/")

# ##################################################################****
#   ** Description: Creates Figure 1
# ##################################################################******/
#   code adapted from Figure1.do
data <- read_dta(paste0(base, "raw_data/ATUS/merged_ATUS_2014.dta"))

# Collapse data: calculate the mean of 'time_less8' weighted by 'wgt', grouped by 'numsearch'
collapsed_data <- data %>%
  group_by(numsearch) %>%
  summarise(time_less8 = weighted.mean(time_less8, wgt, na.rm = TRUE)) %>%
  ungroup()

# Optional: Add descriptive labels for variables (no direct equivalent in R)
# You might use them as axis labels or in annotations later.

# Create the bar chart
p <- ggplot(collapsed_data, aes(x = as.factor(numsearch), y = time_less8)) +
  geom_bar(stat = "identity", fill = "blue", color = "black") +
  labs(
    x = "Number of Search Methods",
    y = "Average Search Time Per Day",
    title = "Figure 1. The Average Minutes (per day) Spent on Job Search Activities by the Number of Search Methods",
    caption = "Notes: Each bin reflects the average search time in minutes per day\nby the number of search methods that the individual reports using in the previous month.\nData is pooled from 2003–2014 and observations are weighted by the individual sample weight."
  ) +
  theme_minimal(base_size = 15) +
  theme(
    panel.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "gray"),
    panel.grid.minor = element_blank()
  )

print(p)

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
  add_years <- tibble()
  for(i in 2015:2024){
    data <- readRDS(here(paste0(base, "final_data/R_final/cps_data_no_time_", as.character(i), ".rds"))) %>% 
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
      select(year, month, num_search, num_nonsearch, num_unemp, num_nonpart, num_emp) # Eventually also numsearch, newwgt,time_create, newwgt
    
    add_years <- rbind(add_years, data)
  }
  # Load the CPS data
  #ORIGINAL:data_fig3 <- readRDS(paste0(base, "final_data/R_final/full_CPS_data.RDS"))
  
  data_fig3 <- readRDS(here(base, "final_data/R_final/temp_full_CPS_data_before_new_years.rds"))
  
  # Calculate weighted sums for different groups
  data_fig3 <- data_fig3 %>%
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
    select(year, month, num_search, num_nonsearch, num_unemp, num_nonpart, num_emp) %>% 
    rbind(add_years)
  
  # Collapse the data by year and month
  collapsed_data_3a <- data_fig3 %>%
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
  
  write.csv(collapsed_data_3a_adj, here(paste0(base, "final_data/R_final/Figure3a_data_extended.csv")))

}

# ##############################################################################
#   Figure 3b: Average search time (methods and Created time)
# ##############################################################################
if(first){
  # Unemployed data
  data_unemp <- data_fig3 %>%
    filter(unemp == 1) %>%
    group_by(year, month) %>%
    summarise(
      numsearch = weighted.mean(numsearch, newwgt, na.rm = TRUE),
      time_create = weighted.mean(time_create, newwgt, na.rm = TRUE)
    ) %>%
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
  
  collapsed_unemp_adj <- data_unemp
  # Apply the seasonal adjustment to each variable
  for (var in c("numsearch", "time_create")) {
    collapsed_unemp_adj <- seasonal_adjust(collapsed_unemp_adj, var)
  }

  write.csv(collapsed_unemp_adj, here(paste0(base, "final_data/R_final/Figure3b_data.csv")))
}

################################################################################
############ Figures 2-4 - Data prep

# Load data (update file paths as needed)
fig2a_data <- read.csv(paste0(base, "int_data/ATUS/Fig2a_data.csv")) 
fig2b_data <- read.csv(paste0(base, "int_data/ATUS/Fig2b_data.csv")) 
figure3a_data <-read.csv(paste0(base, "final_data/R_final/Figure3a_data.csv"))[-1] #read_csv(paste0(base, "int_data/CPS/Figure3a_data.csv")) # collapsed_data_3a_adj <- read.csv(here(paste0(base, "final_data/R_final/Figure3a_data.csv")))
figure3b_data <- read.csv(paste0(base, "final_data/R_final/Figure3b_data.csv"))[-1] #read.csv(paste0(base, "int_data/CPS/Figure3b_data.csv")) # read.csv(data_unemp_adj, here(paste0(base, "final_data/R_final/Figure3b_data.csv")))

fig3a_new <- read.csv(paste0(base, "final_data/R_final/Figure3a_data_extended.csv"))[-1] 

# Data preprocessing
year <- fig2a_data[[1]]
nonemp_base <- fig2a_data[2:3]
unemp_base <- fig2b_data[2:3]
  
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

date_new <- fig3a_new %>% mutate(date = year + (month/12)) %>% pull(date)
searchers_new <- fig3a_new[[3]]
nonsearchers_new <- fig3a_new[[4]]
unemp_new <- fig3a_new[[5]]
nonpart_new <- fig3a_new[[6]]
emp_new <- fig3a_new[[7]]

frac_unemp_new <- unemp_new / (unemp_new + nonpart_new)
time_unemp_new <- figure3b_data[[4]]

effort_unemp_UNE_new <- time_unemp_new * unemp_new / (unemp_new + nonpart_new + emp_new)
unemp_frac_new <- unemp_new / (unemp_new + nonpart_new + emp_new)




# Helper function to add shaded recession areas
add_recession <- function(p) {
  p +
    annotate("rect", xmin = 2007 + 11/12, xmax = 2009.5, ymin = -Inf, ymax = Inf, alpha = 0.2) +
    annotate("rect", xmin = 2001 + 3/12, xmax = 2001 + 11/12, ymin = -Inf, ymax = Inf, alpha = 0.2)
}

##################################
############ Figures 2a-b ########
##################################

# Figure 2a
fig2a <- ggplot() +
  geom_line(aes(x = year, y = nonemp_base[[1]]), color = "red", linetype = "dashed", size = 1) +
  geom_line(aes(x = year, y = nonemp_base[[2]]), color = "blue", size = 1) +
  scale_x_continuous(breaks = 2003:2014) +
  scale_y_continuous(limits = c(0, 10), breaks = seq(0, 10, by = 2)) +
  theme_minimal()
fig2a <- add_recession(fig2a)

# Figure 2b
fig2b <- ggplot() +
  geom_line(aes(x = year, y = unemp_base[[1]]), color = "red", linetype = "dashed", size = 1) +
  geom_line(aes(x = year, y = unemp_base[[2]]), color = "blue", size = 1) +
  scale_x_continuous(breaks = 2003:2014) +
  scale_y_continuous(limits = c(10, 50), breaks = seq(10, 50, by = 10)) +
  theme_minimal()
fig2b <- add_recession(fig2b)

print(fig2a + fig2b + plot_annotation("Figure 2. Actual and Imputed Average Search Time (minutes per day) \nfor All Nonemployed Workers ( panel A) and Unemployed Workers ( panel B)",
                                caption = "Notes: Regressions are estimated in the ATUS from 2003–2014. \nWhile both panels A and B plot the fitted values from the sample regression, panel A plots the actual and imputed search time for all nonemployed, while panel B plots them for just the unemployed. \nObservations are weighted by their ATUS sample weight.",
                                theme=theme(plot.title=element_text(hjust=0.5))))

##################################
############ Figures 3a-b ########
##################################

# Figure 3a
fig3a <- ggplot() +
  geom_line(aes(x = date, y = frac_unemp), color = "blue", size = 1) +
  #scale_x_continuous(breaks = seq(1994, 2014, by = 2)) +
  scale_y_continuous(limits = c(0.05, 0.25), breaks = seq(0.05, 0.25, by = 0.05)) +
  theme_minimal() +
  labs(x = "Date", y = "Extensive Margin")
fig3a <- add_recession(fig3a)

# Figure 3b
fig3b <- ggplot() +
  geom_line(aes(x = date, y = time_unemp), color = "blue", size = 1) +
  #scale_x_continuous(breaks = seq(1994, 2014, by = 2)) +
  scale_y_continuous(limits = c(25, 45), breaks = seq(25, 45, by = 5)) +
  theme_minimal() +
  labs(x = "Date", y = "Intensive Margin")
fig3b <- add_recession(fig3b)

print(fig3a + fig3b + plot_annotation(
  "Figure 3. The Time Series of the Extensive Margin (U/(U + N )) ( panel A)\n and the Intensive Margin ( panel B), \nMeasured by the Average Minutes of Search per Day for Unemployed Workers",
caption = "Notes: Panel A plots the monthly ratio of the number of unemployed (U) to the total number of unemployed (U + N ) in the CPS from 1994–2014. \nPanel B plots the average minutes of search per day, constructed as described in the text. Each observation is weighted by its CPS sample weight.",
theme=theme(plot.title=element_text(hjust=0.5))))


# Figure 3a
fig3anew <- ggplot() +
  geom_line(aes(x = date_new, y = frac_unemp_new), color = "red", size = 1) +
  geom_line(aes(x = date, y = frac_unemp), color = "blue", size = 1) +
  #scale_x_continuous(breaks = seq(1994, 2014, by = 2)) +
  scale_y_continuous(limits = c(0.05, 0.25), breaks = seq(0.05, 0.25, by = 0.05)) +
  theme_minimal() +
  labs(x = "Date", y = "Extensive Margin")
fig3anew <- add_recession(fig3anew)

# # Figure 3b
# fig3bnew <- ggplot() +
#   geom_line(aes(x = date_new, y = time_unemp_new), color = "blue", size = 1) +
#   #scale_x_continuous(breaks = seq(1994, 2014, by = 2)) +
#   scale_y_continuous(limits = c(25, 45), breaks = seq(25, 45, by = 5)) +
#   theme_minimal() +
#   labs(x = "Date", y = "Intensive Margin")
# fig3bnew <- add_recession(fig3bnew)

print(fig3anew + plot_annotation(
  "Figure 3. The Time Series of the Extensive Margin (U/(U + N )) ( panel A)\n ", #and the Intensive Margin ( panel B), \nMeasured by the Average Minutes of Search per Day for Unemployed Workers",
  caption = "Red data is new data. Notes: Panel A plots the monthly ratio of the number of unemployed (U) to the total number of unemployed (U + N ) in the CPS from 1994–2014.", #\nPanel B plots the average minutes of search per day, constructed as described in the text. Each observation is weighted by its CPS sample weight.",
  theme=theme(plot.title=element_text(hjust=0.5))))

##################################
############ Figures 4a-b ########
##################################

# Figure 4a
fig4a <- ggplot() +
  geom_line(aes(x = date, y = effort_unemp_UNE), color = "blue", size = 1) +
  scale_x_continuous(breaks = seq(1994, 2014, by = 2)) +
  scale_y_continuous(limits = c(0, 2.75), breaks = seq(0, 3, by = 0.5)) +
  labs(x = "Date", 
       y = "Total Search Effort (Extensive x Intensive Margin)") + 
       #title ="Panel A: Time Series of Total Search Effort") + 
  theme_minimal()
fig4a <- add_recession(fig4a)

# Figure 4b
fig4b <- ggplot() +
  geom_line(aes(x = date, y = effort_unemp_UNE / effort_unemp_UNE[1]), color = "blue", size = 1) +
  geom_line(aes(x = date, y = unemp_frac / unemp_frac[1]), color = "red", linetype = "dashed", size = 1) +
  scale_x_continuous(breaks = seq(1994, 2014, by = 2)) +
  scale_y_continuous(limits = c(0, 2.75), breaks = seq(0, 2.5, by = 0.5)) +
  labs(x = "Date", 
       y = "Total Search Effort (Extensive x Intensive Margin)") + 
       #title ="Panel B: Time Series of Total Search Effort \n Using the Search Time of Unemployed Workers \n s*(U/(E + U + N)) (blue) \n vs. Using the Number of Unemployed Workers\nU/(E + U + N) (red)") +
  theme_minimal()
fig4b <- add_recession(fig4b)

print(fig4a + fig4b + plot_annotation('Figure 4. Time Series of (Panel A) Total Search Effort and \n(Panel B) Total Search Effort Using the Search Time of\nUnemployed Workers [blue: (s*(U/(E + U + N))] versus \nUsing the Number of Unemployed Workers [red: U/(E + U + N)) (panel B)',
                                theme=theme(plot.title=element_text(hjust=0.5))))

# # Save figures as PDFs
# ggsave("Figure2a.pdf", fig2a, width = 10.5, height = 8, units = "in")
# ggsave("Figure2b.pdf", fig2b, width = 10.5, height = 8, units = "in")
# ggsave("Figure3a.pdf", fig3a, width = 10.5, height = 8, units = "in")
# ggsave("Figure3b.pdf", fig3b, width = 10.5, height = 8, units = "in")
# ggsave("Figure4a.pdf", fig4a, width = 10.5, height = 8, units = "in")
# ggsave("Figure4b.pdf", fig4b, width = 10.5, height = 8, units = "in")


