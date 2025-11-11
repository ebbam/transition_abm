### CLEANING OEWS DATA - industry-specific occupational employment
# Occupational Employment and Wage Statistics
# 
# https://www.bls.gov/oes/tables.htm
# Download the industry-specific table for each

library(tidyverse)
library(here)
library(conflicted)
library(janitor)
library(readxl)
library(assertthat)
conflict_prefer_all("dplyr", quiet = TRUE)
new_data = FALSE

if(new_data){
  df <- tibble()
  for(yr in 1999:2024){
    yr_tmp <- substr(yr, 3,4)
    if(yr == 1999){
      tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in3/nat2d_sic_", as.character(yr), "_dl.xls")), skip = 35) 
    }else if(yr == 2000){
      tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in3/nat2d_sic_", as.character(yr), "_dl.xls")),  skip = 33)
    }else if(yr == 2001){
      tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in3/nat2d_sic_", as.character(yr), ".xls"))) 
    }else if(yr == 2002){
      tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oes", yr_tmp, "in4/nat4d_", as.character(yr), "_dl.xls")))
    }else if(yr == 2003){
      tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/nat3d_may", as.character(yr), "_dl.xls"))) 
    }else if (yr > 2003 & yr < 2008){
      tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/natsector_may",as.character(yr), "_dl.xls"))) 
    }else if(yr >= 2008 & yr < 2014){
      tmp <- read_xls(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/natsector_M",as.character(yr), "_dl.xls"))) 
    }else{
      tmp <- read_xlsx(here(paste0("data/occ_macro_vars/OEWS/industry_specific/oesm", yr_tmp, "in4/natsector_M",as.character(yr), "_dl.xlsx")))  
    }
    
    df <- tmp %>% 
      clean_names %>% 
      mutate(year = yr) %>% 
      select(-contains("pct_rpt")) %>% 
      mutate(across(c(pct_total, a_mean, emp_prse, mean_prse), ~as.numeric(.)),
             annual = as.logical(annual)) %>% 
      bind_rows(df, .)
    
    
  }
  df %>% saveRDS(here("data/occ_macro_vars/OEWS/occ_ind_employment_compiled.RDS"))
  }else{
    df <- readRDS(here("data/occ_macro_vars/OEWS/occ_ind_employment_compiled.RDS"))
  }

print(paste0("Calculating occupational TD for ", nx,"."))

# ABM gets input from occ_td_setting

temp <- df %>% select(year, naics, naics_title, sic, sic_title, occ_code, occ_title, tot_emp, pct_total) %>% 
  filter(naics %in% c('11', '21', '22', '23', '31-33', '42', '44-45', '48-49', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81') & occ_code %in% abm$SOC2010) %>% 
  group_by(naics) %>% 
  # Make NAICS titles consistent with latest attributed title
  mutate(naics_title = last(naics_title),
         tot_emp_char = tot_emp,
         tot_emp = as.numeric(tot_emp_char)) %>% 
  ungroup %>% 
  select(year, naics, naics_title, occ_code, pct_total, tot_emp) %>% 
  distinct

  # group_by(occ_code, naics, naics_title) %>% 
  # fill(pct_total, .direction = "down") %>% 
  # ungroup %>% 

codes <- temp %>% select(naics, naics_title) %>% distinct

temp %>% 
  group_by(occ_code, naics, naics_title) %>% 
  fill(pct_total, .direction = "down") %>% 
  ungroup %>% 
  ggplot() + 
  geom_area(aes(x = year, y = pct_total, fill = occ_code)) +
  facet_wrap_custom(~naics_title) +
  labs(x = "Year", y = "Reported Pct Shares", title = "Occupational Employment Shares by 2-digit NAICS") +
  common_theme +
  theme(legend.position = "none") -> p1

print(p1)

temp %>% 
  group_by(occ_code, naics, naics_title) %>% 
  fill(pct_total, .direction = "down") %>% 
  ungroup %>% 
  group_by(naics_title, year) %>% 
  mutate(tot = sum(pct_total, na.rm = TRUE),
         pct_total_norm = pct_total/tot) %>% 
  ungroup %>% 
  ggplot() + 
  geom_area(aes(x = year, y = pct_total_norm, fill = occ_code), stat = "identity") +
  facet_wrap_custom(~naics_title) +
  labs(x = "Year", title = "Occupational Employment Shares by 2-digit NAICS", y= "Shares of Total") +
  common_theme +
  theme(legend.position = "none") -> p2

print(p2)

shares <- temp %>% 
  # Create total annual occupation-level employment
  group_by(occ_code, year) %>% 
  mutate(occ_total = sum(tot_emp, na.rm = TRUE)) %>% 
  # Create total annual industry-specific occupation-level employment
  group_by(naics_title, occ_code, year) %>% 
  mutate(occ_ind_total = sum(tot_emp, na.rm = TRUE)) %>% 
  ungroup %>% 
  # Create share of annual occupation employment in a specific industry
  mutate(occ_share = occ_ind_total/occ_total) 

# test completeness
shares %>% 
  group_by(year, occ_code) %>% 
  summarise(occ_share = sum(occ_share, na.rm = TRUE)) %>% 
  # Lose one observation in 2021 where occupation was 0 - all NAs for occupation code "33-9093" - "transportation security screeners"
  filter(round(occ_share) == 1)

shares %>% ggplot(aes(x = year, y = occ_share, color = occ_code)) + 
  geom_line() + facet_wrap_custom(~naics_title) + common_theme + theme(legend.position = "none")

shares_ma <- function(s_dat, len){
  plot <- s_dat %>% 
    arrange(naics_title, occ_code, year) %>%       
    group_by(naics_title, occ_code) %>%           
    mutate(
      occ_share_ma = rollmean(
        x      = occ_share,
        k      = len,   
        fill   = NA,      
        align  = "right"       
      )
    ) %>% 
    ungroup() %>% 
    ggplot(aes(x = year, y = occ_share_ma, colour = occ_code)) +
    geom_line() +
    facet_wrap_custom(~ naics_title) +
    labs(y = paste0(as.character(len), "-year moving average of occ_share"),
         title = paste0(as.character(len), "-year moving average of occ_share"),
         x = "Year") +
    common_theme +
    theme(legend.position = "none")
  
  return(plot)
  
}

shares_ma(shares, 3)
shares_ma(shares, 5)
shares_ma(shares, 10)

shares %>% 
  group_by(year) %>% 
  summarise(n_occs = n_distinct(occ_code)) %>% 
  filter(year > 2012)

# Only 2012 and 2013
shares %>% 
  group_by(year) %>% 
  summarise(n_occs = n_distinct(occ_code)) %>% 
  filter(n_occs >= 463)

# I take the years where all/majority (97%) of occ codes are present: between 2012 and 2018
mean_shares <- shares %>% 
  group_by(year) %>% 
  mutate(n_occs = n_distinct(occ_code)) %>% 
  filter(n_occs >= 450) %>% 
  group_by(occ_code, naics_title) %>% 
  summarise(mean_share = mean(occ_share, na.rm = TRUE))

# Testing to see sum
mean_shares %>% 
    arrange(naics_title) %>% 
  group_by(occ_code) %>% 
  summarise(test = sum(mean_share, na.rm= TRUE)) %>% 
  arrange(-test)


mean_shares %>% 
  group_by(naics_title) %>% 
  mutate(mean_order = mean(mean_share),
         max_order = max(mean_share),
         zeros = sum(mean_share != 0)) %>% 
  ungroup %>% 
  # turn the two keys into factors so ggplot keeps a stable order
  mutate(
    occ_code   = factor(occ_code),
    naics_title = fct_reorder(naics_title, zeros)
  ) %>% 
  filter(mean_share >= 0.05) %>% 

  ggplot(aes(x = naics_title, y = occ_code, fill = mean_share)) +
  geom_tile(colour = "grey90") +
  
  # colour bar
  scale_fill_viridis_c(
    option = "magma",      # or "plasma", "viridis"
    labels  = percent_format(accuracy = 0.01),
    trans  = "log10",  
    name    = "Employment\nshare",
    direction = -1
  ) +
  common_theme + 
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid  = element_blank()
  ) +
  labs(
    x = "Industry (NAICS title)",
    y = "Occupation (SOC code)",
    title = "Occupation–Industry Employment Shares",
    subtitle = "The colors display the share of total occupational employment in each industry.\nIndustries are ordered by the total number of occupations they employ in ascending order.\nDisplay only those occupations whose industrial-level employment share is at least 5%."
  ) -> p6

print(p6)

mean_shares %>% 
  group_by(occ_code) %>% 
  summarise(temp = sum(mean_share, na.rm = TRUE)) %>% 
  ggplot() + 
  geom_histogram(aes(x = temp)) +
  labs(
    x = "Total Occupational Employment represented by Mean Shares",
    y = "Occupation Count (n = 463)",
    title = "Representativeness/Accuracy of Mean Share Calculation (sum of industry share of occupation = 1)") +
  common_theme

odd_ones <- mean_shares %>% 
     group_by(occ_code) %>% 
  summarise(total_share = sum(mean_share, na.rm = TRUE)) %>% arrange(-total_share) %>% filter(total_share > 1.1) %>% 
  left_join(., abm, by = c("occ_code" = "SOC2010")) %>% 
  select("SOC2010 Occupational Code" = occ_code, "Occupational Label" = OCC2010_desc, "Sum of Mean Shares" = total_share)

print("Two occupational categories have a sum of mean shares > 1.1:")

odd_ones %>% kable(format = "latex") %>% print(.)


mean_shares_for_bind <- mean_shares %>% 
  ungroup %>% 
  left_join(., codes, by = "naics_title") 

print(abm %>% filter(!(SOC2010 %in% c(mean_shares_for_bind$occ_code))))

