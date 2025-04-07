# Load necessary libraries
library(plyr) # Only need this for rbind.fill - load this first
library(tidyverse)
library(conflicted)
library(data.table)
library(haven)
library(stringr)
library(here)
library(readr)
library(rio)
library(assertthat)
library(patchwork)
library(stats) # for normal distribution functions
conflicted::conflict_prefer_all("dplyr", quiet = TRUE)
conflicts_prefer(here::here)

base <- here("data/behav_params/Mukoyama_Replication/")
paths <- c(
  # pulk is cleaned from the CPS_data ones
  #paste0(base, "mukoyama_all/final_data/R_final/full_CPS_data.RDS"),
  #paste0(base, "mukoyama_all/final_data/R_final/temp_full_CPS_data_before_new_years.RDS"),
  paste0(base, "mukoyama_all/final_data/R_final/temp_full_CPS_2015_19_correct.RDS"),
  paste0(base, "mukoyama_all/final_data/R_final/temp_full_CPS_2020_24_correct.RDS"))

full_emps <- tibble()
for(fpath in paths){
  data <- readRDS(fpath) %>% 
    select(year, month, lfs, pulk, newwgt) %>% 
    filter(lfs == "E") %>% 
    mutate(date = year + (month/12),
           pulk_nas = case_when(pulk == 1 ~ 1, 
                                    pulk == 2 ~ 0, 
                                    TRUE ~ 0),
           pulk_cleaned = case_when(pulk == 1 ~ 1, 
                                    pulk == 2 ~ 0, 
                                    TRUE ~ NA)) %>% 
    group_by(date) %>% 
    summarise(emp_seekers = weighted.mean(pulk_cleaned, newwgt, na.rm = TRUE),
              emp_seekers_counting_na = weighted.mean(pulk_nas, newwgt, na.rm = TRUE),
              n_emp_seekers = sum(pulk_nas))
  
  full_emps <- bind_rows(full_emps, data)
}

reported <- full_emps %>% 
  ggplot() + 
  geom_line(aes(x = date, y = emp_seekers))

all <- full_emps %>% 
  ggplot() + 
  geom_line(aes(x = date, y = emp_seekers_counting_na), color = "blue")

n_seekers <- full_emps %>% 
  ggplot() + 
  geom_line(aes(x = date, y = n_emp_seekers), color = "purple")

reported / all / n_seekers


read_dta(here("data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/CPS/intermediate_199401.dta"))