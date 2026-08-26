# Source : https://www.atlantafed.org/research-and-data/data/wage-growth-tracker
# Wage Growth Tracker
# The Atlanta Feds Wage Growth Tracker is a measure of the nominal wage growth of individuals. It is constructed using microdata from the Current Population Survey (CPS), and is the median percent change in the hourly wage of individuals observed 12 months apart. Our measure is based on methodology developed by colleagues at the San Francisco Fed.

library(here)
library(tidyverse)
library(readxl)
library(janitor)

temp <- read_xlsx(here("data/occ_macro_vars/wages/wage-growth-data.xlsx"), skip = 1) %>% 
  tibble  %>% 
  clean_names %>% 
  rename(date = x1)

temp %>% 
  pivot_longer(!date) %>% 
  mutate(value = as.numeric(value)) %>% 
  filter(name %in% c("overall", "full_time")) %>% 
  ggplot(aes(x = date)) +
  geom_line(aes(y = value, group = name, color = name))
