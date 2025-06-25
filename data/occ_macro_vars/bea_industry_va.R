# Cleaning BEA Value Added by Industry
# Suggested citation: U.S. Bureau of Economic Analysis, "Real Value Added by Industry" (accessed Sunday, June 8, 2025).
library(here)
library(tidyverse)
library(zoo)

# Quarterly Data from 2005-2024
# The year and quarter are not combined in one header requiring a strange read-in process below
header <- read.csv(here("data/occ_macro_vars/bea_va_by_industry_05_24_q.csv"), 
                   nrows = 2, header = FALSE, stringsAsFactors = FALSE, skip = 3)

# Combine year and quarter into proper names
new_names <- paste0(gsub("X", "", header[1, -c(1,2)]), "_", header[2, -c(1,2)])

# Read the rest of the data (starting after header rows)
temp <- read.csv(here("data/occ_macro_vars/bea_va_by_industry_05_24_q.csv"), 
                 skip = 5, header = FALSE, stringsAsFactors = FALSE) 

# Assign clean column names
colnames(temp) <- c("Line", "X", new_names)
final <- tibble(temp) %>% 
  rename("industry" = X) %>% 
  select(-Line) %>% 
  mutate(
    indent_level = str_extract(industry, "^\\s*") |> str_length(),
    label = paste0("Level_", indent_level / 2)  # Assuming 2 spaces per indent level
  ) %>% 
  mutate(label_0 = ifelse(label == "Level_0", trimws(industry), NA),
         label_2 = ifelse(label == "Level_2", trimws(industry), NA),
         label_4 = ifelse(label == "Level_4", trimws(industry), NA),
         label_6 = ifelse(label == "Level_6", trimws(industry), NA),
         label_8 = ifelse(label == "Level_8", trimws(industry), NA)) %>% 
  fill(label_0) %>%
  fill(label_2) %>%
  fill(label_4) %>%
  fill(label_6) %>%
  fill(label_8) %>%
  mutate(label_0 = ifelse(industry == "        Gross domestic product", "Total GDP", label_0), 
         industry = trimws(industry)) %>% 
  pivot_longer(contains('_Q'), names_to = "date", values_to = "real_VA") %>% 
  mutate(date = as.yearqtr(gsub("_", " ", date), format = "%Y Q%q"),
         real_VA = as.numeric(real_VA)) %>% 
  filter(!is.na(real_VA))

final %>% 
  filter(label_0 == "Private industries" & !is.na(label_2)) %>% 
  ggplot() +
  geom_line(aes(x = date, y = real_VA, group = industry, color = label)) +
  facet_wrap(~label_2, scales = "free_y") + 
  theme_minimal() +
  labs(title = "Quarterly Real VA by Industry 2005-2024", subtitle = "Data from Bureau of Economic Analysis Economic Accounts", x = "Year-Quarter", y = "Real Value Added in 2017-chained USD", color = "Industry Level of Aggregation")

# Annual Data from 1997-2024
# The year and quarter are not combined in one header requiring a strange read-in process below
annual <- read.csv(here("data/occ_macro_vars/bea_va_by_industry_97_24_a.csv"), skip = 3) %>% 
  slice(-1)

final <- tibble(annual) %>% 
  rename("industry" = X) %>% 
  select(-Line) %>% 
  mutate(
    indent_level = str_extract(industry, "^\\s*") |> str_length(),
    label = paste0("Level_", indent_level / 2)  # Assuming 2 spaces per indent level
  ) %>% 
  mutate(label_0 = ifelse(label == "Level_0", trimws(industry), NA),
         label_2 = ifelse(label == "Level_2", trimws(industry), NA),
         label_4 = ifelse(label == "Level_4", trimws(industry), NA),
         label_6 = ifelse(label == "Level_6", trimws(industry), NA),
         label_8 = ifelse(label == "Level_8", trimws(industry), NA)) %>% 
  fill(label_0) %>%
  fill(label_2) %>%
  fill(label_4) %>%
  fill(label_6) %>%
  fill(label_8) %>%
  mutate(label_0 = ifelse(industry == "        Gross domestic product", "Total GDP", label_0), 
         industry = trimws(industry),
         across(contains("X"), ~as.numeric(.))) %>%
  pivot_longer(contains('X'), names_to = "date", values_to = "real_VA") %>% 
  mutate(date = as.numeric(gsub("X", "", date)),
         real_VA = as.numeric(real_VA)) %>% 
  filter(!is.na(real_VA))

final %>% 
  filter(label_0 == "Private industries" & !is.na(label_2)) %>% 
  ggplot() +
  geom_line(aes(x = date, y = real_VA, group = industry, color = label)) +
  facet_wrap(~label_2, scales = "free_y") + 
  theme_minimal() +
  labs(title = "Annual Real VA by Industry 1997-2024", 
       subtitle = "Data from Bureau of Economic Analysis Economic Accounts", 
       x = "Year", 
       y = "Real Value Added in 2017-chained USD", 
       color = "Industry Level of Aggregation")


# Picked relevant industries as 2-digit NAICS industries 
# https://www.bls.gov/iag/tgs/iag_index_naics.htm 
# Also matches data on occupational density of each industry as calcualted in occ_emp_industry_oews.R
rel_inds <- c("Agriculture, Forestry, Fishing, and Hunting" = "11",
"Mining" = "21",
"Construction" = "23",
"Manufacturing" = "31-33",
"Wholesale Trade" = "42",
"Retail Trade" = "44-45",
"Transportation and Warehousing" = "48-49",
"Utilities" = "22",
"Information" = "51",
"Finance and Insurance" = "52",
"Real Estate and Rental and Leasing" = "53",
"Professional, Scientific, and Technical Services" = "54",
"Management of Companies and Enterprises" = "55",
"Administrative and waste management services" = "56",
"Educational Services" = "61",
"Health Care and Social Assistance" = "62",
"Arts, Entertainment, and Recreation" = "71",
"Accommodation and Food Services" = "72",
"Other Services, except government" = "81")

rel_inds %>% 
  kable()


final %>% 
  filter(tolower(industry) %in% tolower(names(rel_inds))) %>% 
  ggplot() +
  geom_line(aes(x = date, y = real_VA, color = industry)) +
  facet_wrap(~industry, scales = "free_y") + 
  theme_minimal() +
  labs(title = "Annual Real VA by Industry 1997-2024", subtitle = "Data from Bureau of Economic Analysis Economic Accounts", x = "Year", y = "Real Value Added in 2017-chained USD", color = "Industry Level of Aggregation") +
  theme(legend.position = "none")

final %>% 
  filter(tolower(industry) %in% tolower(names(rel_inds))) %>% 
  ggplot() +
  geom_line(aes(x = date, y = real_VA, color = industry)) +
  #facet_wrap(~industry, scales = "free_y") + 
  scale_y_continuous(trans = "log") +
  theme_minimal() +
  labs(title = "(log) Annual Real VA by Industry 1997-2024", subtitle = "Data from Bureau of Economic Analysis Economic Accounts", x = "Year", y = "log Real Value Added in 2017-chained USD", color = "Industry Level of Aggregation")



