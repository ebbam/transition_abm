# Description: Extracting long-term unemployment rate by occupation
# Cite the following:
# Publications and research reports based on the IPUMS CPS database must cite it appropriately. The citation should include the following:
#   
# Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 11.0 [dataset]. Minneapolis, MN: IPUMS, 2023.
# https://doi.org/10.18128/D030.V11.0
# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

library(tidyverse)
library(here)
library(lubridate)
library(readxl)
source(here("code/formatting/plot_dicts.R"))

if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

cw <- read.csv(here("data/crosswalk_occ_soc_cps_codes.csv")) %>% 
  tibble

source(here("data/occ_soc_cw_2010.R")) # Creates soc_desc and a get_soc_table function that can call by a specific grouping level - the source is listed in the occ_soc_cw_20101.R script
cw_SOC_minor <- get_soc_table('soc_2010_minor') %>% 
  rename_with(~gsub("soc", "SOC", gsub("2010_", "", .))) %>% 
  mutate(SOC_minor = case_when(SOC_minor == "15-1100" ~ "15-1000",
                               SOC_minor == "51-5100" ~ "51-5000",
                               TRUE ~ SOC_minor))
cw_SOC_major <- get_soc_table('soc_2010_major') %>% 
  rename_with(~gsub("soc", "SOC", gsub("2010_", "", .))) %>% 
  mutate(SOC_major = as.numeric(substr(SOC_major, 1, 2)))
cw_SOC_broad <- get_soc_table('soc_2010_broad') %>% 
  rename_with(~gsub("soc", "SOC", gsub("2010_", "", .))) %>% 
  # There are discrepancies between the CPS data and teh SOC crosswalk - I fill in the labels for those "broad categories" that are missing in the crosswalk using this site: https://www.bls.gov/soc/2010/2010_major_groups.htm#25-0000
  bind_rows(
    tribble(
      ~SOC_broad,   ~description,
      "25-1000", "Postsecondary Teachers",
      "25-3000", "Other Teachers and Instructors",
      "25-90X0", "Other Education, Training, and Library Occupations",
      "29-9000", "Other Healthcare Practitioners and Technical Occupations",
      "39-40X0", "Funeral Service Workers",
      "47-50X0", "Miscellaneous Extraction Workers",
         "53-1000", "Supervisors of Transportation and Material Moving Workers", 
         "53-40X0", "Miscellaneous Rail Transportation Workers", 
         "53-60X0", "Miscellaneous Transportation Workers"))
         
cw_SOC2010 <- get_soc_table('soc_2010') %>% 
  rename_with(~gsub("soc_", "SOC", .)) %>% 
  # There are discrepancies between the CPS data and teh SOC crosswalk - I fill in the labels for those "broad categories" that are missing in the crosswalk using this site: https://www.bls.gov/soc/2010/2010_major_groups.htm#25-0000
  bind_rows(
    tribble(
      ~SOC2010,   ~description,
      "15-113X", "Software Developers and Programmers",
      "21-109X", "Miscellaneous Community and Social Service Specialists",
      "25-90XX", "Other Education, Training, and Library Occupations",
      "31-909X", "Miscellaneous Healthcare Support Occupations",
      "33-909X", "Miscellaneous Protective Service Workers",
      "37-201X", "Building Cleaning Workers",
      "39-40XX", "Funeral Service Workers",
      "47-50XX", "Extraction Workers",
      "49-209X", "Miscellaneous Electrical and Electronic Equipment Mechanics, Installers, and Repairers",
      "49-904X", "Industrial Machinery Installation, Repair, and Maintenance Workers",
      "49-909X", "Miscellaneous Installation, Maintenance, and Repair Workers",
      "53-60XX", "Other Transportation Workers" ,
      "53-40XX", "Rail Transportation Workers"))
      
          

cw_short <- cw %>% 
  select(OCC2010_cps, SOC2010, SOC_major, SOC_minor, SOC_broad) %>% 
  group_by(OCC2010_cps) %>% 
  summarise(SOC2010 = first(SOC2010),
            SOC_major = first(SOC_major),
            SOC_minor = first(SOC_minor),
            SOC_broad = first(SOC_broad)) %>% 
  ungroup
# ddi2 <- read_ipums_ddi(here("data/macro_vars/CPS_LTUER/cps_00005.xml"))
# data2 <- read_ipums_micro(ddi)
# data %>% select(names(data2)) %>% identical(data2)

ddi <- read_ipums_ddi(here("data/occ_macro_vars/CPS_LTUER/cps_00016.xml"))
data <- read_ipums_micro(ddi)

ipums_conditions()

# Reference data from FRED
macros <- read.csv(here('data/macro_vars/LNS13025703.csv')) %>%
  tibble %>% 
  rename(LTUER = 2) %>% 
  mutate(YEAR = year(DATE),
         LTUER = as.numeric(LTUER)/100,
         DATE = as.Date(DATE))

filtered <- data %>% 
  left_join(., cw_short, by = join_by(OCC2010 == OCC2010_cps)) %>% 
  filter(YEAR > 2003 & EMPSTAT %in% c(10, 12, 21, 22)) %>%  # Occupation classification scheme was redone in 2002, effective in 2003: https://cps.ipums.org/cps/occ_transition_2002_2010.shtml ; also filtered to include only unemployed workers. 
  select(YEAR, MONTH, WTFINL, OCC2010, DURUNEMP, DURUNEM2, EMPSTAT, HWTFINL, names(cw_short[-1])) %>% # Select only relevant variables.
  mutate(DURUNEMP = ifelse(DURUNEMP == 999, NA, DURUNEMP),
         DURUNEM2 = ifelse(DURUNEM2 == 9, NA, DURUNEM2)) %>% # Replace "99" value with NA - 99 = NIU ie. "Not in universe" or simply not applicable to the correspondent. See explanation here: https://cps.ipums.org/cps-action/faq (Ctrl+F "NIU")
  mutate(UE = EMPSTAT %in% c(21,22),
         LF = EMPSTAT %in% c(10, 12, 21, 22),
         LTUE1 = DURUNEMP >= 27,
         LTUE2 = DURUNEM2 >= 12
         ) # Counts the categories 5 (27-39 weeks), 6 (40+ weeks), and 7 (Over 26 weeks - period 1962-1967) as LTUE
  

library(ggridges)
library(viridis)
ridges <- filtered %>% 
  mutate(DATE = as.factor(as.Date(paste0(YEAR, "-", MONTH, "-01")))) %>% 
  group_by(YEAR, DATE, DURUNEM2) %>% 
  summarise(n = n()) %>% 
  ungroup

ridges %>% 
  group_by(YEAR, DURUNEM2) %>% 
  summarise(n = mean(n, na.rm = TRUE)) %>% 
  ungroup %>% 
  filter(DURUNEM2 != 0) %>% 
  ggplot(., aes(x = DURUNEM2, y = YEAR, height = n, group = YEAR)) + 
  geom_density_ridges(stat = "identity", scale = 1) +
  theme(legend.position = "none") + coord_flip() + common_theme + theme(fill = "lightblue") +
  labs(y = "Year", x = "Unemployment Duration")

ridges2 <- filtered %>% 
  mutate(DATE = as.factor(as.Date(paste0(YEAR, "-", MONTH, "-01")))) %>% 
  group_by(YEAR, DATE, DURUNEMP) %>% 
  summarise(n = n()) %>% 
  ungroup

ridges2 %>% 
  group_by(YEAR, DURUNEMP) %>% 
  summarise(n = mean(n, na.rm = TRUE)) %>% 
  ungroup %>% 
  filter(DURUNEMP != 0) %>% 
  ggplot(., aes(x = DURUNEMP/4, y = YEAR, height = n, group = YEAR)) + 
  geom_density_ridges(stat = "identity", scale = 1) +
  theme(legend.position = "none") + 
  coord_flip() + common_theme + 
  labs(y = "Year", x = "Unemployment Duration (Months)", title = "Distribution of Unemployment Duration (Annual Average)") 

ggsave(here('data/occ_macro_vars/CPS_LTUER/unemp_distributions_cps.png'))

ridges2 %>% 
  group_by(YEAR, DURUNEMP) %>% 
  # summarise(n = mean(n, na.rm = TRUE)) %>% 
  # ungroup %>% 
  filter(DURUNEMP != 0) %>% 
  ggplot(., aes(x = DURUNEMP/4, y = YEAR, group = YEAR)) + 
  geom_violin(trim =FALSE) +
  theme(legend.position = "none") + 
  coord_flip() + common_theme + theme(fill = "lightblue") +
  labs(y = "Year", x = "Unemployment Duration (Months)")

ridges2 %>% 
  group_by(YEAR, DURUNEM) %>% 
  summarise(n = mean(n, na.rm = TRUE)) %>% 
  ungroup %>% 
  filter(DURUNEM2 != 0) %>% 
  ggplot(aes(x = DURUNEM2, y = YEAR, group = YEAR, height = n)) + 
  geom_density_ridges(stat = "binline", bins = 30, scale = 0.2, draw_baseline = TRUE) +
  theme(legend.position = "none") + coord_flip() + common_theme 

filtered %>% 
  mutate(DATE = as.factor(as.Date(paste0(YEAR, "-", MONTH, "-01")))) %>% 
  filter(DURUNEM2 != 0) %>% 
  ggplot(aes(x = DATE, y = DURUNEM2)) +
    geom_density() + 
  common_theme

################################################################################
################### YEARLY DURATION DISTRIBUTION BY SOC MAJOR #################
################################################################################

# dur_dist_yearly_SOC_major <- filtered %>%
#   filter(UE == TRUE, DURUNEMP > 0, !is.na(DURUNEMP), !is.na(SOC_major)) %>%
#   mutate(dur_months = round(DURUNEMP / 4.33)) %>%
#   group_by(YEAR, SOC_major, dur_months) %>%
#   summarise(n_weighted = sum(WTFINL), .groups = "drop") %>%
#   group_by(YEAR, SOC_major) %>%
#   mutate(prop = n_weighted / sum(n_weighted)) %>%
#   ungroup()
# 
# write.csv(dur_dist_yearly_SOC_major,
#           here("data/occ_macro_vars/CPS_LTUER/unemp_duration_dist_yearly_SOC_major.csv"),
#           row.names = FALSE)

################### YEARLY DURATION DISTRIBUTION BY ALL SOC LEVELS ##############
# this new aggrgeation was created more recently to save a separate file for each occupational category

soc_levels <- c("SOC_major", "SOC_minor", "SOC_broad", "SOC2010")

for (level in soc_levels) {
  out <- filtered %>%
    filter(UE == TRUE, DURUNEMP > 0, !is.na(DURUNEMP), !is.na(.data[[level]])) %>%
    mutate(dur_months = round(DURUNEMP / 4.33),
           group_id   = .data[[level]]) %>%
    group_by(YEAR, group_id, dur_months) %>%
    summarise(n_weighted = sum(WTFINL), .groups = "drop") %>%
    group_by(YEAR, group_id) %>%
    mutate(prop = n_weighted / sum(n_weighted)) %>%
    ungroup() %>% 
    left_join(., get(paste0("cw_", level)), by = c("group_id" = level)) %>% 
    mutate(YEAR = as.integer(YEAR))
  
  if(level == "SOC2010"){
    out <- out %>% 
      filter(is.na(description)) %>% 
      select(-description) %>% 
      left_join(., cw_SOC_broad, by = c("group_id" = "SOC_broad")) %>% 
      rbind(filter(out, !is.na(description)), .)
    
    stopifnot(out %>% filter(is.na(description)) %>% nrow(.) == 0)
    stopifnot(out %>% select(-description) %>% arrange(YEAR, group_id) %>% all.equal(tibble(read.csv(here(paste0("data/occ_macro_vars/CPS_LTUER/unemp_duration_dist_yearly_", level, ".csv"))))))
    
  }else{
    stopifnot(out %>% filter(is.na(description)) %>% nrow(.) == 0)
    stopifnot(out %>% select(-description) %>% all.equal(tibble(read.csv(here(paste0("data/occ_macro_vars/CPS_LTUER/unemp_duration_dist_yearly_", level, ".csv"))))))
  }
  
  write.csv(out,
            here(paste0("data/occ_macro_vars/CPS_LTUER/unemp_duration_dist_yearly_",
                        level, "_w_labels.csv")),
            row.names = FALSE)
  cat("Written:", level, "\n")
}

################################################################################
################### YEARLY UER & LTUER RATES BY ALL SOC LEVELS ################
################################################################################

for (level in soc_levels) {
  out_rates <- filtered %>%
    filter(!is.na(.data[[level]])) %>%
    mutate(group_id = .data[[level]]) %>%
    group_by(YEAR, group_id) %>%
    summarise(n_ue   = sum(WTFINL * UE,         na.rm = TRUE),
              n_lf   = sum(WTFINL * LF,         na.rm = TRUE),
              n_ltue = sum(WTFINL * LTUE1 * UE, na.rm = TRUE),
              .groups = "drop") %>%
    mutate(uer   = n_ue / n_lf,
           ltuer = n_ltue / n_ue,
           YEAR  = as.integer(YEAR)) %>%
    left_join(., get(paste0("cw_", level)), by = c("group_id" = level))
  
  if (level == "SOC2010") {
    out_rates <- out_rates %>%
      filter(is.na(description)) %>%
      select(-description) %>%
      left_join(., cw_SOC_broad, by = c("group_id" = "SOC_broad")) %>%
      rbind(filter(out_rates, !is.na(description)), .)
  }
  
  stopifnot(out_rates %>% filter(is.na(description)) %>% nrow(.) == 0)
  
  write.csv(out_rates,
            here(paste0("data/occ_macro_vars/CPS_LTUER/unemp_rate_yearly_",
                        level, ".csv")),
            row.names = FALSE)
  cat("Written UER/LTUER yearly rates:", level, "\n")
}

deseasonalize_group <- function(df, value_col) {
  df <- df %>% arrange(YEAR, MONTH)
  start_yr <- df$YEAR[1];        start_mo <- df$MONTH[1]
  end_yr   <- df$YEAR[nrow(df)]; end_mo   <- df$MONTH[nrow(df)]
  
  full_dates <- seq(as.Date(paste(start_yr, start_mo, 1, sep = "-")),
                    as.Date(paste(end_yr, end_mo, 1, sep = "-")), by = "month")
  if (length(full_dates) < 24) {
    return(tibble(YEAR = year(full_dates), MONTH = month(full_dates),
                  seasonal_decompose = NA_real_, seasonal_stl = NA_real_))
  }
  
  full_df <- tibble(YEAR = year(full_dates), MONTH = month(full_dates)) %>%
    left_join(df %>% select(YEAR, MONTH, value = !!sym(value_col)), by = c("YEAR", "MONTH"))
  
  vals <- full_df$value
  if (any(is.na(vals))) {
    vals <- approx(seq_along(vals)[!is.na(vals)], vals[!is.na(vals)],
                   xout = seq_along(vals), rule = 2)$y
  }
  
  ts_obj <- ts(vals, start = c(start_yr, start_mo), frequency = 12)
  
  full_df$seasonal_decompose <- as.numeric(
    decompose(ts_obj, type = "additive")$seasonal
  )
  full_df$seasonal_stl <- as.numeric(
    stl(ts_obj, s.window = "periodic", robust = TRUE)$time.series[, "seasonal"]
  )
  
  full_df %>% select(YEAR, MONTH, seasonal_decompose, seasonal_stl)
}

for (level in soc_levels_monthly) {
  out_rates <- filtered %>%
    filter(!is.na(.data[[level]])) %>%
    mutate(group_id = .data[[level]]) %>%
    group_by(YEAR, MONTH, group_id) %>%
    summarise(n_ue   = sum(WTFINL * UE,         na.rm = TRUE),
              n_lf   = sum(WTFINL * LF,         na.rm = TRUE),
              n_ltue = sum(WTFINL * LTUE1 * UE, na.rm = TRUE),
              .groups = "drop") %>%
    mutate(uer   = n_ue / n_lf,
           ltuer = n_ltue / n_ue,
           YEAR  = as.integer(YEAR)) %>%
    left_join(., get(paste0("cw_", level)), by = c("group_id" = level))
  
  if (level == "SOC2010") {
    out_rates <- out_rates %>%
      filter(is.na(description)) %>%
      select(-description) %>%
      left_join(., cw_SOC_broad, by = c("group_id" = "SOC_broad")) %>%
      rbind(filter(out_rates, !is.na(description)), .)
  }
  
  stopifnot(out_rates %>% filter(is.na(description)) %>% nrow(.) == 0)
  
  out_rates <- out_rates %>%
    group_by(group_id) %>%
    group_modify(~ {
      seas_uer   <- deseasonalize_group(.x, "uer") %>%
        rename(uer_seasonal = seasonal_decompose, uer_seasonal_stl = seasonal_stl)
      seas_ltuer <- deseasonalize_group(.x, "ltuer") %>%
        rename(ltuer_seasonal = seasonal_decompose, ltuer_seasonal_stl = seasonal_stl)
      .x %>%
        left_join(seas_uer,   by = c("YEAR", "MONTH")) %>%
        left_join(seas_ltuer, by = c("YEAR", "MONTH"))
    }) %>%
    ungroup() %>%
    mutate(uer_sa       = uer   - uer_seasonal,
           uer_sa_stl    = uer   - uer_seasonal_stl,
           ltuer_sa      = ltuer - ltuer_seasonal,
           ltuer_sa_stl  = ltuer - ltuer_seasonal_stl)
  
  write.csv(out_rates,
            here(paste0("data/occ_macro_vars/CPS_LTUER/unemp_rate_monthly_",
                        level, ".csv")),
            row.names = FALSE)
  cat("Written UER/LTUER monthly rates (incl. both seasonal adjustments):", level, "\n")
}


################################################################################
################### NATIONAL RATE ##############################################
################################################################################

ltuer_overall <- filtered %>% 
  # Total LTUER - same calculation as above
  group_by(YEAR, MONTH) %>% 
  summarise(n_ue = sum(WTFINL*UE, na.rm = TRUE),
            n_lf = sum(WTFINL, na.rm = TRUE),
            n_ltue = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer = n_ltue/n_ue, 
         uer = n_ue/n_lf,
         DATE = as.Date(paste0(YEAR, "-", MONTH, "-01")))

ggplot() +
  geom_line(data = ltuer_overall, aes(x = DATE, y = ltuer)) +
  geom_line(data = filter(macros, YEAR >= 2004), aes(x = DATE, y = LTUER), linetype = "dashed")

ggplot() +
  geom_line(data = ltuer_overall, aes(x = DATE, y = uer)) +
  geom_line(data = filter(macros, YEAR >= 2004), aes(x = DATE, y = LTUER), linetype = "dashed")

ltuer_occ <- filtered %>%   # LTUER by Occupation
  group_by(OCC2010) %>% # Group by year, and occupation
  summarise(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
            n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
         uer_occ = n_ue_occ/n_lf_occ,
         ltuer_occ_filled = ifelse(ltuer_occ %in% c(0, 1), NA, ltuer_occ),
         uer_occ_filled = ifelse(uer_occ %in% c(0, 1), NA, ltuer_occ),
         occ_cat = as.factor(zap_labels(OCC2010)),
         spec = as.character(nchar(OCC2010))) # calculate LTUER as N_LTUE/N_LF 

################################################################################
################### OCCUPATIONAL CATEGORIES ####################################
################################################################################
occ_cats <- names(cw_short[-1])
ltuer_list <- list()
for(cat in occ_cats){
  cat = sym(cat)
  temp <- filtered %>%   # LTUER by Occupation
    group_by(!!cat) %>% # Group by year, and occupation
    summarise(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
              n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
              n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
    ungroup %>% 
    mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
           uer_occ = n_ue_occ/n_lf_occ)
           #ltuer_occ_filled = ifelse(ltuer_occ %in% c(0, 1), NA, ltuer_occ))
  print(temp)
  
  p1 <- temp %>% 
    arrange(ltuer_occ) %>% 
    mutate(occupation = factor(!!cat, levels = !!cat)) %>% 
    ggplot() +
    geom_point(aes(x = occupation, y = ltuer_occ)) 
  
  print(p1)
  
}

ltuer_grouped <- filtered %>%
  group_by(OCC2010) %>% # LTUER by Occupation
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
            n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
            n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
         uer_occ = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC2010) %>% # Group by year, and occupation
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc2010 = (n_ltue_occ/n_ue_occ),
         uer_soc2010 = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_major) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_major = (n_ltue_occ/n_ue_occ),
         uer_soc_major = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_minor) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_minor = (n_ltue_occ/n_ue_occ),
         uer_soc_minor = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_broad) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_broad = (n_ltue_occ/n_ue_occ),
         uer_soc_broad = (n_ue_occ/n_lf_occ)) %>% 
  select(OCC2010, SOC2010, SOC_major, SOC_minor, SOC_broad, contains("uer")) %>% 
  distinct

LTUER <- ltuer_grouped %>% 
  arrange(ltuer_occ) %>% 
  mutate(occupation = factor(OCC2010, levels = OCC2010)) %>% 
  ggplot(aes(x = occupation)) +
  geom_point(aes(y = ltuer_occ), color = "black", alpha = 0.5) + 
  geom_point(aes(y = ltuer_soc2010), color = "blue", alpha = 0.5) +
  geom_point(aes(y = ltuer_soc_major), color = "purple", alpha = 0.5) +
  geom_point(aes(y = ltuer_soc_minor), color = "darkorange", alpha = 0.5) +
  geom_point(aes(y = ltuer_soc_broad), color = "darkgreen", alpha = 0.5) +
  labs(x = "Occupational Category", y = "LTUER", title = "LTUER")


UER <- ltuer_grouped %>% 
  filter(!is.na(SOC_broad)) %>% 
  arrange(uer_occ) %>% 
  mutate(occupation = factor(OCC2010, levels = OCC2010)) %>% 
  ggplot(aes(x = occupation)) +
  geom_point(aes(y = uer_occ), color = "black") + 
  geom_point(aes(y = uer_soc2010), color = "blue") +
  geom_point(aes(y = uer_soc_major), color = "purple") +
  geom_point(aes(y = uer_soc_minor), color = "darkorange") +
  geom_point(aes(y = uer_soc_broad), color = "darkgreen") +
  labs(x = "Occupational Category", y = "UER", title = "UER") +
  ylim(c(0,0.25))


ltuer_grouped_ts <- filtered %>%
  group_by(year, month, OCC2010) %>% # LTUER by Occupation
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% 
  ungroup %>% 
  mutate(ltuer_occ = (n_ltue_occ/n_ue_occ),
         uer_occ = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC2010) %>% # Group by year, and occupation
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc2010 = (n_ltue_occ/n_ue_occ),
         uer_soc2010 = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_major) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_major = (n_ltue_occ/n_ue_occ),
         uer_soc_major = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_minor) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_minor = (n_ltue_occ/n_ue_occ),
         uer_soc_minor = (n_ue_occ/n_lf_occ)) %>% 
  group_by(SOC_broad) %>% 
  mutate(n_ue_occ = sum(WTFINL*UE, na.rm = TRUE), # labour force in particular occupation
         n_lf_occ = sum(WTFINL*LF, na.rm = TRUE),
         n_ltue_occ = sum(WTFINL*LTUE1*UE, na.rm = TRUE)) %>% # number of individuals in LTUE per occupation
  ungroup %>% 
  mutate(ltuer_soc_broad = (n_ltue_occ/n_ue_occ),
         uer_soc_broad = (n_ue_occ/n_lf_occ)) %>% 
  select(OCC2010, SOC2010, SOC_major, SOC_minor, SOC_broad, contains("uer")) %>% 
  distinct

library(patchwork)
LTUER + UER + plot_annotation(title = "UER & LTUER by Occupational Category (n = 464)")

# Create real-world observations to compare to in python code
cw %>% 
  left_join(ltuer_grouped, by = c("OCC2010_cps" = "OCC2010"), suffix = c("_orig", "_adj")) %>% 
  write.csv(here("data/occ_macro_vars/CPS_LTUER/occ_uer_ltuer_observed.csv"))

soc_levels <- c("SOC_major", "SOC_minor", "SOC_broad", "SOC2010")
t2 <- read.csv(here("data/occ_macro_vars/CPS_LTUER/occ_uer_ltuer_observed.csv")) %>% tibble

for (level in soc_levels) {
  t1 <- read.csv(here(paste0("data/occ_macro_vars/CPS_LTUER/unemp_duration_dist_yearly_",
                        level, "_w_labels.csv"))) %>% tibble %>% 
    select(group_id, description) %>% 
    distinct %>% 
    tibble %>% 
    rename_with(~paste0(level, "_", .x), description)
  
  stopifnot(length(setdiff(pull(t2, paste0(level,"_adj")), t1$group_id)) == 0)
  stopifnot(length(setdiff(t1$group_id, pull(t2, paste0(level,"_adj")))) == 0)
  cat("Labelled:", level, "\n")
  
  t2 <- left_join(t2, t1, join_by(!!sym(paste0(level, "_adj")) == group_id))
  
  stopifnot(t2 %>% filter(is.na(!!sym(paste0(level, "_description")))) %>% nrow(.) == 0)
}

t2 %>% write.csv(here("data/occ_macro_vars/CPS_LTUER/occ_uer_ltuer_observed_w_labels.csv"))

  
#ltuer_occ_filled = ifelse(ltuer_occ %in% c(0, 1), NA, ltuer_occ))
print(temp)

p1 <- temp %>% 
  arrange(ltuer_occ) %>% 
  mutate(occupation = factor(!!cat, levels = !!cat)) %>% 
  ggplot() +
  geom_point(aes(x = occupation, y = ltuer_occ)) 

print(p1)

################################################################################
################################################################################
################## TEST AGAINST ONET
################################################################################
################################################################################

# THIS IS FINE FOR SOC_major and SOC_minor for now 
# Will need to bring this to SOC2010 and SOC_broad eventually

cw <- read.csv(here("data/occ_macro_vars/CPS_LTUER/occ_uer_ltuer_observed_w_labels.csv")) %>% tibble()

onet_vars <- read.csv("/Users/ebbamark/OneDrive - Nexus365/GenerateOccMobNets/ONET/acs_onet_2010_soc_cw.csv") %>%
  tibble() %>%
  mutate(
    SOC_broad_adj = paste0(substr(soc_2010_code, 1, 6), "0"),
    SOC_major_adj = substr(soc_2010_code, 1, 2),
    SOC_minor_adj = paste0(substr(soc_2010_code, 1, 4), "000"),
    SOC2010_adj   = soc_2010_code
  )

# get description columns from cw — join on each grouping level separately
major_desc <- cw %>% distinct(SOC_major_adj, SOC_major_description) %>% mutate(SOC_major_adj = as.character(SOC_major_adj))
minor_desc <- cw %>% distinct(SOC_minor_adj, SOC_minor_description)
broad_desc <- cw %>% distinct(SOC_broad_adj, SOC_broad_description)
soc2010_desc <- cw %>% distinct(SOC2010_adj, SOC2010_description)

onet_cw <- onet_vars %>%
  left_join(major_desc,  by = "SOC_major_adj") %>%
  left_join(minor_desc,  by = "SOC_minor_adj") %>%
  left_join(broad_desc,  by = "SOC_broad_adj") %>%
  left_join(soc2010_desc, by = "SOC2010_adj") %>%
  select(X, acs_2010_code, soc_2010_code,
         SOC_major_adj, SOC_major_description,
         SOC_minor_adj, SOC_minor_description,
         SOC_broad_adj, SOC_broad_description,
         SOC2010_adj,   SOC2010_description) %>% 
  rename(acs_occ_code = acs_2010_code)
  

write.csv(onet_cw,
          here("calibration_remote/data/occ_uer_ltuer_observed_onet_w_labels.csv"),
          row.names = FALSE)


# SOC2010
setdiff(onet_vars$SOC2010, cw$SOC2010_adj)
setdiff(cw$SOC2010_adj, onet_vars$SOC2010)

# SOC_major - FINE!
setdiff(onet_vars$SOC_major, cw$SOC_major_adj)
setdiff(cw$SOC_major_adj, onet_vars$SOC_major)

# SOC_broad
setdiff(onet_vars$SOC_broad, cw$SOC_broad_adj)
setdiff(cw$SOC_broad_adj, onet_vars$SOC_broad)

#SOC_minor - FINE!
setdiff(onet_vars$SOC_minor, cw$SOC_minor_adj)
setdiff(cw$SOC_minor_adj, onet_vars$SOC_minor)



################################################################################
################### SURVIVAL CURVE #############################################
################################################################################

# calculate economy-wide, occupational (SOC_major (n = 23) and SOC_minor (n = 95)) survival rates
 survival_obs <- filtered %>%
  filter(UE == TRUE, DURUNEMP > 0, !is.na(DURUNEMP)) %>%
  mutate(dur_months = DURUNEMP / 4.33,
         PHASE = case_when(
           YEAR >= 2001 & YEAR <= 2007 ~ "Pre-GFC (2001-2007)",
           YEAR >= 2008 & YEAR <= 2009 ~ "GFC recession (2008-09)",
           YEAR >= 2010 & YEAR <= 2019 ~ "Post-GFC (2010-2019)"
         )) %>%
  filter(!is.na(PHASE)) 


survival_obs_national <- survival_obs %>% 
  group_by(PHASE) %>%
  summarise(
    dur = list(dur_months),
    weights = list(WTFINL)
  ) %>%
  rowwise() %>%
  mutate(
    thresholds = list(seq(1, 24, by = 1)),
    survival = list(sapply(thresholds, function(d)
      sum(weights[dur > d]) / sum(weights)))
  ) %>%
  unnest(c(thresholds, survival)) %>%
  select(PHASE, months = thresholds, survival)

survival_obs_SOC_major <- survival_obs %>% 
  group_by(PHASE, SOC_major) %>%
  summarise(
    dur = list(dur_months),
    weights = list(WTFINL)
  ) %>%
  rowwise() %>%
  mutate(
    thresholds = list(seq(1, 24, by = 1)),
    survival = list(sapply(thresholds, function(d)
      sum(weights[dur > d]) / sum(weights)))
  ) %>%
  unnest(c(thresholds, survival)) %>%
  select(PHASE, SOC_major, months = thresholds, survival)

survival_obs_SOC_minor <- survival_obs %>% 
  group_by(PHASE, SOC_minor) %>%
  summarise(
    dur = list(dur_months),
    weights = list(WTFINL)
  ) %>%
  rowwise() %>%
  mutate(
    thresholds = list(seq(1, 24, by = 1)),
    survival = list(sapply(thresholds, function(d)
      sum(weights[dur > d]) / sum(weights)))
  ) %>%
  unnest(c(thresholds, survival)) %>%
  select(PHASE, SOC_minor, months = thresholds, survival)

survival_obs_SOC_minor %>% ggplot(aes(x = months, y = survival, color = SOC_minor)) + geom_line() + facet_wrap(~PHASE) + theme(legend.position = "none")


write.csv(survival_obs_national, here("data/occ_macro_vars/CPS_LTUER/survival_curves_by_phase_economy_wide.csv"), row.names = FALSE)
# N represents the number of occupational categories
write.csv(survival_obs_SOC_major, here("data/occ_macro_vars/CPS_LTUER/survival_curves_by_phase_occupational_SOC_major_n23.csv"), row.names = FALSE)
write.csv(survival_obs_SOC_minor, here("data/occ_macro_vars/CPS_LTUER/survival_curves_by_phase_occupational_SOC_minor_n95.csv"), row.names = FALSE)


################################################################################
################### ANNUAL & MONTHLY DURATION DISTRIBUTION (AGGREGATE) ########
################################################################################

# Annual: monthly weighted counts averaged over months within year
dur_dist_yearly <- filtered %>%
  mutate(DURUNEMP_MO = round(DURUNEMP / 4.33)) %>%
  group_by(YEAR, MONTH, DURUNEMP_MO) %>%
  summarise(n = sum(WTFINL), .groups = "drop") %>%
  group_by(YEAR, DURUNEMP_MO) %>%
  summarise(n = mean(n), .groups = "drop")

write.csv(dur_dist_yearly,
          here("calibration_remote/data/ltuer_cps_data_new.csv"),
          row.names = FALSE)  # set row.names=TRUE if you want the leading integer index

# Monthly: same but keep MONTH — schema ['YEAR', 'MONTH', 'DURUNEMP_MO', 'n']
dur_dist_monthly <- filtered %>%
  mutate(DURUNEMP_MO = round(DURUNEMP / 4.33)) %>%
  group_by(YEAR, MONTH, DURUNEMP_MO) %>%
  summarise(n = sum(WTFINL), .groups = "drop")

write.csv(dur_dist_monthly,
          here("calibration_remote/data/ltuer_cps_data_monthly.csv"),
          row.names = FALSE)



