# library(haven)
# library(dplyr)
# library(stringr)
# library(ipumsr)
# library(tidyverse)
# library(janitor)
# library(here)
# 
# ddi_17 <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00012.xml"))
# data_17 <- read_ipums_micro(ddi_17) %>%
#   clean_names %>% filter(year == 2017 & month == 1)
# 
# ddi <- read_ipums_ddi(here("data/behav_params/Eeckhout_Replication/cps_data/cps_00011.xml"))
# data <- read_ipums_micro(ddi) %>%
#   clean_names %>%
#   rbind(data_17)
# ipums_conditions()
# rm(ddi_17, data_17)

# -------------------------
# 1. Global Settings
# -------------------------
# destination_path <- "XXXX"  # <--- Replace XXXX with your actual directory path
# data_path <- file.path(destination_path, "Data")
# EEx_path <- file.path(destination_path, "EE_x")
master_file <- data

# -------------------------
# 2. Function Definitions
# -------------------------

# (a) recode_filter(): Filters the master data by years (i, i+1, i+2) and, for each month loop j,
# applies the month–selection and recoding rules. (There are 12 cases; the code for each is written explicitly.)
recode_filter <- function(df, i, j) {
  # Retain only years in the required range:
  df <- df %>% filter(year %in% c(i, i + 1, i + 2))
  
  if(j == 1) {
    df <- df %>% 
      filter((year == i & month %in% c(1, 2, 3, 10, 11, 12)) |
               (year == i + 1 & month %in% c(1, 2, 3))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 1  ~ 111,
        year == i     & month == 2  ~ 112,
        year == i     & month == 3  ~ 113,
        year == i     & month == 10 ~ 114,
        year == i     & month == 11 ~ 115,
        year == i     & month == 12 ~ 116,
        year == i + 1 & month == 1  ~ 117,
        year == i + 1 & month == 2  ~ 118,
        year == i + 1 & month == 3  ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 2) {
    df <- df %>% 
      filter((year == i & month %in% c(2, 3, 4, 11, 12)) |
               (year == i + 1 & month %in% c(1, 2, 3, 4))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 2 ~ 111,
        year == i     & month == 3 ~ 112,
        year == i     & month == 4 ~ 113,
        year == i     & month == 11 ~ 114,
        year == i     & month == 12 ~ 115,
        year == i + 1 & month == 1  ~ 116,
        year == i + 1 & month == 2  ~ 117,
        year == i + 1 & month == 3  ~ 118,
        year == i + 1 & month == 4  ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 3) {
    df <- df %>% 
      filter((year == i & month %in% c(3, 4, 5, 12)) |
               (year == i + 1 & month %in% c(1, 2, 3, 4, 5))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 3  ~ 111,
        year == i     & month == 4  ~ 112,
        year == i     & month == 5  ~ 113,
        year == i     & month == 12 ~ 114,
        year == i + 1 & month == 1  ~ 115,
        year == i + 1 & month == 2  ~ 116,
        year == i + 1 & month == 3  ~ 117,
        year == i + 1 & month == 4  ~ 118,
        year == i + 1 & month == 5  ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 4) {
    df <- df %>% 
      filter((year == i & month %in% c(4, 5, 6)) |
               (year == i + 1 & month %in% c(1, 2, 3, 4, 5, 6))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 4 ~ 111,
        year == i     & month == 5 ~ 112,
        year == i     & month == 6 ~ 113,
        year == i + 1 & month == 1 ~ 114,
        year == i + 1 & month == 2 ~ 115,
        year == i + 1 & month == 3 ~ 116,
        year == i + 1 & month == 4 ~ 117,
        year == i + 1 & month == 5 ~ 118,
        year == i + 1 & month == 6 ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 5) {
    df <- df %>% 
      filter((year == i & month %in% c(5, 6, 7)) |
               (year == i + 1 & month %in% c(2, 3, 4, 5, 6, 7))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 5 ~ 111,
        year == i     & month == 6 ~ 112,
        year == i     & month == 7 ~ 113,
        year == i + 1 & month == 2 ~ 114,
        year == i + 1 & month == 3 ~ 115,
        year == i + 1 & month == 4 ~ 116,
        year == i + 1 & month == 5 ~ 117,
        year == i + 1 & month == 6 ~ 118,
        year == i + 1 & month == 7 ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 6) {
    df <- df %>% 
      filter((year == i & month %in% c(6, 7, 8)) |
               (year == i + 1 & month %in% c(3, 4, 5, 6, 7, 8))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 6 ~ 111,
        year == i     & month == 7 ~ 112,
        year == i     & month == 8 ~ 113,
        year == i + 1 & month == 3 ~ 114,
        year == i + 1 & month == 4 ~ 115,
        year == i + 1 & month == 5 ~ 116,
        year == i + 1 & month == 6 ~ 117,
        year == i + 1 & month == 7 ~ 118,
        year == i + 1 & month == 8 ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 7) {
    df <- df %>% 
      filter((year == i & month %in% c(7, 8, 9)) |
               (year == i + 1 & month %in% c(4, 5, 6, 7, 8, 9))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 7 ~ 111,
        year == i     & month == 8 ~ 112,
        year == i     & month == 9 ~ 113,
        year == i + 1 & month == 4 ~ 114,
        year == i + 1 & month == 5 ~ 115,
        year == i + 1 & month == 6 ~ 116,
        year == i + 1 & month == 7 ~ 117,
        year == i + 1 & month == 8 ~ 118,
        year == i + 1 & month == 9 ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 8) {
    df <- df %>% 
      filter((year == i & month %in% c(8, 9, 10)) |
               (year == i + 1 & month %in% c(5, 6, 7, 8, 9, 10))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 8  ~ 111,
        year == i     & month == 9  ~ 112,
        year == i     & month == 10 ~ 113,
        year == i + 1 & month == 5  ~ 114,
        year == i + 1 & month == 6  ~ 115,
        year == i + 1 & month == 7  ~ 116,
        year == i + 1 & month == 8  ~ 117,
        year == i + 1 & month == 9  ~ 118,
        year == i + 1 & month == 10 ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 9) {
    df <- df %>% 
      filter((year == i & month %in% c(9, 10, 11)) |
               (year == i + 1 & month %in% c(6, 7, 8, 9, 10, 11))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 9  ~ 111,
        year == i     & month == 10 ~ 112,
        year == i     & month == 11 ~ 113,
        year == i + 1 & month == 6  ~ 114,
        year == i + 1 & month == 7  ~ 115,
        year == i + 1 & month == 8  ~ 116,
        year == i + 1 & month == 9  ~ 117,
        year == i + 1 & month == 10 ~ 118,
        year == i + 1 & month == 11 ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 10) {
    df <- df %>% 
      filter((year == i & month %in% c(10, 11, 12)) |
               (year == i + 1 & month %in% c(7, 8, 9, 10, 11, 12))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 10 ~ 111,
        year == i     & month == 11 ~ 112,
        year == i     & month == 12 ~ 113,
        year == i + 1 & month == 7  ~ 114,
        year == i + 1 & month == 8  ~ 115,
        year == i + 1 & month == 9  ~ 116,
        year == i + 1 & month == 10 ~ 117,
        year == i + 1 & month == 11 ~ 118,
        year == i + 1 & month == 12 ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 11) {
    df <- df %>% 
      filter((year == i & month %in% c(11, 12)) |
               (year == i + 1 & month %in% c(1, 8, 9, 10, 11, 12)) |
               (year == i + 2 & month == 1)) %>%
      mutate(month_temp = case_when(
        year == i     & month == 11 ~ 111,
        year == i     & month == 12 ~ 112,
        year == i + 1 & month == 1  ~ 113,
        year == i + 1 & month == 8  ~ 114,
        year == i + 1 & month == 9  ~ 115,
        year == i + 1 & month == 10 ~ 116,
        year == i + 1 & month == 11 ~ 117,
        year == i + 1 & month == 12 ~ 118,
        year == i + 2 & month == 1  ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
    
  } else if(j == 12) {
    df <- df %>% 
      filter((year == i & month == 12) |
               (year == i + 1 & month %in% c(1, 2, 9, 10, 11, 12)) |
               (year == i + 2 & month %in% c(1, 2))) %>%
      mutate(month_temp = case_when(
        year == i     & month == 12 ~ 111,
        year == i + 1 & month == 1  ~ 112,
        year == i + 1 & month == 2  ~ 113,
        year == i + 1 & month == 9  ~ 114,
        year == i + 1 & month == 10 ~ 115,
        year == i + 1 & month == 11 ~ 116,
        year == i + 1 & month == 12 ~ 117,
        year == i + 2 & month == 1  ~ 118,
        year == i + 2 & month == 2  ~ 119,
        TRUE ~ NA_real_
      )) %>%
      mutate(month = as.numeric(recode(as.character(month_temp),
                                       "111" = "1", "112" = "2", "113" = "3",
                                       "114" = "4", "115" = "5", "116" = "6",
                                       "117" = "7", "118" = "8", "119" = "9"))) %>%
      select(-month_temp)
  }
  return(df)
}

# (b) process_set(): For each “set” (1, 2, or 3) the code keeps only the relevant months
# and then recodes them to a 1–5 scale.
process_set <- function(df, set_number) {
  if(set_number == 1) {
    df_set <- df %>% 
      filter(month %in% c(1, 4, 5, 6, 7)) %>%
      mutate(month = case_when(
        month == 1 ~ 1,
        month == 4 ~ 2,
        month == 5 ~ 3,
        month == 6 ~ 4,
        month == 7 ~ 5
      ))
  } else if(set_number == 2) {
    df_set <- df %>% 
      filter(month %in% c(2, 5, 6, 7, 8)) %>%
      mutate(month = case_when(
        month == 2 ~ 1,
        month == 5 ~ 2,
        month == 6 ~ 3,
        month == 7 ~ 4,
        month == 8 ~ 5
      ))
  } else if(set_number == 3) {
    df_set <- df %>% 
      filter(month %in% c(3, 6, 7, 8, 9)) %>%
      mutate(month = case_when(
        month == 3 ~ 1,
        month == 6 ~ 2,
        month == 7 ~ 3,
        month == 8 ~ 4,
        month == 9 ~ 5
      ))
  }
  return(df_set)
}

# (c) process_first_month() and process_new_wage():
#    Extract the “first month” wage (to be later called previous_wage) and the “new wage” 
#    (keeping only observations with earnweek != 9999.99, and dropping non‑labor‐force individuals)
process_first_month <- function(df_set) {
  df_first <- df_set %>%
    filter(month == 1, labforce != 0, earnweek != 9999.99) %>%
    rename(previous_wage = earnweek) %>%
    arrange(cpsidp)
  return(df_first)
}

process_new_wage <- function(df_set) {
  df_new <- df_set %>%
    filter(month == 5, earnweek != 9999.99) %>%
    rename(new_wage = earnweek) %>%
    arrange(cpsidp)
  return(df_new)
}

# (d) merge_wages(): Combine the first and new wage files so that only individuals present in both appear.
merge_wages <- function(df_first, df_new) {
  bind_rows(df_first, df_new) %>%
    group_by(cpsidp) %>%
    filter(n() == 2) %>%  # only keep individuals with exactly 2 observations
    ungroup()
}

# (e) process_ee_flows(): Final selection for EE flows – here the sample is restricted (via a similar
# duplicate‐tagging logic) and then the employer switch conditions are computed.
# (The “job switch” check is based on comparing empstat and empsame in the two observations.)
process_ee_flows <- function(df_set, df_first) {
  df_flow <- df_set %>%
    filter(month %in% c(4, 5)) %>%        # retain the two months for EE comparisons
    mutate(month = if_else(month == 4, 1, 2)) %>%
    filter(labforce != 0) %>%
    group_by(cpsidp) %>%
    filter(n() == 2) %>%                # keep only those observed in both months
    arrange(cpsidp, month) %>%
    mutate(diffEE = if_else(first(empstat) %in% c(10, 12) & last(empstat) %in% c(10, 12), 1, 0),
           diffEEm = if_else(diffEE == 1 & last(empsame) == 1, 1, 0)) %>%
    ungroup() %>%
    filter(diffEEm == 1) %>%
    select(cpsidp, wtfinl, new_wage, earnwt)
  
  left_join(df_first %>% select(cpsidp, previous_wage), df_flow, by = "cpsidp") %>%
    filter(!is.na(new_wage))
}

# (f) Weighted median helper function.
weighted.median <- function(x, w, na.rm = FALSE) {
  if(na.rm) {
    df <- data.frame(x, w) %>% filter(!is.na(x) & !is.na(w))
    x <- df$x; w <- df$w
  }
  if(sum(w) == 0) return(NA)
  idx <- order(x)
  x_sorted <- x[idx]
  w_sorted <- w[idx]
  cumw <- cumsum(w_sorted)
  cutoff <- 0.5 * sum(w_sorted)
  x_sorted[which(cumw >= cutoff)[1]]
}

# (g) compute_wage_stats(): For a given threshold (multiplier 1, 1.05, or 1.1), calculate the weighted mean and median.
compute_wage_stats <- function(df, threshold_multiplier) {
  # Use the condition: previous_wage * multiplier < new_wage, with wage values not 9999.99.
  df_sub <- df %>%
    filter(previous_wage * threshold_multiplier < new_wage,
           previous_wage != 9999.99, new_wage != 9999.99)
  avg_wage <- weighted.mean(df_sub$new_wage, w = df_sub$earnwt, na.rm = TRUE)
  median_wage <- weighted.median(df_sub$new_wage, w = df_sub$earnwt, na.rm = TRUE)
  list(avg = avg_wage, median = median_wage)
}

# (h) compute_ee_percentages(): Compute percentage of EE moves (weighted) by comparing
# total weights of cases with wage increases versus decreases.
compute_ee_percentages <- function(df, threshold_multiplier) {
  total_greater <- sum(df$wtfinl[df$previous_wage * threshold_multiplier < df$new_wage], na.rm = TRUE)
  total_less <- sum(df$wtfinl[df$previous_wage * threshold_multiplier > df$new_wage], na.rm = TRUE)
  total_greater / (total_greater + total_less)
}


# -------------------------
# 3. Main Loop Over Years and Months
# -------------------------
# We iterate over i = 1996:2015 and j = 1:12.

result_list <- list()
counter <- 1
for(i in 1996:1996) {
  for(j in 1:6) {
    
    message("Processing year ", i, " and month index ", j)
    # Load the master data (assumed to be in Stata .dta format)
    #df_master <- master_file #read_dta(master_file)
    
    # Apply the recode and filter for the current i and j.
    df_recoded <- recode_filter(master_file, i, j)
    
    # The “All_11” dataset (your intermediary file) is now stored in df_recoded.
    # Next, process the three sets that have different month selections and recodings.
    df_set1 <- process_set(df_recoded, 1)
    df_set2 <- process_set(df_recoded, 2)
    df_set3 <- process_set(df_recoded, 3)
    
    # For each set, extract the first month wage and the new wage.
    df_first1 <- process_first_month(df_set1)
    df_new1   <- process_new_wage(df_set1)
    # CHANGED DF_NEW1 FROM DF_SET1
    df_ee1    <- process_ee_flows(df_new1, df_first1)
    
    df_first2 <- process_first_month(df_set2)
    df_new2   <- process_new_wage(df_set2)
    # CHANGED DF_NEW2 FROM DF_SET2
    df_ee2    <- process_ee_flows(df_new2, df_first2)
    
    df_first3 <- process_first_month(df_set3)
    df_new3   <- process_new_wage(df_set3)
    # CHANGED DF_NEW3 FROM DF_SET3
    df_ee3    <- process_ee_flows(df_new3, df_first3)
    
    # Combine EE observations from the three sets.
    df_ee_combined <- bind_rows(df_ee1, df_ee2, df_ee3) %>%
      group_by(cpsidp) %>%
      summarise(previous_wage = first(previous_wage),
                new_wage = mean(new_wage, na.rm = TRUE),
                wtfinl   = mean(wtfinl, na.rm = TRUE),
                earnwt   = mean(earnwt, na.rm = TRUE)) %>%
      ungroup()
    
    # For reporting the sample size (observations) in this iteration.
    obs_total <- nrow(df_ee_combined)
    
    # Compute wage summaries for the three thresholds:
    stats_x0  <- compute_wage_stats(df_ee_combined, 1)
    stats_x5  <- compute_wage_stats(df_ee_combined, 1.05)
    stats_x10 <- compute_wage_stats(df_ee_combined, 1.1)
    
    # Compute weighted percentages of EE moves.
    per_more_x0  <- compute_ee_percentages(df_ee_combined, 1)
    per_more_x5  <- compute_ee_percentages(df_ee_combined, 1.05)
    per_more_x10 <- compute_ee_percentages(df_ee_combined, 1.1)
    
    # Add date, year, and month variables as in the Stata code.
    df_ee_combined <- df_ee_combined %>%
      mutate(date  = paste0(i + 1, sprintf("%02d", j)),
             year  = i + 1,
             month = j,
             observations_total = obs_total,
             avg_wage_EE_x_0 = stats_x0$avg,
             median_EE_x_0  = stats_x0$median,
             avg_wage_EE_x_5 = stats_x5$avg,
             median_EE_x_5  = stats_x5$median,
             avg_wage_EE_x_10 = stats_x10$avg,
             median_EE_x_10  = stats_x10$median,
             per_more_x_0  = per_more_x0,
             per_more_x_5  = per_more_x5,
             per_more_x_10 = per_more_x10)
    print(df_ee_combined)
    
    # Save the result for this (i, j) iteration.
    result_list[[counter]] <- df_ee_combined
    counter <- counter + 1
    
    # Clean up objects to free memory.
    rm(df_recoded, df_set1, df_set2, df_set3,
       df_first1, df_new1, df_ee1,
       df_first2, df_new2, df_ee2,
       df_first3, df_new3, df_ee3, df_ee_combined)
    gc()
  }
}

# -------------------------
# 4. Combine and Save Final Results
# -------------------------
final_results <- bind_rows(result_list)

# Write the aggregated data to Stata .dta files.
#saveRDS(final_results, here("data/behav_params/Eeckhout_Replication/new_EE_x_data_for_quarterly_aggregation.RDS"))
saveRDS(final_results, here("data/behav_params/Eeckhout_Replication/new_EE_x.dta"))

message("Process complete. Final aggregated data saved.")
