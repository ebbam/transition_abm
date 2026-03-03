# Cleaning Lightcast vacancy data
library(here)
library(tidyverse)
library(readxl)
library(R.utils)
new_data = FALSE

#temp <- read.csv("/Users/ebbamark/Downloads/data_0_0_0.csv")


# Save variable names to decide what to keep
# temp %>% 
#   names %>% tibble %>% rename("var_label" = 1) %>% 
#   write.xlsx(here("data/lightcast_vacancies/var_labels.xlsx"))


keep_vars <- read_xlsx(here("data/lightcast_vacancies/var_labels.xlsx")) %>% 
  mutate(levels = as.logical(ifelse(is.na(levels), 0, levels))) %>% 
  filter(levels) %>% 
  pull(var_label) %>% 
  unique

# for(yr in 2012:2013){
#   print(yr)
#   for(mth in sprintf("%02d", 1:12)){
#     soc_list <- list()
#     print(mth)
#     sub_file_i = 1
#     file_list = list.files(paste0("/Volumes/projects/SSEE_Lightcast_Data/us/csv/fortnightly/all/20240308/", yr, "/all_for_", yr, "-", mth, "-01/"))
#     if(length(file_list) == 0){
#       stop("Cannot access files in filestore!")
#     }
#     for(file_name in file_list){
#       if(file.exists(here(paste0("data/lightcast_vacancies/lightcast_monthly_", yr, "_", mth,".RDS")))){
#         print("Skipping month")
#         next
#         }
#       print(paste0(sub_file_i, file_name))
#       full_file_name <- paste0("/Volumes/projects/SSEE_Lightcast_Data/us/csv/fortnightly/all/20240308/", yr, "/all_for_", yr, "-", mth, "-01/", file_name)
#       gunzip(paste0(full_file_name), destname = here(paste0("data/lightcast_vacancies/", gsub(".gz", "", file_name))), remove = FALSE, overwrite = TRUE)
#       temp <- read.csv(here(paste0("data/lightcast_vacancies/", gsub(".gz", "", file_name))))
# 
#       for (soc_i in 2:5) {
#         soc_var  <- paste0("soc_", soc_i)
#         name_var <- paste0("soc_", soc_i, "_name")
# 
#         # Assert statements
#         stopifnot(nrow(distinct(select(temp, all_of(c(soc_var, name_var))))) == n_distinct(temp[[soc_var]]))
#         stopifnot(nrow(distinct(select(temp, all_of(c(soc_var, name_var))))) == n_distinct(temp[[name_var]]))
#         stopifnot(n_distinct(temp[[soc_var]]) == n_distinct(temp[[name_var]]))
# 
#         soc_list[[soc_var]] <- temp %>%
#           select(all_of(keep_vars)) %>%
#           tibble() %>%
#           filter(employment_type == 1) %>%
#           select(-contains("2021")) %>%
#           group_by(across(all_of(c(soc_var, name_var)))) %>%
#           summarise(n_ft_vacancies = n(), .groups = "drop") %>% 
#           mutate(year = yr, month = mth, 
#                  sub_file_no = sub_file_i) %>% 
#           rbind(soc_list[[soc_var]], .)
#       }
#       file.remove(here(paste0("data/lightcast_vacancies/", gsub(".gz", "", file_name))))
#       sub_file_i = sub_file_i + 1
#     }
#     saveRDS(soc_list, here(paste0("data/lightcast_vacancies/lightcast_monthly_", yr, "_", mth,".RDS")))
#     print("Save monthly file.")
#   }
#   gc()
# }

if(new_data){
#parallel::mclapply(2024:2024, function(yr) {

  parallel::mclapply(sprintf("%02d", 1:12), function(mth) {#
    yr = 2024
    soc_list <- list()
    print(paste(yr, mth))
    sub_file_i = 1

    file_list = list.files(paste0(
      "/Volumes/projects/SSEE_Lightcast_Data/us/csv/fortnightly/all/20240308/",
      yr, "/all_for_", yr, "-", mth, "-01/"))

    if(length(file_list) == 0){
      stop("Cannot access files in filestore!")
    }

    if(file.exists(here(paste0("data/lightcast_vacancies/lightcast_monthly_", yr, "_", mth, ".RDS")))){
      print("Skipping month")
      next
    }
    for(file_name in file_list){

      print(paste0(sub_file_i, file_name))
      full_file_name <- paste0(
        "/Volumes/projects/SSEE_Lightcast_Data/us/csv/fortnightly/all/20240308/",
        yr, "/all_for_", yr, "-", mth, "-01/", file_name)

      # Unique temp filename per core to avoid collisions
      local_file <- here(paste0("data/lightcast_vacancies/", yr, "_", mth, "_", gsub(".gz", "", file_name)))
      gunzip(full_file_name, destname = local_file, remove = FALSE, overwrite = TRUE)
      temp <- read.csv(local_file)

      for (soc_i in 2:5) {
        soc_var  <- paste0("soc_", soc_i)
        name_var <- paste0("soc_", soc_i, "_name")

        stopifnot(nrow(distinct(select(temp, all_of(c(soc_var, name_var))))) == n_distinct(temp[[soc_var]]))
        stopifnot(nrow(distinct(select(temp, all_of(c(soc_var, name_var))))) == n_distinct(temp[[name_var]]))
        stopifnot(n_distinct(temp[[soc_var]]) == n_distinct(temp[[name_var]]))

        soc_list[[soc_var]] <- temp %>%
          select(all_of(keep_vars)) %>%
          tibble() %>%
          filter(employment_type == 1) %>%
          select(-contains("2021")) %>%
          group_by(across(all_of(c(soc_var, name_var)))) %>%
          summarise(n_ft_vacancies = n(), .groups = "drop") %>%
          mutate(year = yr, month = mth, sub_file_no = sub_file_i) %>%
          rbind(soc_list[[soc_var]], .)
      }

      file.remove(local_file)
      sub_file_i = sub_file_i + 1
    }

    saveRDS(soc_list, here(paste0("data/lightcast_vacancies/lightcast_monthly_", yr, "_", mth, ".RDS")))
    print(paste("Saved:", yr, mth))
  }, mc.cores = parallel::detectCores() - 3)

}

for(yr in 2010:2023){
  if(yr == 2010){
    tot_files = 0
  }
  n_files <- length(list.files("/Volumes/grte3169/vacancy_data", pattern = paste0("lightcast_monthly_", yr, ".*\\.RDS")))
  tot_files = tot_files + n_files
  print(paste0(yr, ": ", ifelse(n_files == 12, "Complete!", n_files)))
  if(yr == 2023){
    print(tot_files)
  }
}


total_soc_list <- list()
for(yr in 2010:2024){
  print(yr)
  annual_soc_list <- list()
  for(mth in sprintf("%02d", 1:12)){
    print(mth)
    if(!file.exists(here(paste0("data/lightcast_vacancies/lightcast_monthly_", yr, "_", mth,".RDS")))){
      print("Skipping month.")
      next
    }
    mo_temp <- readRDS(here(paste0("data/lightcast_vacancies/lightcast_monthly_", yr, "_", mth,".RDS")))
    for (soc_i in 2:5) {
      soc_var  <- paste0("soc_", soc_i)
      name_var <- paste0("soc_", soc_i, "_name")

      tbl <- mo_temp[[soc_var]] %>% 
        group_by(across(all_of(c(soc_var, name_var, "year", "month")))) %>%
        summarise(n_ft_vacancies = sum(n_ft_vacancies, na.rm = TRUE), .groups = "drop") %>%
        ungroup %>% 
        mutate(year = yr, 
               month = mth,
               date = as.Date(paste0(year, "-", month, "-01")))

      annual_soc_list[[soc_var]] <- tbl %>%
        rbind(annual_soc_list[[soc_var]], .)
      
      total_soc_list[[soc_var]] <-  rbind(total_soc_list[[soc_var]], tbl)
    }
  }
  
  p1 <- annual_soc_list$soc_2 %>% ggplot(aes(x = month, y = n_ft_vacancies, group = soc_2)) + geom_line() + theme(legend.position = "none") + labs(title = yr)
  print(p1)
  saveRDS(annual_soc_list, here(paste0("data/lightcast_vacancies/lightcast_annual_", yr, ".RDS")))
  print("Save annual file.")
  gc()
}

for(soc_cat in 2:5){
  soc_cat_var <- paste0("soc_", soc_cat)
  soc_cat_var_name <- paste0(soc_cat_var, "_name")
  print(soc_cat_var)
  
  temp <- total_soc_list[[soc_cat_var]] %>% 
    complete(date, nesting(!!!syms(c(soc_cat_var, soc_cat_var_name))))
  
  stopifnot(n_groups(group_by(temp, date, !!sym(soc_cat_var), !!sym(soc_cat_var_name))) == nrow(temp))
  
  p1 <- temp %>% 
    ggplot(aes(x = date, y = n_ft_vacancies, group = .data[[soc_cat_var]], color = .data[[soc_cat_var]])) + 
    geom_line() + 
    theme(legend.position = "none") + 
    labs(title = paste0("Vacancy Numbers per Occupation by ", soc_cat_var, " Category"))
  
  print(p1)
  
  temp %>% saveRDS(here(paste0("data/lightcast_vacancies/lightcast_", soc_cat_var, "_2010_2024.RDS")))
}
  