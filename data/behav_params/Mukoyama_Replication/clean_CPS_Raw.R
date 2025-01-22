library(data.table)
library(dplyr)
library(stringr)
library(haven)
library(here)
library(httr)

# Set memory and directory paths (adjust as per your system)
raw_CPS <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/CPS")
int_CPS <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/CPS/R_int")

# Function to process each year-month combination
process_cps <- function(x) {
  print(x)
  # Rename variables
  rename_vars <- c(
    HRHHID = "hhid", HRHHID2 = "hhid2", PRTAGE = "age", PESEX = "sex", 
    PTDTRACE = "race", HRMIS = "mis", PEMLR = "mlr", PERACE = "race", 
    PWSSWGT = "wgt", PREDUCA4 = "grdatn", PRUNEDUR = "undur", 
    PEMARITL = "married", PEDWWNTO = "nlfwant", dwwant = "nlfwant", 
    GEREG = "gereg", region = "gereg", GESTCEN = "state", 
    PTERNHLY = "ernhr", PTERNWA = "ernwk", PRUNTYPE = "untype", 
    HRSERSUF = "serial", PEEDUCA = "grdatn", PULINENO = "lineno"
  )
  
  if(x > 201412){
    names(rename_vars) <- tolower(names(rename_vars))
    # Load data
    file_path <- file.path(raw_CPS, paste0("R_int/cpsb", x, ".dta"))
    if (!file.exists(file_path)) return(NULL)
    data <- read_dta(file_path)
    print(data)
  }else{
    # Load data
    file_path <- file.path(raw_CPS, paste0("bas", x, ".dta"))
    if (!file.exists(file_path)) return(NULL)
    data <- read_dta(file_path)
    print(data)
  }
  
  # Add year and month
  data <- data %>%
    mutate(
      YYYYMM = as.character(x),
      year = as.integer(substr(YYYYMM, 1, 4)),
      month = as.integer(substr(YYYYMM, 5, 6))
    ) %>%
    select(-YYYYMM)
  

  data <- data %>% rename_with(~ rename_vars[.x], any_of(names(rename_vars)))
  data <- data %>% mutate(wgt = wgt * 10000)
  
  # Rename lkdk and lkps variables
  for (t in 1:6) {
    lkdk <- paste0("PULKDK", t)
    lkps <- paste0("PULKPS", t)
    if (lkdk %in% names(data)) data <- rename(data, !!paste0("lkdk", t) := !!sym(lkdk))
    if (lkps %in% names(data)) data <- rename(data, !!paste0("lkps", t) := !!sym(lkps))
  }
  
  if ("PELK1" %in% names(data)) data <- rename(data, lkm1 = PELKM1)
  for (t in 2:6) {
    lkm <- paste0("PULKM", t)
    if (lkm %in% names(data)) data <- rename(data, !!paste0("lkm", t) := !!sym(lkm))
  }
  
  # Generate search method variables
  srch_methods <- list(
    empldir = 1, pubemkag = 2, PriEmpAg = 3, FrendRel = 4, SchEmpCt = 5,
    Resumes = 6, Unionpro = 7, Plcdads = 8, Otheractve = 9, lkatads = 10,
    Jbtrnprg = 11, otherpas = 13
  )
  
  for (method in names(srch_methods)) {
    method_code <- srch_methods[[method]]
    data <- data %>% 
      mutate(!!method := as.integer(
        rowSums(across(starts_with("lkm"), ~ .x == method_code), na.rm = TRUE) > 0
      ))
  }
  
  # Replace missing values and filter based on mlr
  for (method in names(srch_methods)) {
    data <- data %>% 
      mutate(
        !!method := ifelse(is.na(!!sym(method)) | mlr != 4, 0, !!sym(method))
      )
  }
  
  # Aggregate search methods
  active_methods <- names(srch_methods)[-length(srch_methods)] # Exclude 'otherpas'
  data <- data %>%
    mutate(
      numsearch = rowSums(select(., all_of(names(srch_methods))), na.rm = TRUE),
      numactive = rowSums(select(., all_of(active_methods)), na.rm = TRUE)
    ) %>%
    select(-starts_with("lkm"), -starts_with("lkdk"), -starts_with("lkps"))
  
  # Generate lfs variable
  data <- data %>%
    mutate(
      lfs = case_when(
        mlr %in% c(1, 2) ~ "E",
        mlr %in% c(3, 4) ~ "U",
        mlr %in% c(5, 6, 7) & nlfwant != 1 ~ "N",
        mlr %in% c(5, 6, 7) & nlfwant == 1 ~ "D", #lfs == "N" & nlfwant == 1 ~ "D",
        TRUE ~ "M"
      )
    )
  
  if(x >= 201412){
    # Compress and save intermediate data
    save_path <- file.path(int_CPS, paste0("intermediate_", x, ".rds"))
    saveRDS(data, save_path)
    return(NULL)
  }else{    # Compress and save intermediate data
    save_path <- file.path(int_CPS, paste0("intermediate_", x, ".rds"))
    saveRDS(data, save_path)
    return(NULL)}
}

# Loop over date range and process files
x <- 199401
#x <- 200301
while (x <= 201212) {
  process_cps(x)
  second <- ifelse((x - 12) %% 100 == 0, x + 89, x + 1)
  x <- second
}

cps_13_14 <- read_dta(file.path(raw_CPS, "cps_pull.dta")))
# Process data from 2013-2014
x <- 201301
while (x <= 201412) {
  # Similar processing for 2013-2014 format adjustments
  # Use the process_cps function but adapt as per data format changes
  process_cps(x)
  second <- ifelse((x - 12) %% 100 == 0, x + 89, x + 1)
  x <- second
}


## Files for additional years downloaded using cps_raw_pull_2015_2024.R
x <- 201501
while (x <= 202411) {
  # Similar processing for 2013-2014 format adjustments
  # Use the process_cps function but adapt as per data format changes
  process_cps(x)
  second <- ifelse((x - 12) %% 100 == 0, x + 89, x + 1)
  x <- second
}
