#CPS Pull Additional Years
library(data.table)
library(dplyr)
library(stringr)
library(haven)
library(here)
library(httr)

# Set memory and directory paths (adjust as per your system)
raw_CPS <- here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/CPS")

## Download files for additional years
x <- 201501
options(timeout=360)
while (x <= 202411) {
  print(x)
  url <- paste0("https://data.nber.org/cps-basic3/dta/", substr(as.character(x), 1, 4), "/cpsb", as.character(x), ".dta")
  download.file(url, here(paste0(raw_CPS, "/R_int/cpsb", as.character(x), ".dta")))
  second <- ifelse((x - 12) %% 100 == 0, x + 89, x + 1)
  x <- second
}