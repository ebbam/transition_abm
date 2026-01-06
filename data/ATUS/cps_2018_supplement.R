#### Importing data from HTML tables on 
## https://www.bls.gov/opub/btn/volume-9/how-do-jobseekers-search-for-jobs.htm#_edn2
library(here)
library(tidyverse)
library(httr)
library(rvest)
library(assertthat)

# Save URL
url <- "https://www.bls.gov/opub/btn/volume-9/how-do-jobseekers-search-for-jobs.htm#_edn2"
# Send GET request
x <- GET(url, add_headers('user-agent' = 'Oxford University Student web scraper ([[ebba.mark@gmail.com]])'))

# Print progress
#cat("Processing URL:", url, "\n")

# Read the content of the webpage
webpage <- tryCatch({
  read_html(x)
}, error = function(e) {
  cat("Failed to load:", url, "\n")
  return(NULL)
})

# read in HTML tables
file_tables <- webpage %>%
  html_nodes("table") %>%
  html_table() # Extract all tables

table_titles <- webpage %>% 
  html_elements("table") %>%  # Select all <table> elements
  html_element("caption") %>% # Find the <caption> within each <table>
  html_text()  

assert_that(length(table_titles) == length(file_tables))

file_tables[[1]][1, ] <- c(file_tables[[1]][1, -1], NA) 


# Write the table to CSV
#write.csv(file_tables, here('data/behav_params/', paste0("cps_2018_supplement_tables.csv")))
