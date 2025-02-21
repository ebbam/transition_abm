# LTUER by Occupation 
# The following file downloads all occupation-specific LTUER data as available from the BLS
# Table 32 of each year represented here: https://www.bls.gov/cps/tables.htm#otheryears

# Load necessary libraries
library(rvest)
library(httr)
library(readxl)
library(here)
library(tidyverse)
library(gridExtra)

### DOWNLOAD FILES ######
# Base URL before the year
base_url <- "https://www.bls.gov/cps/cps_aa"

# Column names for the table in the .txt files
col_names <- c("occupation", "total_unemployed", "less_than_5_weeks", "wks5_14", "wks15plus", 
               "wks15_26", "wks27plus", "wks_mean_duration", "wks_median_duration")

# # Loop through the years 2002 to 2022
# for (year in 2002:2022) {
#   print(year)
#   
#   if (year >= 2011) {
#     # For years 2011-2022 (HTML-based file)
#     if (year > 2014) {
#       url <- paste0(base_url, year, ".htm")
#     } else {
#       url <- paste0("https://www.bls.gov/cps/aa", year, "/cpsaat32.htm")
#     }
#     
#     # Send GET request
#     x <- GET(url, add_headers('user-agent' = 'Oxford University Student web scraper ([[ebba.mark@gmail.com]])'))
#     
#     # Print progress
#     cat("Processing URL:", url, "\n")
#     
#     # Read the content of the webpage
#     webpage <- tryCatch({
#       read_html(x)
#     }, error = function(e) {
#       cat("Failed to load:", url, "\n")
#       return(NULL)
#     })
#     
#     # If the webpage failed to load, skip to the next iteration
#     if (is.null(webpage)) next
#     
#     if (year > 2014) {
#       # Extract the link to the file for years 2015 and above (Excel file)
#       file_link <- webpage %>%
#         html_nodes("a") %>%
#         html_attr("href") %>%
#         .[grepl("\\cpsaat32.xlsx", ., ignore.case = TRUE)]
#       
#       if (length(file_link) == 0) {
#         cat("No file link found for year", year, "\n")
#         next
#       }
#       
#       full_link <- paste0("https://www.bls.gov", file_link)
#       
#       # Specify the output file name for .xlsx download
#       output_file <- here('data/macro_vars/CPS_LTUER/', paste0("cps_ltuer_highlevoccs", year, ".xlsx"))
#       
#       # Download the file
#       GET(full_link, write_disk(output_file, overwrite = TRUE), add_headers('user-agent' = 'Oxford University Student web scraper ([[ebba.mark@gmail.com]])'))
#       
#     } else {
#       # For years 2011-2014 (HTML table processing)
#       file_table <- webpage %>%
#         html_nodes("table") %>%
#         html_table() %>%
#         .[[1]]  # Assuming the first table is the one we want
#       
#       # Write the table to CSV for 2011-2014 years
#       write.csv(file_table, here('data/macro_vars/CPS_LTUER/', paste0("cps_ltuer_highlevoccs", year, ".csv")))
#     }
#     
#     # Print success message
#     cat("Downloaded file for year", year, "\n")
#     
#   } else {
#     # For years 2002-2010 (Text-based file processing)
#     url <- paste0("https://www.bls.gov/cps/aa", year, "/aat32.txt")
#     
#     # Download the .txt file
#     x <- GET(url, add_headers('user-agent' = 'Oxford University Student web scraper ([[ebba.mark@gmail.com]])')) %>%
#       content(., as = "text", encoding = "UTF-8")
#     
#     # Print progress
#     cat("Processing URL:", url, "\n")
#     
#     # Split the text content into lines
#     y <- x %>% strsplit(., "\n") %>% unlist()
#     
#     # Find the index of the first element that starts with "Manage"
#     index <- which(grepl("Manage", y))[1]
#     print(index)
#     
#     # Remove all elements before that index (including that element itself)
#     z <- y[index:length(y)] %>%
#       trimws() %>%
#       lapply(., function(x) gsub("[()]", "", x)) %>%
#       unlist() 
#     
#     # Function to rectify the occupation names running over two rows
#     for (i in 1:(length(z) - 1)) {
#       # Check if the current string does not end with a digit
#       if (grepl("[a-zA-Z]$", z[i])) {
#         # Concatenate the string to the next string in the vector
#         z[i] <- paste(z[i], z[i + 1])
#         # Remove the next string (as it has been merged)
#         z <- z[-(i + 1)]
#       }
#     }
#     
#     w <- z %>%
#       unique %>% 
#       data.frame() %>%
#       separate(., col = `.`, sep = "\\s+(?=\\d)", into = col_names) %>%
#       tibble() 
#     
#     # Write the cleaned data frame to CSV for 2002-2010 years
#     write_csv(w, here('data/macro_vars/CPS_LTUER/', paste0("cps_ltuer_highlevoccs", year, ".csv")))
#     
#     # Print success message
#     cat("Downloaded file for year", year, "\n")
#   }
# }
# 
# cat("All downloads completed.\n")


########### COMBINE FILES ##########

occ_ltuers <- tibble()
for(year in 2022:2003){
  print(year)
  if(year < 2011){
    test <- read.csv(here(paste0('data/macro_vars/CPS_LTUER/cps_ltuer_highlevoccs', year,'.csv'))) %>% 
      tibble %>% 
      mutate(across(col_names[-1], ~as.numeric(gsub(",", "", .))),
             occupation = trimws(gsub(".", "", occupation, fixed = TRUE)),
             occupation = ifelse(occupation == "Mining", "Mining, quarrying, and oil and gas extraction", occupation)) %>%  
      filter(!(rowSums(across(col_names, ~ is.na(.))) > 7)) 
  }
  else if(year >= 2011 & year < 2015){
    test <- read.csv(here(paste0('data/macro_vars/CPS_LTUER/cps_ltuer_highlevoccs', year,'.csv'))) %>% 
      tibble %>% 
      slice(9:nrow(.)) %>% 
      select(-1) %>% 
      set_names(col_names) %>% 
      mutate(across(col_names[-1], ~as.numeric(gsub(",", "", .)))) %>% 
      filter(!(rowSums(across(col_names, ~ is.na(.))) > 7))
      
    
  }else if(year >= 2015){
    test <- read_xlsx(here(paste0('data/macro_vars/CPS_LTUER/cps_ltuer_highlevoccs', year,'.xlsx'))) %>% 
      tibble %>% 
      slice(8:nrow(.)) %>% 
      set_names(col_names) %>% 
      mutate(across(col_names[-1], ~as.numeric(.))) %>% 
      filter(!(rowSums(across(col_names, ~ is.na(.))) > 7)) 

  }

  occ_ltuers <- test %>%
    mutate(year = year) %>%
    bind_rows(., occ_ltuers)

}

occs_interest <- c("Management, business, and financial operations occupations",
"Professional and related occupations",
"Service occupations",
"Sales and related occupations",
"Office and administrative support occupations",
"Farming, fishing, and forestry occupations",
"Construction and extraction occupations",
"Installation, maintenance, and repair occupations",
"Production occupations",
"Transportation and material moving occupations")

ind_interest <- c("Agriculture and related industries",
                  "Mining, quarrying, and oil and gas extraction",
                  "Construction",
                  "Manufacturing",
                  "Durable goods",
                  "Nondurable goods",
                  "Wholesale and retail trade",
                  "Transportation and utilities",
                  "Information",
                  "Financial activities",
                  "Professional and business services",
                  "Education and health services",
                  "Leisure and hospitality",
                  "Other services",
                  "Public administration")

pp1 <- occ_ltuers %>% 
  filter(occupation %in% occs_interest) %>% 
  ggplot(aes(x = year, y = wks_mean_duration, color = occupation)) +
  geom_line() +
  labs(title = "Mean Duration of Unemployment by Occupation")

pp2 <- occ_ltuers %>% 
  filter(occupation %in% occs_interest) %>% 
  mutate(ltuer = wks27plus/total_unemployed) %>% 
  ggplot(aes(x = year, y = ltuer, color = occupation)) +
  geom_line() +
  labs(title = "LTUER by Occupation")

pp3 <- occ_ltuers %>% 
  filter(occupation %in% ind_interest) %>%
  rename(industry = occupation) %>% 
  ggplot(aes(x = year, y = wks_mean_duration, color = industry)) +
  geom_line() +
  labs(title = "Mean Duration of Unemployment by Industry")

pp4 <- occ_ltuers %>% 
  filter(occupation %in% ind_interest) %>% 
  rename(industry = occupation) %>% 
  mutate(ltuer = wks27plus/total_unemployed) %>% 
  ggplot(aes(x = year, y = ltuer, color = industry)) +
  geom_line() +
  labs(title = "LTUER by Industry")


grid.arrange(pp1, pp3, pp2, pp4)










