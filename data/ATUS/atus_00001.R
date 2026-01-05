# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

library(here)
library(tidyverse)

if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

ddi <- read_ipums_ddi(here("data/ATUS/atus_00001.xml"))
data <- read_ipums_micro(ddi)
ipums_conditions()

data %>% 
  select(YEAR, FAMINCOME, EMPSTAT, EMPSTAT_CPS8, LOOKING, RETIRED, LOOKING_CPS8, WANTJOB_CPS8, WNLWNILF_CPS8, FWK_NOTHING, FWK_NOTHING_CPS8)
