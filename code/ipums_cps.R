# IPUMS CPS DATA
# Citation: Sarah Flood, Miriam King, Renae Rodgers, Steven Ruggles, 
# J. Robert Warren, Daniel Backman, Annie Chen, Grace Cooper, Stephanie Richards, 
# Megan Schouweiler, and Michael Westberry. IPUMS CPS: Version 11.0 [dataset]. 
# Minneapolis, MN: IPUMS, 2023. https://doi.org/10.18128/D030.V11.0



library(tidyverse)
library(here)
read.delim(here("data/us_cps/cps_00001.dat"), sep=" ")
           