****This program sets all of the files paths for the project 


/// THIS IS THE ONLY THING NEW USERS HAVE TO CHANGE
local dir "/Users/cpatterson/Dropbox/ATUS/Replication-Files"


global programs "`dir'/programs"

display "$programs"
* Raw data paths
global raw_ATUS "`dir'/raw_data/ATUS"
global raw_ATUS_single "`dir'/raw_data/ATUS"

global raw_CPS "`dir'/raw_data/CPS"
global int_CPS "`dir'/int_data/CPS"
global int_data "`dir'/int_data"
global final_CPS "`dir'/final_data"

global maps "`dir'/raw_data/maps"
global other "`dir'/raw_data/other"

global tables "`dir'/Tables"
global figures "`dir'/Figures"
