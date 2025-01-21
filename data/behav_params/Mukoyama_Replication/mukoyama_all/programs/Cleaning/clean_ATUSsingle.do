
/***************************************************
Description: This file combines and cleans the ATUS single year data for 2003-2014 and creates a dataset at the end to be merged in at the beginning of clean_ATUS.do 
*****************************************************/

clear all 
capture log close
set more off

*Begin by appending the single year atusresp files, 2003-2014; this is where our identifier will come from

cd "$raw_ATUS_single/2003/atusresp_2003/"

use atusresp_2003.dta, clear

forvalues i=2004(1)2014 {
	quietly append using "$raw_ATUS_single/`i'/atusresp_`i'/atusresp_`i'.dta"
}

tab tuyear
isid tucaseid

tempfile resp_single0314
save "`resp_single0314'", replace


*Now append the single year atussum files, 2003-2011; this is where the variable of interest is
use "$raw_ATUS_single/2003/atussum_2003/atussum_2003.dta", clear

gen year = 2003

forvalues i=2004(1)2014 {
	quietly append using "$raw_ATUS_single/`i'/atussum_`i'/atussum_`i'.dta"
	replace year = `i' if year == .
}

tab year
isid tucaseid

tempfile sum_single0314
save "`sum_single0314'", replace

*Combine the file with the identifier and year
use "`resp_single0314'", clear
merge 1:1 tucaseid using "`sum_single0314'"

isid tucaseid
tab _m
tab year tuyear

*Only need to keep the variables related to interview travel time: t170504 (2004) and t180504 (2005-2011)
keep tucaseid tuyear t170504 t180504

*Since it's not the same variable across time, I create one variable for the whole dataset
gen intvwtravel = t170504 if tuyear == 2004 
replace intvwtravel = t180504 if tuyear != 2004 & tuyear != 2003 & intvwtravel==.
count if intvwtravel==.
count if tuyear==2003

drop tuyear

sort tucaseid

save "$raw_ATUS_single/intvwtravel_0314.dta", replace

*Use this dataset, which is unique on tucaseid to merge in the interview travel time in clean_ATUS.do
