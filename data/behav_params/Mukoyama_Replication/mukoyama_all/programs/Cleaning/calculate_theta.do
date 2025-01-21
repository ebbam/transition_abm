********************************************************************************************
** Description: This program calculates market tightness (theta)
********************************************************************************************

* Settings 
clear all
capture log close
set more off

***************** Step 1: Calculating Unemployment Counts 

cd "$int_CPS"
local filenames: dir . files "*intermediate*"
foreach filename in `filenames' {
   append using `filename'
}

keep if mlr==3 | mlr==4  // keeping only the unemployed
/* correcting for recoding of the final weight in CPS utilities */
rename wgt newwgt
replace newwgt = newwgt / 10000
replace newwgt = newwgt / 10000 if year>=2003 & year<=2012

******* 1) Aggregate Unemployment Counts 
preserve
collapse (sum) newwgt, by(year month) 
rename newwgt unemployed
save "$int_CPS/temporary_unemp_aggregate", replace
restore 


***************************************************************Step 2: Calcualting Theta 

*************************************
******Aggregate Theta from JOLTS
*************************************

//JOLTS vacancy data 
use "$other/jolts_openings.dta", clear //Downloaded from DLX: 8/3/2015
gen month=month(date)
gen year=year(date)
drop if year==. 

//merging in unemployment data from the CPS
merge 1:1 year month using  "$int_CPS/temporary_unemp_aggregate"
assert year==2015 if _m==1 
assert year<=2000 if _m==2 
keep if _m==3 
drop _m
gen theta_agg_JOLTS = 1000*job_openings_nsa/unemployed
keep year month theta_agg_JOLTS

// saving 
save "$int_data/theta/JOLTS_agg_theta_monthly", replace

*************************************
******Aggregate Theta from Barnichon
*************************************

import excel "$other/HWI_index.xls",  clear cellrange(A6:B774) first
gen year = substr(Date,1,4)
gen month = substr(Date,6,2)
drop Date 
destring year month, replace
merge 1:1 year month using "$int_CPS/temporary_unemp_aggregate"
* Normalize unemployment in January 1994 
sum unemployed if year==1994 & month==1
local normalize = r(mean)
replace unemployed = 100* unemployed /`normalize'
gen theta_agg_BARNICHON = compositeHWI/unemployed
keep year month theta_agg_BAR
keep if year>=1994
save "$int_data/theta/BARNICHON_agg_theta_monthly", replace

erase  "$int_CPS/temporary_unemp_aggregate.dta"




