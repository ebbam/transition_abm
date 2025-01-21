/********************************************************
** Description: Creates Figure A5, showing raw ATUS data 
**********************************************************/

clear all
capture log close
set more off
set mem 1g


***************************************************
*** loading ATUS dataset
***************************************************
cd "$raw_ATUS"
use merged_ATUS_2014.dta, clear

gen date = yq(year, quarter)
gen emp = 1 if mlr==1 | mlr==2
replace emp = 0 if emp ==. 
gen nonpart = 1 if mlr==5
replace nonpart= 0 if nonpart!=1
bysort year quarter searchers: egen num_search = sum(wgt)
replace num_search = num_search / 91 /*this is the number of days in a quarter */
replace num_search = . if searchers!=1
bysort year quarter nonsearchers: egen num_nonsearch = sum(wgt)
replace num_nonsearch = num_nonsearch / 91 
replace num_nonsearch = . if nonsearcher!=1
bysort year quarter unemp: egen num_unemp = sum(wgt)
replace num_unemp = num_unemp / 91 
replace num_unemp = . if unemp!=1
bysort year quarter nonpar: egen num_nonpart = sum(wgt)
replace num_nonpart = num_nonpart / 91 
replace num_nonpart = . if nonpart!=1
bysort year quarter emp: egen num_emp = sum(wgt)
replace num_emp = num_emp / 91 
replace num_emp = . if emp!=1


// seasonally adjusting simply using a regression with quarter dummies
foreach var in num_search num_nonsearch num_unemp num_nonpart num_emp {
	reg `var' i.quarter, r
	replace `var' = `var' - _b[2.quarter] if quarter==2 
	replace `var' = `var' - _b[3.quarter] if quarter==3 
	replace `var' = `var' - _b[4.quarter] if quarter==4 
}

  
preserve
bysort year quarter: gen n = _n
keep if n==1
drop n 
keep year quarter num_search num_nonsearch num_unemp num_nonpart num_emp
order year quarter num_search num_nonsearch num_unemp num_nonpart num_emp
outsheet using "$int_data/ATUS/FigureA9_data.csv", comma replace
restore
   

*****************************************************
****Part 3: Average Search Time in ATUS 
****************************************************

drop if mlr==1 | mlr==2 // keeping only the unemployed
gen time_less8_unemp = time_less8 if mlr==3 | mlr==4
gen time_less8_srch = time_less8 if searchers==1
gen time_less8_nonemp = time_less8 
gen methods_unemp = numsearch if mlr==3 | mlr==4
gen methods_srch = numsearch if searchers==1
gen methods_nonemp = numsearch 
gen time_less8_travel_unemp = time_less8_travel if mlr==3 | mlr==4
gen time_less8_travel_srch = time_less8_travel if searchers==1
gen time_less8_travel_nonemp = time_less8_travel

preserve
collapse (mean) time_less8_unemp time_less8_srch time_less8_nonemp  methods_unemp methods_srch methods_nonemp time_less8_travel_unemp time_less8_travel_srch time_less8_travel_nonemp [pw=wgt], by(year)
order year time_less8_unemp time_less8_srch time_less8_nonemp methods_unemp methods_srch methods_nonemp
outsheet using "$int_data/ATUS/FigureA5_data.csv", comma replace
restore
 

