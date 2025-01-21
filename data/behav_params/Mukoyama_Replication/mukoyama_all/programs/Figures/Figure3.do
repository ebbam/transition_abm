/********************************************************
** Description: Creates the data needed for Figure 3 
**********************************************************/

clear
capture log close
set more off


cd "$final_CPS"

****************************************************
****Figure 3a: fraction of unemployed (U/U+N)
****************************************************

use full_CPS_data, clear

bysort year month searchers: egen num_search = sum(newwgt)
replace num_search = . if searcher!=1
bysort year month nonsearchers: egen num_nonsearch = sum(newwgt)
replace num_nonsearch = . if nonsearcher!=1
bysort year month unemp: egen num_unemp = sum(newwgt)
replace num_unemp = . if unemp!=1
bysort year month emp: egen num_emp = sum(newwgt)
replace num_emp = . if emp!=1
bysort year month nonpar: egen num_nonpart = sum(newwgt)
replace num_nonpart = . if nonpart!=1

preserve
collapse (mean) num_search num_nonsearch num_unemp num_nonpart num_emp, by(year month)

// seasonally adjusting simply using a regression with monthly dummies
foreach var in num_search num_nonsearch num_unemp num_nonpart num_emp {
	reg `var' i.month, r
	forvalues count = 2(1)12 {
		replace `var' = `var' - _b[`count'.month] if month==`count' 
	}
}

outsheet using "$int_data/CPS/Figure3a_data.csv", comma replace
restore


****************************************************
****Figure 3b: Average search time (methods and Created time)
****************************************************
cd "$final_CPS"
use full_CPS_data.dta, clear

// unemployed
preserve
keep if unemp==1
collapse (mean) numsearch time_create [pweight = newwgt], by(year month)
//seasonally adjusting using month dummies
foreach var in numsearch time_create {
	reg `var' i.month, r
	forvalues count = 2(1)12 {
		replace `var' = `var' - _b[`count'.month] if month==`count' 
	}
}

outsheet using "$int_data/CPS/Figure3b_data.csv", comma replace
restore

// searchers only
preserve
keep if mlr==4
collapse (mean) numsearch time_create [pweight = newwgt], by(year month)
//seasonally adjusting using month dummies
foreach var in numsearch time_create {
	reg `var' i.month, r
	forvalues count = 2(1)12 {
		replace `var' = `var' - _b[`count'.month] if month==`count' 
	}
}

outsheet using "$int_data/CPS/FigureA8_data.csv", comma replace
restore

