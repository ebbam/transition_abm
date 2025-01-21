/********************************************************
** Description: Creates Figure A14, showing the fraction of workers that are eligible for UI over time
**********************************************************/

use "$final_CPS/full_CPS_data", clear 
keep if mlr==4 
gen elig = 1 if untype==1 | untype==2 | untype==3
replace elig = 0 if untype==4 | untype==5

//creating plots of fractions eligible for appendix
preserve
bysort year month: egen num_elig = sum(newwgt) if elig==1
bysort year month: egen num_unemp = sum(newwgt)
gen frac_elig = num_elig/num_unemp
keep if year>=2000 & year<=2012

collapse (mean) frac_elig , by(year month)
outsheet using "$int_data/CPS/FigureA14_data.csv", comma replace
restore
