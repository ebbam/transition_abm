
/********************************************************
** Description: Creates Figure 1
**********************************************************/

clear all
capture log close
set more off
set mem 1g

*** loading ATUS dataset
cd "$raw_ATUS"
use merged_ATUS_2014.dta, clear

collapse (mean) time_less8  [aw=wgt], by(numsearch)
label var time_less8 "Average Search Time" 
label var numsearch "Number of Search methods"
graph bar time_less8, over(numsearch) graphregion(fcolor(white) lcolor(white)) ytitle("Average Search Time Per Day")
graph export "$figures/Figure1.pdf", replace
