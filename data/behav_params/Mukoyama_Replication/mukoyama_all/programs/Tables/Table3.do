/********************************************************
** Description: Creates Table 3
**********************************************************/

clear all
capture log close
set more off
set mem 1g


*** loading ATUS dataset

cd "$raw_ATUS"
use merged_ATUS_2014.dta, clear

***********************************************************
**creating summary tables 
***********************************************************
   log using "$tables/Table3.log", replace

   foreach x in timesearch time_less8 {

   /*by labor force status*/
      sum `x' [aw = wgt]
      local `x'_total = r(mean)
      sum `x' if mlr==1 | mlr==2 [aw = wgt]
      local `x'_emp = r(mean)
      sum `x' if mlr==3 | mlr==4 | mlr==5 [aw = wgt]
      local `x'_nonemp = r(mean)
      sum `x' if mlr==3 | mlr==4 [aw = wgt]
      local `x'_unemp = r(mean)
      sum `x' if mlr==5 [aw = wgt] 
      local `x'_nonpart = r(mean) 
      sum `x' if searchers==1 [aw = wgt] 
      local `x'_srch = r(mean)
      sum `x' if attached==1  [aw = wgt] 
      local `x'_attached = r(mean)
      sum `x' if mlr==3 [aw = wgt] 
      local `x'_layoff = r(mean)
      sum `x' if (mlr==5 & attached!=1) [aw = wgt] 
      local `x'_retired = r(mean)
      sum `x' if attached==1 & searchers!=1 [aw = wgt] 
      local `x'_nonsearch = r(mean)

   }
log close
log using "$tables/Table3.log", replace
macro list _all
log close

 
 
