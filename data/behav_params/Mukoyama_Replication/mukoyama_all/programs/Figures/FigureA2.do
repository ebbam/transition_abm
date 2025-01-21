/********************************************************
** Description: Creates Figure A2, showing ATUS data with data quality measures
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

//using only the non-employed
keep if mlr==4 | mlr==3 |mlr==5 

//dropping the outliers 
drop if time_less8==.

**************************************************
**creating local variables for regressions
**************************************************

local srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
local observables age age2 age3 age4 female hs somecol college black married marriedfemale
local lfs np_other layoff nonsearchers

***************************************************
**cleaning variable names for regression output
***************************************************

/*cleaning variable names for regression output */
label var empldir "Contacted Employed Directly"
label var pubemkag "Contacted Public Employment Agency "
label var PriEmpAg "Contacted Private Employment Agency "
label var FrendRel "Contacted Friends or Relatives "
label var SchEmpCt "Contacted School Employment Center "
label var  Resumes "Sent Out Resumes"
label var  Unionpro "Checked Union Registers"
label var  Plcdads "Placed/Answered Ads"
label var  Otheractve "Other Active"
label var  Jbtrnprg "Attended Job Training"
label var  otherpas "Other Passive"
label var lkatads "looked at ads"
label var sex female
label var age age
label var year year

*****************************************************
**Baseline regression
****************************************************

gen dummy_search = (time_less8>0)
gen ltime_less8 = log(time_less8)

***********************Unemployed (U)

** First Stage Probit
probit dummy_search `srch_mth' `lfs' `observables' i.day missing [pw=wgt] 
predict pi, xb // this is linear
predict prob_search // this is non-linear probability you search
gen invmills=normalden(pi)/normal(pi)

** Second Stage (adding inverse mills)
reg ltime_less8  `srch_mth' `lfs' `observables' invmills i.day missing [pw=wgt] 
predict search_cond, xb
predict resid, r
sum resid, d
gen sigma_hat = r(Var)
replace search_cond = exp(search_cond + sigma_hat / 2)

gen time_create = prob_search * search_cond 

**creating plots of fitted values
preserve
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigA2a_data.csv", comma replace
restore

preserve
keep if mlr==3 | mlr==4
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigA2b_data.csv", comma replace
restore





