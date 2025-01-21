/********************************************************
** Description: Creates Figure A6, showing time variation in ATUS data 
**********************************************************/

clear all
capture log close
set more off

cd "$raw_ATUS"
use merged_ATUS_2014.dta, clear
/*keep only non-employed*/
keep if mlr==3 | mlr==4 | mlr==5 
drop if time_less8==.
gen dummy_search = (time_less8>0)
gen ltime_less8 = log(time_less8)

local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
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
**Regression with pre- recession data only
****************************************************
gen pre_rec = 1 if year<=2007
replace pre_rec = 0 if year>2007

est clear
/*first stage regression */
probit dummy_search `srch_mth'  `lfs' `observables' [pw=wgt] if pre_rec==1, r
predict pi, xb // this is linear
predict prob_search // this is non-linear probability you search
gen invmills=normalden(pi)/normal(pi)

/*second-stage regression */
reg ltime_less8 `srch_mth' `lfs' `observables' invmills [pw=wgt] if pre_rec==1, r
predict lsearch_fitted
predict resid, r
sum resid, d
gen sigma_hat = r(Var)
gen search_cond = exp(lsearch_fitted + sigma_hat / 2)

/*final fitted values */
gen time_create = prob_search * search_cond 


preserve
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigureA6a_data_prerec.csv", comma replace
restore

preserve
keep if mlr==3 | mlr==4
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigureA6b_data_prerec.csv", comma replace
restore

drop time_create sigma_hat prob_search search_cond lsearch_fitted resid invmills pi 


*****************************************************
**Regression with post- recession data only
****************************************************
est clear
/*first stage regression */
probit dummy_search `srch_mth' `lfs' `observables' [pw=wgt] if pre_rec==0, r
predict pi, xb // this is linear
predict prob_search // this is non-linear probability you search
gen invmills=normalden(pi)/normal(pi)

/*second-stage regression */
reg ltime_less8 `srch_mth' `lfs' `observables' invmills [pw=wgt] if pre_rec==0, r
predict lsearch_fitted
predict resid, r
sum resid, d
gen sigma_hat = r(Var)
gen search_cond = exp(lsearch_fitted + sigma_hat / 2)

/*final fitted values */
gen time_create = prob_search * search_cond

preserve
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigureA6a_data_postrec.csv", comma replace
restore

preserve
keep if mlr==3 | mlr==4
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigureA6b_data_postrec.csv", comma replace
restore

drop time_create sigma_hat prob_search search_cond lsearch_fitted resid invmills pi 






