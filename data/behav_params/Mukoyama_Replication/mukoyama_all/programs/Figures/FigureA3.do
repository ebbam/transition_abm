/****************************************************************************
* Description: Creates the files necessary for crating Figure A3 (OLS and two-step imputation method)
*******************************************************************************/

clear all
capture log close
set more off

cd "$raw_ATUS"
use merged_ATUS_2014.dta, clear
keep if mlr==3 | mlr==4 | mlr==5 /*keep only non-employed*/

drop if time_less8==.
gen dummy_search = (time_less8>0)
gen ltime_less8 = log(time_less8)

**creating local variables for regressions
local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
local observables age age2 age3 age4 female hs somecol college black married marriedfemale
local lfs np_other layoff nonsearchers
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


**Baseline OLS regression
est clear
reg time_less8 `srch_mth'  `observables' `lfs' [pw=wgt], r
  matrix b = e(b)
   /*saving estimates in the local variables */
   local count = 0
   foreach x of local srch_mth {
      local coef_`x'= b[1,`count' + 1]
      local count = `count' + 1
   }
   foreach x of local observables {
      local coef_`x'= b[1,`count' + 1]
      local count = `count' + 1
   }
   foreach x of local lfs {
      local coef_`x'= b[1,`count' + 1]
      local count = `count' + 1
   }
   local coef_constant = b[1,27]

   gen time_create = `coef_constant'
   foreach x of local srch_mth {
      replace time_create = time_create + `coef_`x'' if `x' == 1 
   }
   foreach x of local observables {
      replace time_create = time_create + `coef_`x'' if `x' == 1 
   }
   foreach x of local lfs {
      replace time_create = time_create + `coef_`x'' if `x' == 1 
   }
   forvalue x = 25/70 {
      replace time_create = time_create + `x' * `coef_age' + `x' ^2 * `coef_age2' + `x' ^3 * `coef_age3' + `x' ^4 * `coef_age4'  if age==`x'
}

gen timesearch_unemp_less8 = time_less8 if mlr==3 | mlr==4
gen timesearch_total_less8 = time_less8
gen time_create_unemp_less8 = time_create if mlr==3 | mlr==4
gen time_create_total_less8 = time_create
drop time_create

preserve
collapse (mean) timesearch_unemp_less8- time_create_total_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigureA3_data_ols.csv", comma replace
restore

**********************************************
**Table with year dummies 
**********************************************
log using "$tables/TableA1.log", replace
reg time_less8 `srch_mth'  `observables' `lfs' i.year [pw=wgt], r
log close

**********************************************
**Two-Step Regressions 
**********************************************
/*first stage regression */
probit dummy_search `srch_mth' `lfs' `observables' [pw=wgt], r
predict pi, xb // this is linear
predict prob_search // this is non-linear probability you search
/*second-stage regression */
reg ltime_less8 `srch_mth' `lfs' `observables' [pw=wgt]
predict lsearch_fitted, xb
predict resid, r
sum resid, d
gen sigma_hat = r(Var)
gen search_cond = exp(lsearch_fitted + sigma_hat / 2)

/*final fitted values */
gen time_create = prob_search * search_cond 

preserve
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigureA3a_data.csv", comma replace
restore

preserve
keep if mlr==3 | mlr==4
collapse (mean) time_create time_less8 [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigureA3b_data.csv", comma replace
restore

