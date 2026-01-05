/********************************************************
** Description: Creates Figure A4, showing ATUS data with additional controls
**********************************************************/


clear all
capture log close
set more off

***************************************************
*** Step 1: Creating Impulation in ATUS data
***************************************************
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


**Regression with Methods and demographics interactions
est clear
local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
foreach method of local srch_mth {
	gen `method'_gender = `method'*female
	gen `method'_age = `method'*age
	local srch_mth_obs `srch_mth_obs' `method'_gender `method'_age 
}

/*first stage regression */
probit dummy_search `srch_mth' `lfs' `observables' `srch_mth_obs' [pw=wgt], r
predict pi, xb // this is linear
predict prob_search // this is non-linear probability you search
gen invmills=normalden(pi)/normal(pi)
matrix b = e(b)
/*saving estimates in the local variables */
local count = 0
foreach x of local srch_mth {
  gen  pweight_`x'= b[1,`count' + 1]
  local count = `count' + 1
}

foreach x of local lfs {
  gen pweight_`x'= b[1,`count' + 1]
  local count = `count' + 1
}
foreach x of local observables {
  gen pweight_`x'= b[1,`count' + 1]
  local count = `count' + 1
}
foreach x of local srch_mth_obs {
  gen  pweight_`x'= b[1,`count' + 1]
  local count = `count' + 1
}   
gen pweight_constant = b[1,51]

/*second-stage regression */
reg ltime_less8 `srch_mth' `lfs' `observables' `srch_mth_obs' invmills [pw=wgt], r
predict lsearch_fitted
predict resid, r
sum resid, d
gen sigma_hat = r(Var)
gen search_cond = exp(lsearch_fitted + sigma_hat / 2)

matrix b = e(b)
/*saving estimates in the local variables */
local count = 0
foreach x of local srch_mth {
  gen  sweight_`x'= b[1,`count' + 1]
  local count = `count' + 1
}

foreach x of local lfs {
  gen sweight_`x'= b[1,`count' + 1]
  local count = `count' + 1
}
foreach x of local observables {
  gen sweight_`x'= b[1,`count' + 1]
  local count = `count' + 1
}
foreach x of local srch_mth_obs {
gen  sweight_`x'= b[1,`count' + 1]
local count = `count' + 1
}      
gen sweight_invmills = b[1,51]
gen sweight_constant = b[1,52]

/*final fitted values */
gen time_create = prob_search * search_cond

preserve
keep *weight* sigma_hat
keep if _n==1
save "$int_data/ATUS/time_method_reweight_interactions", replace
restore


****************************************************
**** Step 2: Imputing Search time in the CPS
****************************************************

use "$final_CPS/full_CPS_data", clear

//dropping the employed 
drop if mlr==1 | mlr==2

*** Dropping variables that I don't need to reduce the size of the dataset 
drop  wks_tot wksleft_tot wksused_tot untype undur 

/*initiating additional varaibles */
local count = 0


est clear
local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
local srch_mth_obs "" 
foreach method of local srch_mth {
	gen `method'_gender = `method'*female
	gen `method'_age = `method'*age
	local srch_mth_obs `srch_mth_obs' `method'_gender `method'_age 
}

append using "$int_data/ATUS/time_method_reweight_interactions"
erase "$int_data/ATUS/time_method_reweight_interactions.dta"

local variables empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas `srch_mth_obs' age age2 age3 age4 female hs somecol college black married marriedfemale np_other layoff nonsearchers constant

/*defining weights */
sum year
local count = `count' + 1
local last_obs = r(N)+`count'
foreach x of local variables {
	local pweight_`x' = pweight_`x'[`last_obs']
	local sweight_`x' = sweight_`x'[`last_obs']
}
display `pweight_constant'
local sweight_invmills = sweight_invmills[`last_obs']
local sigma_hat = sigma_hat[`last_obs']

gen pi  = empldir*`pweight_empldir' + pubemkag * `pweight_pubemkag' + PriEmpAg *`pweight_PriEmpAg' + FrendRel*`pweight_FrendRel' + SchEmpCt*`pweight_SchEmpCt' + Unionpro*`pweight_Unionpro' + Resumes*`pweight_Resumes' + Plcdads*`pweight_Plcdads' + Otheractve * `pweight_Otheractve' + lkatads *`pweight_lkatads' + Jbtrnprg*`pweight_Jbtrnprg' + otherpas*`pweight_otherpas' + age*`pweight_age' + age2 * `pweight_age2' + age3 * `pweight_age3' + age4 * `pweight_age4' + female*`pweight_female' + hs*`pweight_hs' + somecol*`pweight_somecol' + college*`pweight_college' + black*`pweight_black' + married * `pweight_married' + marriedfemale* `pweight_marriedfemale' + np_other* `pweight_np_other' + layoff* `pweight_layoff' + nonsearchers * `pweight_nonsearchers' + `pweight_constant'  + empldir_age*`pweight_empldir_age' + pubemkag_age * `pweight_pubemkag_age' + PriEmpAg_age *`pweight_PriEmpAg_age' + FrendRel_age*`pweight_FrendRel_age' + SchEmpCt_age*`pweight_SchEmpCt_age' + Unionpro_age*`pweight_Unionpro_age' + Resumes_age*`pweight_Resumes_age' + Plcdads_age*`pweight_Plcdads_age' + Otheractve_age * `pweight_Otheractve_age' + lkatads_age *`pweight_lkatads_age' + Jbtrnprg_age*`pweight_Jbtrnprg_age' + otherpas_age*`pweight_otherpas_age' + empldir_gender*`pweight_empldir_gender' + pubemkag_gender * `pweight_pubemkag_gender' + PriEmpAg_gender *`pweight_PriEmpAg_gender' + FrendRel_gender*`pweight_FrendRel_gender' + SchEmpCt_gender*`pweight_SchEmpCt_gender' + Unionpro_gender*`pweight_Unionpro_gender' + Resumes_gender*`pweight_Resumes_gender' + Plcdads_gender*`pweight_Plcdads_gender' + Otheractve_gender * `pweight_Otheractve_gender' +  lkatads_gender *`pweight_lkatads_gender' +  Jbtrnprg_gender*`pweight_Jbtrnprg_gender' + otherpas_gender*`pweight_otherpas_gender' 
gen invmills=normalden(pi)/normal(pi)
gen psearch = normprob(empldir*`pweight_empldir' + pubemkag * `pweight_pubemkag' + PriEmpAg *`pweight_PriEmpAg' + FrendRel*`pweight_FrendRel' + SchEmpCt*`pweight_SchEmpCt' + Unionpro*`pweight_Unionpro' + Resumes*`pweight_Resumes' + Plcdads*`pweight_Plcdads' + Otheractve * `pweight_Otheractve' + lkatads *`pweight_lkatads' + Jbtrnprg*`pweight_Jbtrnprg' + otherpas*`pweight_otherpas' + age*`pweight_age' + age2 * `pweight_age2' + age3 * `pweight_age3' + age4 * `pweight_age4' + female*`pweight_female' + hs*`pweight_hs' + somecol*`pweight_somecol' + college*`pweight_college' + black*`pweight_black' + married * `pweight_married' + marriedfemale* `pweight_marriedfemale' + np_other* `pweight_np_other' + layoff* `pweight_layoff' + nonsearchers * `pweight_nonsearchers' + `pweight_constant'  + empldir_age*`pweight_empldir_age' + pubemkag_age * `pweight_pubemkag_age' + PriEmpAg_age *`pweight_PriEmpAg_age' + FrendRel_age*`pweight_FrendRel_age' + SchEmpCt_age*`pweight_SchEmpCt_age' + Unionpro_age*`pweight_Unionpro_age' + Resumes_age*`pweight_Resumes_age' + Plcdads_age*`pweight_Plcdads_age' + Otheractve_age * `pweight_Otheractve_age' + lkatads_age *`pweight_lkatads_age' + Jbtrnprg_age*`pweight_Jbtrnprg_age' + otherpas_age*`pweight_otherpas_age' + empldir_gender*`pweight_empldir_gender' + pubemkag_gender * `pweight_pubemkag_gender' + PriEmpAg_gender *`pweight_PriEmpAg_gender' + FrendRel_gender*`pweight_FrendRel_gender' + SchEmpCt_gender*`pweight_SchEmpCt_gender' + Unionpro_gender*`pweight_Unionpro_gender' + Resumes_gender*`pweight_Resumes_gender' + Plcdads_gender*`pweight_Plcdads_gender' + Otheractve_gender * `pweight_Otheractve_gender' +  lkatads_gender *`pweight_lkatads_gender' +  Jbtrnprg_gender*`pweight_Jbtrnprg_gender' + otherpas_gender*`pweight_otherpas_gender' )
gen search_cond = exp(empldir*`sweight_empldir' + pubemkag*`sweight_pubemkag' + PriEmpAg * `sweight_PriEmpAg' + FrendRel*`sweight_FrendRel' + SchEmpCt*`sweight_SchEmpCt' + Unionpro*`sweight_Unionpro' + Resumes*`sweight_Resumes' + Plcdads*`sweight_Plcdads' + Otheractve *`sweight_Otheractve' + lkatads*`sweight_lkatads' + Jbtrnprg*`sweight_Jbtrnprg' + otherpas*`sweight_otherpas' + age*`sweight_age' + age2 * `sweight_age2' + age3 * `sweight_age3' + age4 * `sweight_age4' + female*`sweight_female' + hs*`sweight_hs' + somecol*`sweight_somecol' + college*`sweight_college' + black*`sweight_black' + married * `sweight_married' + marriedfemale* `sweight_marriedfemale' + np_other* `sweight_np_other' + layoff* `sweight_layoff' + nonsearchers * `sweight_nonsearchers' + invmills*`sweight_invmills' + `sweight_constant' + `sigma_hat'/2 + empldir_age*`sweight_empldir_age' + pubemkag_age * `sweight_pubemkag_age' + PriEmpAg_age *`sweight_PriEmpAg_age' + FrendRel_age*`sweight_FrendRel_age' + SchEmpCt_age*`sweight_SchEmpCt_age' + Unionpro_age*`sweight_Unionpro_age' + Resumes_age*`sweight_Resumes_age' + Plcdads_age*`sweight_Plcdads_age' + Otheractve_age * `sweight_Otheractve_age' + lkatads_age *`sweight_lkatads_age' + Jbtrnprg_age*`sweight_Jbtrnprg_age' + otherpas_age*`sweight_otherpas_age' + empldir_gender*`sweight_empldir_gender' + pubemkag_gender * `sweight_pubemkag_gender' + PriEmpAg_gender *`sweight_PriEmpAg_gender' + FrendRel_gender*`sweight_FrendRel_gender' + SchEmpCt_gender*`sweight_SchEmpCt_gender' + Unionpro_gender*`sweight_Unionpro_gender' + Resumes_gender*`sweight_Resumes_gender' + Plcdads_gender*`sweight_Plcdads_gender' + Otheractve_gender * `sweight_Otheractve_gender' +  lkatads_gender *`sweight_lkatads_gender' +  Jbtrnprg_gender*`sweight_Jbtrnprg_gender' + otherpas_gender*`sweight_otherpas_gender' ) 
gen time_create_method = psearch * search_cond

/*Employed searchers are not included in the ATUS regression - replace their search time with missing */
replace time_create_method = . if mlr==1 | mlr==2 | mlr==-1 | mlr==. 

/*get rid of regression varaibles */
drop *weight* sigma_hat psearch search_cond pi invmills

******************************************************
**calculating averages and plots comparing imputed minutes 
******************************************************

keep if  unemp==1
collapse (mean) time_create* [pweight = newwgt], by(year month)
//seasonally adjusting using month dummies
foreach var in time_create_method time_create {
	reg `var' i.month, r
	forvalues count = 2(1)12 {
		replace `var' = `var' - _b[`count'.month] if month==`count' 
	}
}
order year month  time_create_method time_create 
outsheet using "$int_data/CPS/FigureA4_data.csv", comma replace










