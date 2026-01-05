/********************************************************
** Description: Creates Figure A7, showing ATUS imputation with additional cyclical indicators
**********************************************************/

clear all
capture log close
set more off
set mem 1g

***************************************************
*** loading ATUS dataset
***************************************************
use "$raw_ATUS/merged_ATUS_2014.dta", clear
keep if mlr==3 | mlr==4 | mlr==5 /*keep only non-employed*/


local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
local observables age age2 age3 age4 female hs somecol college black married marriedfemale
local lfs np_other layoff nonsearchers

**************************************************
***Merging in Theta, GDP and urate
**************************************************

merge m:1 year month using "$int_data/theta/BARNICHON_agg_theta_monthly"
keep if _m==3 /*this will drop years and months that are not in the ATUS */
drop _m
rename theta_agg_BAR theta
merge m:1 year month using "$other/monthly_urate"
keep if _m==3
drop _m

merge m:1 year quarter using "$other//quarterly_gdp"
keep if _m==3
drop _m

foreach method of local srch_mth {
	gen `method'_theta = `method'* theta
	gen `method'_lgdp = `method'* lgdp
	gen `method'_lgdp_cycle = `method'* lgdp_cycle
	gen `method'_urate = `method'* urate
}

/*defining local variables */
local  srch_mth_theta empldir_theta pubemkag_theta PriEmpAg_theta FrendRel_theta SchEmpCt_theta Unionpro_theta Resumes_theta Plcdads_theta Otheractve_theta lkatads_theta  Jbtrnprg_theta otherpas_theta
local  srch_mth_gdp empldir_gdp pubemkag_gdp PriEmpAg_gdp FrendRel_gdp SchEmpCt_gdp Unionpro_gdp Resumes_gdp Plcdads_gdp Otheractve_gdp lkatads_gdp  Jbtrnprg_gdp otherpas_gdp
local  srch_mth_urate empldir_urate pubemkag_urate PriEmpAg_urate FrendRel_urate SchEmpCt_urate Unionpro_urate Resumes_urate Plcdads_urate Otheractve_urate lkatads_urate  Jbtrnprg_urate otherpas_urate
local  srch_mth_lgdp empldir_lgdp pubemkag_lgdp PriEmpAg_lgdp FrendRel_lgdp SchEmpCt_lgdp Unionpro_lgdp Resumes_lgdp Plcdads_lgdp Otheractve_lgdp lkatads_lgdp  Jbtrnprg_lgdp otherpas_lgdp
local  srch_mth_lgdp_cycle empldir_lgdp_cycle pubemkag_lgdp_cycle PriEmpAg_lgdp_cycle FrendRel_lgdp_cycle SchEmpCt_lgdp_cycle Unionpro_lgdp_cycle Resumes_lgdp_cycle Plcdads_lgdp_cycle Otheractve_lgdp_cycle lkatads_lgdp_cycle  Jbtrnprg_lgdp_cycle otherpas_lgdp_cycle

***************************************************
**cleaning variable names for regression output
**************************************************

gen dummy_search = 1 if time_less8>0 & time_less8!=.
replace dummy_search = 0 if time_less8<=0 & time_less8!=.
gen ltime_less8 = log(time_less8)

/*looping through aggregate variables */
local var theta lgdp_cycle urate

foreach agg of local var {

	*****************************************************
	**Baseline regression with theta interacted with methods 
	****************************************************

	/*part 1 p(search)*/
	probit dummy_search `srch_mth' `srch_mth_`agg'' `agg' `lfs' `observables' [pw=wgt], r
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
		  foreach x of local srch_mth_`agg' {
		  gen  pweight_`x'= b[1,`count' + 1]
		  local count = `count' + 1
	   }
	   
		  gen  pweight_`agg' = b[1,`count' + 1]
		  local count = `count' + 1
		  
	   foreach x of local lfs {
		  gen pweight_`x'= b[1,`count' + 1]
		  local count = `count' + 1
	}
	   foreach x of local observables {
		  gen pweight_`x'= b[1,`count' + 1]
		  local count = `count' + 1
	   }
	   gen pweight_constant = b[1,40]

	/*part 2 (search | search>0 ) */
	reg ltime_less8 `srch_mth' `srch_mth_`agg'' `agg' `lfs' `observables' invmills [pw=wgt], r
	predict search_cond, xb
	predict resid, r
	sum resid, d
	gen sigma_hat = r(Var)
	replace search_cond = exp(search_cond + sigma_hat / 2)
	
	matrix b = e(b)
	   /*saving estimates in the local variables */
	   local count = 0
	   foreach x of local srch_mth {
		  gen  sweight_`x'= b[1,`count' + 1]
		  local count = `count' + 1
	   }
			 foreach x of local srch_mth_`agg' {
		  gen  sweight_`x'= b[1,`count' + 1]
		  local count = `count' + 1
	   }
		  gen  sweight_`agg' = b[1,`count' + 1]
		  local count = `count' + 1
	   
	   foreach x of local lfs {
		  gen sweight_`x'= b[1,`count' + 1]
		  local count = `count' + 1
	}
	   foreach x of local observables {
		  gen sweight_`x'= b[1,`count' + 1]
		  local count = `count' + 1
	   }
		gen sweight_invmills = b[1,40]
	   gen sweight_constant = b[1,41]
	   
	   preserve
	   keep *weight* sigma_hat
	   keep if _n==1
	   save "$int_data/time_method_reweight_`agg'", replace
	   restore
	   
	gen time_create = prob_search * search_cond 
	drop time_create sigma_hat prob_search search_cond resid sweight* pweight* pi invmills

}




****************************************************
****Intensive Margin - Summing average search time (methods and Created time)
****************************************************

use "$final_CPS/full_CPS_data", clear

//dropping the employed 
drop if mlr==1 | mlr==2

/*merging in theta, gdp and urate*/
merge m:1 year month using "$int_data/theta/BARNICHON_agg_theta_monthly"
assert _m==3 
drop _m
rename theta_agg_BAR theta

merge m:1 year month using "$other/monthly_urate"
keep if _m==3
drop _m

gen quarter = 1 if month==1 | month==2 | month==3
replace quarter = 2 if month==4 | month==5 | month==6
replace quarter = 3 if month==7 | month==8 | month==9
replace quarter = 4 if month==10 | month==11 | month==12

merge m:1 year quarter using "$other/quarterly_gdp"
keep if _m==3
drop _m

*** Dropping variables that I don't need to reduce the size of the dataset 
drop wks_tot wksleft_tot wksused_tot untype undur

***LOOPING THROUGH THE AGGREGATE VARIABLES*****
local var theta urate lgdp_cycle

/*initiating additional varaibles */
local count = 0

foreach agg of local var {

***************************************************
****Reweighting to construct Minutes series with aggregate variables
***************************************************

	append using "$int_data/time_method_reweight_`agg'"
	erase "$int_data/time_method_reweight_`agg'.dta"

	local variables empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas empldir_`agg' pubemkag_`agg' PriEmpAg_`agg' FrendRel_`agg' SchEmpCt_`agg' Unionpro_`agg' Resumes_`agg' Plcdads_`agg' Otheractve_`agg' lkatads_`agg'  Jbtrnprg_`agg' otherpas_`agg' `agg' age age2 age3 age4 female hs somecol college black married marriedfemale np_other layoff nonsearchers constant

	/*defining weights */
	sum year
	local count = `count' + 1
	local last_obs = r(N)+`count'
	foreach x of local variables {
	local pweight_`x' = pweight_`x'[`last_obs']
	local sweight_`x' = sweight_`x'[`last_obs']
	}

	local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas

	foreach method of local srch_mth {
		gen `method'_`agg' = `method'* `agg'
	}

	local sweight_invmills = sweight_invmills[`last_obs']
	local sigma_hat = sigma_hat[`last_obs']


	gen pi  = empldir*`pweight_empldir' + pubemkag * `pweight_pubemkag' + PriEmpAg *`pweight_PriEmpAg' + FrendRel*`pweight_FrendRel' + SchEmpCt*`pweight_SchEmpCt' + Unionpro*`pweight_Unionpro' + Resumes*`pweight_Resumes' + Plcdads*`pweight_Plcdads' + Otheractve * `pweight_Otheractve' + lkatads *`pweight_lkatads' + Jbtrnprg*`pweight_Jbtrnprg' + otherpas*`pweight_otherpas' + empldir_`agg'*`pweight_empldir_`agg'' + pubemkag_`agg' * `pweight_pubemkag_`agg'' + PriEmpAg_`agg' *`pweight_PriEmpAg_`agg'' + FrendRel_`agg'*`pweight_FrendRel_`agg'' + SchEmpCt_`agg'*`pweight_SchEmpCt_`agg'' + Unionpro_`agg'*`pweight_Unionpro_`agg'' + Resumes_`agg'*`pweight_Resumes_`agg'' + Plcdads_`agg'*`pweight_Plcdads_`agg'' + Otheractve_`agg' * `pweight_Otheractve_`agg'' + lkatads_`agg' *`pweight_lkatads_`agg'' + Jbtrnprg_`agg'*`pweight_Jbtrnprg_`agg'' + otherpas_`agg'*`pweight_otherpas_`agg'' + `agg'*`pweight_`agg'' + age*`pweight_age' + age2 * `pweight_age2' + age3 * `pweight_age3' + age4 * `pweight_age4' + female*`pweight_female' + hs*`pweight_hs' + somecol*`pweight_somecol' + college*`pweight_college' + black*`pweight_black' + married * `pweight_married' + marriedfemale* `pweight_marriedfemale' + np_other* `pweight_np_other' + layoff* `pweight_layoff' + nonsearchers * `pweight_nonsearchers' + `pweight_constant'
	gen invmills=normalden(pi)/normal(pi)
	gen psearch = normprob(empldir*`pweight_empldir' + pubemkag * `pweight_pubemkag' + PriEmpAg *`pweight_PriEmpAg' + FrendRel*`pweight_FrendRel' + SchEmpCt*`pweight_SchEmpCt' + Unionpro*`pweight_Unionpro' + Resumes*`pweight_Resumes' + Plcdads*`pweight_Plcdads' + Otheractve * `pweight_Otheractve' + lkatads *`pweight_lkatads' + Jbtrnprg*`pweight_Jbtrnprg' + otherpas*`pweight_otherpas' + empldir_`agg'*`pweight_empldir_`agg'' + pubemkag_`agg' * `pweight_pubemkag_`agg'' + PriEmpAg_`agg' *`pweight_PriEmpAg_`agg'' + FrendRel_`agg'*`pweight_FrendRel_`agg'' + SchEmpCt_`agg'*`pweight_SchEmpCt_`agg'' + Unionpro_`agg'*`pweight_Unionpro_`agg'' + Resumes_`agg'*`pweight_Resumes_`agg'' + Plcdads_`agg'*`pweight_Plcdads_`agg'' + Otheractve_`agg' * `pweight_Otheractve_`agg'' + lkatads_`agg' *`pweight_lkatads_`agg'' + Jbtrnprg_`agg'*`pweight_Jbtrnprg_`agg'' + otherpas_`agg'*`pweight_otherpas_`agg'' + `agg'*`pweight_`agg'' + age*`pweight_age' + age2 * `pweight_age2' + age3 * `pweight_age3' + age4 * `pweight_age4' + female*`pweight_female' + hs*`pweight_hs' + somecol*`pweight_somecol' + college*`pweight_college' + black*`pweight_black' + married * `pweight_married' + marriedfemale* `pweight_marriedfemale' + np_other* `pweight_np_other' + layoff* `pweight_layoff' + nonsearchers * `pweight_nonsearchers' + `pweight_constant')
	gen search_cond = exp(empldir*`sweight_empldir' + pubemkag*`sweight_pubemkag' + PriEmpAg * `sweight_PriEmpAg' + FrendRel*`sweight_FrendRel' + SchEmpCt*`sweight_SchEmpCt' + Unionpro*`sweight_Unionpro' + Resumes*`sweight_Resumes' + Plcdads*`sweight_Plcdads' + Otheractve *`sweight_Otheractve' + lkatads*`sweight_lkatads' + Jbtrnprg*`sweight_Jbtrnprg' + otherpas*`sweight_otherpas' + empldir_`agg'*`sweight_empldir_`agg'' + pubemkag_`agg'*`sweight_pubemkag_`agg'' + PriEmpAg_`agg' * `sweight_PriEmpAg_`agg'' + FrendRel_`agg'*`sweight_FrendRel_`agg'' + SchEmpCt_`agg'*`sweight_SchEmpCt_`agg'' + Unionpro_`agg'*`sweight_Unionpro_`agg'' + Resumes_`agg'*`sweight_Resumes_`agg'' + Plcdads_`agg'*`sweight_Plcdads_`agg'' + Otheractve_`agg' *`sweight_Otheractve_`agg'' + lkatads_`agg'*`sweight_lkatads_`agg'' + Jbtrnprg_`agg'*`sweight_Jbtrnprg_`agg'' + otherpas_`agg'*`sweight_otherpas_`agg'' + `agg'*`sweight_`agg'' + age*`sweight_age' + age2 * `sweight_age2' + age3 * `sweight_age3' + age4 * `sweight_age4' + female*`sweight_female' + hs*`sweight_hs' + somecol*`sweight_somecol' + college*`sweight_college' + black*`sweight_black' + married * `sweight_married' + marriedfemale* `sweight_marriedfemale' + np_other* `sweight_np_other' + layoff* `sweight_layoff' + nonsearchers * `sweight_nonsearchers' + invmills*`sweight_invmills' + `sweight_constant' + `sigma_hat'/2)
	gen time_create_`agg'_method = psearch * search_cond


	/*Employed searchers are not included in the ATUS regression - replace their search time with missing */
	replace time_create_`agg'_method = . if mlr==1 | mlr==2 | mlr==-1 | mlr==. 

	/*get rid of regression varaibles */
	drop *weight* sigma_hat psearch search_cond pi invmills

}

******************************************************
**calculating averages and plots comparing imputed minutes 
******************************************************

preserve
keep if  unemp==1
collapse (mean) time_create* [pweight = newwgt], by(year month)
//seasonally adjusting using month dummies
foreach var in time_create_theta_method time_create_lgdp_cycle_method time_create_urate_method {
	reg `var' i.month, r
	forvalues count = 2(1)12 {
		replace `var' = `var' - _b[`count'.month] if month==`count' 
	}
}
order year month time_create_theta_method time_create_lgdp_cycle_method time_create_urate_method
outsheet using "$int_data/CPS/FigureA7_data.csv", comma replace
restore 



