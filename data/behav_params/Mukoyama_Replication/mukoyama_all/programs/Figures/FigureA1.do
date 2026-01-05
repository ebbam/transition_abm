/********************************************************
** Description: Creates Figure A1, showing ATUS data with travel time to interviews
**********************************************************/


clear all
capture log close
set more off
set mem 1g

*****************************************************
***Step 1: Calculating the imputation weights in the ATUS
*****************************************************

use "$raw_ATUS/merged_ATUS_2014.dta", clear

keep if mlr==3 | mlr==4 | mlr==5 /*keep only non-employed*/

//dropping the outliers 
drop time_less8
drop if time_less8_travel==.

local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
local observables age age2 age3 age4 female hs somecol college black married marriedfemale
local lfs np_other layoff nonsearchers

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
gen dummy_search = (time_less8_travel>0)
gen ltime_less8 = log(time_less8_travel)

** First Stage Probit
probit dummy_search `srch_mth' `lfs' `observables' [pw=wgt]
predict pi, xb // this is linear
predict prob_search // this is non-linear probability you search
gen invmills=normalden(pi)/normal(pi)

** Second Stage (adding inverse mills)
reg ltime_less8  `srch_mth' `lfs' `observables' invmills [pw=wgt] 
predict search_cond, xb
predict resid, r
sum resid, d
gen sigma_hat = r(Var)
replace search_cond = exp(search_cond + sigma_hat / 2)

gen time_create = prob_search * search_cond 

**creating weights for the CPS sample
/*part 1 p(search)*/
probit dummy_search `srch_mth' `lfs' `observables' [pw=wgt]
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
   gen pweight_constant = b[1,27]


/*part 2 (search | search>0 ) */

reg ltime_less8 `srch_mth' `lfs' `observables' invmills [pw=wgt]
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
gen sweight_invmills = b[1,27]
gen sweight_constant = b[1,28]


preserve
keep *weight* sigma_hat
keep if _n==1
save "$int_data/ATUS/time_method_reweight_travel", replace
restore

*** For the descriptive charts
keep if mlr==3 | mlr==4
collapse (mean) time_less8_travel [pw=wgt], by(year)
outsheet using "$int_data/ATUS/FigA1a_data.csv", comma replace


*****************************************************
***Step 2: Calculating Imputed minutes in the CPS
*****************************************************
use "$final_CPS/full_CPS_data", clear
keep if mlr>=3

/*initiating additional varaibles */
local count = 0
capture drop invmills

****Reweighting to construct Minutes series with travel time
append using "$int_data/ATUS/time_method_reweight_travel"
erase "$int_data/ATUS/time_method_reweight_travel.dta"

local variables empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas age age2 age3 age4 female hs somecol college black married marriedfemale np_other layoff nonsearchers constant

sum year
local last_obs = r(N)+1

foreach x of local variables {
local pweight_`x' = pweight_`x'[`last_obs']
local sweight_`x' = sweight_`x'[`last_obs']
}

local sweight_invmills  sweight_invmills[`last_obs']
local sigma_hat = sigma_hat[`last_obs']

* Step 1: Calculating the Inverse Mills Ration 
gen pi  = empldir*`pweight_empldir' + pubemkag * `pweight_pubemkag' + PriEmpAg *`pweight_PriEmpAg' + FrendRel*`pweight_FrendRel' + SchEmpCt*`pweight_SchEmpCt' + Unionpro*`pweight_Unionpro' + Resumes*`pweight_Resumes' + Plcdads*`pweight_Plcdads' + Otheractve * `pweight_Otheractve' + lkatads *`pweight_lkatads' + Jbtrnprg*`pweight_Jbtrnprg' + otherpas*`pweight_otherpas' + age*`pweight_age' + age2 * `pweight_age2' + age3 * `pweight_age3' + age4 * `pweight_age4' + female*`pweight_female' + hs*`pweight_hs' + somecol*`pweight_somecol' + college*`pweight_college' + black*`pweight_black' + married * `pweight_married' + marriedfemale* `pweight_marriedfemale' + np_other* `pweight_np_other' + layoff* `pweight_layoff' + nonsearchers * `pweight_nonsearchers' + `pweight_constant'
gen invmills=normalden(pi)/normal(pi)
gen psearch = normprob(empldir*`pweight_empldir' + pubemkag * `pweight_pubemkag' + PriEmpAg *`pweight_PriEmpAg' + FrendRel*`pweight_FrendRel' + SchEmpCt*`pweight_SchEmpCt' + Unionpro*`pweight_Unionpro' + Resumes*`pweight_Resumes' + Plcdads*`pweight_Plcdads' + Otheractve * `pweight_Otheractve' + lkatads *`pweight_lkatads' + Jbtrnprg*`pweight_Jbtrnprg' + otherpas*`pweight_otherpas' + age*`pweight_age' + age2 * `pweight_age2' + age3 * `pweight_age3' + age4 * `pweight_age4' + female*`pweight_female' + hs*`pweight_hs' + somecol*`pweight_somecol' + college*`pweight_college' + black*`pweight_black' + married * `pweight_married' + marriedfemale* `pweight_marriedfemale' + np_other* `pweight_np_other' + layoff* `pweight_layoff' + nonsearchers * `pweight_nonsearchers' + `pweight_constant')
gen search_cond = exp(empldir*`sweight_empldir' + pubemkag*`sweight_pubemkag' + PriEmpAg * `sweight_PriEmpAg' + FrendRel*`sweight_FrendRel' + SchEmpCt*`sweight_SchEmpCt' + Unionpro*`sweight_Unionpro' + Resumes*`sweight_Resumes' + Plcdads*`sweight_Plcdads' + Otheractve *`sweight_Otheractve' + lkatads*`sweight_lkatads' + Jbtrnprg*`sweight_Jbtrnprg' + otherpas*`sweight_otherpas' + age*`sweight_age' + age2 * `sweight_age2' + age3 * `sweight_age3' + age4 * `sweight_age4' + female*`sweight_female' + hs*`sweight_hs' + somecol*`sweight_somecol' + college*`sweight_college' + black*`sweight_black' + married * `sweight_married' + marriedfemale* `sweight_marriedfemale' + np_other* `sweight_np_other' + layoff* `sweight_layoff' + nonsearchers * `sweight_nonsearchers' + `sweight_constant' + invmills*`sweight_invmills' + `sigma_hat'/2)
gen time_create_travel_time = psearch * search_cond 

/*Employed searchers are not included in the ATUS regression - replace their search time with missing */

/*get rid of regression varaibles */
drop *weight* sigma_hat psearch search_cond pi invmills
drop if final_id==. //Note that this is only going to drop the observation I added with the coefficint merge 


preserve
keep if  unemp==1
collapse (mean) time_create time_create_travel_time [pweight = newwgt], by(date)
outsheet using "$int_data/CPS/FigureA1b_data.csv", comma replace
restore 




