
/********************************************************
** Description: Creates Table 4 , the correlation of state search effort and a measure of cyclicality
**********************************************************/

clear all
set more off
cap log close
est clear
set matsize 10000


use "$final_CPS/full_CPS_data.dta", clear

// calculating the state population shares
collapse (sum) newwgt, by(year month state)
// converting from state abbreviations to FIPS codes
merge m:1 state using "$maps/US_FIPS_Codes"
drop if _m==2
drop _m
bysort year month: egen total_pop = sum(newwgt)
gen pop_share = newwgt/total_pop
keep year month fips pop_share total_pop
save "$other/state_population", replace

use "$final_CPS/full_CPS_data.dta", clear

/*keeping only the unemployed */
keep if unemp==1

/*collapse to the average job search effort by state over time */
collapse (mean) numsearch time_create [pweight = newwgt], by(year month state)
merge m:1 state using "$maps/US_FIPS_Codes"
drop if _m==2
drop _m


* merging on state-level aggregates */ 
merge m:1 year month fips using "$other/state_population"
assert _m==3 
drop _m
erase "$other/state_population.dta"

merge m:1 year month fips using "$other/urate_by_state_nsa"
drop if _m==2 & year!=1997
drop _m


merge m:1 year month fips using "$other/state_ue"
drop if _m==2 & year!=1997
assert _m ==3 
drop _m

rename fips statefips
merge m:1 year month statefips using "$int_data/theta/HWOL_state_theta_monthly"
drop if _m==2 & year!=1997
assert year<=2005 if _m==1
drop _m ym statecensus statename
rename statefips fips

drop if fips==0 //national data
tostring fips, replace 

encode state, gen(state_num)
drop if state_num==.

gen date = ym(year,month)
sort date

************************************************************
****Part 1: Monthly 
************************************************************

gen ltime_create = log(time_create)
keep year month state_num urate pop_share ltime_create theta ue_rate
reshape wide urate pop_share ltime_create theta ue_rate, i(year month) j(state_num)
gen date = ym(year,month)

tsset date
// hp-filtering series

forvalue x = 1/51{

	// A fix for Janurary 1997, which has missing time 
	replace ltime_create`x' = (ltime_create`x'[_n-1] + ltime_create`x'[_n+1])/2 if ltime_create`x'==.
	replace urate`x' = (urate`x'[_n-1] + urate`x'[_n+1])/2 if urate`x'==.
	replace theta`x' = (theta`x'[_n-1] + theta`x'[_n+1])/2 if theta`x'==.
	replace ue_rate`x' = (ue_rate`x'[_n-1] + ue_rate`x'[_n+1])/2 if ue_rate`x'==.
	
	replace pop_share`x' = (pop_share`x'[_n-1] + pop_share`x'[_n+1])/2 if pop_share`x'==.
	
	* Seasonally Adjusting the Series with Month dummies	
	foreach var in ltime_create urate theta ue_rate {
	
		if "`var'"!="ue_rate" { // these are already seasonally adjusted
			reg `var'`x' i.month
			predict temp_`var'`x', resid
			replace `var'`x' = temp_`var'`x' + _b[_cons]
			drop temp_`var'`x'
		}
		
		if "`var'"=="ue_rate" {
			tsfilter hp temp_`var'`x' = `var'`x' if year>1995, smooth(10) trend(smooth_`var'`x')		
		}
		else {
			tsfilter hp temp_`var'`x' = `var'`x', smooth(10) trend(smooth_`var'`x')		
		}		
	}	
}

drop temp*
reshape long   ltime_create smooth_ltime_create ///
			urate  smooth_urate pop_share    ///
			theta  smooth_theta ue_rate smooth_ue_rate, i(year month date) j(state_num)
					
bysort state: egen avg_pop_share = mean(pop_share)
drop if year==1997 & month==1

xi i.state_num i.date
forvalues x = 2(1)51 {
	gen date_Istate_num_`x' =    _Istate_num_`x'*date
}
sort state_num date
foreach var in ltime_create smooth_ltime_create theta smooth_theta  urate smooth_urate smooth_ue_rate ue_rate {
	gen d3_`var' = `var' - `var'[_n-3] if date==date[_n-3]+3 & state_num==state_num[_n-3]
	gen d6_`var' = `var' - `var'[_n-6] if date==date[_n-6]+6 & state_num==state_num[_n-6]
}

// regression analysis at the state*month level
est clear

// Unemployment Rate 
rename d3_smooth_urate lu_var
eststo d3Ul_month_t_wgt: eststo:      reg d3_smooth_ltime_create lu_var _Istate_num_* _Idate_* date_Istate_num_* [aweight=avg_pop_share], cluster(state_num)
estadd local year "1994-2014" , replace
rename lu_var d3_smooth_urate 
rename d6_smooth_urate lu_var
eststo d6Ul_month_t_wgt: eststo:      reg d6_smooth_ltime_create lu_var _Istate_num_* _Idate_* date_Istate_num_* [aweight=avg_pop_share], cluster(state_num)
estadd local year "1994-2014" , replace
rename lu_var d6_smooth_urate 

// Theta 
rename d3_smooth_theta ltheta_var
eststo d3Tl_month_t_wgt: eststo:      reg d3_smooth_ltime_create ltheta_var _Istate_num_* _Idate_* date_Istate_num_* [aweight=avg_pop_share], cluster(state_num)
estadd local source "HWOL" , replace
estadd local year "2005-2014" , replace
rename ltheta_var d3_smooth_theta 
rename d6_smooth_theta ltheta_var
eststo d6Tl_month_t_wgt: eststo:      reg d6_smooth_ltime_create ltheta_var _Istate_num_* _Idate_* date_Istate_num_* [aweight=avg_pop_share], cluster(state_num)
estadd local source "HWOL" , replace
estadd local year "2005-2014" , replace
rename ltheta_var d6_smooth_theta 


// UE Rate 
rename d3_smooth_ue_rate ue_var
eststo d3UE_month_t_wgt: eststo:       reg d3_smooth_ltime_create ue_var _Istate_num_* _Idate_* date_Istate_num_* [aweight=avg_pop_share], cluster(state_num)
estadd local source "CPS" , replace
estadd local year "1994-2014" , replace
rename ue_var d3_smooth_ue_rate
rename d6_smooth_ue_rate ue_var
eststo d6UE_month_t_wgt: eststo:       reg d6_smooth_ltime_create ue_var _Istate_num_* _Idate_* date_Istate_num_* [aweight=avg_pop_share], cluster(state_num)
estadd local source "CPS" , replace
estadd local year "1994-2014" , replace
rename ue_var d6_smooth_ue_rate


gen ltheta_var = 1
gen lu_var = 1
gen emp_var = 1
gen ue_var = 1
label var ltheta_var "$\Delta \frac{v}{u}$"
label var lu_var "$\Delta  \text{unemployment rate}$"
label var ue_var "$\Delta  \text{U-to-E Rate}$"

 
 * Alternate Table - Employment and UE Rates - Referee Note
file open mytable using "$tables/Table4.tex", write text replace
file write mytable "\begin{table}[htbp]" _newline
file write mytable "\caption{Exploiting State Level Variation} \label{tab:state_level}" _newline
file write mytable "\begin{center}" _newline
file write mytable "\scriptsize" _newline
file write mytable "\begin{tabular}{  l  c  c  c c  c  c  c  c  c c c c }" _newline
file write mytable "\hline \hline" _newline
file write mytable "\\" _newline
file write mytable " && 3-Month & 6-Month   && 3-Month & 6-Month  && 3-Month & 6-Month  \\" _newline
file write mytable "&& Change & Change && Change & Change  && Change & Change  \\" _newline
file write mytable "\\" _newline
file close mytable


estout  d3Ul_month_t_wgt d6Ul_month_t_wgt d3Tl_month_t_wgt  d6Tl_month_t_wgt  d3UE_month_t_wgt  d6UE_month_t_wgt using  "$tables/Table4.tex", append ///
  style(tex) ///
  mlabels(none) collabels(none) ///
  cells(b(star fmt(4)) se(par fmt(4))) ///
  starlevels($^{*}$ 0.10 $^{**}$ 0.05 $^{***}$ 0.01) ///
  keep(lu_var ltheta_var ue_var) ///
  order(lu_var ltheta_var ue_var) ///
  label  ///
  prefoot("\\") ///
  stats(year N r2, label("Years" Observations "R-Squared") fmt(0 0 3))  ///
  extracols(1 3 5 7) varwidth(36) modelwidth(0 14 0 14)
  

file open mytable using "$tables/Table4.tex", write text append
file write mytable "\hline \hline" _newline
file write mytable "\end{tabular}" _newline
file write mytable "\end{center}" _newline
file write mytable "\end{table}" _newline
file close mytable
 
