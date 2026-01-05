/********************************************************
** Description: Creates Table A4 and A5, estimates of the cobb-douglas matching function
**********************************************************/
clear
set more off

*******************************
**Step 1: Collecting Data
*******************************

use "$final_CPS/full_CPS_data.dta", clear
keep if mlr==3 | mlr==4  // keeping only the unemployed

******* 1) Aggregate Counts 
preserve
collapse (sum) newwgt, by(year month) 
rename newwgt unemployed
save "$int_data/CPS/unemp_aggregate", replace
restore 
preserve
collapse (sum) newwgt, by(year month state) 
rename newwgt unemployed
save "$int_data/CPS/unemp_state", replace
restore 

* 2) Search time (of the unemployed)

use "$final_CPS/full_CPS_data.dta", clear
keep if  unemp==1
preserve
collapse (mean) time_create numsearch [pweight = newwgt], by(year month)
save "$int_data/CPS/search_time_aggregate", replace
restore 
preserve
collapse (mean) time_create numsearch [pweight = newwgt], by(year month state)
save "$int_data/CPS/search_time_state", replace
restore

* 3) Assembling the data and running the matching function estimation

foreach level in aggregate state {

	if "`level'"=="aggregate" {	
		**Aggregate 
		use "$int_data/CPS/unemp_aggregate", clear
		merge 1:1 year month using  "$int_data/CPS/search_time_aggregate"
		assert _m==3
		drop _m			
		merge 1:1 year month using  "$other/hires_aggregate"
		assert _m==3 | _m==1 
		keep if _m==3
		drop _m
		merge 1:1 year month using  "$int_data/theta/JOLTS_agg_theta_monthly"
		assert _m==3
		drop _m	
		rename theta_agg_JOLTS theta 
		local add "i.month"
		gen date = ym(year, month)
		local level_num 4
	}
		
	if "`level'"=="state" {	
		**State 
		use "$int_data/CPS/unemp_state", clear
		merge 1:1 year month state using  "$int_data/CPS/search_time_state"	
		assert _m==3
		drop _m		
		merge m:1 state using "$maps/US_FIPS_Codes"
		drop if _m==2
		drop _m	
		keep if year>=2005 
		drop if year==2005 & month<=4
		rename fips statefips
		merge 1:1 year month statefips using  "$int_data/theta/HWOL_state_theta_monthly"
		assert year==2015 if _m==2 
		drop if _m==2 
		drop _m
		
		gen quarter = 1 if month==1 | month==2 | month==3 
		replace quarter = 2 if month==4 | month==5 | month==6 
		replace quarter = 3 if month==7 | month==8 | month==9 
		replace quarter = 4 if month==10 | month==11 | month==12 
		rename statefips fips
		merge m:1 year quarter fips using  "$other/hires_state"
		drop if year==2005 & quarter==1
		drop if state=="MA" //not in QWI
		drop if year==2014 & quarter==4 //this is when QWI ends
		drop _m				
		
		//collapsing everything to quarterly 
		collapse (sum) unemployed hires (mean) time_create theta, by(year fips quarter)
		local add "i.fips i.date"
		gen date = yq(year, quarter)
		local level_num 5
	}
		
	/*logging variables */
	foreach x in  unemployed time_create hires theta {
		gen l`x' = ln(`x')
		gen l`x'_t_1 = l`x'[_n-1]
		gen `x'_t_1 = `x'[_n-1]
	}	
	
	gen post_rec = 1 if year>=2008
	replace post_rec = 0 if year<2008
		
	gen lhires_rate = ln(hires/unemployed_t_1)
	gen lsearch_time_rate = ln(time_create_t_1*unemployed/unemployed_t_1)
	*gen lsearch_time_rate = ln(time_create)
	gen ltheta_rec = ltheta*post_rec
	gen lsearch_time_rate_rec = lsearch_time_rate*post_rec

	label var ltheta "log $\frac{v_t}{u_{t}}$"
	label var lsearch_time_rate "log $\bar{s_t}$"
	label var ltheta_rec "log $\frac{h_t}{u_{t}}$ * Recession Dummy"
	label var lsearch_time_rate_rec "log $\bar{s_t}$  * Recession Dummy"
	
	est clear

	eststo base: eststo: reg lhires_rate ltheta `add', r
	eststo base_rec: eststo: reg lhires_rate ltheta ltheta_rec `add', r
	eststo base_time: eststo: reg lhires_rate ltheta lsearch_time_rate `add', r
	eststo base_time_rec: eststo: reg lhires_rate ltheta ltheta_rec lsearch_time_rate `add', r
	eststo base_time_rec2: eststo: reg lhires_rate ltheta ltheta_rec lsearch_time_rate lsearch_time_rate_rec `add', r


	file open mytable using "$tables/TableA`level_num'.tex", write text replace
	file write mytable "\begin{table}" _newline
	file write mytable "\begin{center}" _newline
	file write mytable "\footnotesize" _newline
	file write mytable "\begin{tabular}{l*{6}{l}} \label{tab:matching_function_`level'} \\" _newline
	file write mytable "\hline \hline" _newline
	file write mytable "&& Basic & Dummy & Search & Search Dummy & Search Dummy \\" _newline
	file write mytable "\\" _newline
	file close mytable

	estout base base_rec base_time base_time_rec base_time_rec2 using "$tables/TableA`level_num'.tex", append ///
	  style(tex) ///
	  mlabels(none) collabels(none) ///
	  cells(b(star fmt(3)) se(par fmt(3))) ///
	  starlevels($^{*}$ 0.10 $^{**}$ 0.05 $^{***}$ 0.01) ///
	  keep(ltheta ltheta_rec lsearch_time_rate lsearch_time_rate_rec) ///
	  order(ltheta ltheta_rec lsearch_time_rate lsearch_time_rate_rec) ///
	  label  ///
	  prefoot("\\") ///
	  stats(r2 N, label(R$^2$ Observations) fmt(3 0))  ///
	  extracols(1) varwidth(20) modelwidth(0 10 0 10)
	  
	 
	file open mytable using "$tables/TableA`level_num'.tex", write text append
	file write mytable "\hline \hline" _newline
	file write mytable "\end{tabular}" _newline
	if "`level'"=="state" {
		file write mytable "\noindent \footnotesize \\ $^{*}$, $^{**}$, $^{***}$: significant at the 10, 5, and 1 percent level, respectively. Regressions include quarter, year, and state fixed effects. Robust standard errors." _newline
	}
	else {
		file write mytable "\noindent \footnotesize \\ $^{*}$, $^{**}$, $^{***}$: significant at the 10, 5, and 1 percent level, respectively. Regressions include month fixed effects. Robust standard errors." _newline
	}
	file write mytable "\end{center}" _newline
	file write mytable "\end{table}" _newline
	file close mytable
		
	erase 	 "$int_data/CPS/unemp_`level'.dta"
	erase 	 "$int_data/CPS/search_time_`level'.dta"
}

