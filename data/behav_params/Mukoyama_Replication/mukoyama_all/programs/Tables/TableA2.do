/********************************************************
** Description: This program calculates Table A2 (Appendix Table 2)  
**********************************************************/

clear all
capture log close
set more off
set mem 1g
local M 1000

********Loading all of the estimates to calculate the final standard errors 
use "$tables/bootstraps/regression_estimates", clear 


* We use Rubin's combination rules to arrive at our final estimates
local variable_list lcycle age age2 age3 age4  black married female marriedfemale hs somecol college soc2 soc3 soc4 undur undur2 undur3 undur4 elig wksleft_tot
    
foreach var of local variable_list {   
	
	replace `var' = . if `var'==0
	replace `var'_std = . if `var'_std==0
	
	* The MI predication of the coefficient of interest is the average of the coefficients
	bysort spec type_cycle: egen `var'_final = mean(`var')

	* The MI standard error is the square root of the total MI variance: 
	*      T = Average within-imputation variance + (1+1/M) between imputation variance 

	replace `var'_std = `var'_std^2
	bysort spec type_cycle: egen `var'_std_avg = mean(`var'_std)
	gen B = (`var' - `var'_final)^2
	bysort spec type_cycle: egen B_i = sum(B)
	gen se_`var'_final = sqrt(`var'_std_avg + (1+1/`M')*B_i*(1/(`M'-1)))
	drop B `var'_std_avg B_i `var' `var'_std
}

drop draw
duplicates drop

foreach var of local variable_list {   
	gen sig10_`var' = (`var'_final<0 & ((`var'_final + 1.645*se_`var'_final)<0)) | (`var'_final>0 & ((`var'_final - 1.645*se_`var'_final)>0)) 
	gen sig5_`var' = (`var'_final<0 & ((`var'_final + 1.96*se_`var'_final)<0)) | (`var'_final>0 & ((`var'_final - 1.96*se_`var'_final)>0)) 
	gen sig1_`var' = (`var'_final<0 & ((`var'_final + 2.58*se_`var'_final)<0)) | (`var'_final>0 & ((`var'_final - 2.58*se_`var'_final)>0)) 
	rename `var'_final beta_`var'
	rename se_`var'_final se_`var'
}

********************************************************************************
********* Exporting Tables with these estimates 
********************************************************************************

replace spec = spec + "_"
replace spec = "nop_nox_" if spec=="nopanel_nox_"
replace spec = "nop_" if spec=="nopanel_"
replace type_cycle = "BAR" if type_cycle=="theta_agg_BAR"
replace type_cycle = "JOLTS" if type_cycle=="theta_agg_JOLTS"
replace type_cycle = "HWOL" if type_cycle=="theta_state_HWOL"
rename observation N_

* In order to do this using tabstat, I need to reshape the data 
reshape long sig10_ sig5_ sig1_ beta_ se_ , i(spec type_cycle N_) j(xvar) str
reshape wide beta_* se_* sig10_* sig5_* sig1_* N_*, i(xvar type_cycle) j(spec) str
reshape wide beta_* se_* sig10_* sig5_* sig1_* N_*, i(xvar) j(type_cycle) str

* Creating Variable Labels 
gen var_label  = " log ($\theta$)" if xvar=="lcycle"
replace var_label = "Age" if xvar=="age" 
replace var_label = "Age$^2$" if xvar=="age2" 
replace var_label = "Age$^3$" if xvar=="age3" 
replace var_label = "Age$^4$" if xvar=="age4" 
replace var_label = "Unemployment Duration" if xvar=="undur" 
replace var_label = "Unemployment Duration$^2$" if xvar=="undur2" 
replace var_label = "Unemployment Duration$^3$" if xvar=="undur3" 
replace var_label = "Unemployment Duration$^4$" if xvar=="undur4" 
replace var_label = "Black" if xvar=="black" 
replace var_label = "Married" if xvar=="married" 
replace var_label = "Female" if xvar=="female" 
replace var_label = "Married x Female" if xvar=="marriedfemale" 
replace var_label = "High School" if xvar=="hs" 
replace var_label = "Some College" if xvar=="somecol" 
replace var_label = "College" if xvar=="college" 
replace var_label = "Cognitive Routine" if xvar=="soc2" 
replace var_label = "Manual Non-Routine" if xvar=="soc3" 
replace var_label = "Manual Routine" if xvar=="soc4" 
replace var_label = "Eligible for UI" if xvar=="elig" 
replace var_label = "Weeks of Benefits Remaining" if xvar=="wksleft_tot" 


*********************************Table A2 : Panel Regression with Alternate Cyclical Indicators

file open mytable using "$tables/TableA2.tex", write text replace
file write mytable "\begin{table}[htbp]" _newline
file write mytable "\begin{center}" _newline
file write mytable "\scriptsize" _newline
file write mytable "\begin{tabular}{l*{6}{c}} \label{tab:alternate_indicators}" _newline
file write mytable "\\" _newline
file write mytable " && State HWOL &  S \& P  & House Price & State House Price & Payroll Employment \\" _newline
file write mytable "\\ \hline" _newline
file close mytable

replace var_label  = " log (\text{Indicator})" if xvar=="lcycle"

local variable_list lcycle undur undur2 undur3 undur4 

foreach var of local variable_list {

	preserve
	keep if xvar=="`var'"
	local label = var_label[1]
	
	estpost tabstat beta_panel_FE_HWOL beta_panel_FE_sp beta_panel_FE_hp beta_panel_FE_hp_state  beta_panel_FE_payroll if xvar=="`var'" , statistics(mean) missing

	estout . using "$tables/TableA2.tex" ,  append ///	
	style(tex) ///
	cells("beta_panel_FE_HWOL(fmt(3)) beta_panel_FE_sp(fmt(3)) beta_panel_FE_hp(fmt(3)) beta_panel_FE_hp_state(fmt(3)) beta_panel_FE_payroll(fmt(3))") /// 
	mlabels(none) collabels(none) ///
	varlabels(mean "`label'") /// 
	extracols(1) varwidth(36) modelwidth(0 14 0 14)

	estpost tabstat se_panel_FE_HWOL se_panel_FE_sp se_panel_FE_hp se_panel_FE_hp_state se_panel_FE_payrol if xvar=="`var'" , statistics(mean) missing

	estout . using "$tables/TableA2.tex" ,  append ///	
	style(tex) ///
	cells("se_panel_FE_HWOL(fmt(3) par) se_panel_FE_sp(fmt(3)) se_panel_FE_hp(fmt(3)) se_panel_FE_hp_state(fmt(3)) se_panel_FE_payroll(fmt(3))") /// 
	mlabels(none) collabels(none) ///
	varlabels(mean " ") /// 
	extracols(1) varwidth(36) modelwidth(0 14 0 14)	
	
	restore
	
	file open mytable using "$tables/TableA2.tex", write text append
	file write mytable "\\" _newline
	file close mytable
	
}

//adding number of observations
estpost tabstat N_panel_FE_HWOL N_panel_FE_sp N_panel_FE_hp N_panel_FE_hp_state N_panel_FE_payroll if xvar=="lcycle" , statistics(mean) missing

estout . using "$tables/TableA2.tex" ,  append ///	
style(tex) ///
cells("N_panel_FE_HWOL N_panel_FE_sp N_panel_FE_hp N_panel_FE_hp_state N_panel_FE_payroll") /// 
mlabels(none) collabels(none) ///
varlabels(mean "No. Obs.") /// 
extracols(1) varwidth(36) modelwidth(0 14 0 14)		
	
	
//closing the table 
file open mytable using "$tables/TableA2.tex", write text append
file write mytable "\hline \hline" _newline
file write mytable "\end{tabular}" _newline
file write mytable "\end{center}" _newline
file write mytable "\noindent \footnotesize \\ All regressions include individual fixed effects. Coefficients and standard errors are calculated using multiple impurtation methods described in detail in Appendix \ref{micro_regression_appendix}. Column 1 includes data from May 2005-Dec 2014, and columns 2-4 include data from Jan 1994-Dec 2014." _newline
file write mytable "\end{table}" _newline
file close mytable






