/********************************************************
** Description: Creates Figure A11, showing search by individual method
**********************************************************/
clear all
set more off
cap log close

****************************************************
****Part 1: Plotting the fraction of the unemployed searchers who use each search method over time 
****************************************************
cd "$final_CPS"
use full_CPS_data.dta, clear

keep if  searchers==1
local methods empldir pubemkag PriEmpAg FrendRel SchEmpCt Unionpro Resumes Plcdads Otheractve lkatads  Jbtrnprg otherpas
bysort year month: egen total = sum(newwgt)
foreach var of local methods {
		gen temp = `var'*newwgt
		bysort year month: egen count_`var' = sum(temp)
		gen frac_`var' = count_`var' /total
		drop temp
}

keep year month frac_* count_*
duplicates drop
outsheet using "$int_data/CPS/FigureA11_data.csv", comma replace

