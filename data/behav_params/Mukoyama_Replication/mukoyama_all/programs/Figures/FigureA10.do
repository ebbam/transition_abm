/********************************************************
** Description: Creates Figure A10, showing historical search methods in the CPS
**********************************************************/

clear
set more off
cap log close

***************************************************
****Appending datasets to be merged
***************************************************

cd "$raw_CPS"

gen t = .
gen month = .

forvalue year = 1976(1)1993 {
   forvalue month = 1(1)12 {
      if `month' < 10 {
         append using bas`year'0`month'.dta
         replace t = `year'0`month' if t == . 
		 replace month=`month' if month== . | month>12
      }
      if `month' >= 10 {
         append using bas`year'`month'.dta
         replace t = `year'`month' if t == .
		 replace month=`month' if month == . | month>12

      }
   }
}

keep if age >= 25 & age <= 70
keep if lkwk==1

sort t
drop year
rename _year year

***************************************************
**** counting search methods
***************************************************
local method fndwk1 fndwk2 fndwk3 fndwk4 fndwk5 fndwk6
foreach x of local method {
	replace `x'= 0 if `x'!=1
}
gen numsearch6 = fndwk1 + fndwk2 + fndwk3 + fndwk4 + fndwk5 + fndwk6
gen numsearch5 = fndwk1 + fndwk2 + fndwk3 + fndwk4 + fndwk5

keep numsearch6 numsearch5 fndwk1 fndwk2 fndwk3 fndwk4 fndwk5 fndwk6 wgt year t month

save "$int_data/CPS/historical", replace

***************************************************
**** appending to the post-94 data for number of methods plots
***************************************************
use "$final_CPS/full_CPS_data.dta", clear
keep if mlr==4

gen fndwk1 = pubemkag
replace fndwk1 = . if mlr!=4

gen fndwk2 = 1 if PriEmpAg==1 
replace fndwk2 = 0 if mlr==4 & fndwk2!=1
replace fndwk2 = . if mlr!=4

gen fndwk3 = empldir
replace fndwk3 = . if mlr!=4

gen fndwk4 = FrendRel 
replace fndwk4 = . if mlr!=4

gen fndwk5 = 1 if Plcdads==1 | lkatads==1
replace fndwk5 = 0 if mlr==4 & fndwk5!=1
replace fndwk5 = . if mlr!=4

gen fndwk6 = 1 if Resumes==1 | Otheractve==1 | Jbtrnprg==1 | otherpas==1 | Unionpro==1 | SchEmpCt==1
replace fndwk6 = 0 if mlr==4 & fndwk6!=1
replace fndwk6 = . if mlr!=4

gen numsearch6 = fndwk1 + fndwk2 + fndwk3 + fndwk4 + fndwk5 + fndwk6

gen numsearch5 = fndwk1 + fndwk2 + fndwk3 + fndwk4 + fndwk5

rename newwgt wgt

append using "$int_data/CPS/historical"

collapse (mean) fndwk1 fndwk2 fndwk3 fndwk4 fndwk5 fndwk6  numsearch6 numsearch5 numsearch [pw=wgt] , by(year month)
gen date = ym(year,month)
label var fndwk1 "Public Employment Agancy"  
label var fndwk2 "Private Employment Agancy"  
label var fndwk3 "Employer"
label var fndwk4 "Friends and Relatives"  
label var fndwk5 "Ads"  
label var fndwk6 "Other" 
label var numsearch "All 12 methods"
label var numsearch6 "All 6 methods"
label var numsearch5 " 5 methods - excluding other category"

keep year month numsearch5 numsearch6
outsheet using "$int_data/CPS/FigureA10_data.csv", comma replace

//deleting the big file I just created
erase "$int_data/CPS/historical.dta"




