***************************************************
**Description: This file combines and cleans the ATUS data for 2003-2011 combined files for the search intensity over the business cycle project (Sahin, Mukoyama and Patterson)
*****************************************************

clear all 
capture log close
set more off
set mem 1g

***************************************************
*** merging the relevant ATUS data sets
***************************************************

cd "$raw_ATUS"

use atusresp_0314.dta, clear

// merging in information on time spend on activities in a day
merge 1:1 tucaseid using atussum_0314.dta, nogen

// merging in labor force information
merge 1:1  tucaseid tulineno using atusrost_0314.dta
/* the entries that are not matched is the information on the other household members of the respondents */
drop if _m==2
drop _m

/*merging in CPS merge information */
merge 1:1 tucaseid tulineno using atuscps_0314.dta
drop if _m==2 /*this drops individuals who are not in the respondent data set */
drop _m

/*merging in the interview travel time variable */
merge 1:1 tucaseid using intvwtravel_0314.dta
drop _m

***************************************************
**** cleaning the number of search methods variables
***************************************************
foreach x in 1 2 3 4  {
   rename tulkdk`x' lkdk`x'
   rename tulkps`x' lkps`x'
}
foreach x in 2 3 4 5 6 {
   rename tulkm`x' lkm`x'
}
rename telkm1 lkm1

gen empldir = 1 if   lkm1==1 | lkm2==1 | lkm3==1 | lkm4==1 | lkm5==1 | lkm6==1 | lkps1==1 | lkps2==1 |  lkps3==1 |  lkps4==1 |   lkdk1==1 | lkdk2==1 |  lkdk3==1 |  lkdk4==1 
gen pubemkag  = 1 if lkm1==2 | lkm2==2 | lkm3==2 | lkm4==2 | lkm5==2 | lkm6==2 | lkps1==2 | lkps2==2 |  lkps3==2 |  lkps4==2 | lkdk1==2 | lkdk2==2 |  lkdk3==2 |  lkdk4==2
gen PriEmpAg = 1 if  lkm1==3 | lkm2==3 | lkm3==3 | lkm4==3 | lkm5==3 | lkm6==3 | lkps1==3 | lkps2==3 |  lkps3==3 |  lkps4==3 | lkdk1==3 | lkdk2==3 |  lkdk3==3 |  lkdk4==3
gen FrendRel = 1 if  lkm1==4 | lkm2==4 | lkm3==4 | lkm4==4 | lkm5==4 | lkm6==4 | lkps1==4 | lkps2==4 |  lkps3==4 |  lkps4==4 | lkdk1==4 | lkdk2==4 |  lkdk3==4 |  lkdk4==4
gen SchEmpCt = 1 if  lkm1==5 | lkm2==5 | lkm3==5 | lkm4==5 | lkm5==5 | lkm6==5 | lkps1==5 | lkps2==5 |  lkps3==5 |  lkps4==5 | lkdk1==5 | lkdk2==5 |  lkdk3==5 |  lkdk4==5
gen Resumes = 1 if  lkm1==6 | lkm2==6 | lkm3==6 | lkm4==6 | lkm5==6 | lkm6==6 | lkps1==6 | lkps2==6 |  lkps3==6 |  lkps4==6 |  lkdk1==6 | lkdk2==6 |  lkdk3==6 |  lkdk4==6 
gen Unionpro = 1 if  lkm1==7 | lkm2==7 | lkm3==7 | lkm4==7 | lkm5==7 | lkm6==7 | lkps1==7 | lkps2==7 |  lkps3==7 |  lkps4==7 | lkdk1==7 | lkdk2==7 |  lkdk3==7 |  lkdk4==7
gen Plcdads = 1 if  lkm1==8 | lkm2==8 | lkm3==8 | lkm4==8 | lkm5==8 | lkm6==8 | lkps1==8 | lkps2==8 |  lkps3==8 |  lkps4==8 |  lkdk1==8 | lkdk2==8 |  lkdk3==8 |  lkdk4==8 
gen Otheractve = 1 if  lkm1==9 | lkm2==9 | lkm3==9 | lkm4==9 | lkm5==9 | lkm6==9 | lkps1==9 | lkps2==9 |  lkps3==9 |  lkps4==9 | lkdk1==9 | lkdk2==9 |  lkdk3==9 |  lkdk4==9
gen lkatads = 1 if  lkm1==10 | lkm2==10 | lkm3==10 | lkm4==10 | lkm5==10 | lkm6==10 | lkps1==10 | lkps2==10 |  lkps3==10 |  lkps4==10 | lkdk1==10 | lkdk2==10 |  lkdk3==10 |  lkdk4==10 
gen Jbtrnprg = 1 if  lkm1==11 | lkm2==11 | lkm3==11 | lkm4==11 | lkm5==11 | lkm6==11 | lkps1==11 | lkps2==11 |  lkps3==11 |  lkps4==11 |  lkdk1==11 | lkdk2==11 |  lkdk3==11 |  lkdk4==11 
gen otherpas = 1 if  lkm1==13 | lkm2==13 | lkm3==13 | lkm4==13 | lkm5==13 | lkm6==13 | lkps1==13 | lkps2==13 |  lkps3==13 |  lkps4==13 |  lkdk1==13 | lkdk2==13 |  lkdk3==13 |  lkdk4==13 
local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Resumes Unionpro Plcdads Otheractve lkatads  Jbtrnprg otherpas
rename telfs mlr
foreach x of local srch_mth {
      replace `x' = 0 if `x' ==.
      replace `x' = 0 if mlr!=4
}

egen numsearch = rowtotal(empldir- otherpas)

*********************************************************
****collecting the search time variables
*********************************************************
/* these are the variables under job search activities */

gen timesearch_travel = t050481 + t050405 + t050404 + t050403 + t050499 + intvwtravel if tuyear != 2003
gen timesearch = t050481 + t050405 + t050404 + t050403 + t050499

gen timesearch_old = t050481 + t050405 + t050404 + t050403 + t050499

label var t050481 "Job Search Activities"
label var t050405 "Security related to job interviews"
label var t050404 "Waiting for interview" 
label var t050403 "Interviewing" 
label var t050499 "Job Search Unclassified" 

******************************************
****generating the date variable
******************************************
rename tuyear year
rename tumonth month
gen date = ym(year, month)
tostring year month, replace
replace month = "0"+month if length(month)==1
gen date2=""
replace date2 = year + month
replace date2 = "" if date2==".0."
destring year month date2, replace

********************************************
*****defining searchers
********************************************
/* define a searcher to be someone who has labor force status = 4 (unemployed and looking for work */

rename tufnwgtp wgt
rename teage age
rename tesex sex
rename ptdtrace race
rename pruntype untype

gen quarter = 1 if month==1 | month==2 | month==3
replace quarter = 2 if month==4 | month==5 | month==6
replace quarter = 3 if month==7 | month==8 | month==9
replace quarter = 4 if month==10 | month==11 | month==12
gen attached = 1 if  mlr==3 | mlr==4 | (mlr==5 & prwntjob==1)
gen nonemployed = 1 if  mlr==3 | mlr==4 | mlr==5 
gen black = 0
replace black = 1 if (race==2 | race==6 | race==10 | race==11 | race==12 | race==15 | race==19)

// searchers are people actively looking for a job (all unemployed who are looking)
gen searchers = 1 if mlr==4
gen nonsearchers = 1 if mlr==5 & prwntjob==1
replace searchers = 0 if searchers!=1
replace nonsearchers = 0 if nonsearchers!=1


gen married = 1 if prmarsta==1 | prmarsta==2 | prmarsta==3
replace married = 0 if prmarsta==4 | prmarsta==5 | prmarsta==6 | prmarsta==7
gen educ = 1 if peeduca<=38
replace educ =2 if peeduca==39
replace educ = 3 if peeduca>=40 & peeduca <43
replace educ = 4 if peeduca>=43
rename peio1icd ind
rename prunedur undur
rename gestfips state
rename peio1ocd occ
rename state fips
*merge m:1 fips using "N:\Mismatch\KEYS\state_fips"
merge m:1 fips using "$maps/state_FIPS.dta"
drop if _m==2
drop _m

/*cleaning occupation variabe */
gen occ_2011 = occ if year==2011 | year==2012 | year==2013 | year==2014 //this is the 2010 occupation codes
replace occ = . if year==2011 | year==2012 | year==2013 | year==2014

*merge m:1 occ using the occ-to-soc conversion 
merge m:1 occ using "$maps/OCC-SOC_conversion.dta"
replace soc = 11 if occ==400
replace soc = 55 if  occ==9840 /*this renames the military */
rename soc soc_2010
drop occupationname soc6 soc3 _merge occ
rename occ_2011 occ
merge m:1 occ using "$maps/OCC-SOC_conversion_2011.dta"
replace soc = 11 if year==2011 & occ<=430 & occ>=10
replace soc = 13 if year==2011 & occ==560 | occ==620 | occ==730
replace soc = 55 if year==2011 & occ==9840
replace soc = 19 if year==2011 & occ==1960
replace soc = 21 if year==2011 & occ==2020
replace soc = 23 if year==2011 & occ==2140 | occ==2150
replace soc = 27 if year==2011 & occ==2820
replace soc = 29 if year==2011 & occ>=3000 & occ<=3540
replace soc = 31 if year==2011 & occ==3650
replace soc = 33 if year==2011 & occ==3920 | occ==3950
replace soc = 39 if year==2011 & occ==4550
replace soc = 41 if year==2011 & occ==4960
replace soc = 43 if year==2011 & occ==5930
replace soc = 45 if year==2011 & occ==6000
replace soc = 47 if year==2011 & occ>=6200 & occ<=6940
replace soc = 49 if year==2011 & occ==7310 | occ==7620
replace soc = 51 if year==2011 & occ>=7700 & occ<=8965 
replace soc = soc_2010 if year<=2010
replace soc = . if soc==99

gen weekend = 1 if tudiaryday==1 | tudiaryday==7
replace weekend = 0 if  tudiaryday>=2 & tudiaryday<=6
rename tudiaryday day
rename t500106 missing 

/* keep only the variables we need for the analysis */
keep tucaseid weekend year month ind soc educ married searchers nonsearchers black attached quarter race sex age wgt mlr numsearch timesearch timesearch_travel undur gereg state /// 
	empldir pubemkag PriEmpAg FrendRel SchEmpCt Resumes Unionpro Plcdads Otheractve lkatads Jbtrnprg otherpas untype pelklwo /// 
	t050481 t050405 t050404  t050403  t050499 day missing

/*cleaning labor force varaibles */
gen layoff = 1 if mlr==3
replace layoff = 0 if mlr!=3
gen unemp = 1 if mlr==3 | mlr==4
replace unemp = 0 if unemp!=1
gen np_other = 1 if mlr==5 & nonsearcher!=1
replace np_other = 0 if np_other!=1
gen nonemp = 1 if mlr==3 | mlr==4 | mlr==5

/*cleaning demographics*/
gen female = 1 if sex == 2
replace female = 0 if sex==1
gen marriedfemale = married*female
gen age2 = age*age
gen age3 = age*age2
gen age4 = age*age3

gen hs = 1 if educ==2
replace hs = 0 if hs!=1
gen somecol = 1 if educ==3
replace somecol = 0 if somecol!=1
gen college = 1 if educ==4
replace college = 0 if college!=1

/*dropping outliers */
gen time_less8 = timesearch
gen time_less8_travel = timesearch_travel
replace time_less8 = . if time_less8>=480
replace time_less8_travel = . if time_less8_travel>=480

drop if age<25 | age>70
drop if age==. 
save merged_ATUS_2014.dta, replace


