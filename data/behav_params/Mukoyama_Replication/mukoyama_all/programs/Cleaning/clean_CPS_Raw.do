/********************************************************************************************/
*Description - Clean CPS raw data from CPS utilities and Dataferret       
/********************************************************************************************/


clear
capture log close
set more off
set mem 1g

/* Declare local variables */
local x = 199401
while `x' <= 201212 {
   
	cd "$raw_CPS"

   // this cleans up the file names
   use "bas`x'", clear

   capture gen YYYYMM = `x'
   tostring YYYYMM, replace
   gen year = substr(YYYYMM,1,4)
   gen month = substr(YYYYMM,5,2)
   drop YYYYMM
   destring year month, replace
   
   /* renaming dataferret variables to match CPS variable names */
   capture rename HRHHID hhid
   capture rename HRHHID2 hhid2
   capture drop OCCURNUM
   capture drop HRMONTH
   capture rename PRTAGE age
   capture rename PESEX sex
   capture rename PTDTRACE race
   capture rename HRMIS mis
   capture rename PEMLR mlr
   capture rename PERACE race
   capture rename PWSSWGT wgt
   capture rename PREDUCA4 grdatn
   capture replace wgt= wgt*10000
   capture rename PRUNEDUR undur
   capture rename  PEMARITL married
   capture rename PEDWWNTO nlfwant
   capture rename dwwant nlfwant
   capture rename  GEREG gereg
   capture rename  region gereg
   capture rename  GESTCEN state
   capture rename PTERNHLY ernhr
   capture rename  PTERNWA ernwk
   capture rename  PRUNTYPE untype
   capture rename HRSERSUF serial
   capture rename PEEDUCA grdatn
   capture rename PULINENO lineno
   forval t = 1(1)6 {
	capture rename PULKDK`t' lkdk`t'
	capture rename PULKPS`t' lkps`t'
   }
	capture rename  PELKM1 lkm1
   forval t = 2(1)6 {
	capture rename  PULKM`t' lkm`t'
   }
  
   /// counting up search methods - counting if people EVER did option 1
    gen empldir = 1 if   lkm1==1 | lkm2==1 | lkm3==1 | lkm4==1 | lkm5==1 | lkm6==1 | lkps1==1 | lkps2==1 |  lkps3==1 |  lkps4==1 |  lkps5==1 | lkps6==1 | lkdk1==1 | lkdk2==1 |  lkdk3==1 |  lkdk4==1 |  lkdk5==1 | lkdk6==1

   gen pubemkag  = 1 if lkm1==2 | lkm2==2 | lkm3==2 | lkm4==2 | lkm5==2 | lkm6==2 | lkps1==2 | lkps2==2 |  lkps3==2 |  lkps4==2 |  lkps5==2 | lkps6==2 | lkdk1==2 | lkdk2==2 |  lkdk3==2 |  lkdk4==2 |  lkdk5==2 | lkdk6==2

   gen PriEmpAg = 1 if  lkm1==3 | lkm2==3 | lkm3==3 | lkm4==3 | lkm5==3 | lkm6==3 | lkps1==3 | lkps2==3 |  lkps3==3 |  lkps4==3 |  lkps5==3 | lkps6==3 | lkdk1==3 | lkdk2==3 |  lkdk3==3 |  lkdk4==3 |  lkdk5==3 | lkdk6==3

   gen FrendRel = 1 if  lkm1==4 | lkm2==4 | lkm3==4 | lkm4==4 | lkm5==4 | lkm6==4 | lkps1==4 | lkps2==4 |  lkps3==4 |  lkps4==4 |  lkps5==4 | lkps6==4 | lkdk1==4 | lkdk2==4 |  lkdk3==4 |  lkdk4==4 |  lkdk5==4 | lkdk6==4

   gen SchEmpCt = 1 if  lkm1==5 | lkm2==5 | lkm3==5 | lkm4==5 | lkm5==5 | lkm6==5 | lkps1==5 | lkps2==5 |  lkps3==5 |  lkps4==5 |  lkps5==5 | lkps6==5 | lkdk1==5 | lkdk2==5 |  lkdk3==5 |  lkdk4==5 |  lkdk5==5 | lkdk6==5

   gen Resumes = 1 if  lkm1==6 | lkm2==6 | lkm3==6 | lkm4==6 | lkm5==6 | lkm6==6 | lkps1==6 | lkps2==6 |  lkps3==6 |  lkps4==6 |  lkps5==6 | lkps6==6 | lkdk1==6 | lkdk2==6 |  lkdk3==6 |  lkdk4==6 |  lkdk5==6 | lkdk6==6

   gen Unionpro = 1 if  lkm1==7 | lkm2==7 | lkm3==7 | lkm4==7 | lkm5==7 | lkm6==7 | lkps1==7 | lkps2==7 |  lkps3==7 |  lkps4==7 |  lkps5==7 | lkps6==7 | lkdk1==7 | lkdk2==7 |  lkdk3==7 |  lkdk4==7 |  lkdk5==7 | lkdk6==7

   gen Plcdads = 1 if  lkm1==8 | lkm2==8 | lkm3==8 | lkm4==8 | lkm5==8 | lkm6==8 | lkps1==8 | lkps2==8 |  lkps3==8 |  lkps4==8 |  lkps5==8 | lkps6==8 | lkdk1==8 | lkdk2==8 |  lkdk3==8 |  lkdk4==8 |  lkdk5==8 | lkdk6==8

   gen Otheractve = 1 if  lkm1==9 | lkm2==9 | lkm3==9 | lkm4==9 | lkm5==9 | lkm6==9 | lkps1==9 | lkps2==9 |  lkps3==9 |  lkps4==9 |  lkps5==9 | lkps6==9 | lkdk1==9 | lkdk2==9 |  lkdk3==9 |  lkdk4==9 |  lkdk5==9 | lkdk6==9

   gen lkatads = 1 if  lkm1==10 | lkm2==10 | lkm3==10 | lkm4==10 | lkm5==10 | lkm6==10 | lkps1==10 | lkps2==10 |  lkps3==10 |  lkps4==10 |  lkps5==10 | lkps6==10 | lkdk1==10 | lkdk2==10 |  lkdk3==10 |  lkdk4==10 |  lkdk5==10 | lkdk6==10

   gen Jbtrnprg = 1 if  lkm1==11 | lkm2==11 | lkm3==11 | lkm4==11 | lkm5==11 | lkm6==11 | lkps1==11 | lkps2==11 |  lkps3==11 |  lkps4==11 |  lkps5==11 | lkps6==11 | lkdk1==11 | lkdk2==11 |  lkdk3==11 |  lkdk4==11 |  lkdk5==11 | lkdk6==11

   gen otherpas = 1 if  lkm1==13 | lkm2==13 | lkm3==13 | lkm4==13 | lkm5==13 | lkm6==13 | lkps1==13 | lkps2==13 |  lkps3==13 |  lkps4==13 |  lkps5==13 | lkps6==13 | lkdk1==13 | lkdk2==13 |  lkdk3==13 |  lkdk4==13 |  lkdk5==13 | lkdk6==13

 local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Resumes Unionpro Plcdads Otheractve lkatads  Jbtrnprg otherpas
   foreach y of local srch_mth {
      replace `y' = 0 if `y' ==.
      replace `y' = 0 if mlr!=4
   }   

   egen numsearch = rowtotal(empldir- otherpas)
   egen numactive = rowtotal(empldir-Otheractve)
   drop lkm* lkdk* lkps*
   

   gen str1 lfs = "E" if mlr == 1 | mlr == 2
   replace lfs = "U" if mlr == 3 | mlr == 4
   replace lfs = "N" if mlr == 5 | mlr == 6 | mlr == 7 
   replace lfs = "M" if mlr == . 
   replace lfs = "D" if lfs == "N" & nlfwant == 1
   
   compress
   
   
   save "$int_CPS/intermediate_`x'", replace

   local second = `x' + 1
   if (`x'-12)/100 == int((`x'-12)/100) {
      local second = `x' + 89
   }
   local x = `second'
   clear
}


****************************************
** Data from 2013 and 2014 were downloaded later and therefore have a slightly different format
****************************************

/* Declare local variables */
local x = 201301
while `x' <= 201412 {

	cd "$raw_CPS"

	use "cps_pull", clear

	// keeping only the relevant year and month
	gen temp = `x'
	tostring temp, gen(temp2)
	gen year_keep = substr(temp2,1,4)
	gen month_keep = substr(temp2,5,2)
	destring month_keep year_keep, replace
	keep if month==month_keep & year==year_keep 
	drop temp temp2 month_keep year_keep
	
  /// counting up search methods - counting if people EVER did option 1
   gen empldir = 1 if   lkm1==1 | lkm2==1 | lkm3==1 | lkm4==1 | lkm5==1 | lkm6==1 | lkps1==1 | lkps2==1 |  lkps3==1 |  lkps4==1 |  lkps5==1 | lkps6==1 | lkdk1==1 | lkdk2==1 |  lkdk3==1 |  lkdk4==1 |  lkdk5==1 | lkdk6==1

   gen pubemkag  = 1 if lkm1==2 | lkm2==2 | lkm3==2 | lkm4==2 | lkm5==2 | lkm6==2 | lkps1==2 | lkps2==2 |  lkps3==2 |  lkps4==2 |  lkps5==2 | lkps6==2 | lkdk1==2 | lkdk2==2 |  lkdk3==2 |  lkdk4==2 |  lkdk5==2 | lkdk6==2

   gen PriEmpAg = 1 if  lkm1==3 | lkm2==3 | lkm3==3 | lkm4==3 | lkm5==3 | lkm6==3 | lkps1==3 | lkps2==3 |  lkps3==3 |  lkps4==3 |  lkps5==3 | lkps6==3 | lkdk1==3 | lkdk2==3 |  lkdk3==3 |  lkdk4==3 |  lkdk5==3 | lkdk6==3

   gen FrendRel = 1 if  lkm1==4 | lkm2==4 | lkm3==4 | lkm4==4 | lkm5==4 | lkm6==4 | lkps1==4 | lkps2==4 |  lkps3==4 |  lkps4==4 |  lkps5==4 | lkps6==4 | lkdk1==4 | lkdk2==4 |  lkdk3==4 |  lkdk4==4 |  lkdk5==4 | lkdk6==4

   gen SchEmpCt = 1 if  lkm1==5 | lkm2==5 | lkm3==5 | lkm4==5 | lkm5==5 | lkm6==5 | lkps1==5 | lkps2==5 |  lkps3==5 |  lkps4==5 |  lkps5==5 | lkps6==5 | lkdk1==5 | lkdk2==5 |  lkdk3==5 |  lkdk4==5 |  lkdk5==5 | lkdk6==5

   gen Resumes = 1 if  lkm1==6 | lkm2==6 | lkm3==6 | lkm4==6 | lkm5==6 | lkm6==6 | lkps1==6 | lkps2==6 |  lkps3==6 |  lkps4==6 |  lkps5==6 | lkps6==6 | lkdk1==6 | lkdk2==6 |  lkdk3==6 |  lkdk4==6 |  lkdk5==6 | lkdk6==6

   gen Unionpro = 1 if  lkm1==7 | lkm2==7 | lkm3==7 | lkm4==7 | lkm5==7 | lkm6==7 | lkps1==7 | lkps2==7 |  lkps3==7 |  lkps4==7 |  lkps5==7 | lkps6==7 | lkdk1==7 | lkdk2==7 |  lkdk3==7 |  lkdk4==7 |  lkdk5==7 | lkdk6==7

   gen Plcdads = 1 if  lkm1==8 | lkm2==8 | lkm3==8 | lkm4==8 | lkm5==8 | lkm6==8 | lkps1==8 | lkps2==8 |  lkps3==8 |  lkps4==8 |  lkps5==8 | lkps6==8 | lkdk1==8 | lkdk2==8 |  lkdk3==8 |  lkdk4==8 |  lkdk5==8 | lkdk6==8

   gen Otheractve = 1 if  lkm1==9 | lkm2==9 | lkm3==9 | lkm4==9 | lkm5==9 | lkm6==9 | lkps1==9 | lkps2==9 |  lkps3==9 |  lkps4==9 |  lkps5==9 | lkps6==9 | lkdk1==9 | lkdk2==9 |  lkdk3==9 |  lkdk4==9 |  lkdk5==9 | lkdk6==9

   gen lkatads = 1 if  lkm1==10 | lkm2==10 | lkm3==10 | lkm4==10 | lkm5==10 | lkm6==10 | lkps1==10 | lkps2==10 |  lkps3==10 |  lkps4==10 |  lkps5==10 | lkps6==10 | lkdk1==10 | lkdk2==10 |  lkdk3==10 |  lkdk4==10 |  lkdk5==10 | lkdk6==10

   gen Jbtrnprg = 1 if  lkm1==11 | lkm2==11 | lkm3==11 | lkm4==11 | lkm5==11 | lkm6==11 | lkps1==11 | lkps2==11 |  lkps3==11 |  lkps4==11 |  lkps5==11 | lkps6==11 | lkdk1==11 | lkdk2==11 |  lkdk3==11 |  lkdk4==11 |  lkdk5==11 | lkdk6==11

   gen otherpas = 1 if  lkm1==13 | lkm2==13 | lkm3==13 | lkm4==13 | lkm5==13 | lkm6==13 | lkps1==13 | lkps2==13 |  lkps3==13 |  lkps4==13 |  lkps5==13 | lkps6==13 | lkdk1==13 | lkdk2==13 |  lkdk3==13 |  lkdk4==13 |  lkdk5==13 | lkdk6==13

	local  srch_mth empldir pubemkag PriEmpAg FrendRel SchEmpCt Resumes Unionpro Plcdads Otheractve lkatads  Jbtrnprg otherpas
	foreach y of local srch_mth {
      replace `y' = 0 if `y' ==.
      replace `y' = 0 if mlr!=4
   }   

   egen numsearch = rowtotal(empldir- otherpas)
   egen numactive = rowtotal(empldir-Otheractve)
   drop lkm* lkdk* lkps*
   
   // renaming variables to match the previous definitions 
   rename region gereg

   gen str1 lfs = "E" if mlr == 1 | mlr == 2
   replace lfs = "U" if mlr == 3 | mlr == 4
   replace lfs = "N" if mlr == 5 | mlr == 6 | mlr == 7 
   replace lfs = "M" if mlr == . 
   replace lfs = "D" if lfs == "N" & dwwant == 1
   
   compress	
	
   save "$int_CPS/intermediate_`x'", replace

   local second = `x' + 1
   if (`x'-12)/100 == int((`x'-12)/100) {
      local second = `x' + 89
   }
   local x = `second'
   clear	

}
	
	
	
	
	
	
	
	
	
