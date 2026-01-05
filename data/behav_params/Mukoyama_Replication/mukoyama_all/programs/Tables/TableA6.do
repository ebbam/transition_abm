clear all
mat drop _all
pause on
set more off
set maxiter 150
set matsize 800
set linesize 255

// ESTIMATION OF BENCHMARK ARMA SPECIFICATIONS FOR JOLTS NON-SEASONALLY ADJUSTED DATA
// (SECTIONS 5, 6.1 AND 6.3)

*** Constructiong the data for this analysis 

use "$final_CPS/full_CPS_data", clear
keep if unemp==1
preserve
collapse (sum) newwgt, by(year month) 
rename newwgt unemployed
save "$int_data/CPS/unemp_aggregate", replace
restore 
collapse (mean) time_create [pweight = newwgt], by(year month)
* Merging on theta data 
merge 1:1 year month using "$int_data/theta/JOLTS_agg_theta_monthly"
keep if _m==3 
drop _m
gen s = log(time_create)
gen theta = log(theta)
drop theta_agg time_create
forvalues x = 2/12 {
	gen m_`x' = (month==`x')
}
* Merging on job finding rate data
merge 1:1 year month using  "$other/hires_aggregate"
assert _m==3 
drop _m
sort year month
merge 1:1 year month using "$int_data/CPS/unemp_aggregate"
sort year month
gen jfr = ln(hires/unemployed[_n-1])
keep if _m==3 
drop _m

/*aggregate S&P index */
capture drop _m
quietly merge m:1 year month using "$other/sp_500"
drop if _m==2
drop _m

/* aggregate House Price index */
quietly merge m:1 year month using "$other/houseprice_index"
rename home_price_total hp
drop if _m==2
drop _m
rename sp_500 SP
rename hp HP
drop unemployed
gen date = mdy(month,1,year)
drop if year>2012 
gen t = _n

qui{

	 *drop if year==2012
	// Breaks for JOLTS
	gen bk1 = date>mdy(9,1,2001)
	gen bk2 = date>mdy(10,1,2008)
	gen dum1 = date<mdy(7,1,2009)

	// Prepare the sample for estimation
	keep year month date t jfr theta s m_* bk* SP HP dum1
	drop if jfr==.
}

cap program drop estim_grid
program define estim_grid
syntax [, P(real 1) Q(real 1) ADDLAGSTH(integer 1) LAGSJFR(integer 1) BK(integer 1) PMAX(integer 1) SELECT(string) ETA01(real 1) ETA02(real 1) GRAPH] 
	
	preserve
	
	if "`select'"~=""	keep if `select'
	
	local first = `q' + 1
	//p+2=5 parameters, (p+1)*2=8 instruments
	local last_th = `q' + `p' + 3 + `addlagsth'
	local laglist_th "`first'/`last_th'"
	
	if `lagsjfr'>0	{
		local last_jfr = `q' + `lagsjfr'
		local laglist_jfr "`first'/`last_jfr'"
					}
		
	// list of instruments 
	if `lagsjfr'>0	local inst "l(`laglist_th').theta l(`laglist_th').s l(`laglist_jfr').jfr m_* dum1"
	else local inst "l(`laglist_th').theta l(`laglist_th').s m_* dum1"
	
	if `bk'==1 local inst "`inst' bk*"
	
	local addobs = 100
		
	// Instruments = 0 if missing
	local new_n = _N + `addobs'
	set obs `new_n'
	recode * (.=0)
	sort t
	replace t = _n  
	sort t
	tsset t
	gen insamp = (t>=`addobs' + max(`q'+2,`p'+1))
	
	// Proper IV imposing common factor restriction , full sample
	local urtest "(-1)"

	local esteq "jfr - {eta1}*theta - {eta2}*s - {mu} - {gam}*dum1"
	forval m = 2/12	{
		local esteq "`esteq' - {tau`m'}*m_`m'"
					}		
	forval l = 1/`p'	{
		
		local urtest "[rho`l']_cons + `urtest'"
		
		local esteq "`esteq' - {rho`l'}*(l`l'.jfr - {eta1}*l`l'.theta - {eta2}*l`l'.s - {gam}*l`l'.dum1"
		forval m = 2/12	{
			local esteq "`esteq' - {tau`m'}*l`l'.m_`m'"
						}	
		local esteq "`esteq')"
						}
						
	if `bk'==1	{
		local esteq "`esteq' - {b1}*bk1 - {b2}*bk2"
		forval l = 1/`p'	{
			local esteq "`esteq' + {rho`l'}*( {b1}*l`l'.bk1 + {b2}*l`l'.bk2 )"
							}
				}
						
	local esteq "(`esteq')"
	local urtest "`urtest' == 0"
		
	mat m = J(5 + 2*(`pmax'+2) ,1,.)
	mat m[1,1] = `p'
	mat m[2,1] = `q'
	
	cap	{
		noi gmm `esteq' if insamp, instruments(`inst') twostep vce(unadjusted) wmatrix(unadjusted) from(mu 0 eta1 `eta01' eta2 `eta02')
		
		
		mat V = e(V)
		
		// Retrieve the actual constant and its SE
		matrix V = V["mu:_cons","mu:_cons".."rho`p':_cons"] \ V["rho1:_cons".."rho`p':_cons","mu:_cons".."rho`p':_cons"]
		matrix V = V[1...,"mu:_cons"] , V[1...,"rho1:_cons".."rho`p':_cons"]
		local denom = 1
		forval arp = 1/`p'	{
			local denom = `denom'-[rho`arp']_b[_cons]
							}
							
		local mu = [mu]_b[_cons]/`denom'
		
		mat G = 1/`denom' \ J(`p',1,`mu'/`denom')
		mat SE = G'*V*G
				
		matrix m[3,1] = sqrt(SE[1,1]) \ `mu' 
		* matrix m[3,1] = [mu]_se[_cons] \ [mu]_b[_cons]  
		//only eta1 recorded
		matrix m[5,1] = [eta1]_se[_cons] \ [eta1]_b[_cons] 
		forv arp = 1/`p'	{
			/*
			local t = [rho`arp']_b[_cons] / [rho`arp']_se[_cons]
			matrix m[6 + 2*`arp'-1 ,1] = `t' \ [rho`arp']_b[_cons]
			*/
			
			matrix m[6 + 2*`arp'-1 ,1] = [rho`arp']_se[_cons] \ [rho`arp']_b[_cons] 
			
							}
			
		test "`urtest'"
		matrix m[6 + 2*`pmax' + 1,1] = r(p)
		
		noi estat overid
		matrix m[6 + 2*`pmax' + 2,1] = r(J) \ r(J_p)
				
				
		// graph part, ignored.		
		if "`graph'"~=""	{
			predict omega if insamp
			noi ac omega if insamp, lag(18) level(90) text(-.15 14 "(p,q) = (`p',`q')", box place(e) margin(medsmall)) /*
				*/ note("") xlab(0(2)18) scheme(s1mono) 
							}
		}
	restore
end

// Table 1, cols 4 and 5
estim_grid, p(3) q(3) pmax(3) addlagsth(0) lagsjfr(0) bk(0) eta01(0.7) eta02(0.5) 
 matrix list m
estim_grid, p(3) q(3) pmax(3) addlagsth(0) lagsjfr(1) bk(0) eta01(0.7) eta02(0.5) 
 matrix list m


