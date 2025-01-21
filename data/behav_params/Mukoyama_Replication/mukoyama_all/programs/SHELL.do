*****************************************************************
** Description:  This is a script that will run the data cleaning and aggregate data descriptions. When you run this program, 
*			     you will be able to reproduce all of the analysis ncessary for the charts and tables 
*                in the "Job Search over the Business Cycle" by Mukoyama, Patterson and Sahin. Some final figure are created in matlab, and those can be created
*                by the accompanying SHELL_matlab. 

* 				There are generally 3 steps: 
*					1) Using the ATUS to create a mapping between the number of search minutes and the amount of search time 
*					2) Mapping this into the CPS to create a measure of job search effort over time, and exploring the cyclicality of this measure. 
*					3) Using this measure, decompose the aggregate movements into that which is due to composition and that which is due to within-individual movements


**Created by: Christina Patterson 
*****************************************************************

* Setting locals to run parts of the analysis 
local raw_data = 1
local data_build = 1
local figures = 1
local tables = 1
local AppendixFigures = 1
local AppendixTables = 1

// Change all file paths in the accompanying do file to reflect your local directory
do file_paths.do 

**************************************************************************************************************************************
*******************************************************************Preliminary: Raw Data Cleaning
*************************************************************************************************************************************
if `raw_data'== 1{
	cd "$programs/Cleaning"
	do clean_ATUSsingle.do 
	
	cd "$programs/Cleaning"
	do clean_ATUS.do 

	cd "$programs/Cleaning" 
	do clean_CPS_Raw.do //This file cleans the raw CPS files so they are consistent over years/data sources

	cd "$programs/Cleaning" 
	do calculate_theta.do //This uses the CPS and vacancy data to calculate theta
}

if `data_build' ==1 {
	cd "$programs/Build"
	do ATUS_imputation.do // This calcualtes the relationship between search time and minutes in the ATUS

	cd "$programs/Build"
	do Merge_CPS.do // this merges together the ATUS estimates and the CPS data to created imputed search minutes for each observation in the CPS sample
	
	cd "$programs/Build"
	do regressions.do // This runs the individual regressions 
}

if `figures' ==1 { 

	cd "$programs/Figures"
	do Figure1.do
	*** Figure 2 is produced in matlab 
	cd "$programs/Figures"
	do Figure3.do 
	
}

if `tables'==1 { 

	cd "$programs/Tables"
	do Table3.do
	cd "$programs/Tables"
	do Table4.do
	cd "$programs/Tables"
	do Table5_6.do	
	
}

if `AppendixFigures' ==1 { 

	cd "$programs/Figures"
	do FigureA1.do		
	cd "$programs/Figures"
	do FigureA2.do			
	cd "$programs/Figures"
	do FigureA3.do	
	cd "$programs/Figures"
	do FigureA4.do	
	cd "$programs/Figures"
	do FigureA5.do	 
	cd "$programs/Figures"
	do FigureA6.do		
	cd "$programs/Figures"
	do FigureA7.do		
	* Figure A8 and A9 are created in matlab
	cd "$programs/Figures"
	do FigureA10.do			
	cd "$programs/Figures"
	do FigureA11.do		
	* Figure A12 and A13 are created in matlab
	cd "$programs/Figures"
	do FigureA14.do		
	
}

if `AppendixTables' ==1 { 

	cd "$programs/Tables"
	do TableA2.do	
	cd "$programs/Tables"
	do TableA3.do			
	cd "$programs/Tables"
	do TableA4_5.do			
}

