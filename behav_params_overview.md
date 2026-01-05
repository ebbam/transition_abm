---
title: "Data Scoping: Job Search Behaviour"
author: "Ebba Mark"
date: "2025-06-12"
output: 
  pdf_document:
    keep_tex: true
    latex_engine: xelatex
editor_options: 
  chunk_output_type: console
  markdown: 
    wrap: 72
---

# Overview {.tabset}

The following document summarises current progress on identifying data
sources to inform the job search behaviour in our labour market ABM.

**Goals:**

1.  Identify parameters relevant to agent search behaviour in the ABM.

2.  Assess data quality for deriving empirical estimates of these
    parameters.

We have narrowed the list of behavioural adjustments to the following:\
- **Duration-dependent search effort**\
- **Reservation Wage Adjustment Rates**\
- **Cyclical On-the-Job Search**\
- **Risk Aversion**: For now this is randomised to ensure variation in
vacancy targeting by similar workers. This is not yet supported by data.

<!-- - *Dynamic selection* (heterogeneity of job seekers - fitness & skills match for the current labour market) -->

<!-- - *Duration dependence* (job-finding probability decreases over time spent unemployed indendent of job-seeker characteristics) -->

<!-- - *Search effort* (individuals can exert greater search effort to counteract duration dependence - generally exhibits a countercyclical behaviour - job search effort increases as the economy recedes) -->

<!-- **Current idea:** If we view *dynamic selection* simply as worker heterogeneity, *duration dependence* as an endogenous process of skill deterioration or signal to employers (up to a certain point), and *search effort* as responsive to labour market conditions, we will have quasi-independent elements in the job-finding probability of workers in the model! -->



## **Relevant Analyses** {.tabset}

**Data we have decided to keep:**

1.  Current Population Survey 2018 & 2022: Information on applications
    sent by unemployment duration.             
    
2.  Mukoyama et al. data on the intensive margin of unemployed search
    effort (in minutes searched) over the business cycle. We have chosen
    to include this as a validation exercise of our application effort
    imposition.               

3.  [Eeckhout et al. 2019 Unemployment
    Cycles](https://www.aeaweb.org/articles?id=10.1257/mac.20180105): We
    derive the sensitivity of employed job seekers to the business cycle
    from the employment-to-employment transitions data as used in
    Eeckhout et al. Due to unreliable component parts of the Eeckhout
    analysis, we decided to abandon using their estimated parameters
    (search intensity for employed workers).               
    
4.   Displaced Worker Supplement: As part of the Current Population Survey, the US Census Bureau conducts an annual Displaced Worker Supplement in which workers who have lost their job in the last three years are asked additional questions about their unemployment experiences and (if re-employed) their re-employment conditions. From this we draw a reservation wage adjustment rate as a function of unemployment duration.                  

5.   [**Mueller et al. 2021: Job Seekers' Perceptions and Employment
    Prospects: Heterogeneity, Duration Dependence and
    Bias**](https://www.aeaweb.org/articles?id=10.1257/aer.20190808)           

**Additional analyses that we have decided to exclude as data inputs due
to lack of relevance or poor data quality are in the "Discarded
Analyses" tab.**

### Applications Sent {.tabset}

#### Current Population Survey - 2018 & 2020 Supplement

This 2020 ["Beyond the Numbers"
issue](https://www.bls.gov/opub/btn/volume-9/how-do-jobseekers-search-for-jobs.htm#_edn5)
distills insights from a 2018 Supplement to the Current Population
Survey. The below plots show the highlights relevant to our
decision-making on the job search process. In nearly all cases, the
results are "binned" into intervals (ie. number of people sending 81 or
more applications or unemployment duration of between 5 and 14 weeks)
which means that any line plots (or linear interpretation of the bar
graph) should be done carefully. Preliminary results using the raw data
are found in the next section.

-   Figure 1: Shows the proportion of all individuals sending x amount
    of applications receiving y amount of interviews. The plot indicates
    a "consistent" return to sending more applications, although as
    demonstrated in Figure 3, the number of interviews received does not
    necessarily equate to receiving a job offer.

-   Figure 2: Demonstrates the number of applications sent (red),
    interviews received (green), average interview:applicaiton ratio
    (blue), and probability of receiving a job offer (purple) by
    individuals in each category of unemployment duration. There is some
    indication (although, again, interpretation is difficult without the
    raw data) that both effort and success seems to increase and then
    decline with time spent in unemployment, apart from success as
    measured by receiving a job offer which seems to consistently
    decline with time spent in unemployment.

-   Figure 3: Percentage of jobseekers receiving an offer seems to
    increase as a function of the number of applications sent, until a
    certain point.


```
## Processing URL: https://www.bls.gov/opub/btn/volume-9/how-do-jobseekers-search-for-jobs.htm#_edn2
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1-1.png)

It turns out that the 2018 supplement was also run in 2022, giving us
two sets of years to compare (including pre- and post-Covid). The below
looks at the [raw
data](https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-unemployment.2022.html#list-tab-1269561637)
that underlies the plotting immediately above, plus the additional data
from 2022. Below find a preliminary scatter plot of applications sent
versus unemployment duration. Each individual is asked how many
applications they sent in the last two months (two-month periods are
indicated by the grey gridlines, for reference). This does NOT include
data in on the job search.

**Data Source:** Unemployment Insurance Nonfilers Supplement conducted
in 2018 (n = 3,268) & 2022 (n = 1,901) where individuals who are
unemployed but have not filed for unemployment insurance are asked the
following:

![Survey Question: Unemployment
Duration](CPS_BLS_Supplement_18_22/udur_survey_question.png)

![Survey Question: Applications
Sent](CPS_BLS_Supplement_18_22/apps_sent_question.png)

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2-1.png)

Below, I display the results of an exploration of the probability of
reporting a specific number of applications sent (in the bins as in the
survey question above) using various specifications of an ordinal
logistic regression. I test specifications varying three different
model parameters: 
1. link function                 
2. linear vs. quadratic unemploymentduration,              
3. with and without demographic control variables (education,
gender, age, family income - race excluded because of lack of
statistical significance though this can be revisited.)              

We estimate an ordinal logistic regression model for reported applications sent
$Y_i in {0, 1, 2, 3, 4}$ testing four different link
functions: the complementary log-log (cloglog),
logistic, log-log, and probit link functions.
Let $X_i^\top \beta$ denote the predictor variable. The cumulative
probability of observing response category $j$ or below,
$\Pr(Y_i \leq j \mid X_i)$, is modeled as follows for each link
function:

\begin{align*}
\text{Complementary log-log (cloglog):} \quad & \Pr(Y_i \leq j \mid X_i) = 1 - \exp\left( -\exp\left( \tau_j - X_i^\top \beta \right) \right) \\
\text{Logistic (logit):} \quad & \Pr(Y_i \leq j \mid X_i) = \frac{1}{1 + \exp\left( -(\tau_j - X_i^\top \beta) \right)} \\
\text{Loglog:} \quad & \Pr(Y_i \leq j \mid X_i) = \exp\left( -\exp\left( -(\tau_j - X_i^\top \beta) \right) \right) \\
\text{Probit:} \quad & \Pr(Y_i \leq j \mid X_i) = \Phi(\tau_j - X_i^\top \beta)
\end{align*}

Here, $\Phi(\cdot)$ denotes the cumulative distribution function of the
standard normal distribution. The estimated coefficients $\beta$ are
interpreted conditional on the choice of link function where $X_i$ is either:

$X_i = \left( \text{Unemp.Dur.}_i \right)$

$X_i = \left( \text{Unemp.Dur.}_i^2 \right)$

$X_i = \left( \text{Unemp.Dur.}_i, \text{Unemp.Dur.}_i^2 \right)$

with and without control variables (education, gender, age, family income).

Assumptions about the probability distribution of the errors associated with each link function:              
- *Logit:* good when the response behavior is symmetric around the middle category.             
- *Probit:* When you’re assuming a normal latent error distribution or want closer fit to Gaussian processes.             
- *Complementary log-log:* When the likelihood of being in a higher category increases sharply but asymmetrically, or you expect hazard-like dynamics.             
- *Log-log:* When early categories are of more importance and need sharper separation.             

*Preliminary hypothesis:* Best fit will be with a complementary log-log as we care more about distinguishing between lower-level bins and there are few observations in the highest-level bins. 


![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png)![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-2.png)![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-3.png)

Using an AIC information criterion to compare the fit across all models, the following results are clear:            
1. Models with control variables consistently perform better than those without.                                        
2. Looking at the plots above, the relationship between unemployment duration and the predicted probability of reporting each application effort bin is very consistent except in the case of the log-log link function (blue in the panels above). In the plot below comparing the AIC the log-log link function (represented by the square symbol below) is consistently worse than all other link functions. This indicates consistency in the results reported above. Intuitively, the log-log link function is likely to be an unreasonable fit for the latent variable as we care more about shifts in the lower-level categories than higher-level categories.                  
3. A complementary log-log specification for the latent variable is most suitable. This follows logically from the fact that the probability of being in the highest-level categories is relatively low.             
4. Finally, a specification with a linear and quadratic estimator is consistently better than either the specification with simply a linear OR quadratic unemployment duration estimator indicating that the probability distributions represented in the final panel above are likely to be the best fit.          


*Result:* For each additional quarter of unemployment, an individual’s odds of dropping to a lower-level application category decreases by ~.1%. This is statistically significant across all specifications at the 0.1% level.


![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png)


![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png)

#### Validation: Mukoyama et al. Job Search and the Business Cycle

[Mukoyama: Job Search Over the Business
Cycle](https://www.aeaweb.org/articles?id=10.1257/mac.20160202)

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png)![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-2.png)![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-3.png)![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-4.png)

### Reservation Wage Adjustment {.tabset}












As part of the Current Population Survey, the US Census Bureau conducts an annual [Displaced Worker Supplement](https://cps.ipums.org/cps/dw_sample_notes.shtml) in which workers who have lost their job in the last three years are asked additional questions about their unemployment experiences and (if re-employed) their re-employment conditions.

From link above: "The universe for the Displaced Workers Supplement is civilians 20 or older. Respondents are further categorized as a"displaced worker" if they meet additional characteristics (see DWSTAT). After 1998, displaced workers are those who lost or left a job due to layoffs or shutdowns within the past 3 years...were not self-employed, and did not expect to be recalled to work within the next six months.

The data used below is from annual survey responses between 2000-2025.
I use the supplement sample weights in all results below.
I note where I have clipped the sample for outliers (wage ratio between [0.25, 2] and unemployment duration less than 96 weeks (\~24 months).

Below I:

1.  **Data Cleaning Procedure:** Show data cleaning just for reference (feel free to ignore)!.
2.  **Descriptives:** Show some descriptives about the data itself.
3.  **Regression Results on Non-Uniform Sample:** Regression results with ratio of new wage to wage at the lost job ($W_{h}$ and $W_{w}$) regressed (cross-sectionally) on unemployment duration with and without various combinations of control variables (whether or not an individual received unemployment compensation, age, race, sex, marital status, education, previous wage level. ) Note that the wages are reported in hourly and weekly values but. this reporting is inconsistent across observations. In other words, though most individuals (4600/6198) report their wage in both units, 270 report only hourly and 1328 report only weekly. I have not reconciled the inconsistency so I use hourly wage ratios in majority of the below document. I could try to reconcile this.
4.  Outline some considerations for further improvement of the analysis:
    1.  **Reweighted Samples:** The sample is non-uniform in unemployment duration (less observations as unemployment duration increases). Try two methods of reweighting to address selection issues (Heckman Selection correction - though I think this is inappropriate for this particular selection issue) and non-uniform (entropy-balancing to deal with representativeness of population over unemployment durations) sample confirm regression results in non-uniform sample.
    2.  **Representativeness of the Sample (Education, Age, Gender, and Wage):** Representativeness of the data to motivate data limitations and inform the ultimate reweighting scheme.

**Overall result (at the moment):** Individuals accept a \~1-percentage point change in the wage ratio per additional month of unemployment.
Variations using model reweighting, different samples, combinations of control variables, reported hourly and weekly wage ratios do not seem to affect the result.
However, the data seems to follow a non-linear relationship (we see little satisficing until around \~12 months of unemployment) after which the wage ratio begins to decrease.
Individuals seem to accept a below-1 relative wage ratio (current wage:wage at lost job) following a year of unemployment.
If we fit this model with a quadratic fit this could inform our reservation wage adjustment parameter in the model.

**Important Considerations/Limitations:**

1.  **Displaced worker classification as outlined above.** Can we generalise from this definition to all unemployed workers?
2.  **The reported 'current wage' is not necessarily the the realised wage post-re-employment.** Individuals report the wage at the lost job, the amount of time unemployed until they were re-employed, and the wage they hold at their current job. However, it is not indicated whether the current job is the same job as the first they were re-employed at. Given various comments in the literature about finding "stop-gap" employment, this might not be a problem in the sense that the "current wage" would more accurately indicate the wage an individual has "landed" at post-unemployment spell. But curious what you think about the defensibility of this.
3.  **Outcome variable:** How do we feel about the outcome variable as the ratio of current to latest held job? Might we want to take the log or consider simply the (log) level regressed on the previous wage. Wondering if a ratio-based outcome variable might muddle interpretation. Curious for your reactions.



#### Data cleaning

Feel free to ignore this code chunk immediately below.
I include it for your info on binning and outlier trimming.


``` r
# From the original dataset, I include only those that reported having lost a FT job in the last three years
df <- readRDS(here("data/behav_params/cps_displaced_worker_supplement/cps_disp_filtered.RDS")) %>% 
  select(hwtfinl, cpsid, wtfinl, age, sex, race, marst, educ, # age, sex, race, marital status, educational attainment
         dwsuppwt, # Survey weight
         dwyears, # Years worked at lost job
         dwben, # Received unemployment benefits
         dwexben, # Exhausted unemployment benefits
         dwlastwrk, # Time since worked at last job
         dwweekc, # Weekly earnings at current job
         dwweekl, # Weekly earnings at lost job
         dwwagel, # Hourly earnings at lost job
         dwwagec, # Hourly wage at current job
         dwhrswkc, # Hours worked each week at current job
         dwresp, # Eligibility and interview status for Displaced Worker Supplement
         # Interestingly the unemployment duration is not directly linked to CURRENT job and we cannot see the wage of the start of the next job...thought this feels problematic, it does indicate more accurately the ultimate "recovered" wage...will need to declare as a limitation but also not completely indefensible
         dwwksun) %>%  # Number of weeks not working between between end of lost or left job and start of next job
  # I remove anyone who is Not in Universe (99) and declaring greater than 160 weeks unemployed between jobs
filter(dwhrswkc != 99 & dwwksun <= 160) %>% 
  # Replacing NIU values with NA values
  mutate(dwwagel = ifelse(round(dwwagel) == 100, NA, dwwagel),
         dwwagec = ifelse(round(dwwagec) == 100, NA, dwwagec),
         dwweekl = ifelse(round(dwweekl) == 10000, NA, dwweekl),
         dwweekc = ifelse(round(dwweekc) == 10000, NA, dwweekc),
         # dwwage_rec_l = ifelse(is.na(dwagel) & !is.na(dweekl) ~ dwweekl),
         # dwweekc = ifelse(round(dwweekc) == 10000, NA, dwweekc),
         # Binning educational categories
         educ_cat = case_when(educ %in% c(1) ~ NA, # (NIU)
                              educ > 1 & educ <= 71 ~ "Less than HS", # Includes "None" - Grade 12 no diploma (8 subcategories (grade 1-11 etc))
                              educ %in% c(73, 81) ~ "HS Diploma", # Includes "High school Diploma or equivalent" and "some college, but no degree"
                              educ %in% c(91, 92) ~ "Associate's", # Include "[Associate's degree, occupational/vocational program]" and "Associate's         [Associate's degree, academic program]"
                              educ %in% c(111) ~ "Bachelor's", # Bachelor's degree
                              educ > 111 ~ "Postgraduate Degree" # Includes Master's, Professional School, and Doctorate degree
                              ),
         # Marital status to binary indicator
         marst = case_when(marst == 1 ~ 1, # Married with a present spouse
                           # Might consider dividing this differently
                           TRUE ~ 0), # Married with absent spouse, separated, divorced, widowed, never married/single
         # gender to 0,1 values
         female = sex == 2,
         # race to higher-level categories w binary values
         white = race == 100,
         black = race == 200,
         mixed = race %in% c(801, 802, 803, 804, 805, 806, 810, 812, 813, 820, 830),
         aapi = race %in% c(650, 651, 652, 808, 809),
         native = race == 300
         # age is a continuous variable which seems fine for now...binning likely unnecessary
         ) %>% 
        # Ratio of hourly wage of current job to lost job
  mutate(ratio_wage = dwwagec/dwwagel,
         # Ratio of weekly wage of current job to lost job
         ratio_weekly = dwweekc/dwweekl,
         # Reconciling missing reporting between weekly and hourly wage. Take either the min, max or mean value. 
         ratio_reconciled_min = case_when(is.na(ratio_wage) ~ ratio_weekly, 
                                          is.na(ratio_weekly) ~ ratio_wage, 
                                          TRUE ~ pmin(ratio_weekly, ratio_wage)), 
         ratio_reconciled_max = case_when(is.na(ratio_wage) ~ ratio_weekly, 
                                          is.na(ratio_weekly) ~ ratio_wage, 
                                          TRUE ~ pmax(ratio_weekly, ratio_wage)), 
         ratio_reconciled_mean = case_when(is.na(ratio_wage) ~ ratio_weekly, 
                                          is.na(ratio_weekly) ~ ratio_wage, 
                                          TRUE ~ rowMeans(across(c(ratio_wage, ratio_weekly)), na.rm = TRUE)), 
         # Create monthly unemployment duration for continuous
         dwmosun = floor(dwwksun/4),
         # Unemployment duration (reported as time between lost job and start of next job)
         # I bin in...
         # monthly intervals (4 weeks) from 1-6 months
         # quarterly intervals (12 weeks) from 7 mos-1 year
         # half-year interval from 1-2.5 years
         # single bin for anyone about 120 weeks
         dwwksun_bin = case_when(
           # Monthly intervals (4 weeks) from 1-6 months
           dwwksun <= 4 ~ 1, #"Less than 4 weeks",
                                 dwwksun > 4 & dwwksun <= 8 ~ 2,
                                 dwwksun > 8 & dwwksun <= 12 ~ 3,
                                 dwwksun > 12 & dwwksun <= 16 ~ 4, 
                                 dwwksun > 16 & dwwksun <= 20 ~ 5,
                                 dwwksun > 20 & dwwksun <= 24 ~ 6,
                                 # Quarterly Intervals (12 weeks) from 6+ mos - 1 year
                                 dwwksun > 24 & dwwksun <= 36 ~ 7,
                                 dwwksun > 36 & dwwksun <= 48 ~ 8, 
                                 # Half-year Intervals (24 weeks) from 1-2.5 years
                                 dwwksun > 48 & dwwksun <= 72 ~ 9, 
                                 dwwksun > 72 & dwwksun <= 96 ~ 10, 
                                 dwwksun > 96 & dwwksun <= 120 ~ 11, 
                                 # Anyone above - recall this is capped at 160 weeks as per filter above
                                 dwwksun > 120 ~ 12),
         # Bin labels
         dwwksun_bin_labs = case_when(dwwksun_bin == 1 ~ "<= 1 mo.", #"Less than 4 weeks",
                                 dwwksun_bin == 2 ~ "1-2 mos.",
                                 dwwksun_bin == 3 ~ "2-3 mos.",
                                 dwwksun_bin == 4 ~ "3-4 mos.", 
                                 dwwksun_bin == 5 ~ "4-5 mos.",
                                 dwwksun_bin == 6 ~ "5-6 mos.",
                                 # Quarterly Intervals (12 weeks) from 6+ mos - 1 year
                                 dwwksun_bin == 7 ~ "6-9 mos.",
                                 dwwksun_bin == 8 ~ "9-12 mos.", 
                                 # Half-year Intervals (24 weeks) from 1-2.5 years
                                 dwwksun_bin == 9 ~ "12-18 mos.", 
                                 dwwksun_bin == 10 ~ "18-24 mos.", 
                                 dwwksun_bin == 11 ~ "24-30 mos.", 
                                 # Anyone above - recall this is capped at 160 weeks as per filter above
                                 dwwksun_bin == 12 ~ "30+ mos."),
         log_ratio_wage = log(ratio_wage),
         log_ratio_weekly = log(ratio_weekly),
         # I clip the sample to an accepted wage ratio between [0.5, 2] and less than 96 weeks of unemployment
         clipped_sample_hwage = ratio_wage >= 0.5 & ratio_wage <= 2 & dwwksun_bin < 11,
         clipped_sample_wwage = ratio_weekly >= 0.5 & ratio_weekly <= 2  & dwwksun_bin < 11,
         clipped_sample_rec_min = ratio_reconciled_min >= 0.5 & ratio_reconciled_min <= 2 & dwwksun_bin < 11,
         clipped_sample_rec_max = ratio_reconciled_max >= 0.5 & ratio_reconciled_max <= 2 & dwwksun_bin < 11,
         clipped_sample_rec_mean = ratio_reconciled_mean >= 0.5 & ratio_reconciled_mean <= 2 & dwwksun_bin < 11)
```

#### Descriptives

All descriptives below us the us the Displaced Worker Sample Weights.

Histogram: sample is skewed (see reweighting alternatives at end of document).

Box plots: Looking at the reported wage ratios in weekly and hourly values, the mean is fixed near 1 until \>12 mos of unemployment in hourly wage reporting.
In weekly wage reporting, the "satisficing" seems to start earlier in unemployment duration (sample size is larger for weekly reporting - might be worth focusing on those wages).

Scatter plot: I fit a linear and spline fit to the scatted plot of the wage ratio to unemployment duration before using the regression.
Indicates decline in the wage ratio with unemployment duration that has a potentially non-linear fit.

![plot of chunk unnamed-chunk-24](figure/unnamed-chunk-24-1.png)![plot of chunk unnamed-chunk-24](figure/unnamed-chunk-24-2.png)![plot of chunk unnamed-chunk-24](figure/unnamed-chunk-24-3.png)![plot of chunk unnamed-chunk-24](figure/unnamed-chunk-24-4.png)![plot of chunk unnamed-chunk-24](figure/unnamed-chunk-24-5.png)

![plot of chunk unnamed-chunk-25](figure/unnamed-chunk-25-1.png)

#### Regressions (non-uniform sample) {.tabset}

Next, (ignoring for now the non-uniformity of the sample ie. that there are less observations present for higher unemployment durations) I run the following regression (with various modifications to sample and control variables).
$W_{i} = \alpha_{i} + \beta_{1} d_{i} + \beta_{2}UI_{i} + \beta_{3}X_{i} + \epsilon_{i}$

where $W_{i}$: Ratio of accepted wage to wage at lost job (hourly values).

$d_{i}$: Unemployment duration (continuous or binned).

$UI_{i}$: Control variable for having used or exhausted unemployment benefits.

$X_{i}$: Vector of control variables (sex, age, race (white, black, mixed), marital status (married or not), whether individual used UI benefits, whether individual exhausted UI benefits, education level, and previous wage level).

There are 48 models present with all combinations of the following:

-   **Continuous vs. Discrete Treatment Variable (2 alternatives):** Continuous (monthly) versus binned unemployment duration.

-   **w. UI vs w. Exhausted UI (3 alternatives):** The data includes a variable for whether individuals USE and/or EXHAUST unemployment benefits.
    I run the regressions without these UI controls, with control for having used UI, with control for having exhausted UI.

-   **w. Controls (2 alternatives):** With or without additional demographic controls (sex, age, race, married, education)

-   **w. Wage Level (2 alternatives):** With or without wage level of lost job to control for income.
    The level of the previous wage likely affects the wage ratio.

-   **Outlier clipped sample (2 alternatives):** (As described in the intro section) Remove outliers where the wage ratio is within [0.25, 2.5] and reported unemploymetn duration is below 96 weeks (\~ 2 years).

I include the full set of coefficients (again, apologies for verbose output) in case you find the coefficients on the controls interesting (I think the coefficient on age and holding a Bachelor's degree particularly interesting).
But I highlight in blue our main interest in $\beta_{1}$.



Across all models in the tabs below we see a consistently negative coefficient on unemployment duration (\~0.7-1 percentage point increase in the wage ratio for each additional month spent in unemployment).
If we look more closely at the performance of our model with continuous unemployment duration, UI use (not exhaustion), all controls, wage levels, and outlier correction we see that the model performs fairly well across various diagnostic tests.


```
## [1] "Continuous U Duration. w. UI Control w. demographic controls (clipped sample)"
```

![plot of chunk unnamed-chunk-27](figure/unnamed-chunk-27-1.png)

##### Continuous UE Duration {.tabset}

Continuous UE duration treatment is reported in monthly values.
A one-unit increase in the treatment variable = 1 additional month of unemployment.

###### W.O. Wage Level Control

<!--html_preserve--><div id="kgiplnsuzs" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#kgiplnsuzs table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#kgiplnsuzs thead, #kgiplnsuzs tbody, #kgiplnsuzs tfoot, #kgiplnsuzs tr, #kgiplnsuzs td, #kgiplnsuzs th {
  border-style: none;
}

#kgiplnsuzs p {
  margin: 0;
  padding: 0;
}

#kgiplnsuzs .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#kgiplnsuzs .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#kgiplnsuzs .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#kgiplnsuzs .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#kgiplnsuzs .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#kgiplnsuzs .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#kgiplnsuzs .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#kgiplnsuzs .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#kgiplnsuzs .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#kgiplnsuzs .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#kgiplnsuzs .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#kgiplnsuzs .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#kgiplnsuzs .gt_spanner_row {
  border-bottom-style: hidden;
}

#kgiplnsuzs .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#kgiplnsuzs .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#kgiplnsuzs .gt_from_md > :first-child {
  margin-top: 0;
}

#kgiplnsuzs .gt_from_md > :last-child {
  margin-bottom: 0;
}

#kgiplnsuzs .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#kgiplnsuzs .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#kgiplnsuzs .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#kgiplnsuzs .gt_row_group_first td {
  border-top-width: 2px;
}

#kgiplnsuzs .gt_row_group_first th {
  border-top-width: 2px;
}

#kgiplnsuzs .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#kgiplnsuzs .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#kgiplnsuzs .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#kgiplnsuzs .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#kgiplnsuzs .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#kgiplnsuzs .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#kgiplnsuzs .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#kgiplnsuzs .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#kgiplnsuzs .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#kgiplnsuzs .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#kgiplnsuzs .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#kgiplnsuzs .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#kgiplnsuzs .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#kgiplnsuzs .gt_left {
  text-align: left;
}

#kgiplnsuzs .gt_center {
  text-align: center;
}

#kgiplnsuzs .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#kgiplnsuzs .gt_font_normal {
  font-weight: normal;
}

#kgiplnsuzs .gt_font_bold {
  font-weight: bold;
}

#kgiplnsuzs .gt_font_italic {
  font-style: italic;
}

#kgiplnsuzs .gt_super {
  font-size: 65%;
}

#kgiplnsuzs .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#kgiplnsuzs .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#kgiplnsuzs .gt_indent_1 {
  text-indent: 5px;
}

#kgiplnsuzs .gt_indent_2 {
  text-indent: 10px;
}

#kgiplnsuzs .gt_indent_3 {
  text-indent: 15px;
}

#kgiplnsuzs .gt_indent_4 {
  text-indent: 20px;
}

#kgiplnsuzs .gt_indent_5 {
  text-indent: 25px;
}

#kgiplnsuzs .katex-display {
  display: inline-flex !important;
  margin-bottom: 0.75em !important;
}

#kgiplnsuzs div.Reactable > div.rt-table > div.rt-thead > div.rt-tr.rt-tr-group-header > div.rt-th-group:after {
  height: 0px !important;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <!--/html_preserve--><caption class='gt_caption'>Continuous UE Duration w.o Wage Level Control</caption><!--html_preserve-->
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="a-"> </th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.">Cont.</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-(clipped)">Cont. (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI">Cont. w. UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI-(clipped)">Cont. w. UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI">Cont. w. exhausted UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI-(clipped)">Cont. w. exhausted UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-controls">Cont. w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-controls-(clipped)">Cont. w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI-w.-controls">Cont. w. UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI-w.-controls-(clipped)">Cont. w. UI w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI-w.-controls">Cont. w. exhausted UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI-w.-controls-(clipped)">Cont. w. exhausted UI w. controls (clipped)</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers=" " class="gt_row gt_left">Intercept</td>
<td headers="Cont." class="gt_row gt_center">1.053***</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">1.045***</td>
<td headers="Cont. w. UI" class="gt_row gt_center">1.053***</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">1.045***</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">1.006***</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">1.006***</td>
<td headers="Cont. w. controls" class="gt_row gt_center">1.211***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">1.154***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">1.211***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">1.154***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">1.156***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">1.108***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center">(0.006)</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">(0.004)</td>
<td headers="Cont. w. UI" class="gt_row gt_center">(0.006)</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">(0.004)</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">(0.010)</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.032)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.032)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.033)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.023)</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="background-color: #ADD8E6;">Unemployment Duration (Months)</td>
<td headers="Cont." class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Cont. (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.005***</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.004***</td>
<td headers="Cont. w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.004***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.004***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center">(0.001)</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Received Unemployment Compensation</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center">-0.000</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.000</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.000</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.000</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">Exhausted Unemployment Compensation</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.001***</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.001***</td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.001***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.000***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Female</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.003</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.003</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.003</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.003</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.003</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.003</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Age</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.003***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.002***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.003***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.002***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.003***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.002***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">White</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.035</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.052**</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.035</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.052**</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.033</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.051**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Black</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.048+</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.057**</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.048+</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.057**</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.045+</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.055**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Mixed</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.014</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.070**</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.014</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.070*</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.017</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.068*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.040)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.040)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.040)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.027)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Married</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.005</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.005</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.005</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.012+</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Bachelor's Degree</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.048*</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.075***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.048*</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.075***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.047*</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.075***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">High School</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.026</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.010</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.026</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.010</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.027</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.009</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Less than HS</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.032</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.009</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.032</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.009</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.038+</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.005</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Postgraduate Degree</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.082+</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.039</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.082+</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.039</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.085+</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.042</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont." class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.045)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.031)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.045)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.031)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.045)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.031)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Num.Obs.</td>
<td headers="Cont." class="gt_row gt_center">4870</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. UI" class="gt_row gt_center">4870</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">4870</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. controls" class="gt_row gt_center">4870</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">4644</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2</td>
<td headers="Cont." class="gt_row gt_center">0.009</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">0.012</td>
<td headers="Cont. w. UI" class="gt_row gt_center">0.009</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.012</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.017</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.022</td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.025</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.032</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.025</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.032</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.030</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.040</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2 Adj.</td>
<td headers="Cont." class="gt_row gt_center">0.009</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">0.012</td>
<td headers="Cont. w. UI" class="gt_row gt_center">0.009</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.016</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.022</td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.022</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.029</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.022</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.029</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.028</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.037</td></tr>
    <tr><td headers=" " class="gt_row gt_left">F</td>
<td headers="Cont." class="gt_row gt_center">46.344</td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center">23.169</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">41.487</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">11.151</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">10.220</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">12.521</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">RMSE</td>
<td headers="Cont." class="gt_row gt_center">0.38</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. UI" class="gt_row gt_center">0.38</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.24</td></tr>
  </tbody>
  <tfoot class="gt_sourcenotes">
    <tr>
      <td class="gt_sourcenote" colspan="13">+ p &lt; 0.1, * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</td>
    </tr>
  </tfoot>
  
</table>
</div><!--/html_preserve-->

###### W. Wage Level Control

<!--html_preserve--><div id="wcsdntwngj" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#wcsdntwngj table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#wcsdntwngj thead, #wcsdntwngj tbody, #wcsdntwngj tfoot, #wcsdntwngj tr, #wcsdntwngj td, #wcsdntwngj th {
  border-style: none;
}

#wcsdntwngj p {
  margin: 0;
  padding: 0;
}

#wcsdntwngj .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#wcsdntwngj .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#wcsdntwngj .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#wcsdntwngj .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#wcsdntwngj .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#wcsdntwngj .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#wcsdntwngj .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#wcsdntwngj .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#wcsdntwngj .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#wcsdntwngj .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#wcsdntwngj .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#wcsdntwngj .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#wcsdntwngj .gt_spanner_row {
  border-bottom-style: hidden;
}

#wcsdntwngj .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#wcsdntwngj .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#wcsdntwngj .gt_from_md > :first-child {
  margin-top: 0;
}

#wcsdntwngj .gt_from_md > :last-child {
  margin-bottom: 0;
}

#wcsdntwngj .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#wcsdntwngj .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#wcsdntwngj .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#wcsdntwngj .gt_row_group_first td {
  border-top-width: 2px;
}

#wcsdntwngj .gt_row_group_first th {
  border-top-width: 2px;
}

#wcsdntwngj .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#wcsdntwngj .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#wcsdntwngj .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#wcsdntwngj .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#wcsdntwngj .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#wcsdntwngj .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#wcsdntwngj .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#wcsdntwngj .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#wcsdntwngj .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#wcsdntwngj .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#wcsdntwngj .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#wcsdntwngj .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#wcsdntwngj .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#wcsdntwngj .gt_left {
  text-align: left;
}

#wcsdntwngj .gt_center {
  text-align: center;
}

#wcsdntwngj .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#wcsdntwngj .gt_font_normal {
  font-weight: normal;
}

#wcsdntwngj .gt_font_bold {
  font-weight: bold;
}

#wcsdntwngj .gt_font_italic {
  font-style: italic;
}

#wcsdntwngj .gt_super {
  font-size: 65%;
}

#wcsdntwngj .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#wcsdntwngj .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#wcsdntwngj .gt_indent_1 {
  text-indent: 5px;
}

#wcsdntwngj .gt_indent_2 {
  text-indent: 10px;
}

#wcsdntwngj .gt_indent_3 {
  text-indent: 15px;
}

#wcsdntwngj .gt_indent_4 {
  text-indent: 20px;
}

#wcsdntwngj .gt_indent_5 {
  text-indent: 25px;
}

#wcsdntwngj .katex-display {
  display: inline-flex !important;
  margin-bottom: 0.75em !important;
}

#wcsdntwngj div.Reactable > div.rt-table > div.rt-thead > div.rt-tr.rt-tr-group-header > div.rt-th-group:after {
  height: 0px !important;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <!--/html_preserve--><caption class='gt_caption'>Continuous UE Duration w. Wage Level Control</caption><!--html_preserve-->
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="a-"> </th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.">Cont.</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-(clipped)">Cont. (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI">Cont. w. UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI-(clipped)">Cont. w. UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI">Cont. w. exhausted UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI-(clipped)">Cont. w. exhausted UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-controls">Cont. w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-controls-(clipped)">Cont. w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI-w.-controls">Cont. w. UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-UI-w.-controls-(clipped)">Cont. w. UI w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI-w.-controls">Cont. w. exhausted UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Cont.-w.-exhausted-UI-w.-controls-(clipped)">Cont. w. exhausted UI w. controls (clipped)</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers=" " class="gt_row gt_left">Intercept</td>
<td headers="Cont." class="gt_row gt_center">1.185***</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">1.131***</td>
<td headers="Cont. w. UI" class="gt_row gt_center">1.186***</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">1.130***</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">1.145***</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">1.094***</td>
<td headers="Cont. w. controls" class="gt_row gt_center">1.347***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">1.243***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">1.347***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">1.243***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">1.300***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">1.202***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center">(0.011)</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">(0.008)</td>
<td headers="Cont. w. UI" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">(0.008)</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">(0.014)</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">(0.010)</td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.032)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.032)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.034)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.023)</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="background-color: #ADD8E6;">Hourly Wage of Lost Job</td>
<td headers="Cont." class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Cont. (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center">(0.001)</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="background-color: #ADD8E6;">Unemployment Duration (Months)</td>
<td headers="Cont." class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Cont. (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.005***</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.004***</td>
<td headers="Cont. w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.005***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.004***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center">(0.001)</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Received Unemployment Compensation</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center">-0.000</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.000</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.000</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.000</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">Exhausted Unemployment Compensation</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.001***</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.000***</td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.000***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.000***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Female</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.028**</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.023**</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.028**</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.023**</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.028**</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.023**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Age</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.002***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.001***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.002***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.001***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.001***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.001***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">White</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.034</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.050**</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.034</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.050**</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.032</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.049**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Black</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.058*</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.061***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.058*</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.061***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.055*</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.060***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Mixed</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.016</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.067*</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.016</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.067*</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.019</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.065*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.039)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.039)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.039)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.026)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Married</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.013</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.018*</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.013</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.018*</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.013</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.018*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.010)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.010)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.010)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Bachelor's Degree</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.077***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.094***</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.077***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.094***</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.076***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.094***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">High School</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.051**</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.008</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.051**</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.008</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.051**</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.008</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Less than HS</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">-0.084***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">-0.027+</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">-0.084***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">-0.027+</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">-0.088***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.029*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Postgraduate Degree</td>
<td headers="Cont." class="gt_row gt_center"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.160***</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.093**</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.160***</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.093**</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.161***</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.094**</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont." class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Cont. w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.044)</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.030)</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.044)</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.030)</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.044)</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.030)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Num.Obs.</td>
<td headers="Cont." class="gt_row gt_center">4870</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. UI" class="gt_row gt_center">4870</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">4870</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. controls" class="gt_row gt_center">4870</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">4644</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2</td>
<td headers="Cont." class="gt_row gt_center">0.048</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">0.046</td>
<td headers="Cont. w. UI" class="gt_row gt_center">0.048</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.046</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.052</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.053</td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.069</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.073</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.069</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.073</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.073</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.079</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2 Adj.</td>
<td headers="Cont." class="gt_row gt_center">0.047</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">0.046</td>
<td headers="Cont. w. UI" class="gt_row gt_center">0.047</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.046</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.051</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.053</td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.067</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.071</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.067</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.071</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.070</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.077</td></tr>
    <tr><td headers=" " class="gt_row gt_left">F</td>
<td headers="Cont." class="gt_row gt_center">121.551</td>
<td headers="Cont. (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI" class="gt_row gt_center">81.034</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">88.352</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. controls" class="gt_row gt_center">30.216</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">27.890</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">29.347</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">RMSE</td>
<td headers="Cont." class="gt_row gt_center">0.37</td>
<td headers="Cont. (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. UI" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. exhausted UI" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. exhausted UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Cont. w. controls" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. controls (clipped)" class="gt_row gt_center">0.23</td>
<td headers="Cont. w. UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. UI w. controls (clipped)" class="gt_row gt_center">0.23</td>
<td headers="Cont. w. exhausted UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Cont. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.23</td></tr>
  </tbody>
  <tfoot class="gt_sourcenotes">
    <tr>
      <td class="gt_sourcenote" colspan="13">+ p &lt; 0.1, * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</td>
    </tr>
  </tfoot>
  
</table>
</div><!--/html_preserve-->

##### Binned UE Duration {.tabset}

Binned UE duration treatment is reported in bins as indicated in the box plots and code cleaning above.

###### W.O. Wage Level Control

<!--html_preserve--><div id="nzwimfwpub" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#nzwimfwpub table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#nzwimfwpub thead, #nzwimfwpub tbody, #nzwimfwpub tfoot, #nzwimfwpub tr, #nzwimfwpub td, #nzwimfwpub th {
  border-style: none;
}

#nzwimfwpub p {
  margin: 0;
  padding: 0;
}

#nzwimfwpub .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#nzwimfwpub .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#nzwimfwpub .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#nzwimfwpub .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#nzwimfwpub .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#nzwimfwpub .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nzwimfwpub .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#nzwimfwpub .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#nzwimfwpub .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#nzwimfwpub .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#nzwimfwpub .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#nzwimfwpub .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#nzwimfwpub .gt_spanner_row {
  border-bottom-style: hidden;
}

#nzwimfwpub .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#nzwimfwpub .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#nzwimfwpub .gt_from_md > :first-child {
  margin-top: 0;
}

#nzwimfwpub .gt_from_md > :last-child {
  margin-bottom: 0;
}

#nzwimfwpub .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#nzwimfwpub .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#nzwimfwpub .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#nzwimfwpub .gt_row_group_first td {
  border-top-width: 2px;
}

#nzwimfwpub .gt_row_group_first th {
  border-top-width: 2px;
}

#nzwimfwpub .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#nzwimfwpub .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#nzwimfwpub .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#nzwimfwpub .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nzwimfwpub .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#nzwimfwpub .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#nzwimfwpub .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#nzwimfwpub .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#nzwimfwpub .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nzwimfwpub .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#nzwimfwpub .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#nzwimfwpub .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#nzwimfwpub .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#nzwimfwpub .gt_left {
  text-align: left;
}

#nzwimfwpub .gt_center {
  text-align: center;
}

#nzwimfwpub .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#nzwimfwpub .gt_font_normal {
  font-weight: normal;
}

#nzwimfwpub .gt_font_bold {
  font-weight: bold;
}

#nzwimfwpub .gt_font_italic {
  font-style: italic;
}

#nzwimfwpub .gt_super {
  font-size: 65%;
}

#nzwimfwpub .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#nzwimfwpub .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#nzwimfwpub .gt_indent_1 {
  text-indent: 5px;
}

#nzwimfwpub .gt_indent_2 {
  text-indent: 10px;
}

#nzwimfwpub .gt_indent_3 {
  text-indent: 15px;
}

#nzwimfwpub .gt_indent_4 {
  text-indent: 20px;
}

#nzwimfwpub .gt_indent_5 {
  text-indent: 25px;
}

#nzwimfwpub .katex-display {
  display: inline-flex !important;
  margin-bottom: 0.75em !important;
}

#nzwimfwpub div.Reactable > div.rt-table > div.rt-thead > div.rt-tr.rt-tr-group-header > div.rt-th-group:after {
  height: 0px !important;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <!--/html_preserve--><caption class='gt_caption'>Binned UE Duration w.o Wage Level Control</caption><!--html_preserve-->
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="a-"> </th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.">Disc.</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-(clipped)">Disc. (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI">Disc. w. UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI-(clipped)">Disc. w. UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI">Disc. w. exhausted UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI-(clipped)">Disc. w. exhausted UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-controls">Disc. w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-controls-(clipped)">Disc. w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI-w.-controls">Disc. w. UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI-w.-controls-(clipped)">Disc. w. UI w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI-w.-controls">Disc. w. exhausted UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI-w.-controls-(clipped)">Disc. w. exhausted UI w. controls (clipped)</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers=" " class="gt_row gt_left">Intercept</td>
<td headers="Disc." class="gt_row gt_center">1.069***</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">1.055***</td>
<td headers="Disc. w. UI" class="gt_row gt_center">1.069***</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">1.055***</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">1.016***</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">1.010***</td>
<td headers="Disc. w. controls" class="gt_row gt_center">1.223***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">1.162***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">1.223***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">1.161***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">1.165***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">1.111***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center">(0.008)</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">(0.005)</td>
<td headers="Disc. w. UI" class="gt_row gt_center">(0.008)</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">(0.005)</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">(0.012)</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">(0.008)</td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.032)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.032)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.034)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.023)</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="background-color: #ADD8E6;">Unemployment Duration (Binned)</td>
<td headers="Disc." class="gt_row gt_center" style="background-color: #ADD8E6;">-0.013***</td>
<td headers="Disc. (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Disc. w. UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.013***</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.008***</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.005***</td>
<td headers="Disc. w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.008***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.008***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.005***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center">(0.002)</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Received Unemployment Compensation</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center">-0.000</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.000</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.000</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.000</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">Exhausted Unemployment Compensation</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.001***</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.001***</td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.001***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.001***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Female</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.003</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.003</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.003</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.003</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.003</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.003</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Age</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.003***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.002***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.003***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.002***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.003***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.002***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">White</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.035</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.052**</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.035</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.052**</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.033</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.050**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Black</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.047+</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.056**</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.047+</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.056**</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.045+</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.055**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Mixed</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.014</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.070**</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.014</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.070*</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.017</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.068*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.040)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.040)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.040)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.027)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Married</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.004</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.004</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.005</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.012</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Bachelor's Degree</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.049*</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.076***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.049*</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.076***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.049*</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.076***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">High School</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.026</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.010</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.026</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.010</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.027</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.009</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Less than HS</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.033</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.009</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.033</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.009</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.038+</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.005</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Postgraduate Degree</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.083+</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.039</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.083+</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.039</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.086+</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.042</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc." class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.045)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.031)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.045)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.031)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.045)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.031)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Num.Obs.</td>
<td headers="Disc." class="gt_row gt_center">4870</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. UI" class="gt_row gt_center">4870</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">4870</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. controls" class="gt_row gt_center">4870</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">4644</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2</td>
<td headers="Disc." class="gt_row gt_center">0.010</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Disc. w. UI" class="gt_row gt_center">0.010</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.016</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.021</td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.025</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.031</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.025</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.031</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.030</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.039</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2 Adj.</td>
<td headers="Disc." class="gt_row gt_center">0.009</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">0.011</td>
<td headers="Disc. w. UI" class="gt_row gt_center">0.009</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.010</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.016</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.021</td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.022</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.028</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.022</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.028</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.027</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.036</td></tr>
    <tr><td headers=" " class="gt_row gt_left">F</td>
<td headers="Disc." class="gt_row gt_center">47.638</td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center">23.816</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">40.199</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">11.165</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">10.232</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">12.314</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">RMSE</td>
<td headers="Disc." class="gt_row gt_center">0.37</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. UI" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.24</td></tr>
  </tbody>
  <tfoot class="gt_sourcenotes">
    <tr>
      <td class="gt_sourcenote" colspan="13">+ p &lt; 0.1, * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</td>
    </tr>
  </tfoot>
  
</table>
</div><!--/html_preserve-->

###### W. Wage Level Control

<!--html_preserve--><div id="nncmtvpjxy" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#nncmtvpjxy table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#nncmtvpjxy thead, #nncmtvpjxy tbody, #nncmtvpjxy tfoot, #nncmtvpjxy tr, #nncmtvpjxy td, #nncmtvpjxy th {
  border-style: none;
}

#nncmtvpjxy p {
  margin: 0;
  padding: 0;
}

#nncmtvpjxy .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#nncmtvpjxy .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#nncmtvpjxy .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#nncmtvpjxy .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#nncmtvpjxy .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#nncmtvpjxy .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nncmtvpjxy .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#nncmtvpjxy .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#nncmtvpjxy .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#nncmtvpjxy .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#nncmtvpjxy .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#nncmtvpjxy .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#nncmtvpjxy .gt_spanner_row {
  border-bottom-style: hidden;
}

#nncmtvpjxy .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#nncmtvpjxy .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#nncmtvpjxy .gt_from_md > :first-child {
  margin-top: 0;
}

#nncmtvpjxy .gt_from_md > :last-child {
  margin-bottom: 0;
}

#nncmtvpjxy .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#nncmtvpjxy .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#nncmtvpjxy .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#nncmtvpjxy .gt_row_group_first td {
  border-top-width: 2px;
}

#nncmtvpjxy .gt_row_group_first th {
  border-top-width: 2px;
}

#nncmtvpjxy .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#nncmtvpjxy .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#nncmtvpjxy .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#nncmtvpjxy .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nncmtvpjxy .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#nncmtvpjxy .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#nncmtvpjxy .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#nncmtvpjxy .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#nncmtvpjxy .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#nncmtvpjxy .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#nncmtvpjxy .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#nncmtvpjxy .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#nncmtvpjxy .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#nncmtvpjxy .gt_left {
  text-align: left;
}

#nncmtvpjxy .gt_center {
  text-align: center;
}

#nncmtvpjxy .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#nncmtvpjxy .gt_font_normal {
  font-weight: normal;
}

#nncmtvpjxy .gt_font_bold {
  font-weight: bold;
}

#nncmtvpjxy .gt_font_italic {
  font-style: italic;
}

#nncmtvpjxy .gt_super {
  font-size: 65%;
}

#nncmtvpjxy .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#nncmtvpjxy .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#nncmtvpjxy .gt_indent_1 {
  text-indent: 5px;
}

#nncmtvpjxy .gt_indent_2 {
  text-indent: 10px;
}

#nncmtvpjxy .gt_indent_3 {
  text-indent: 15px;
}

#nncmtvpjxy .gt_indent_4 {
  text-indent: 20px;
}

#nncmtvpjxy .gt_indent_5 {
  text-indent: 25px;
}

#nncmtvpjxy .katex-display {
  display: inline-flex !important;
  margin-bottom: 0.75em !important;
}

#nncmtvpjxy div.Reactable > div.rt-table > div.rt-thead > div.rt-tr.rt-tr-group-header > div.rt-th-group:after {
  height: 0px !important;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <!--/html_preserve--><caption class='gt_caption'>Binned UE Duration w. Wage Level Control</caption><!--html_preserve-->
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="a-"> </th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.">Disc.</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-(clipped)">Disc. (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI">Disc. w. UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI-(clipped)">Disc. w. UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI">Disc. w. exhausted UI</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI-(clipped)">Disc. w. exhausted UI (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-controls">Disc. w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-controls-(clipped)">Disc. w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI-w.-controls">Disc. w. UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-UI-w.-controls-(clipped)">Disc. w. UI w. controls (clipped)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI-w.-controls">Disc. w. exhausted UI w. controls</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Disc.-w.-exhausted-UI-w.-controls-(clipped)">Disc. w. exhausted UI w. controls (clipped)</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers=" " class="gt_row gt_left">Intercept</td>
<td headers="Disc." class="gt_row gt_center">1.198***</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">1.139***</td>
<td headers="Disc. w. UI" class="gt_row gt_center">1.199***</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">1.139***</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">1.154***</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">1.098***</td>
<td headers="Disc. w. controls" class="gt_row gt_center">1.357***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">1.251***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">1.357***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">1.251***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">1.308***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">1.206***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center">(0.012)</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">(0.008)</td>
<td headers="Disc. w. UI" class="gt_row gt_center">(0.012)</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">(0.008)</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">(0.016)</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.032)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.033)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.035)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.024)</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="background-color: #ADD8E6;">Hourly Wage of Lost Job</td>
<td headers="Disc." class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Disc. (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Disc. w. UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Disc. w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center">(0.001)</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="background-color: #ADD8E6;">Unemployment Duration (Binned)</td>
<td headers="Disc." class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Disc. w. UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.009***</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.008***</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.005***</td>
<td headers="Disc. w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.011***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.008***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.010***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.008***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.007***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.005***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center">(0.002)</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.002)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Received Unemployment Compensation</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center">-0.000</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.000</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.000</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.000</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.001)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">Exhausted Unemployment Compensation</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.000***</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.000***</td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.000***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.000***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Female</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.028**</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.023**</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.028**</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.023**</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.028**</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.023**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Age</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.002***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.001***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.002***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.001***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.001***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.001***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.000)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">White</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.034</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.050**</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.034</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.050**</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.032</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.049**</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.023)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.016)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Black</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.057*</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.061***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.057*</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.061***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.054*</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.059***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.026)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.018)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Mixed</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.017</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.067*</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.017</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.067*</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.019</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.065*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.039)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.039)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.027)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.039)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.026)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Married</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.013</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.017*</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.013</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.017*</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.013</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.018*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.010)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.010)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.010)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.007)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Bachelor's Degree</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.079***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.095***</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.079***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.095***</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.078***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.094***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.022)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.015)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">High School</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.051**</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.008</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.051**</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.008</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.050**</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.008</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.017)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.011)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Less than HS</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">-0.085***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">-0.027+</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">-0.085***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">-0.027+</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">-0.088***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">-0.030*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">(0.021)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">(0.014)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Postgraduate Degree</td>
<td headers="Disc." class="gt_row gt_center"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.161***</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.093**</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.161***</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.093**</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.162***</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.094**</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc." class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Disc. w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.044)</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.030)</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.044)</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.030)</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.044)</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.030)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Num.Obs.</td>
<td headers="Disc." class="gt_row gt_center">4870</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. UI" class="gt_row gt_center">4870</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">4870</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. controls" class="gt_row gt_center">4870</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">4644</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">4870</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">4644</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2</td>
<td headers="Disc." class="gt_row gt_center">0.047</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">0.045</td>
<td headers="Disc. w. UI" class="gt_row gt_center">0.047</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.045</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.051</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.052</td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.069</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.072</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.069</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.072</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.072</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.078</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2 Adj.</td>
<td headers="Disc." class="gt_row gt_center">0.047</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">0.045</td>
<td headers="Disc. w. UI" class="gt_row gt_center">0.047</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.045</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.050</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.051</td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.067</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.070</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.067</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.070</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.070</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.076</td></tr>
    <tr><td headers=" " class="gt_row gt_left">F</td>
<td headers="Disc." class="gt_row gt_center">120.632</td>
<td headers="Disc. (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI" class="gt_row gt_center">80.422</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">86.995</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. controls" class="gt_row gt_center">30.090</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">27.774</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center"></td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">29.084</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">RMSE</td>
<td headers="Disc." class="gt_row gt_center">0.37</td>
<td headers="Disc. (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. UI" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. exhausted UI" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. exhausted UI (clipped)" class="gt_row gt_center">0.24</td>
<td headers="Disc. w. controls" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. controls (clipped)" class="gt_row gt_center">0.23</td>
<td headers="Disc. w. UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. UI w. controls (clipped)" class="gt_row gt_center">0.23</td>
<td headers="Disc. w. exhausted UI w. controls" class="gt_row gt_center">0.37</td>
<td headers="Disc. w. exhausted UI w. controls (clipped)" class="gt_row gt_center">0.23</td></tr>
  </tbody>
  <tfoot class="gt_sourcenotes">
    <tr>
      <td class="gt_sourcenote" colspan="13">+ p &lt; 0.1, * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</td>
    </tr>
  </tfoot>
  
</table>
</div><!--/html_preserve-->

#### Additional Considerations {.tabset}

Below I:

1.  Show results from some sample reweighting to address the non-uniformity of our cross-sectional data.
2.  Histogram of an additional explanatory variable that might be interesting - the tenure of the lost job. How long (in years) did the individual hold the job they lost.
3.  Additionally, I show some rough graphs/figures about the sample population (age, education, gender, wage distributions). I am still working on an occupational distribution graph to understand the "skills"/occupational distribution of the sample.

##### Selection Issues with Non-Random Sample {.tabset}

NOTE: Skip ahead to "Regression Results with Sample Reweighting for Regression Results if you don't wish to look at the reweighting details below.

One of the challenges with this data is that the sample grows significantly smaller for higher reported of unemployment duration (see scatter plots in Descriptives section).
One option is a sample reweighting (beyond the census weights) to ensure population similarity across bins (below I choose GLM propensity score matching & entropy-balancing) or a Heckman Selection.
Again, I include the code below (apologies for verbose output), mainly because I am not yet 100% sure of the implementation as I have never implemented such sample correction in a cross-sectional study).
Open to suggestions and corrections :)

**Conclusion:** With this implementation (which may very well be wrong for now!), the coefficients on unemployment duration remain stable.

###### Entropy Balancing

Entropy balancing simply reweights observations to ensure population matching across the key dependent variable.


``` r
# Apply entropy balancing using dwsuppwt sample weights
# Reweight according to observable characteristics using "ebalance" 
eb <- weightit(
  formula = dwmosun ~ female + age + white + black + mixed + marst + educ_cat,
  data = df,
  method = "ebalance",
  s.weights = df$dwsuppwt
)

# All covariates are balanced at the mean with tight threshold
bal.tab(eb, stats = c("m", "v"), thresholds = c(m = .001))
```

```
## Balance Measures
##                                 Type Diff.Target.Adj      M.Threshold
## female                        Binary          0.0000 Balanced, <0.001
## age                          Contin.         -0.0000 Balanced, <0.001
## white                         Binary         -0.0000 Balanced, <0.001
## black                         Binary         -0.0000 Balanced, <0.001
## mixed                         Binary          0.0001 Balanced, <0.001
## marst                         Binary         -0.0000 Balanced, <0.001
## educ_cat_Associate's          Binary         -0.0000 Balanced, <0.001
## educ_cat_Bachelor's           Binary         -0.0000 Balanced, <0.001
## educ_cat_HS Diploma           Binary         -0.0000 Balanced, <0.001
## educ_cat_Less than HS         Binary         -0.0000 Balanced, <0.001
## educ_cat_Postgraduate Degree  Binary          0.0001 Balanced, <0.001
## 
## Balance tally for target mean differences
##                      count
## Balanced, <0.001        11
## Not Balanced, >0.001     0
## 
## Variable with the greatest target mean difference
##  Variable Diff.Target.Adj      M.Threshold
##     mixed          0.0001 Balanced, <0.001
## 
## Effective sample sizes
##              Total
## Unadjusted 4747.86
## Adjusted   4634.14
```

``` r
# Add the new weights to the dataframe
df$eb_weight <- eb$weights

# Run weighted linear regression using entropy-balanced weights
mod_eb_reweight <- lm(
  formula = ratio_wage ~ dwmosun + female + age + white + black + mixed + marst + educ_cat,
  data = df,
  weights = eb_weight
)
```


```
## [1] "Diagnostic Tests for Entropy-balanced Reweighted Sample"
```

![plot of chunk unnamed-chunk-33](figure/unnamed-chunk-33-1.png)

###### Propensity Score Weighting with GLM Estimator


``` r
# More conventional propensity scoring with a GLM estimator
glm <- weightit(
  formula = dwmosun ~ female + age + white + black + mixed + marst + educ_cat,
  data = df,
  method = "glm",
  s.weights = df$dwsuppwt
)

# All covariates are balanced at the mean with less tight threshold (0.001, very few variables pass with glm esimator)
bal.tab(glm, stats = c("m", "v"), thresholds = c(m = .01))
```

```
## Balance Measures
##                                 Type Diff.Target.Adj     M.Threshold
## female                        Binary         -0.0032 Balanced, <0.01
## age                          Contin.         -0.0057 Balanced, <0.01
## white                         Binary          0.0022 Balanced, <0.01
## black                         Binary         -0.0005 Balanced, <0.01
## mixed                         Binary         -0.0041 Balanced, <0.01
## marst                         Binary          0.0041 Balanced, <0.01
## educ_cat_Associate's          Binary          0.0032 Balanced, <0.01
## educ_cat_Bachelor's           Binary          0.0023 Balanced, <0.01
## educ_cat_HS Diploma           Binary         -0.0003 Balanced, <0.01
## educ_cat_Less than HS         Binary         -0.0042 Balanced, <0.01
## educ_cat_Postgraduate Degree  Binary         -0.0017 Balanced, <0.01
## 
## Balance tally for target mean differences
##                     count
## Balanced, <0.01        11
## Not Balanced, >0.01     0
## 
## Variable with the greatest target mean difference
##  Variable Diff.Target.Adj     M.Threshold
##       age         -0.0057 Balanced, <0.01
## 
## Effective sample sizes
##              Total
## Unadjusted 4747.86
## Adjusted   4637.07
```

``` r
df$glm_weight <- glm$weights

mod_glm_reweight <- lm(
  formula = ratio_wage ~ dwmosun + female + age + white + black + mixed + marst + educ_cat,
  data = df,
  weights = glm_weight
)
```


```
## [1] "Diagnostic Tests for Propensity Score Matching (GLM) Reweighted Sample"
```

![plot of chunk unnamed-chunk-35](figure/unnamed-chunk-35-1.png)

###### Heckman Selection

Another option is a Heckman Selection correction though I do not think this addresses the particular selection concern we have where there are simply less observations in longer unemployment durations.


``` r
# Create selection indicator (1 if ratio_wage is observed)
df$observe_wage <- as.numeric(!is.na(df$ratio_wage))

# Define selection and outcome equations
selection_eq <- observe_wage ~ female + age + white + black + mixed + marst + educ_cat
outcome_eq   <- ratio_wage ~ dwmosun + female + age + white + black + mixed + marst + educ_cat

# Run Heckman 
heckman_model <- selection(
  selection = selection_eq,
  outcome = outcome_eq,
  data = df,
  method = "2step",       
  weights = df$dwsuppwt   # Include weights from CPS
)
```

###### Regression Results with Sample Reweighting

<!--html_preserve--><div id="kpiihoaepq" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#kpiihoaepq table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#kpiihoaepq thead, #kpiihoaepq tbody, #kpiihoaepq tfoot, #kpiihoaepq tr, #kpiihoaepq td, #kpiihoaepq th {
  border-style: none;
}

#kpiihoaepq p {
  margin: 0;
  padding: 0;
}

#kpiihoaepq .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#kpiihoaepq .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#kpiihoaepq .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#kpiihoaepq .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#kpiihoaepq .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#kpiihoaepq .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#kpiihoaepq .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#kpiihoaepq .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#kpiihoaepq .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#kpiihoaepq .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#kpiihoaepq .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#kpiihoaepq .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#kpiihoaepq .gt_spanner_row {
  border-bottom-style: hidden;
}

#kpiihoaepq .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#kpiihoaepq .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#kpiihoaepq .gt_from_md > :first-child {
  margin-top: 0;
}

#kpiihoaepq .gt_from_md > :last-child {
  margin-bottom: 0;
}

#kpiihoaepq .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#kpiihoaepq .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#kpiihoaepq .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#kpiihoaepq .gt_row_group_first td {
  border-top-width: 2px;
}

#kpiihoaepq .gt_row_group_first th {
  border-top-width: 2px;
}

#kpiihoaepq .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#kpiihoaepq .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#kpiihoaepq .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#kpiihoaepq .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#kpiihoaepq .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#kpiihoaepq .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#kpiihoaepq .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#kpiihoaepq .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#kpiihoaepq .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#kpiihoaepq .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#kpiihoaepq .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#kpiihoaepq .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#kpiihoaepq .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#kpiihoaepq .gt_left {
  text-align: left;
}

#kpiihoaepq .gt_center {
  text-align: center;
}

#kpiihoaepq .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#kpiihoaepq .gt_font_normal {
  font-weight: normal;
}

#kpiihoaepq .gt_font_bold {
  font-weight: bold;
}

#kpiihoaepq .gt_font_italic {
  font-style: italic;
}

#kpiihoaepq .gt_super {
  font-size: 65%;
}

#kpiihoaepq .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#kpiihoaepq .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#kpiihoaepq .gt_indent_1 {
  text-indent: 5px;
}

#kpiihoaepq .gt_indent_2 {
  text-indent: 10px;
}

#kpiihoaepq .gt_indent_3 {
  text-indent: 15px;
}

#kpiihoaepq .gt_indent_4 {
  text-indent: 20px;
}

#kpiihoaepq .gt_indent_5 {
  text-indent: 25px;
}

#kpiihoaepq .katex-display {
  display: inline-flex !important;
  margin-bottom: 0.75em !important;
}

#kpiihoaepq div.Reactable > div.rt-table > div.rt-thead > div.rt-tr.rt-tr-group-header > div.rt-th-group:after {
  height: 0px !important;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" scope="col" id="a-"> </th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Heckman-Correction">Heckman Correction</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="Entropy-Balanced-Reweight">Entropy Balanced Reweight</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_center" rowspan="1" colspan="1" scope="col" id="GLM-Reweight">GLM Reweight</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers=" " class="gt_row gt_left">Intercept</td>
<td headers="Heckman Correction" class="gt_row gt_center">1.053***</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">1.154***</td>
<td headers="GLM Reweight" class="gt_row gt_center">1.149***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.093)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.034)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.034)</td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="background-color: #ADD8E6;">Unemployment Duration (Months)</td>
<td headers="Heckman Correction" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td>
<td headers="GLM Reweight" class="gt_row gt_center" style="background-color: #ADD8E6;">-0.006***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.001)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.001)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.001)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Female</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.018</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.001</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.001</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.014)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.011)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.011)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Age</td>
<td headers="Heckman Correction" class="gt_row gt_center">-0.007***</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">-0.002***</td>
<td headers="GLM Reweight" class="gt_row gt_center">-0.002***</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.002)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.000)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.000)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">White</td>
<td headers="Heckman Correction" class="gt_row gt_center">-0.162*</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">-0.027</td>
<td headers="GLM Reweight" class="gt_row gt_center">-0.023</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.074)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.025)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.025)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Black</td>
<td headers="Heckman Correction" class="gt_row gt_center">-0.125*</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">-0.040</td>
<td headers="GLM Reweight" class="gt_row gt_center">-0.036</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.050)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.030)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.030)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Mixed</td>
<td headers="Heckman Correction" class="gt_row gt_center">-0.054</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.003</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.007</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.055)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.044)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.044)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Married</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.003</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.005</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.004</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.011)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.011)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.011)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Bachelor's Degree</td>
<td headers="Heckman Correction" class="gt_row gt_center">-0.139</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.047*</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.048*</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.105)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.023)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.023)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">High School</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.064</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">-0.021</td>
<td headers="GLM Reweight" class="gt_row gt_center">-0.020</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.053)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.017)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.017)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Less than HS</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.078</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">-0.007</td>
<td headers="GLM Reweight" class="gt_row gt_center">-0.006</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.064)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.022)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.022)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Postgraduate Degree</td>
<td headers="Heckman Correction" class="gt_row gt_center">-0.401</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.076</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.080+</td></tr>
    <tr><td headers=" " class="gt_row gt_left"></td>
<td headers="Heckman Correction" class="gt_row gt_center">(0.270)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">(0.048)</td>
<td headers="GLM Reweight" class="gt_row gt_center">(0.047)</td></tr>
    <tr><td headers=" " class="gt_row gt_left">Inverse Mills Ratio</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.870+</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center"></td>
<td headers="GLM Reweight" class="gt_row gt_center"></td></tr>
    <tr><td headers=" " class="gt_row gt_left" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="Heckman Correction" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;">(0.479)</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td>
<td headers="GLM Reweight" class="gt_row gt_center" style="border-bottom-width: 1px; border-bottom-style: solid; border-bottom-color: #000000;"></td></tr>
    <tr><td headers=" " class="gt_row gt_left">Num.Obs.</td>
<td headers="Heckman Correction" class="gt_row gt_center">4870</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">4870</td>
<td headers="GLM Reweight" class="gt_row gt_center">4870</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.893</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.014</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.015</td></tr>
    <tr><td headers=" " class="gt_row gt_left">R2 Adj.</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.893</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.012</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.013</td></tr>
    <tr><td headers=" " class="gt_row gt_left">F</td>
<td headers="Heckman Correction" class="gt_row gt_center"></td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">6.487</td>
<td headers="GLM Reweight" class="gt_row gt_center">6.798</td></tr>
    <tr><td headers=" " class="gt_row gt_left">RMSE</td>
<td headers="Heckman Correction" class="gt_row gt_center">0.37</td>
<td headers="Entropy Balanced Reweight" class="gt_row gt_center">0.37</td>
<td headers="GLM Reweight" class="gt_row gt_center">0.37</td></tr>
  </tbody>
  <tfoot class="gt_sourcenotes">
    <tr>
      <td class="gt_sourcenote" colspan="4">+ p &lt; 0.1, * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</td>
    </tr>
  </tfoot>
  
</table>
</div><!--/html_preserve-->

##### Job Tenure

We have information on the tenure spent at the last job which could impact the result.
This could speak to the "adaptability" of individuals.
Wage ratio seems to decrease (although not sure if meaningfully) with tenure at previous job.

![plot of chunk unnamed-chunk-38](figure/unnamed-chunk-38-1.png)

##### Representation

Although the survey does provide sample weights which we use above, it's still likely that those who are laid off might be systematically more susceptible to layoffs (lower-wage, low-skill occupation, male, etc).
Below, some (very rough) graphs to indicate what the sample looks like.

**Headline result:** it seems the sample over-represents below-mean wage earners and women.
Age looks reasonably accurate (in relation to a simple median though....have not checked spread).
Have not yet checked match to educational attainment.
Individuals with only a HS diploma is strong majority in sample - not sure how accurate this is (likely correlated with wage however...so this might be cause for concern and confirm a skewed sample in that sense).

If we wish to pursue this data, I could improve on the below but it will have to do for now.

![plot of chunk unnamed-chunk-39](figure/unnamed-chunk-39-1.png)

### OTJ Search {.tabset}

#### Eeckhout et al. 2019 Unemployment Cycles

[Source](https://www.aeaweb.org/articles?id=10.1257/mac.20180105)

![Employed Search Effort Fit](Eeckhout_Replication/emp_search_effort.png)



```
##  [1] 0.2000000 0.2315789 0.2631579 0.2947368 0.3263158 0.3578947 0.3894737 0.4210526 0.4526316 0.4842105 0.5157895 0.5473684 0.5789474 0.6105263 0.6421053 0.6736842
## [17] 0.7052632 0.7368421 0.7684211 0.8000000
##  [1] 0.2000000 0.2315789 0.2631579 0.2947368 0.3263158 0.3578947 0.3894737 0.4210526 0.4526316 0.4842105 0.5157895 0.5473684 0.5789474 0.6105263 0.6421053 0.6736842
## [17] 0.7052632 0.7368421 0.7684211 0.8000000
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png)![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-2.png)![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-3.png)![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-4.png)

```
## 
## Call:
## lm(formula = as.formula(forms[which(names(forms) == form)]))
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.014882 -0.006066 -0.003639  0.007309  0.026123 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -0.17447    0.03519  -4.958 3.19e-06 ***
## x            0.23294    0.03499   6.656 1.93e-09 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.01036 on 93 degrees of freedom
## Multiple R-squared:  0.3227,	Adjusted R-squared:  0.3154 
## F-statistic: 44.31 on 1 and 93 DF,  p-value: 1.925e-09
## 
## 
## Call:
## lm(formula = as.formula(forms[which(names(forms) == form)]))
## 
## Residuals:
##        Min         1Q     Median         3Q        Max 
## -0.0102059 -0.0031620 -0.0001317  0.0039334  0.0079867 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -5.268e-02  1.773e-02  -2.970  0.00379 ** 
## x            1.285e-01  1.731e-02   7.423 5.61e-11 ***
## trend       -3.507e-04  1.918e-05 -18.285  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.004839 on 92 degrees of freedom
## Multiple R-squared:  0.8538,	Adjusted R-squared:  0.8507 
## F-statistic: 268.7 on 2 and 92 DF,  p-value: < 2.2e-16
## 
## 
## Call:
## lm(formula = as.formula(forms[which(names(forms) == form)]))
## 
## Residuals:
##        Min         1Q     Median         3Q        Max 
## -0.0068610 -0.0016116  0.0001739  0.0018603  0.0046844 
## 
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -0.049270   0.008149  -6.046 3.05e-08 ***
## x            0.049021   0.008104   6.049 3.02e-08 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.002399 on 93 degrees of freedom
## Multiple R-squared:  0.2824,	Adjusted R-squared:  0.2747 
## F-statistic: 36.59 on 1 and 93 DF,  p-value: 3.018e-08
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-5.png)

### Learning Rate {.tabset}

#### Mueller et al. Job Seekers' Perceptions and Employment Prospects: Heterogeneity, Duration Dependence and Bias

[Mueller et al: Job Seekers' Perceptions and Employment
Prospects](https://www.aeaweb.org/articles?id=10.1257/aer.20190808)

*The authors claim to disentangle the effects of duration dependence and
dynamic selection by using job seekers' elicited beliefs about
job-finding. Assuming (and confirming empirically) that job-seekers have
realistic initial beliefs about job-finding they isolate the
heterogeneity in jobseekers from true duration dependence. Ultimately,
they find that dynamic selection selection explains most of the negative
duration dependence (rather than pure, true duration dependence).*

*Findings: Results are remarkably consistent even when including
additional data from 2019-2024.*

We aim to include this information in our theoretical model of the job
search effort as a learning rate (ie. individuals learn about their
re-employment probability with repeated failures in the job search).


```
## 
## Descriptive Statistics (SCE)
## ===============================================================
## Variable                          Orig. 2013-19 2013-24 2020-24
## ---------------------------------------------------------------
## High-School Degree or Less            44.5       40.6    36.9  
## Some College Education                32.4       34.9    37.6  
## College Degree or More                23.1       24.6    25.6  
## Age 20-34                             25.4       27.2    30.0  
## Age 35-49                             33.5       33.6    35.3  
## Age 50-65                             41.1       39.2    34.8  
## Female                                59.3       61.2    60.8  
## Black                                 19.1       17.9    16.4  
## Hispanic                              12.5       13.0    12.6  
## UE transition rate                    18.7       19.1    18.2  
## UE transition rate: ST                25.8       26.5    24.3  
## UE transition rate: LT                12.7       12.7    12.3  
## # respondents                          948       1,367    433  
## # respondents w/ at least 2 u obs      534        780     252  
## # observations                        2,597      3,926   1,347 
## ---------------------------------------------------------------
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png)![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-2.png)

```
## [1] "Table 2—Regressions of Realized on Elicited 3-Month Job-Finding Probabilities (SCE)"
## [1] "Panel A. Contemporaneous elicitations"
```

```
## 
## ========================================================================
##                                     Dependent variable:                 
##                     ----------------------------------------------------
##                                T+3 UE Transitions (3-Months)            
##                       Orig. 2013-19        2013-24          2020-24     
##                            (1)               (2)              (3)       
## ------------------------------------------------------------------------
## find_job_3mon           0.464***          0.396***          0.265***    
##                          (0.045)           (0.036)          (0.067)     
##                                                                         
## 1 | userid                                                              
##                                                                         
##                                                                         
## Constant                 -0.104            -0.080            -0.136     
##                          (0.169)           (0.137)          (0.267)     
##                                                                         
## ------------------------------------------------------------------------
## Observations              1,201             1,911             673       
## R2                        0.218             0.139            0.105      
## Adjusted R2               0.207             0.132            0.083      
## Residual Std. Error 0.467 (df = 1184) 0.475 (df = 1894) 0.478 (df = 656)
## ========================================================================
## Note:                                        *p<0.1; **p<0.05; ***p<0.01
```

```
## 
## ==========================================================================
##                                       Dependent variable:                 
##                       ----------------------------------------------------
##                                  T+3 UE Transitions (3-Months)            
##                         Orig. 2013-19        2013-24          2020-24     
##                              (1)               (2)              (3)       
## --------------------------------------------------------------------------
## find_job_3mon             0.501***          0.418***          0.391***    
##                            (0.061)           (0.051)          (0.094)     
##                                                                           
## findjob_3mon_longterm     -0.258***         -0.170**         -0.360***    
##                            (0.088)           (0.071)          (0.133)     
##                                                                           
## longterm_unemployed        -0.078           -0.127***          -0.043     
##                            (0.051)           (0.041)          (0.075)     
##                                                                           
## 1 | userid                                                                
##                                                                           
##                                                                           
## Constant                   -0.062            -0.063            -0.402     
##                            (0.175)           (0.139)          (0.266)     
##                                                                           
## --------------------------------------------------------------------------
## Observations                1,201             1,911             673       
## R2                          0.259             0.182            0.155      
## Adjusted R2                 0.248             0.174            0.132      
## Residual Std. Error   0.455 (df = 1182) 0.464 (df = 1892) 0.465 (df = 654)
## ==========================================================================
## Note:                                          *p<0.1; **p<0.05; ***p<0.01
## [1] "Panel B. Lagged elicitations"
```

```
## 
## ======================================================================
##                                    Dependent variable:                
##                     --------------------------------------------------
##                               T+3 UE Transitions (3-Months)           
##                      Orig. 2013-19       2013-24          2020-24     
##                           (1)              (2)              (3)       
## ----------------------------------------------------------------------
## tplus3_percep_3mon      0.332***         0.241***         0.203**     
##                         (0.067)          (0.056)          (0.102)     
##                                                                       
## 1 | userid                                                            
##                                                                       
##                                                                       
## Constant                 0.304           0.490**           0.451      
##                         (0.270)          (0.207)          (0.394)     
##                                                                       
## ----------------------------------------------------------------------
## Observations              474              798              300       
## R2                       0.168            0.090            0.179      
## Adjusted R2              0.139            0.071            0.132      
## Residual Std. Error 0.398 (df = 457) 0.436 (df = 781) 0.447 (df = 283)
## ======================================================================
## Note:                                      *p<0.1; **p<0.05; ***p<0.01
```

```
## 
## ======================================================================
##                                    Dependent variable:                
##                     --------------------------------------------------
##                               T+3 UE Transitions (3-Months)           
##                      Orig. 2013-19       2013-24          2020-24     
##                           (1)              (2)              (3)       
## ----------------------------------------------------------------------
## find_job_3mon           0.301***         0.205***          -0.035     
##                         (0.069)          (0.058)          (0.110)     
##                                                                       
## 1 | userid                                                            
##                                                                       
##                                                                       
## Constant                 0.201           0.422**           0.361      
##                         (0.274)          (0.207)          (0.400)     
##                                                                       
## ----------------------------------------------------------------------
## Observations              474              798              300       
## R2                       0.159            0.083            0.168      
## Adjusted R2              0.129            0.064            0.121      
## Residual Std. Error 0.400 (df = 457) 0.437 (df = 781) 0.450 (df = 283)
## ======================================================================
## Note:                                      *p<0.1; **p<0.05; ***p<0.01
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-3.png)![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-4.png)

```
## [1] "Table 4—Linear Regressions of Elicited Job-Finding Probabilities on Duration of Unemployment"
## +--------------------------------+----------+----------+----------+----------+
## |                                | (1)      | (2)      | (3)      | (4)      |
## +================================+==========+==========+==========+==========+
## | Orig. 2013-19                                                              |
## +--------------------------------+----------+----------+----------+----------+
## | Unemployment Duration (Months) | -0.0057  | -0.0050  | -0.0043  | 0.0022   |
## +--------------------------------+----------+----------+----------+----------+
## |                                | (0.0007) | (0.0007) | (0.0006) | (0.0049) |
## +--------------------------------+----------+----------+----------+----------+
## | Num.Obs.                       | 882      | 2281     | 2281     | 2281     |
## +--------------------------------+----------+----------+----------+----------+
## | R2                             | 0.110    | 0.090    | 0.155    | 0.824    |
## +--------------------------------+----------+----------+----------+----------+
## | 2013-24                                                                    |
## +--------------------------------+----------+----------+----------+----------+
## | Unemployment Duration (Months) | -0.0050  | -0.0048  | -0.0042  | -0.0026  |
## +--------------------------------+----------+----------+----------+----------+
## |                                | (0.0006) | (0.0006) | (0.0005) | (0.0034) |
## +--------------------------------+----------+----------+----------+----------+
## | Num.Obs.                       | 1265     | 3423     | 3399     | 3423     |
## +--------------------------------+----------+----------+----------+----------+
## | R2                             | 0.067    | 0.065    | 0.109    | 0.817    |
## +--------------------------------+----------+----------+----------+----------+
## | 2020-24                                                                    |
## +--------------------------------+----------+----------+----------+----------+
## | Unemployment Duration (Months) | -0.0011  | -0.0035  | -0.0039  | -0.0077  |
## +--------------------------------+----------+----------+----------+----------+
## |                                | (0.0013) | (0.0012) | (0.0013) | (0.0036) |
## +--------------------------------+----------+----------+----------+----------+
## | Num.Obs.                       | 395      | 1150     | 1140     | 1150     |
## +--------------------------------+----------+----------+----------+----------+
## | R2                             | 0.002    | 0.019    | 0.118    | 0.838    |
## +================================+==========+==========+==========+==========+
## | Standard errors are clustered at the user or spell level as indicated.     |
## +================================+==========+==========+==========+==========+
## Table: Table 4 - Panel A: Linear Regressions of Elicited Job-Finding Probabilities on Duration of Unemployment (SCE)
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-5.png)![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-6.png)![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-7.png)

## Discarded Analyses {.tabset}

<!-- I first focus on replicating and extending the empirical analysis that speaks to the three determinants above (dynamic selection, duration dependence, search effort) -->

1.  (Replicated with additional data for unemployed jobseekers)
    [**Mukoyama et al. 2018: Job Search Over the Business
    Cycle**](https://www.aeaweb.org/articles?id=10.1257/mac.20160202)

*They provide a novel measure of job search effort exploiting the
American Time Use and Current Population Surveys which can be reduced to
just the intensive margin (changes in search effort by worker!). At the
moment, I think this will be the most useful input for our model.*

*Abstract: We examine the cyclicality of search effort using
time-series, cross-state, and individual variation and find that it is
countercyclical. We then set up a search and matching model with
endogenous search effort and show that search effort does not amplify
labor market fluctuations but rather dampens them. Lastly, we examine
the role of search effort in driving recent unemployment dynamics and
show that the unemployment rate would have been 0.5 to 1 percentage
points higher in the 2008–2014 period had search effort not increased.*

4.  **Survey of Consumer Expectations Reservation Wages, Accepted Wages,
    and Wage Expectations** The data is unfortunately sparse and linking
    outcomes to reservation wages is difficult. However, in a
    cross-sectional setting we are able to deduce some weak
    relationships between Unemployment Duration and Absolute Reservation
    Wages and Wage Expectations.\*

<!-- 3. (Forthcoming) **[Kroft et al. 2016: Long-Term Unemployment and the Great Recession: The Role of Composition, Duration Dependence, and Nonparticipation](https://www.jstor.org/stable/26588430)** -->

<!-- *Abstract: We explore the role of composition, duration dependence, and labor force nonparticipation in accounting for the sharp increase in the incidence of long-term unemployment (LTU) during the Great Recession. We show that compositional shifts account for very little of the observed increase in LTU. Using panel data from the Current Population Survey for 2002–7, we calibrate a matching model that allows for duration dependence in unemployment and transitions between employment, unemployment, and nonpartici- pation. The calibrated model accounts for almost all of the increase in LTU and much of the observed outward shift in the Beveridge curve between 2008 and 2013.* -->

### Reservation Wages

Exploring the effect of unemployment duration on reservation wages,
accepted wages, and expected wage offers.

**Survey of Consumer Expectations Reservation Wages, Accepted Wages, and
Wage Expectations** (2014-2022) *The data is unfortunately sparse and
linking outcomes to reservation wages is difficult. However, in a
cross-sectional setting we are able to deduce some weak relationships
between Unemployment Duration and Absolute Reservation Wages and Wage
Expectations.*


``` r
source(here("data/behav_params/SCE Labour Market Survey/sce_res_wage_analysis.R"))
```

```
## [1] "Plots of RESERVATION WAGE versus latest, current wage"
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-1.png)![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-2.png)

```
## [1] "Plots of EXPECTED OFFER versus latest, current, reservation wage"
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-3.png)

```
## [1] "Plots of ACCEPTED SALARY versus latest, current, reservation wage"
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-4.png)![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10-5.png)

```
## 
## +-------------+--------------+---------------+---------------+-----------------------+------------------+--------------------------+
## |             | Accpt:Latest | AccptWage w.c | Accpt:ResWage | AccptWage:ResWage w.c | Accpt:EffResWage | AccptWage:EffResWage w.c |
## +=============+==============+===============+===============+=======================+==================+==========================+
## | (Intercept) | 0.826***     | 1.743***      | 0.933***      | 1.199***              | 0.826***         | 1.002***                 |
## +-------------+--------------+---------------+---------------+-----------------------+------------------+--------------------------+
## |             | (0.108)      | (0.260)       | (0.045)       | (0.141)               | (0.051)          | (0.177)                  |
## +-------------+--------------+---------------+---------------+-----------------------+------------------+--------------------------+
## | udur_bins   | 0.050        | -0.005        | -0.048*       | -0.053**              | 0.008            | -0.001                   |
## +-------------+--------------+---------------+---------------+-----------------------+------------------+--------------------------+
## |             | (0.045)      | (0.048)       | (0.019)       | (0.020)               | (0.023)          | (0.024)                  |
## +-------------+--------------+---------------+---------------+-----------------------+------------------+--------------------------+
## | Num.Obs.    | 56           | 56            | 160           | 159                   | 184              | 183                      |
## +-------------+--------------+---------------+---------------+-----------------------+------------------+--------------------------+
## | R2          | 0.022        | 0.430         | 0.040         | 0.118                 | 0.001            | 0.042                    |
## +-------------+--------------+---------------+---------------+-----------------------+------------------+--------------------------+
## | RMSE        | 0.40         | 0.35          | 0.30          | 0.30                  | 0.34             | 0.34                     |
## +=============+==============+===============+===============+=======================+==================+==========================+
## | + p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001                                                                                |
## +=============+==============+===============+===============+=======================+==================+==========================+
## Table: Accepted Wages and Unemployment Duration 
## 
## +-------------+------------+-------------+------------------+----------------------+
## |             | ResWage    | ResWage w.c | ResWage/LastWage | ResWage/LastWage w.c |
## +=============+============+=============+==================+======================+
## | (Intercept) | 10.173***  | 9.945***    | 0.825***         | 0.750***             |
## +-------------+------------+-------------+------------------+----------------------+
## |             | (0.044)    | (0.071)     | (0.022)          | (0.040)              |
## +-------------+------------+-------------+------------------+----------------------+
## | udur_bins   | 0.107***   | 0.083***    | 0.027***         | 0.023***             |
## +-------------+------------+-------------+------------------+----------------------+
## |             | (0.012)    | (0.011)     | (0.006)          | (0.006)              |
## +-------------+------------+-------------+------------------+----------------------+
## | female      |            | -0.275***   |                  | 0.009                |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.022)     |                  | (0.012)              |
## +-------------+------------+-------------+------------------+----------------------+
## | age         |            | 0.005***    |                  | 0.001*               |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.001)     |                  | (0.000)              |
## +-------------+------------+-------------+------------------+----------------------+
## | hhinc_2     |            | 0.230***    |                  | -0.008               |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.026)     |                  | (0.014)              |
## +-------------+------------+-------------+------------------+----------------------+
## | hhinc_3     |            | 0.427***    |                  | -0.017               |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.030)     |                  | (0.017)              |
## +-------------+------------+-------------+------------------+----------------------+
## | hhinc_4     |            | 0.759***    |                  | -0.008               |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.033)     |                  | (0.019)              |
## +-------------+------------+-------------+------------------+----------------------+
## | education_2 |            | -0.247***   |                  | 0.050+               |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.045)     |                  | (0.026)              |
## +-------------+------------+-------------+------------------+----------------------+
## | education_3 |            | -0.122**    |                  | 0.007                |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.047)     |                  | (0.027)              |
## +-------------+------------+-------------+------------------+----------------------+
## | education_4 |            | -0.046      |                  | 0.052+               |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.051)     |                  | (0.029)              |
## +-------------+------------+-------------+------------------+----------------------+
## | education_5 |            | 0.027       |                  | 0.008                |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.049)     |                  | (0.028)              |
## +-------------+------------+-------------+------------------+----------------------+
## | education_6 |            | 0.111*      |                  | 0.054+               |
## +-------------+------------+-------------+------------------+----------------------+
## |             |            | (0.054)     |                  | (0.031)              |
## +-------------+------------+-------------+------------------+----------------------+
## | Num.Obs.    | 7937       | 7824        | 6294             | 6224                 |
## +-------------+------------+-------------+------------------+----------------------+
## | R2          | 0.010      | 0.169       | 0.003            | 0.007                |
## +-------------+------------+-------------+------------------+----------------------+
## | R2 Adj.     | 0.010      | 0.168       | 0.003            | 0.005                |
## +-------------+------------+-------------+------------------+----------------------+
## | AIC         | 191435.4   | 187281.4    | 9054.4           | 8961.7               |
## +-------------+------------+-------------+------------------+----------------------+
## | BIC         | 191456.4   | 187372.0    | 9074.6           | 9049.3               |
## +-------------+------------+-------------+------------------+----------------------+
## | Log.Lik.    | -11923.451 | -11075.843  | -4524.195        | -4467.857            |
## +-------------+------------+-------------+------------------+----------------------+
## | RMSE        | 0.98       | 0.90        | 0.44             | 0.44                 |
## +=============+============+=============+==================+======================+
## | + p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001                                |
## +=============+============+=============+==================+======================+
## Table: Reservation Wages and Unemployment Duration 
## 
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             | AccptWage | AccptWage w.c | AccptWage/ResWage | AccptWage/ResWage w.c |
## +=============+===========+===============+===================+=======================+
## | (Intercept) | 10.568*** | 11.705***     | 0.924***          | 1.303***              |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             | (0.106)   | (0.255)       | (0.048)           | (0.132)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | udur_bins   | -0.006    | -0.037        | -0.031+           | -0.036+               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             | (0.040)   | (0.037)       | (0.018)           | (0.018)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | female      |           | -0.164        |                   | -0.073                |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.102)       |                   | (0.050)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | age         |           | -0.008*       |                   | -0.005**              |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.004)       |                   | (0.002)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | hhinc_2     |           | 0.260+        |                   | 0.043                 |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.139)       |                   | (0.067)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | hhinc_3     |           | 0.272+        |                   | 0.042                 |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.138)       |                   | (0.069)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | hhinc_4     |           | 0.377*        |                   | -0.052                |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.150)       |                   | (0.075)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | education_2 |           | -0.996***     |                   | -0.043                |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.224)       |                   | (0.122)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | education_3 |           | -0.940***     |                   | -0.128                |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.223)       |                   | (0.122)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | education_4 |           | -1.036***     |                   | -0.176                |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.226)       |                   | (0.123)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | education_5 |           | -0.827***     |                   | -0.141                |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.224)       |                   | (0.124)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | education_6 |           | -0.551*       |                   | -0.095                |
## +-------------+-----------+---------------+-------------------+-----------------------+
## |             |           | (0.228)       |                   | (0.127)               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | Num.Obs.    | 127       | 126           | 164               | 163                   |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | R2          | 0.000     | 0.299         | 0.017             | 0.133                 |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | R2 Adj.     | -0.008    | 0.232         | 0.011             | 0.070                 |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | AIC         | 2933.2    | 2884.9        | 110.6             | 109.9                 |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | BIC         | 2941.7    | 2921.7        | 119.9             | 150.1                 |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | Log.Lik.    | -123.204  | -99.911       | -52.283           | -41.957               |
## +-------------+-----------+---------------+-------------------+-----------------------+
## | RMSE        | 0.58      | 0.53          | 0.32              | 0.32                  |
## +=============+===========+===============+===================+=======================+
## | + p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001                                   |
## +=============+===========+===============+===================+=======================+
## Table: Accepted Wages and Unemployment Duration 
## 
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             | ExpWage/ResWage | ExpWage/ResWage w.c | ExpWage/LastWage | ExpWage/LastWage w.c |
## +=============+=================+=====================+==================+======================+
## | (Intercept) | 1.057***        | 1.226***            | 1.087***         | 1.257***             |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             | (0.020)         | (0.040)             | (0.029)          | (0.059)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | udur_bins   | -0.022***       | -0.009              | -0.024**         | -0.008               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             | (0.006)         | (0.006)             | (0.008)          | (0.009)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | female      |                 | -0.022+             |                  | 0.064***             |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.013)             |                  | (0.019)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | age         |                 | -0.003***           |                  | -0.004***            |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.000)             |                  | (0.001)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | hhinc_2     |                 | 0.004               |                  | -0.038               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.016)             |                  | (0.023)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | hhinc_3     |                 | 0.004               |                  | -0.001               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.018)             |                  | (0.026)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | hhinc_4     |                 | 0.000               |                  | -0.005               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.019)             |                  | (0.027)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | education_2 |                 | -0.035              |                  | -0.032               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.027)             |                  | (0.040)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | education_3 |                 | -0.008              |                  | -0.056               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.028)             |                  | (0.041)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | education_4 |                 | 0.004               |                  | -0.031               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.030)             |                  | (0.044)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | education_5 |                 | 0.011               |                  | -0.090*              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.029)             |                  | (0.042)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | education_6 |                 | 0.021               |                  | 0.002                |
## +-------------+-----------------+---------------------+------------------+----------------------+
## |             |                 | (0.032)             |                  | (0.046)              |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | Num.Obs.    | 3114            | 3070                | 2721             | 2690                 |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | R2          | 0.005           | 0.028               | 0.003            | 0.029                |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | R2 Adj.     | 0.005           | 0.024               | 0.003            | 0.025                |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | AIC         | 2803.9          | 2733.2              | 4079.4           | 3986.5               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | BIC         | 2822.1          | 2811.6              | 4097.2           | 4063.1               |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | Log.Lik.    | -1398.968       | -1353.588           | -2036.722        | -1980.241            |
## +-------------+-----------------+---------------------+------------------+----------------------+
## | RMSE        | 0.34            | 0.34                | 0.46             | 0.45                 |
## +=============+=================+=====================+==================+======================+
## | + p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001                                             |
## +=============+=================+=====================+==================+======================+
## Table: Expected Wages and Unemployment Duration
```

<!-- ## Kroft et al. Long-Term Unemployment and the Great Recession: The Role of Composition, Duration Dependence, and Nonparticipation -->

<!-- [Kroft et al. Long-Term Unemployment and the Great Recession](https://www.jstor.org/stable/26588430) -->

<!-- ```{r, echo = FALSE} -->

<!-- ``` -->

<!-- ## Own work on microdata... {.tabset} -->

### Survey on Consumer Expectations - Job Search Supplement

The Federal Reserve Bank of New York compiles the nationally
representative Survey on Consumer Expectations annually in October.
Since 2013, they have run a Job Search Supplement which includes
questions on the time spent searching for work, and unemployment
duration. The job search supplement has plenty more questions that we
can look at incorporating, listed
[here](https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/SCE-Labor-Market-Survey-Data-Codebook.pdf?sc_lang=en).
For now, I plot the relationship between time spent searching and time
out of work. The table below also indicates the number of people
unemployed in the dataset and the number of people unemployed and
searching.




| Year| N Unemployed| N Unemp & Searching|
|----:|------------:|-------------------:|
| 2014|          383|                  70|
| 2015|          321|                  44|
| 2016|          339|                  46|
| 2017|          350|                  38|
| 2018|          354|                  41|
| 2019|          343|                  32|
| 2020|          304|                  45|
| 2021|          330|                  50|

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png)![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-2.png)

### On the job search

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-13-1.png)

### (TBD) American Time Use Survey

The American Time Use Survey gives no indication of time spent in
unemployment. It shows how much time is spent searching but does not
link to time spent in unemployment. Therefore, I prioritised the
datasets above. [Krueger & Mueller
2010](https://www.sciencedirect.com/science/article/abs/pii/S0047272709001625)
impute duration spent unemployed from the ATUS in the following way
which could be worth considering.

"Unfortunately, the ATUS interview does not collect information on
unemployment duration. Consequently, we derive unemployment duration by
taking the unemployment duration reported in the last CPS interview and
adding the number of weeks that elapsed between the CPS interview and
the ATUS interview. Eighty-six percent of the ATUS interviews were
conducted within 3 months of the last CPS interview. For those who were
not unemployed at the time of the CPS interview, we impute duration of
unemployment by taking half the number of weeks between the CPS and the
ATUS interviews. We do not show the weekly LOWESS plot for 13 weeks or
less, but simply report the average time allocated to search, as the
imputed unemployment duration are quite noisy for those who become
unemployed after their last CPS interview."

<!-- ### Occupational Restructuring -->

<!-- ```{r, cache = TRUE} -->

<!-- #read.xls(here("data/macro_vars/OEWS/") -->

<!-- ``` -->
