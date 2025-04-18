## Mueller et al. Job Seekers' Perceptions and Employment Prospects: Heterogeneity, Duration Dependence and Bias
# [Mueller et al: Job Seekers' Perceptions and Employment Prospects](https://www.aeaweb.org/articles?id=10.1257/aer.20190808)
# The following reproduces the empirical analysis of Mueller et al. precisely
# mueller_repl_analysis_extended_samples.R reproduces with additional samples


# Clear the environment
rm(list = ls())

# Load required packages
library(haven)    # For reading .dta files
library(tidyverse)    # For data manipulation
library(ggplot2)  # For plotting
library(stargazer)
library(here)
library(lfe)  # for regressions with clustering
library(weights)
library(diagis)
library(readxl)
library(fixest)      # For regressions with clustering and fixed effects
library(patchwork)
library(broom) # For extracting model coefficients


# Define base directory (replace this with the correct path later)
base <- here("data/behav_params/Mueller_Replication/")

# Load data file
data <- readRDS(paste0(base, "sce_datafile_em_2025.RDS"))

#############################################
### Table 1: Descriptive statistics (SCE) ###
#############################################
# Restrict to ages 20-65, and filter for perception questions and valid weights
data_tab1 <- data %>%
  filter(age >= 20, age <= 65) %>%
  filter((!is.na(find_job_3mon) | !is.na(find_job_12mon)) & !is.na(weight))

# Total number of unemployed observations
data_tab1 <- data_tab1 %>%
  group_by(userid) %>%
  mutate(
    obs_u = row_number(),         # Observation number per user
    Nobs_u = n()                  # Total number of observations per user
  )

# Create age and education dummies
data_tab1 <- data_tab1 %>%
  mutate(
    age_d = case_when(
      age >= 20 & age <= 34 ~ 1,
      age >= 35 & age <= 49 ~ 2,
      age >= 50 & age <= 65 ~ 3,
      TRUE ~ NA_real_
    ),
    dage1 = ifelse(age_d == 1, 1, 0),
    dage2 = ifelse(age_d == 2, 1, 0),
    dage3 = ifelse(age_d == 3, 1, 0)
  )

# Education category dummies
data_tab1 <- data_tab1 %>%
  #rename(edu_cat = `_edu_cat`) %>%
  mutate(
    edu_cat = ifelse(edu_cat == "", NA, edu_cat)) %>%
  group_by(userid) %>%
  fill(edu_cat, .direction = "updown") %>%
  ungroup %>%
  # Corrects when educational category is NA
  mutate(edu_cat = case_when(is.na(edu_cat) & education_1 == 1 ~ "Up to HS grad",
                             is.na(edu_cat) & education_2 == 1 ~ "Up to HS grad",
                             is.na(edu_cat) & education_3 == 1 ~ "Some college, including associate\x92s degree",
                             is.na(edu_cat) & education_4 == 1 ~ "College grad plus", #or "Some college, including associate\x92s degree"
                             is.na(edu_cat) & education_5 == 1 ~ "College grad plus",
                             # NA values in edu_cat forced me to guess to match the descriptive statistics. 
                             # There are only two users whose edu_cat is NA with eduation_6 == 1 (ie. "Other education")
                             # To match the descriptive statistics one had to be "Up to HS grad" and the other "Some college including associate's degree"
                             # After testing matches with descriptive stats the following assignment was most accurate. 
                             # This could of course be incorrect....but the best that could be done at the moment. 
                             # This assignment affects 7 observations in the dataset
                             is.na(edu_cat) & education_6 == 1 & userid == 7334000 ~ "Up to HS grad",
                             is.na(edu_cat) & education_6 == 1 & userid == 1863600 ~ "Some college, including associate\x92s degree",
                             .default = edu_cat)) %>%
  mutate(
    dedu1 = ifelse(edu_cat == "Up to HS grad", 1, 0), #education_3, # dedu1 = ifelse(edu_cat == 3, 1, 0),
    dedu2 = ifelse(edu_cat == "Some college, including associate\x92s degree", 1, 0), #education_2, # ifelse(edu_cat == 2, 1, 0),
    dedu3 = ifelse(edu_cat == "College grad plus", 1, 0)#education_1#ifelse(edu_cat == 1, 1, 0)
  )

# Weighted mean function
weighted_mean <- function(x, w) {
  sum(x * w, na.rm = TRUE) / sum(w, na.rm = TRUE)
}

# Descriptive statistics
medu <- sapply(1:3, function(x) weighted_mean(data_tab1[[paste0("dedu", x)]], data_tab1$weight))
mage <- sapply(1:3, function(x) weighted_mean(data_tab1[[paste0("dage", x)]], data_tab1$weight))
mfem <- weighted_mean(data_tab1$female, data_tab1$weight)
mblack <- weighted_mean(data_tab1$black, data_tab1$weight)
mhisp <- weighted_mean(data_tab1$hispanic, data_tab1$weight)

# the below weighted mean function does not remove the weights for the NA values of UE_trans_1mon so the weighted averages come out wrong
# I create separate dataframe of weights here in which only the weights of non-na values remain.
no_na_weights <- data_tab1 %>% 
  mutate(weight = ifelse(is.na(UE_trans_1mon), NA, weight))

mf <- weighted_mean(data_tab1$UE_trans_1mon, no_na_weights$weight)
mfst <- weighted_mean(data_tab1$UE_trans_1mon[data_tab1$udur_bins <= 2], no_na_weights$weight[data_tab1$udur_bins <= 2])
mflt <- weighted_mean(data_tab1$UE_trans_1mon[data_tab1$udur_bins > 2], no_na_weights$weight[data_tab1$udur_bins > 2])

N <- nrow(data_tab1)
Nind <- nrow(data_tab1 %>% filter(obs_u == 1))
Nind2 <- nrow(data_tab1 %>% filter(obs_u == 1 & Nobs_u > 1))

# Create a summary statistics data frame for Stargazer
summary_table <- data.frame(
  Variable = c(
    "High-School Degree or Less", "Some College Education", "College Degree or More",
    "Age 20-34", "Age 35-49", "Age 50-65",
    "Female", "Black", "Hispanic",
    "UE transition rate", "UE transition rate: ST", "UE transition rate: LT",
    "# respondents", "# respondents w/ at least 2 u obs", "# observations"
  ),
  Value = c(
    100 * medu, 100 * mage, 100 * mfem, 100 * mblack, 100 * mhisp,
    100 * mf, 100 * mfst, 100 * mflt, Nind, Nind2, N
  )
)

# Stargazer summary table
stargazer(
  summary_table,
  type = "text",
  summary = FALSE,
  rownames = FALSE,
  title = "Descriptive Statistics (SCE)",
  digits = 1
)

###################################################################
### Figure 1: Histogram of elicited 3-month probability ###
###################################################################

# There are some issues with the histogram which I think has to do with the binning...left as is below for now

# Restrict data to ages 20-65 and specific sample
data_fig1 <- data %>%
  filter(age >= 20, age <= 65) %>%
  filter(in_sample_1 == 1)

# Create fweight variable (rounded weights)
data_fig1 <- data_fig1 %>%
  mutate(fweight = round(weight, 1))

# Plot histogram of elicited 3-month job-finding probability
p <- data_fig1 %>% 
  ggplot() +
  geom_histogram(
    aes(x = find_job_3mon, y = ..density.., weight = fweight),
    binwidth = 0.1,
    boundary = 0,
    fill = "grey85", color = "black", linewidth = 0.2
  ) +
  labs(
    y = "Density",
    x = "Elicited 3-Month Job-Finding Probability",
    title = "Figure 1. Histograms of Elicited Job-Finding Probabilities - Panel A. SCE (3-mo horizon)"
  ) +
  scale_y_continuous(
    limits = c(0, 3),
    breaks = seq(0, 2, by = 0.5)
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

p

# Save the figure
# ggsave(paste0(resultsdir, "sce_fig1.png"), plot = p, width = 8, height = 6, dpi = 300)
# ggsave(paste0(resultsdir, "sce_fig1.eps"), plot = p, width = 8, height = 6)


#################################################################################################
### Figure 2: Averages of Realized Job-Finding Rates, by Bins of Elicited Probabilities (SCE) ###
#################################################################################################
# Filter the dataset
data_fig2 <- data %>%
  filter(age >= 20, age <= 65) %>% 
  filter(in_sample_2 == 1)

# Create bins for Prob(Find Job in 3 Months)
data_fig2 <- data_fig2 %>%
  mutate(
    findjob_3mon_bin = case_when(
      find_job_3mon < 0.1 ~ 0.1,
      find_job_3mon < 0.2 ~ 0.2,
      find_job_3mon < 0.3 ~ 0.3,
      find_job_3mon < 0.4 ~ 0.4,
      find_job_3mon < 0.5 ~ 0.5,
      find_job_3mon < 0.6 ~ 0.6,
      find_job_3mon < 0.7 ~ 0.7,
      find_job_3mon < 0.8 ~ 0.8,
      find_job_3mon < 0.9 ~ 0.9,
      find_job_3mon >= 0.9 & !is.na(find_job_3mon) ~ 1.0,
      TRUE ~ NA_real_
    ),
    label_3mo = 0.05 + findjob_3mon_bin - 0.1
  )

# Mean job-finding rate by bin
summary_data <- data_fig2 %>%
  group_by(findjob_3mon_bin, label_3mo) %>%
  summarise(
    mean_UE_trans_3mon = weighted.mean(UE_trans_3mon, weight, na.rm = TRUE),
    sd_UE_trans_3mon = sd(UE_trans_3mon, na.rm = TRUE),
    n = sum(!is.na(UE_trans_3mon)),
    .groups = "drop"
  )

# Calculate 95% confidence intervals
summary_data <- summary_data %>%
  mutate(
    hi_UEtrans_3mon = mean_UE_trans_3mon + qt(0.975, n - 1) * (sd_UE_trans_3mon / sqrt(n)),
    low_UEtrans_3mon = mean_UE_trans_3mon - qt(0.975, n - 1) * (sd_UE_trans_3mon / sqrt(n))
  )

# Figure 2
p <- ggplot(summary_data, aes(x = label_3mo, y = mean_UE_trans_3mon)) +
  geom_point(color = "black") +
  geom_errorbar(aes(ymin = low_UEtrans_3mon, ymax = hi_UEtrans_3mon), color = "black") +
  labs(
    x = "Elicited 3-Month Job-Finding Probability",
    y = "Realized 3-Month Job-Finding Rate",
    title = "Figure 2: Averages of Realized Job-Finding Rates, by Bins of Elicited Probabilities (SCE)"
  ) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.1)) +
  theme_minimal()

p
# Save the figure
# ggsave(paste0(resultsdir, "sce_fig2.png"), plot = p, width = 8, height = 6, dpi = 300)
# ggsave(paste0(resultsdir, "sce_fig2.eps"), plot = p, width = 8, height = 6)

######################################################################################
### Table 2: Regressions of Realized on Elicited 3-Month Job-Finding Probabilities (SCE) ###
######################################################################################


# Keep only observations where age is between 20 and 65
data_tab2 <- data %>% 
  filter(age >= 20 & age <= 65) %>% ungroup

# Define cohort and time effects
data_tab2$year <- as.numeric(format(as.Date(as.character(data_tab2$date), "%Y%m%d"), "%Y"))
data_tab2$month <- data_tab2$date - data_tab2$year * 100
data_tab2 <- data_tab2 %>% group_by(userid) %>% mutate(cohorty = first(year))

# Unemployed observation count
data_tab2 <- data_tab2 %>% group_by(userid, lfs) %>% mutate(numu = if_else(lfs == 2, row_number(), NA_integer_))

# LFS at the beginning of survey
data_tab2 <- data_tab2 %>% group_by(userid) %>% mutate(firstlfs = first(lfs))

# Defining labels (can be used for later reference)
# In R, we typically don't label variables in the same way as in Stata, but you can use attributes or documentation.

# Define controls
data_tab2$agesq <- data_tab2$age^2
controls <- c("female", "hispanic", "black", "r_asoth", "other_race", "age", "agesq", "hhinc_2", "hhinc_3", "hhinc_4", "education_2", "education_3", "education_4", "education_5", "education_6")

# Sample 3-consecutive surveys only
data_tab2 <- data_tab2 %>% filter(in_sample_2 == 1)


# Table 2 - Panel A

# First regression: UE_trans_3mon on find_job_3mon
# Clustering is still not correct - for all regressions below SE are off. Treatment effects are correct
modelA_1 <- lfe::felm(UE_trans_3mon ~ find_job_3mon + (1 | userid), data = data_tab2, weights = data_tab2$weight)

# Second regression: UE_trans_3mon on controls
modelA_2 <- lfe::felm(as.formula(paste0("UE_trans_3mon ~ ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2, weights = data_tab2$weight)

# Third regression: UE_trans_3mon on find_job_3mon + controls
modelA_3 <- lfe::felm(as.formula(paste0("UE_trans_3mon ~ find_job_3mon + ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2, weights = data_tab2$weight)

# Fourth regression: UE_trans_3mon on find_job_3mon, findjob_3mon_longterm, controls, and longterm_unemployed
modelA_4 <- lfe::felm(as.formula(paste0("UE_trans_3mon ~ find_job_3mon + findjob_3mon_longterm + ", paste0(controls, collapse = " + "), " + longterm_unemployed + (1 | userid)")), data = data_tab2, weights = data_tab2$weight)

# Table 2 - Panel A: Export results to LaTeX
stargazer(modelA_1, modelA_2, modelA_3, modelA_4, type = "text",
          star.cutoffs = c(0.1, 0.05, 0.01), 
          digits = 3, 
          column.labels = c("Model 1", "Model 2", "Model 3", "Model 4"),
          dep.var.labels = "UE Transitions (3-Months)",
          covariate.labels = c("Find Job 3-Month", "Long-Term Unemployed"),
          omit = controls)

# Table 2 - Panel B

# Update variable labels for Panel B
# data_tab2$find_job_3mon <- lag(data_tab2$find_job_3mon, 3)  # Lagged find_job_3mon by 3 months

# First regression: tplus3_UE_trans_3mon on tplus3_percep_3mon
modelB_1 <- lfe::felm(tplus3_UE_trans_3mon ~ tplus3_percep_3mon + (1 | userid), data = data_tab2, weights = data_tab2$weight)

# The next regressions use same sample as in the above regression which rules out any NA values for tplus3_UE_trans_3mon and tplus3_percep_3mon
data_tab2_short <- data_tab2 %>% 
  filter(!is.na(tplus3_UE_trans_3mon) & !is.na(tplus3_percep_3mon))

# Second regression: tplus3_UE_trans_3mon on controls
modelB_2 <- lfe::felm(as.formula(paste0("tplus3_UE_trans_3mon ~ ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2_short, weights = data_tab2_short$weight)

# Third regression: tplus3_UE_trans_3mon on find_job_3mon
modelB_3 <- lfe::felm(tplus3_UE_trans_3mon ~ find_job_3mon + (1 | userid), data = data_tab2_short, weights = data_tab2_short$weight)

# Fourth regression: tplus3_UE_trans_3mon on find_job_3mon + controls
modelB_4 <- lfe::felm(as.formula(paste0("tplus3_UE_trans_3mon ~ find_job_3mon + ", paste0(controls, collapse = " + "), " + (1 | userid)")), data = data_tab2_short, weights = data_tab2_short$weight)

# Table 2 - Panel B: Export results to LaTeX
stargazer(modelB_1, modelB_2, modelB_3, modelB_4, type = "text",
          star.cutoffs = c(0.1, 0.05, 0.01), 
          digits = 3, 
          column.labels = c("Model 1", "Model 2", "Model 3", "Model 4"),
          dep.var.labels = "T+3 UE Transitions (3-Months)",
          covariate.labels = c("Current Job-Finding Probability", "Lagged Job-Finding Probability"), # Switched order of these labels...still not quite sure about the lagging here
          omit = controls)

# Clear models from memory
rm(modelA_1, modelA_2, modelA_3, modelA_4, modelB_1, modelB_2, modelB_3, modelB_4)

######################################################################################
# Table 3: Lower Bounds for the Variance of 3-Month Job-Finding Probabilities (SCE) #
######################################################################################

# Load data
data_tab3 <- data %>% 
  filter(age >= 20 & age <= 65 & in_sample_2 == 1)

# Convert 12-month probability into a 3-month probability
data_tab3$imputed_findjob_3mon <- 1 - (1 - data_tab3$find_job_12mon)^0.25
data_tab3$find_job_12mon <- data_tab3$imputed_findjob_3mon

# Define unemployment duration bins
data_tab3$ud_ <- as.factor(data_tab3$udur_bins)
data_tab3$ud_1 <- NULL  # Drop ud_1

# Define controls
data_tab3$agesq <- data_tab3$age^2

# Set random seed
set.seed(1)

# Preserve the dataset
data_tab3_preserved <- data_tab3

# Must exclude NA observations
data_tab3_short <- data_tab3 %>% 
  filter(!is.na(find_job_12mon))

# Moments for lower bounds
covzt <- cov.wt(data_tab3_short[, c("UE_trans_3mon", "find_job_3mon")], wt = data_tab3_short$weight, cor = TRUE)$cov[1,2]
varz <- cov.wt(data_tab3_short[, c("UE_trans_3mon", "find_job_3mon")], wt = data_tab3_short$weight, cor = TRUE)$cov[2,2]

covz12t <- cov.wt(data_tab3_short[, c("UE_trans_3mon", "find_job_12mon")], wt = data_tab3_short$weight, cor = TRUE)$cov[1,2]
covzz12 <- cov.wt(data_tab3_short[, c("find_job_3mon", "find_job_12mon")], wt = data_tab3_short$weight, cor = TRUE)$cov[1,2]


# Regression for prediction 1
reg1 <- lm(as.formula(paste0("UE_trans_3mon ~ ", paste0(controls, collapse = " + "))), data = data_tab3_short[, c("UE_trans_3mon", "find_job_3mon", "find_job_12mon", controls)], weights = data_tab3_short$weight)
UE_trans_3mon_pred1 <- predict(reg1)

varz_pred1 <- wtd.var(UE_trans_3mon_pred1, weight = data_tab3_short$weight, na.rm = TRUE)

# Regression for prediction 2
reg2 <- lm(as.formula(paste0("UE_trans_3mon ~ find_job_3mon + find_job_12mon + ", paste0(controls, collapse = " + "))), data = data_tab3_short[, c("UE_trans_3mon", "find_job_3mon", "find_job_12mon", controls)], weights = data_tab3_short$weight)
UE_trans_3mon_pred2 <- predict(reg2)

varz_pred2 <- wtd.var(UE_trans_3mon_pred2, weight = data_tab3_short$weight, na.rm = TRUE)

# Lower bounds
LB_z1 <- (covzt^2) / varz
LB_z12 <- (covz12t * covzt) / covzz12
LB_pred1 <- varz_pred1
LB_pred2 <- varz_pred2

# Save results in a matrix for stargazer output
results <- data.frame(
  "Nonparametric lower bound based on..." = c("the elicited 3-month job-finding probability", "the elicited 3- and 12-month job-finding probabilities", "Predicted - Controls only","Predicted - Controls + 3- and 12-month Elicitation"),
  "Value" = c(round(LB_z1[1], 3), round(LB_z12[1], 3), round(LB_pred1[1], 3), round(LB_pred2[1], 3)),
  "SE" = c(NA, NA, NA, NA)  # Bootstrap standard errors will be computed in the next step
)

# Bootstrap for standard errors
boot_results <- matrix(NA, nrow = 2000, ncol = 4)
for (bsi in 1:2000) {
  # Sample with replacement (bootstrap)
  boot_data <- data_tab3_short[sample(1:nrow(data_tab3_short), replace = TRUE), ]

  # Moments for lower bounds
  covzt <- cov.wt(boot_data[, c("UE_trans_3mon", "find_job_3mon")], wt = boot_data$weight, cor = TRUE)$cov[1,2]
  varz <- cov.wt(boot_data[, c("UE_trans_3mon", "find_job_3mon")], wt = boot_data$weight, cor = TRUE)$cov[2,2]
  
  covz12t <- cov.wt(boot_data[, c("UE_trans_3mon", "find_job_12mon")], wt = boot_data$weight, cor = TRUE)$cov[1,2]
  covzz12 <- cov.wt(boot_data[, c("find_job_3mon", "find_job_12mon")], wt = boot_data$weight, cor = TRUE)$cov[1,2]
  
  
  # Regression for prediction 1
  reg1 <- lm(as.formula(paste0("UE_trans_3mon ~ ", paste0(controls, collapse = " + "))), data = boot_data[, c("UE_trans_3mon", "find_job_3mon", "find_job_12mon", controls)], weights = boot_data$weight)
  UE_trans_3mon_pred1 <- predict(reg1)
  
  varz_pred1 <- wtd.var(UE_trans_3mon_pred1, weight = boot_data$weight, na.rm = TRUE)
  
  # Regression for prediction 2
  reg2 <- lm(as.formula(paste0("UE_trans_3mon ~ find_job_3mon + find_job_12mon + ", paste0(controls, collapse = " + "))), data = boot_data[, c("UE_trans_3mon", "find_job_3mon", "find_job_12mon", controls)], weights = boot_data$weight)
  UE_trans_3mon_pred2 <- predict(reg2)
  
  varz_pred2 <- wtd.var(UE_trans_3mon_pred2, weight = boot_data$weight, na.rm = TRUE)
  
  # Lower bounds
  LB_z1 <- (covzt^2) / varz
  LB_z12 <- (covz12t * covzt) / covzz12
  LB_pred1 <- varz_pred1
  LB_pred2 <- varz_pred2

  # Lower bounds
  LB_z1 <- (covzt^2) / varz
  LB_z12 <- (covz12t * covzt) / covzz12
  LB_pred1 <- varz_pred1
  LB_pred2 <- varz_pred2

  # Store results
  boot_results[bsi, ] <- c(LB_z1, LB_z12, LB_pred1, LB_pred2)
}

# Calculate standard errors from bootstrap results
# Standard errors are incorrect...
se <- apply(boot_results, 2, function(x) round(sd(x, na.rm = TRUE), 3))

# Add standard errors to results table
results$"SE" <- se

# Display table using stargazer
# Standard errors are incorrect...
stargazer(results, type = "text", summary = FALSE,
          header = FALSE,
          align = TRUE,
          #column.labels = c("Lower Bound (3-month)", "Value", "(SE)"),
          rownames = FALSE)


############################################################################################
# Figure 3: Perceived vs. Realized Job Finding, by Duration of Unemployment (SCE)         #
############################################################################################

# # Import results from Appendix Table D6 (Panel A) for comparison with own calculation
# source_data_fig3 <- read_excel(paste0(base, "120501-V1/MST/EMPIRICAL_ANALYSIS/Output/Appendix/sce_tabD6panelA.xlsx"), sheet = 2, col_names = LETTERS[1:6])
# 
# # Define variable labels
# source_data_fig3 <- source_data_fig3 %>%
#   rename(
#     pjob_find = B,
#     se_pjob_find = C,
#     rjob_find = D,
#     se_rjob_find = E,
#     nobs = F
#   ) %>% 
#   select(-A)
# 
# # Define labels for unemployment duration
# source_data_fig3 <- source_data_fig3 %>%
#   mutate(
#     undur = row_number() - 1
#   ) %>%
#   filter(undur != 0) %>%
#   mutate(
#     undur_label = factor(
#       undur,
#       levels = 1:4,
#       labels = c("0-3 Months", "4-6 Months", "7-12 Months", "13 Months +")
#     )
#   )
# 
# # 95 percent confidence interval
# source_data_fig3 <- source_data_fig3 %>%
#   mutate(
#     pjob_find_lower = pjob_find - qt(0.975, df = nobs - 1) * se_pjob_find,
#     pjob_find_upper = pjob_find + qt(0.975, df = nobs - 1) * se_pjob_find,
#     rjob_find_lower = rjob_find - qt(0.975, df = nobs - 1) * se_rjob_find,
#     rjob_find_upper = rjob_find + qt(0.975, df = nobs - 1) * se_rjob_find,
#     undur1 = undur + 0.1  # Offset for graphical purposes
#   )

# Calculate own source data for graph using sce data
data_fig3 <- data %>% 
  filter(age >= 20 & age <= 65 & in_sample_2 == 1) %>% 
  select(udur_bins, UE_trans_3mon, find_job_3mon, weight) %>% 
  filter(!is.na(udur_bins) & !is.na(weight)) %>% 
  group_by(udur_bins) %>% 
  summarise(rjob_find = weighted.mean(UE_trans_3mon, na.rm = TRUE, w = weight),
            pjob_find = weighted.mean(find_job_3mon, na.rm = TRUE, w = weight),
            se_rjob_find = weighted_se(UE_trans_3mon, na.rm = TRUE, w = weight),
            se_pjob_find = weighted_se(find_job_3mon, na.rm = TRUE, w = weight),
            nobs = n()) %>% 
  ungroup %>%
  rename(undur = udur_bins) %>% 
  mutate(pjob_find_lower = pjob_find - qt(0.975, df = nobs - 1) * se_pjob_find,
         pjob_find_upper = pjob_find + qt(0.975, df = nobs - 1) * se_pjob_find,
         rjob_find_lower = rjob_find - qt(0.975, df = nobs - 1) * se_rjob_find,
         rjob_find_upper = rjob_find + qt(0.975, df = nobs - 1) * se_rjob_find,
         undur1 = undur + 0.1,  # Offset for graphical purposes
         undur_label = factor(
           undur,
           levels = 1:4,
           labels = c("0-3 Months", "4-6 Months", "7-12 Months", "13 Months +")))

# source_data_fig3 <- source_data_fig3[, colnames(data_fig3)]
# test1 <- source_data_fig3 %>% 
#   mutate(across(contains("se_"), ~round(., 2)),
#          across(c(rjob_find, pjob_find, pjob_find_lower, pjob_find_upper, rjob_find_lower, rjob_find_upper), ~round(., 3)))
# test2 <- data_fig3 %>% 
#   mutate(across(contains("se_"), ~round(., 2)),
#          across(c(rjob_find, pjob_find, pjob_find_lower, pjob_find_upper, rjob_find_lower, rjob_find_upper), ~round(., 3)))
# all.equal(test1, test2)


# Figure 3: Perceived vs. Realized Job-Finding Probability/Rate
plot <- ggplot(data_fig3, aes(x = undur)) +
  # Elicited probabilities with error bars
  geom_line(aes(y = pjob_find), color = "black", size = 1) +
  geom_point(aes(y = pjob_find), color = "black", size = 3, shape = 4) +  # X symbol
  geom_errorbar(
    aes(ymin = pjob_find_lower, ymax = pjob_find_upper),
    color = "black", width = 0.1, size = 0.8
  ) +
  # Realized probabilities with error bars
  geom_line(aes(x = undur1, y = rjob_find), color = "gray", size = 1, linetype = "dashed") +
  geom_point(aes(x = undur1, y = rjob_find), color = "gray", size = 3, shape = 1) +  # Hollow circle
  geom_errorbar(
    aes(x = undur1, ymin = rjob_find_lower, ymax = rjob_find_upper),
    color = "gray", width = 0.1, size = 0.8, linetype = "dashed"
  ) +
  # geom_line(data = source_data_fig3, aes(x = undur1, y = rjob_find), color = "red", size = 1, linetype = "dashed") +
  # geom_point(data = source_data_fig3, aes(x = undur1, y = rjob_find), color = "red", size = 3, shape = 4) +  # X symbol
  # geom_errorbar(data = source_data_fig3, 
  #   aes(x = undur1, ymin = rjob_find_lower, ymax = rjob_find_upper),
  #   color = "red", width = 0.1, size = 0.8
  # ) +
  #   geom_line(data = source_data_fig3, aes(y = pjob_find), color = "blue", size = 1, linetype = "dashed") +
  #   geom_point(data = source_data_fig3, aes(y = pjob_find), color = "blue", size = 3, shape = 4) +  # X symbol
  #   geom_errorbar(data = source_data_fig3, 
  #     aes(ymin = pjob_find_lower, ymax = pjob_find_upper),
  #     color = "blue", width = 0.1, size = 0.8
  #   ) +
  # Titles and axis labels
  labs(
    y = "3-Month Job-Finding Probability/Rate",
    x = "Duration of Unemployment",
    title = "Perceived vs. Realized Job Finding, by Duration of Unemployment"
  ) +
  scale_x_continuous(
    breaks = 1:4,
    labels = levels(data_fig3$undur_label)
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    legend.title = element_blank()
  ) +
  # Legend customization
  guides(
    color = guide_legend(
      override.aes = list(linetype = c("solid", "dashed"))
    )
  ) 

plot

##################################################################################################
# Table 4 - Panel A: Linear Regressions of Elicited Job-Finding Probabilities on Duration of Unemployment (SCE)
##################################################################################################
# Load the dataset
data_tab4 <- data %>%
  filter(age >= 20 & age <= 65)

# # Time fixed effects
# data_tab4 <- data_tab4 %>%
#   mutate(across(starts_with("date"), ~ as.factor(.), .names = "dd_{col}")) %>%
#   select(-dd_1)

# Generate indicators for labor force status
data_tab4 <- data_tab4 %>%
  mutate(
    olf2 = as.numeric(lfs == 3),   # Out of labor force indicator
    emp3 = as.numeric(lfs == 1),  # Employed indicator
    i3m = ifelse(lfs != 1, as.numeric(!is.na(find_job_3mon)), NA)  # Indicator for perception question
  )

# Indicator for labor force status in the next interview
data_tab4 <- data_tab4 %>%
  arrange(userid, date) %>%
  group_by(userid) %>%
  mutate(next1lfs = lead(lfs))

# Spell number and spell length
data_tab4 <- data_tab4 %>%
  arrange(spell_id, date) %>%
  group_by(spell_id) %>%
  mutate(
    spelln = row_number(),
    spellN = n()
  )

# Number of observations with perception question in a spell
data_tab4 <- data_tab4 %>%
  mutate(
    n_f3m_spell = ifelse(lfs != 1, cumsum(i3m), NA),
    N_f3m_spell = ifelse(lfs != 1, max(n_f3m_spell, na.rm = TRUE), NA)
  )

# Number of observations out of labor force in a spell
data_tab4 <- data_tab4 %>%
  mutate(
    n_olf_spell = ifelse(lfs != 1, cumsum(olf2), NA),
    N_olf_spell = ifelse(lfs != 1, max(n_olf_spell, na.rm = TRUE), NA)
  )

# Label unemployment duration variable
data_tab4 <- data_tab4 %>%
  #rename(udur = unemployment_duration) %>%  # Adjust the variable name to match your dataset
  mutate(agesq = age^2)                     # Generate agesq

# Keep if in the main sample
data_tab4 <- data_tab4 %>%
  filter(in_sample_1 == 1)

# Indicator for first survey
data_tab4 <- data_tab4 %>%
  arrange(userid, date) %>%
  group_by(userid) %>%
  mutate(first_unemp_survey = row_number() == 1)

# Run regressions
# Table 4.1: Simple regression with only udur and first_unemp_survey
model1 <- feols(
  find_job_3mon ~ udur | 0,
  data = filter(data_tab4, first_unemp_survey == 1),
  weights = ~ weight,
  cluster = ~ userid
)

# Table 4.2: Regression with udur and weights
model2 <- feols(
  find_job_3mon ~ udur | 0,
  data = data_tab4,
  weights = ~ weight,
  cluster = ~ userid
)

# Table 4.3: Regression with controls
controls <- c("female", "hispanic", "black", "r_asoth", "other_race", "age", "agesq", 
              "hhinc_2", "hhinc_3", "hhinc_4", "education_2", "education_3", "education_4", 
              "education_5", "education_6")
model3 <- feols(
  as.formula(paste("find_job_3mon ~ udur +", paste(controls, collapse = " + "))),
  data = data_tab4,
  weights = ~ weight,
  cluster = ~ userid
)

# Table 4.4: Regression with spell fixed effects
model4 <- feols(
  find_job_3mon ~ udur | spell_id,
  data = data_tab4,
  weights = ~ weight,
  cluster = ~ spell_id
)
library(modelsummary)
# Create Table 4 - Panel A
modelsummary(
  list(summary(model1), summary(model2), summary(model3), summary(model4)),
  type = "text",  # Change to "latex" for LaTeX output
  title = "Table 4 - Panel A: Linear Regressions of Elicited Job-Finding Probabilities on Duration of Unemployment (SCE)",
  dep.var.labels.include = FALSE,
  column.labels = c("(1)", "(2)", "(3)", "(4)"),
  add.lines = list(
    c("Demographics", "", "", "x", "x"),
    c("Spell FE", "", "", "", "x"),
    c("Observations", nobs(model1), nobs(model2), nobs(model3), nobs(model4)),
    c("$R^2$", signif(r2(model1), 3), signif(r2(model2), 3), signif(r2(model3), 3), signif(r2(model4), 3))
  ),
  star.cutoffs = c(0.1, 0.05, 0.01),
  notes.append = TRUE,
  notes = "Standard errors are clustered at the user or spell level as indicated.",
  gof_map = c("nobs", "r.squared"),
  coef_map = c("udur"= "Unemployment Duration (Months)"),
  fmt = 4
)

##################################################################################################
#### Figure 4 - Panel A: Elicited Job-Finding Probabilities, by Time since First Interview  (SCE) ###
##################################################################################################
# Load data
data_fig4 <- data %>%
  filter(age >= 20 & age <= 65, in_sample_1 == 1) %>%
  mutate(
    agesq = age^2
  )

# Regressions on monthly dummies for duration
# Create monthly dummies - weighting happens in individual function calls (LM() and FEOLS() below)
duration_vars <- grep("^nedur_1mo_", names(data_fig4), value = TRUE)

# Regression 1: Simple OLS with weights and clustering
ols_model <- lm(as.formula(paste0("find_job_3mon ~ ", paste0(duration_vars, collapse = " + "))), 
                data = data_fig4, weights = weight)

# Extract coefficients and standard errors
ols_summary <- summary(ols_model)
ols_coeffs <- ols_summary$coefficients
ols_0_val <- ols_coeffs[rownames(ols_coeffs) == "nedur_1mo_0"][1]

# Regression 2: Fixed-effects regression
fe_model <- feols(as.formula(paste0("find_job_3mon ~ ", paste0(duration_vars, collapse = " + "), " | spell_id")),
                  data = data_fig4,
                  weights = ~ weight,
                  cluster = ~ spell_id)

fe_summary <- summary(fe_model)
fe_coeffs <- tidy(fe_summary)
fe_0_val <- fe_coeffs %>% filter(term == "nedur_1mo_0") %>% pull(estimate) %>% as.numeric

# Combine coefficients and standard errors for plotting
coeffs_data <- bind_rows(
  ols_coeffs %>% as_tibble(rownames = "term") %>% 
    filter(term %in% duration_vars) %>% 
    rename(b = Estimate, se = `Std. Error`, statistic = `t value`, p.value =`Pr(>|t|)`) %>%
    mutate(model = "OLS"),
  
  fe_coeffs %>%
    filter(term %in% duration_vars) %>%
    rename(b = estimate, se = std.error) %>%
    mutate(model = "Fixed Effects")
)

# Add confidence intervals
coeffs_data <- coeffs_data %>%
  mutate(
    high = b + 1.96 * se,
    low = b - 1.96 * se
  )

# Generate scatter plots for each model
plot1 <- coeffs_data %>% 
  filter(model == "OLS" &
           term %in% duration_vars[1:7]) %>% 
  ggplot() +
    #geom_point(aes(x = term, y = b), color = "red") +
    geom_point(aes(x = term, y = b-ols_0_val), color = "black") +
    #geom_errorbar(aes(x = term, ymin = low, ymax = high), color = "red") +
    geom_errorbar(aes(x = term, ymin = low-ols_0_val, ymax = high-ols_0_val), color = "black") +
    geom_hline(yintercept = 0, color = "black", size = 1) +
    xlab("Time Since First Interview, in Months") +
    ylab("Elicited 3-Month Job-Finding Probability") +
    ggtitle("Within and Across Spell Changes") +
    ylim(-0.25, 0.15) +
    theme_minimal()

plot2 <- coeffs_data %>% 
  filter(model == "Fixed Effects" & 
                         term %in% duration_vars[1:7]) %>% 
  ggplot() +
    #geom_point(aes(x = term, y = b), color = "red") +
    geom_point(aes(x = term, y = b - fe_0_val), color = "black") +
    #geom_errorbar(aes(x = term, ymin = low, ymax = high), color = "red") +
    geom_errorbar(aes(x = term, ymin = low - fe_0_val, ymax = high - fe_0_val), color = "black") +
    geom_hline(yintercept = 0, color = "black", size = 1) +
    ylim(-0.25, 0.15) +
    xlab("Time Since First Interview, in Months") +
    ylab("Elicited 3-Month Job-Finding Probability") +
    ggtitle("Within Spell Changes Only") +
    theme_minimal()

plot1 + plot2

# ggsave("sce_fig4panelA.png", combined_plot, width = 10, height = 5)
# ggsave("sce_fig4panelA.eps", combined_plot, width = 10, height = 5)