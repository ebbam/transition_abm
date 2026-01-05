# CPS Displaced Worker Supplement
# https://cps.ipums.org/cps/dw_sample_notes.shtml#:~:text=The%20Displaced%20Worker%20Supplement%20collects,Tenure%20and%20Occupational%20Mobility%20Supplement.
# Codebooks: https://cps.ipums.org/cps/codebooks.shtml#dw_codebooks

library(ipumsr)
library(tidyverse)
library(haven)
library(here)
library(janitor)
library(modelsummary)
library(broom)
library(performance)
library(WeightIt)
library(survey)
# 
# ddi <- read_ipums_ddi(here("data/behav_params/cps_displaced_worker_supplement/cps_00019.xml"))
# data <- read_ipums_micro(ddi) %>% 
#   clean_names %>% 
#   # Include only observations where dwlostjob is true and FT workers
#   filter(dwlostjob == 2 & dwfulltime == 2) 
# 
# #saveRDS(data, here("data/behav_params/cps_displaced_worker_supplement/cps_disp_filtered.RDS"))

df <- readRDS(data, here("data/behav_params/cps_displaced_worker_supplement/cps_disp_filtered.RDS")) %>% 
  select(hwtfinl, cpsid, wtfinl, age, sex, race, marst, educ,
         dwsuppwt, dwyears,
         dwben,
         dwlastwrk, # Time since worked at last job
         dwweekc, # (Weekly earnings at current job)
         dwweekl,
         dwwagel,
         dwwagec, #DWWAGEC (Hourly wage at current job)
         dwhrswkc, #DWHRSWKC (Hours worked each week at current job)
         dwresp, # DWRESP (Eligibility and interview status for Displaced Worker Supplement)
         dwsuppwt, #DWSUPPWT (Displaced workers supplement weight)
         dwwksun) %>%  #DWWKSUN (Number of weeks not working between between end of lost or left job and start of next job))
filter(dwhrswkc != 99 & dwwksun <= 160) %>% 
  mutate(dwwagel = ifelse(round(dwwagel) == 100, NA, dwwagel),
         dwwagec = ifelse(round(dwwagec) == 100, NA, dwwagec),
         dwweekl = ifelse(round(dwweekl) == 10000, NA, dwweekl),
         dwweekc = ifelse(round(dwweekc) == 10000, NA, dwweekc),
         educ_cat = case_when(educ %in% c(1) ~ NA,
                              educ > 1 & educ <= 71 ~ "Less than HS",
                              educ %in% c(73, 81) ~ "HS Diploma",
                              educ %in% c(91, 92) ~ "Associate's",
                              educ %in% c(111) ~ "Bachelor's",
                              educ > 111 ~ "Postgraduate Degree"
                              )) %>% 
  mutate(ratio_wage = dwwagec/dwwagel,
         ratio_weekly = dwweekc/dwweekl,
         dwwksun_bin = case_when(dwwksun <= 4 ~ 1, #"Less than 4 weeks",
                                 dwwksun > 4 & dwwksun <= 8 ~ 2, #"Less than 2 months",
                                 dwwksun > 8 & dwwksun <= 12 ~ 3, #"Less than 3 months",
                                 dwwksun > 12 & dwwksun <= 16 ~ 4, #"Less than 4 months",
                                 dwwksun > 16 & dwwksun <= 20 ~ 5, #"Less than 5 months",
                                 dwwksun > 20 & dwwksun <= 24 ~ 6, #"Less than 6 months",
                                 dwwksun > 24 & dwwksun <= 36 ~ 7, # 6-9 months ",
                                 dwwksun > 36 & dwwksun <= 48 ~ 8, # 9-12 months",
                                 dwwksun > 48 & dwwksun <= 72 ~ 9, # 12-18 months"
                                 dwwksun > 72 & dwwksun <= 104 ~ 10, # 18-24 months
                                 dwwksun > 104 & dwwksun <= 120 ~ 11, # 2-2.5 years
                                 dwwksun > 120 ~ 12),
         clipped_sample_hwage = ratio_wage >= 0.5 & ratio_wage <= 2 & dwwksun_bin <= 10,
         clipped_sample_wwage = ratio_weekly >= 0.5 & ratio_weekly <= 2  & dwwksun_bin <= 10)
  # 2,692 observations where the ratios are not the same
  # filter(ratio_wage != ratio_weekly) %>% 

controls <- " + sex + age + race + marst"
reg_forms <- c("Cont." = "dwwksun",
               "Cont. w. UI" = "dwwksun + dwben",
               "Disc." = "dwwksun_bin",
               "Disc. w. UI" = "dwwksun_bin + dwben")
reg_forms_controls <- paste0(reg_forms, controls)
names(reg_forms_controls) <- paste0(names(reg_forms), " w. controls")
reg_forms_full <- c(reg_forms, reg_forms_controls)

mod_list <- list()
for(rf in names(reg_forms_full)){
  #### HOURLY WAGE RATIO OF ACCEPTED TO PREVIOUS
  ## Unclipped sample
  mod_list[[paste0("HWR ", rf)]] <- lm(data = df, as.formula(paste0("ratio_wage ~ ", reg_forms_full[rf])), weights = dwsuppwt)
  
  ## Clipped sample
  mod_list[[paste0("HWR ", rf, " (clipped)")]] <- df %>% 
    filter(clipped_sample_hwage) %>% 
    lm(data = ., as.formula(paste0("ratio_wage ~ ", reg_forms_full[rf])), weights = dwsuppwt)
  
  #### WEEKLY WAGE RATIO OF ACCEPTED TO PREVIOUS
  ## Unclipped sample
  mod_list[[paste0("WWR ", rf)]] <-filter(df, ratio_weekly != Inf) %>% 
  lm(data = ., as.formula(paste0("ratio_weekly ~ ", reg_forms_full[rf])), weights = dwsuppwt)
  
  ## Clipped sample
  mod_list[[paste0("WWR ", rf, " (clipped)")]] <- filter(df, ratio_weekly != Inf & clipped_sample_wwage) %>% 
    lm(data = ., as.formula(paste0("ratio_weekly ~ ", reg_forms_full[rf])), weights = dwsuppwt)
}

df %>% 
  ggplot() + 
  geom_histogram(aes(x = dwwksun))

###############################################################################
#### Time spent at lost job ###################################################
#######
df %>%
  mutate(dwyears = ifelse(round(dwyears) == 100, NA, dwyears)) %>% 
  ggplot() + geom_histogram(aes(x = dwyears), fill = "lightblue") + 
  labs(x = "Years spent at lost job", title = "Histogram of Reported Tenure at Lost Job") +
  theme_minimal()

################################################################################
################################################################################
##### Histograms of wage ratio by unemployment duration bin
##### HOURLY WAGE
means <- df %>%
  filter(clipped_sample_hwage) %>%
  group_by(dwwksun_bin) %>%
  summarise(mean_ratio = mean(ratio_wage, na.rm = TRUE))

df %>%
  filter(clipped_sample_hwage) %>% 
  ggplot(aes(x = ratio_wage, fill = as.factor(dwwksun_bin))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  geom_segment(
    data = means,
    aes(x = mean_ratio, xend = mean_ratio, y = 0, yend = 660),
    color = "darkgrey",
    linetype = "dashed",
    size = 1
  ) +
  geom_text(
    data = means,
    aes(x = mean_ratio, y = 660, label = as.factor(dwwksun_bin)),
    size = 6
  ) +
  labs(
    title = "Accepted Hourly Wage by Group",
    x = "Ratio HWage",
    y = "Count",
    fill = "Unemp. Duration Bin",
    color = "Group Mean"
  ) +
  theme_minimal()
  
##### WEEKLY WAGE
means <- df %>%
  filter(clipped_sample_wwage) %>%
  group_by(dwwksun_bin) %>%
  summarise(mean_ratio = mean(ratio_weekly, na.rm = TRUE))

df %>%
  filter(clipped_sample_wwage) %>% 
  ggplot(aes(x = ratio_weekly, fill = as.factor(dwwksun_bin))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  geom_segment(
    data = means,
    aes(x = mean_ratio, xend = mean_ratio, y = 0, yend = 450),
    color = "darkgrey",
    linetype = "dashed",
    size = 1
  ) +
  geom_text(
    data = means,
    aes(x = mean_ratio, y = 450, label = as.factor(dwwksun_bin)),
    size = 6
  ) +
  labs(
    title = "Accepted Weekly Wage by Group",
    x = "Ratio WWage",
    y = "Count",
    fill = "Unemp. Duration Bin"
  ) +
  theme_minimal()
################################################################################
################################################################################

modelsummary(mod_list, gof_omit = c("Log.Lik.|BIC|AIC"), stars = TRUE)

df %>% 
  filter(clipped_sample_hwage) %>% 
  ggplot(aes(x = as.factor(dwwksun_bin), y = ratio_wage)) +
  geom_boxplot(aes(weight = dwsuppwt, fill = as.factor(dwwksun_bin)), alpha = 0.3)+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Wage ratio") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5)) +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos",  "7-12 mos", ">12 mos"))

df %>% 
  filter(clipped_sample_wwage) %>% 
  ggplot(aes(x = as.factor(dwwksun_bin), y = ratio_weekly)) +
  geom_boxplot(aes(weight = dwsuppwt, fill = as.factor(dwwksun_bin), alpha = 0.3))+
  geom_hline(yintercept = 1, color = "darkblue", linetype = "dashed") +
  labs(title = "Wage ratio") +
  theme_minimal() +
  ylim(0,2.5) +
  theme(plot.title=element_text(hjust=0.5)) +
  scale_x_discrete(labels = c("<4 mos", "4-6 mos",  "7-12 mos", ">12 mos"))

for(k in 1:length(mod_list)){
  mod_list[[k]] %>% 
    augment %>% 
    ggplot(aes(dwwksun, ratio_wage)) +
    geom_point() +
    stat_smooth(method = lm, se = FALSE) +
    geom_segment(aes(xend = dwwksun, yend = .fitted), color = "red", size = 0.3)
  
  mod_list[[k]] %>% 
    check_model()
}

################################################################################
################################################################################
############ Balancing the sample ##############################################

####### One of the challenges with this data is that the sample grows significantly smaller for higher reported times of unemployment duration. 
####### One option is a reweighting to ensure population similarity across bins
# Convert unemployment duration to factor

# Apply entropy balancing using dwsuppwt sample weights ---
eb <- weightit(
  formula = dwwksun_bin ~ sex + age + race + marst,
  data = df,
  method = "ebalance",
  s.weights = df$dwsuppwt
)

# Add the new weights to the dataframe
df$eb_weight <- eb$weights

# Run weighted linear regression using entropy-balanced weights ---
mod2b <- lm(
  formula = ratio_wage ~ dwwksun_bin + sex + age + race + marst,
  data = df,
  weights = eb_weight
)

# --- Step 5: Output summary of the model ---
summary(mod2b)
mod2b %>% check_model()

# --- Optional: Check covariate balance after weighting ---
summary(eb, digits = 3)


################################################################################
################################################################################
########## TRYING WITH ADDITIONAL CONTROLLS FOR INCOME AND EDUCATION ###########
controls <- " + sex + age + race + marst + educ_cat + dwwagel"
reg_forms <- c("Cont." = "dwwksun",
               "Cont. w. UI" = "dwwksun + dwben",
               "Disc." = "dwwksun_bin",
               "Disc. w. UI" = "dwwksun_bin + dwben")
reg_forms_controls <- paste0(reg_forms, controls)
names(reg_forms_controls) <- paste0(names(reg_forms), " w. controls")
reg_forms_full <- c(reg_forms, reg_forms_controls)

mod_list <- list()
for(rf in names(reg_forms_full)){
  #### HOURLY WAGE RATIO OF ACCEPTED TO PREVIOUS
  ## Unclipped sample
  mod_list[[paste0("HWR ", rf)]] <- lm(data = df, as.formula(paste0("ratio_wage ~ ", reg_forms_full[rf])), weights = dwsuppwt)
  
  ## Clipped sample
  mod_list[[paste0("HWR ", rf, " (clipped)")]] <- df %>% 
    filter(clipped_sample_hwage) %>% 
    lm(data = ., as.formula(paste0("ratio_wage ~ ", reg_forms_full[rf])), weights = dwsuppwt)
  
  #### WEEKLY WAGE RATIO OF ACCEPTED TO PREVIOUS
  ## Unclipped sample
  mod_list[[paste0("WWR ", rf)]] <-filter(df, ratio_weekly != Inf) %>% 
    lm(data = ., as.formula(paste0("ratio_weekly ~ ", reg_forms_full[rf])), weights = dwsuppwt)
  
  ## Clipped sample
  mod_list[[paste0("WWR ", rf, " (clipped)")]] <- filter(df, ratio_weekly != Inf & clipped_sample_wwage) %>% 
    lm(data = ., as.formula(paste0("ratio_weekly ~ ", reg_forms_full[rf])), weights = dwsuppwt)
}

df %>% 
  ggplot() + 
  geom_histogram(aes(x = dwwksun))

modelsummary(mod_list, gof_omit = c("Log.Lik.|BIC|AIC"), stars = TRUE)
