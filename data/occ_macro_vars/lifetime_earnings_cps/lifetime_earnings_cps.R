# Early-career earnings profile from the CPS ASEC: a validation target for the lifetime earnings of
# new entrants in calibration_remote/exposure_entry_policy_exercise.py
#
# Data: IPUMS-CPS ASEC extract cps_00021 (data/occ_macro_vars/ipums_vars/), ASEC 2001-2024, i.e. annual
# wage income (INCWAGE) for income years 2000-2023.
#
# Synthetic cohort: civilian labour-force participants (all agents in the model are in the labour force),
# by years since labour-market entry = age - entry age, where the entry age follows the education-based
# mapping used for occupations in BLS_entry_exit_reqs/entry_exit_indicators_full_omn.R. Annual wage income
# includes zeros (participants who earned nothing in the income year), matching the model's convention that
# a year unemployed at the annual snapshot counts as $0.
#
# Dollars: the model's occupation wages are the unweighted mean of NOMINAL OEWS wages over 2000-2023
# (OEWS/oews_wage_distributions.R). The CPS profile is built the same way -- nominal, computed for each
# income year 2000-2023 and then averaged across years with equal weight -- so both sides are in the same
# mix of dollars and no price index is needed.
#
# Means, not medians: the cumulative MEAN of a synthetic cohort equals the sum of the annual means, so it is
# directly comparable with the model's mean cumulative earnings. The cumulative median is not (the median
# of individual totals is not the sum of annual medians); annual medians are saved for reference only.
#
# Output (this folder):
#   cps_earnings_by_experience.csv          profile by years since entry (0-20), averaged over income years
#   cps_earnings_by_experience_by_year.csv  the same, for each income year
library(tidyverse)
library(here)
library(ipumsr)
library(assertthat)

dir <- here("data/occ_macro_vars/lifetime_earnings_cps")

ddi <- read_ipums_ddi(here("data/occ_macro_vars/ipums_vars/cps_00021.xml"))
data <- read_ipums_micro(ddi, vars = c("YEAR", "ASECWT", "AGE", "EDUC", "EMPSTAT", "INCWAGE"),
                         verbose = FALSE) %>%
  filter(!is.na(ASECWT)) %>%                          # ASEC observations only
  mutate(across(everything(), as.numeric))

# Entry age by education, as in entry_exit_indicators_full_omn.R (ed_req -> entry_age)
entry_age_by_educ <- function(educ) {
  case_when(educ >= 2   & educ <= 73  ~ 18,           # no credential / high school diploma
            educ %in% c(80, 81, 90, 100, 110) ~ 21,    # some college, no degree
            educ %in% c(91, 92)  ~ 23,                 # associate's degree
            educ == 111          ~ 21,                 # bachelor's degree
            educ %in% c(120, 121, 122, 123) ~ 23,      # master's degree (or 5+ years of college)
            educ %in% c(124, 125) ~ 30,                # professional or doctoral degree
            TRUE ~ NA_real_)
}

cps <- data %>%
  mutate(income_year = YEAR - 1,                      # ASEC reports the previous calendar year's income
         entry_age = entry_age_by_educ(EDUC),
         years_since_entry = AGE - entry_age) %>%
  filter(income_year >= 2000, income_year <= 2023,
         EMPSTAT %in% c(10, 12, 20, 21, 22),          # civilian labour force: employed or unemployed
         !INCWAGE %in% c(99999998, 99999999),         # missing / NIU
         !is.na(entry_age),
         years_since_entry >= 0, years_since_entry <= 20)

print(paste0("Observations: ", nrow(cps), " (income years ", min(cps$income_year), "-", max(cps$income_year), ")"))

weighted_median <- function(x, w) {
  o <- order(x); x <- x[o]; w <- w[o]
  x[which(cumsum(w) >= sum(w) / 2)[1]]
}

by_year <- cps %>%
  group_by(income_year, years_since_entry) %>%
  summarise(mean_annual_earnings   = weighted.mean(INCWAGE, ASECWT),
            median_annual_earnings = weighted_median(INCWAGE, ASECWT),
            share_positive_earnings = weighted.mean(INCWAGE > 0, ASECWT),
            n_obs = n(),
            .groups = "drop")

profile <- by_year %>%
  group_by(years_since_entry) %>%
  summarise(mean_annual_earnings    = mean(mean_annual_earnings),       # equal weight per income year
            median_annual_earnings  = mean(median_annual_earnings),
            share_positive_earnings = mean(share_positive_earnings),
            n_obs = sum(n_obs),
            n_income_years = n(),
            .groups = "drop") %>%
  arrange(years_since_entry) %>%
  mutate(cum_mean_earnings = cumsum(mean_annual_earnings))

assert_that(all(profile$n_income_years == 24))         # every income year 2000-2023 present at every step
assert_that(nrow(profile) == 21)

print(profile %>% mutate(across(c(mean_annual_earnings, median_annual_earnings, cum_mean_earnings), round)),
      n = 21)

write.csv(by_year, file.path(dir, "cps_earnings_by_experience_by_year.csv"), row.names = FALSE)
write.csv(profile, file.path(dir, "cps_earnings_by_experience.csv"), row.names = FALSE)
