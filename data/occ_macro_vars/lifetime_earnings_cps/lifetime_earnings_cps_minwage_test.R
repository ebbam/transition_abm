# Quick test variant of lifetime_earnings_cps.R: the same CPS ASEC synthetic cohort, but dropping anyone with
# positive annual wage income below a full year of full-time work at the federal minimum wage (hourly minimum x
# 2,080 hours). Zeros are kept, as in the model (an entrant unemployed at a snapshot earns $0).
#
# Purpose: a rough check of how much of the low early-career CPS profile comes from part-year, part-time and
# student earnings, which the model does not have (its wages are full-time-equivalent OEWS wages), until the
# point-in-time, full-time-equivalent profile in lifetime_earnings_cps_fte.R can be built from a basic-monthly
# extract. Everything else (sample, entry ages, nominal dollars averaged over 2000-2023, means) is unchanged.
#
# Federal minimum wage in effect on 1 January of each income year (the lower rate in the three years it rose
# mid-year, so as few observations as possible are dropped): $5.15 to 2007, $5.85 in 2008, $6.55 in 2009,
# $7.25 from 2010. State minimums above the federal rate are ignored.
#
# Output (this folder; lifetime_earnings_cps.R's outputs are left as they are):
#   cps_earnings_by_experience_minwage.csv   profile by years since entry (0-20), averaged over income years
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

# Entry age by education, as in lifetime_earnings_cps.R
entry_age_by_educ <- function(educ) {
  case_when(educ >= 2   & educ <= 73  ~ 18,
            educ %in% c(80, 81, 90, 100, 110) ~ 21,
            educ %in% c(91, 92)  ~ 23,
            educ == 111          ~ 21,
            educ %in% c(120, 121, 122, 123) ~ 23,
            educ %in% c(124, 125) ~ 30,
            TRUE ~ NA_real_)
}

federal_min_wage <- function(income_year) {
  case_when(income_year <= 2007 ~ 5.15,
            income_year == 2008 ~ 5.85,
            income_year == 2009 ~ 6.55,
            TRUE                ~ 7.25)
}

cps <- data %>%
  mutate(income_year = YEAR - 1,
         entry_age = entry_age_by_educ(EDUC),
         years_since_entry = AGE - entry_age) %>%
  filter(income_year >= 2000, income_year <= 2023,
         EMPSTAT %in% c(10, 12, 20, 21, 22),
         !INCWAGE %in% c(99999998, 99999999),
         !is.na(entry_age),
         years_since_entry >= 0, years_since_entry <= 20) %>%
  mutate(annual_min_wage = federal_min_wage(income_year) * 2080,
         below_min = INCWAGE > 0 & INCWAGE < annual_min_wage)

dropped <- cps %>%
  group_by(years_since_entry) %>%
  summarise(share_dropped = weighted.mean(below_min, ASECWT), .groups = "drop")
print("Weighted share dropped (0 < earnings < full-time minimum-wage year), by years since entry:")
print(dropped %>% mutate(share_dropped = round(share_dropped, 3)), n = 21)

cps <- cps %>% filter(!below_min)

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
  summarise(mean_annual_earnings    = mean(mean_annual_earnings),
            median_annual_earnings  = mean(median_annual_earnings),
            share_positive_earnings = mean(share_positive_earnings),
            n_obs = sum(n_obs),
            n_income_years = n(),
            .groups = "drop") %>%
  arrange(years_since_entry) %>%
  mutate(cum_mean_earnings = cumsum(mean_annual_earnings)) %>%
  left_join(dropped, by = "years_since_entry")

assert_that(all(profile$n_income_years == 24))
assert_that(nrow(profile) == 21)

print(profile %>% mutate(across(c(mean_annual_earnings, median_annual_earnings, cum_mean_earnings), round)),
      n = 21)

write.csv(profile, file.path(dir, "cps_earnings_by_experience_minwage.csv"), row.names = FALSE)
