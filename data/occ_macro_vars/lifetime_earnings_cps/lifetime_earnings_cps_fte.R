# Early-career earnings profile from the basic monthly CPS, built to match how the model measures entrant
# earnings: a validation target for calibration_remote/exposure_entry_policy_exercise.py (Section 16).
#
# Model side: every agent is a labour-force participant. At each annual snapshot an entrant earns the annual
# wage of the job they hold -- an OEWS full-time-equivalent wage (hourly x 2,080) -- or $0 if unemployed at
# that moment. There is no part-time or part-year work and no school enrolment.
#
# CPS side, to match (unlike lifetime_earnings_cps.R, which uses last year's ASEC INCWAGE):
#   - Point in time: labour-force status and pay at the interview, not the previous calendar year's income,
#     so there is no one-year offset between age and the earnings year.
#   - Who: civilian labour-force participants (employed or unemployed), excluding students enrolled in high
#     school or college (SCHLCOLL, asked of 16-24 year olds) and the self-employed (no earnings questions;
#     the model has no self-employment).
#   - Pay if employed: full-time-equivalent annual pay = hourly pay x 2,080, where hourly pay is HOURWAGE for
#     hourly-paid workers and EARNWEEK / usual hours at the job (UHRSWORKORG) otherwise. Part-time workers
#     are kept at their FTE rate rather than dropped, so the sample is not selected towards better jobs.
#   - Pay if unemployed: $0.
#   - Earnings questions are asked only in the outgoing rotation months (MISH 4 and 8), so the sample is
#     restricted to those months for employed and unemployed alike, weighted by WTFINL.
#   - Dollars: nominal, computed for each calendar year 2000-2023 and then averaged with equal weight, as in
#     lifetime_earnings_cps.R and the model's OEWS wages.
#   - Means: the cumulative mean of a synthetic cohort is the sum of the annual means (see lifetime_earnings_cps.R).
#
# Requires an IPUMS-CPS extract of the BASIC MONTHLY samples, Jan 2000 - Dec 2023, with (at least):
#   YEAR, MONTH, MISH, WTFINL, AGE, EDUC, EMPSTAT, CLASSWKR, SCHLCOLL,
#   EARNWEEK, HOURWAGE, PAIDHOUR, UHRSWORKORG, QEARNWEE, QHOURWAG
# Case selection to keep it small: MISH 4 and 8; AGE 18-50 (entry at 18-30, followed for up to 20 years).
# Set EXTRACT below to its file stem. Code values for the variables are as in the IPUMS codebook; check
# them against the extract's DDI (the asserts below catch the main NIU codes).
#
# Output (this folder; lifetime_earnings_cps.R's outputs are left as they are):
#   cps_fte_earnings_by_experience.csv          profile by years since entry (0-20), averaged over years
#   cps_fte_earnings_by_experience_by_year.csv  the same, for each calendar year
library(tidyverse)
library(here)
library(ipumsr)
library(assertthat)

dir <- here("data/occ_macro_vars/lifetime_earnings_cps")
EXTRACT <- "cps_000XX"   # TODO: the new basic-monthly extract

ddi <- read_ipums_ddi(here("data/occ_macro_vars/ipums_vars", paste0(EXTRACT, ".xml")))
data <- read_ipums_micro(ddi, vars = c("YEAR", "MONTH", "MISH", "WTFINL", "AGE", "EDUC", "EMPSTAT", "CLASSWKR",
                                       "SCHLCOLL", "EARNWEEK", "HOURWAGE", "PAIDHOUR", "UHRSWORKORG",
                                       "QEARNWEE", "QHOURWAG"),
                         verbose = FALSE) %>%
  mutate(across(everything(), as.numeric))

# Entry age by education, as in lifetime_earnings_cps.R / entry_exit_indicators_full_omn.R
entry_age_by_educ <- function(educ) {
  case_when(educ >= 2   & educ <= 73  ~ 18,           # no credential / high school diploma
            educ %in% c(80, 81, 90, 100, 110) ~ 21,    # some college, no degree
            educ %in% c(91, 92)  ~ 23,                 # associate's degree
            educ == 111          ~ 21,                 # bachelor's degree
            educ %in% c(120, 121, 122, 123) ~ 23,      # master's degree (or 5+ years of college)
            educ %in% c(124, 125) ~ 30,                # professional or doctoral degree
            TRUE ~ NA_real_)
}

employed   <- c(10, 12)
unemployed <- c(20, 21, 22)
wage_salary <- 20:28                                  # CLASSWKR: private and government employees
enrolled    <- 1:4                                    # SCHLCOLL: high school or college, full or part time

cps <- data %>%
  filter(YEAR >= 2000, YEAR <= 2023,
         MISH %in% c(4, 8),                           # outgoing rotation: earnings questions asked
         EMPSTAT %in% c(employed, unemployed),
         !SCHLCOLL %in% enrolled) %>%
  mutate(entry_age = entry_age_by_educ(EDUC),
         years_since_entry = AGE - entry_age) %>%
  filter(!is.na(entry_age), years_since_entry >= 0, years_since_entry <= 20,
         EMPSTAT %in% unemployed | CLASSWKR %in% wage_salary) %>%
  mutate(
    hourly = case_when(
      PAIDHOUR == 2 & HOURWAGE > 0 & HOURWAGE < 999 ~ HOURWAGE,              # paid by the hour
      EARNWEEK > 0 & EARNWEEK < 9999 & UHRSWORKORG > 0 & UHRSWORKORG < 997 ~ EARNWEEK / UHRSWORKORG,
      TRUE ~ NA_real_),
    fte_pay = if_else(EMPSTAT %in% unemployed, 0, hourly * 2080))

# Employed workers without usable pay (missing, or hours that vary) are dropped rather than set to zero.
print(paste0("Employed, pay not recoverable (dropped): ",
             round(100 * mean(is.na(cps$fte_pay[cps$EMPSTAT %in% employed])), 1), "%"))
cps <- cps %>% filter(!is.na(fte_pay))
assert_that(all(cps$fte_pay >= 0))

print(paste0("Observations: ", nrow(cps), " (years ", min(cps$YEAR), "-", max(cps$YEAR), ")"))

weighted_median <- function(x, w) {
  o <- order(x); x <- x[o]; w <- w[o]
  x[which(cumsum(w) >= sum(w) / 2)[1]]
}

by_year <- cps %>%
  group_by(year = YEAR, years_since_entry) %>%
  summarise(mean_annual_earnings   = weighted.mean(fte_pay, WTFINL),
            median_annual_earnings = weighted_median(fte_pay, WTFINL),
            share_employed         = weighted.mean(EMPSTAT %in% employed, WTFINL),
            mean_fte_pay_employed  = weighted.mean(fte_pay[EMPSTAT %in% employed], WTFINL[EMPSTAT %in% employed]),
            share_pay_imputed      = weighted.mean(QEARNWEE %in% 1:9 | QHOURWAG %in% 1:9, WTFINL),
            n_obs = n(),
            .groups = "drop")

profile <- by_year %>%
  group_by(years_since_entry) %>%
  summarise(across(c(mean_annual_earnings, median_annual_earnings, share_employed, mean_fte_pay_employed,
                     share_pay_imputed), mean),        # equal weight per calendar year
            n_obs = sum(n_obs),
            n_years = n(),
            .groups = "drop") %>%
  arrange(years_since_entry) %>%
  mutate(cum_mean_earnings = cumsum(mean_annual_earnings))

assert_that(all(profile$n_years == 24))               # every year 2000-2023 present at every step
assert_that(nrow(profile) == 21)

print(profile %>% mutate(across(c(mean_annual_earnings, median_annual_earnings, mean_fte_pay_employed,
                                  cum_mean_earnings), round)),
      n = 21)

write.csv(by_year, file.path(dir, "cps_fte_earnings_by_experience_by_year.csv"), row.names = FALSE)
write.csv(profile, file.path(dir, "cps_fte_earnings_by_experience.csv"), row.names = FALSE)
