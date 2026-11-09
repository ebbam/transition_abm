# Full-time variant of lifetime_earnings_cps.R: the same CPS ASEC synthetic cohort (labour-force participants,
# income years 2000-2023, education-based entry ages, nominal dollars averaged with equal weight over income
# years, means), but restricted to full-time workers so it is comparable with the model, whose wages are
# full-time-equivalent OEWS wages and which has no part-time or part-year work.
#
# Data: IPUMS-CPS extract cps_00023 (data/occ_macro_vars/ipums_vars/), ASEC 2001-2024 samples. The extract also
# holds basic monthly samples (2022-2025), which are dropped here.
#
# Sample: labour-force participants at the March interview (EMPSTAT employed or unemployed), as in
# lifetime_earnings_cps.R. Anyone with zero wage income last year is kept at $0, as in the model (an entrant
# unemployed at a snapshot earns $0). Of those with positive wage income, part-time workers last year
# (FULLPART = 2, or unknown) are dropped. Full time = usually 35+ hours a week in the weeks worked.
#
# Earnings measures, one column each in the output:
#   ft_reported    full-time workers' INCWAGE as reported (part-year workers earn less than a full year)
#   ft_annualised  full-time workers' INCWAGE scaled to 52 weeks: INCWAGE x 52 / WKSWORK1
#   ftfy           full-time, full-year (WKSWORK1 >= 50) workers only, INCWAGE as reported
#   fte_all        no part-time exclusion: everyone with positive earnings at a full-time-equivalent rate,
#                  INCWAGE / (WKSWORK1 x UHRSWORKLY) x 2,080 (the model's OEWS annualisation)
#   all            no full-time restriction at all; should reproduce cps_earnings_by_experience.csv
#                  (lifetime_earnings_cps.R on cps_00021), as a check that the two extracts agree
# Zero earners are in every measure's sample.
#
# Output (this folder; the other scripts' outputs are left as they are):
#   cps_earnings_by_experience_fulltime.csv   profile by years since entry (0-20), averaged over income years
library(tidyverse)
library(here)
library(ipumsr)
library(assertthat)

dir <- here("data/occ_macro_vars/lifetime_earnings_cps")

ddi <- read_ipums_ddi(here("data/occ_macro_vars/ipums_vars/cps_00023.xml"))
data <- read_ipums_micro(ddi, vars = c("YEAR", "ASECWT", "AGE", "EDUC", "EMPSTAT", "INCWAGE", "FULLPART",
                                       "WKSWORK1", "UHRSWORKLY"),
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

cps <- data %>%
  mutate(income_year = YEAR - 1,
         entry_age = entry_age_by_educ(EDUC),
         years_since_entry = AGE - entry_age) %>%
  filter(income_year >= 2000, income_year <= 2023,
         EMPSTAT %in% c(10, 12, 20, 21, 22),
         !INCWAGE %in% c(99999998, 99999999),
         !is.na(entry_age),
         years_since_entry >= 0, years_since_entry <= 20) %>%
  mutate(zero       = INCWAGE == 0,
         full_time  = INCWAGE > 0 & FULLPART == 1,
         weeks_ok   = WKSWORK1 >= 1 & WKSWORK1 <= 52,
         hours_ok   = UHRSWORKLY >= 1 & UHRSWORKLY < 999,
         ft_reported   = if_else(zero, 0, if_else(full_time, INCWAGE, NA_real_)),
         ft_annualised = if_else(zero, 0, if_else(full_time & weeks_ok, INCWAGE * 52 / WKSWORK1, NA_real_)),
         ftfy          = if_else(zero, 0, if_else(full_time & WKSWORK1 >= 50, INCWAGE, NA_real_)),
         fte_all       = if_else(zero, 0, if_else(weeks_ok & hours_ok,
                                                  INCWAGE / (WKSWORK1 * UHRSWORKLY) * 2080, NA_real_)),
         all           = INCWAGE)

measures <- c("ft_reported", "ft_annualised", "ftfy", "fte_all", "all")

# Positive earners outside each measure's sample, by years since entry
print("Weighted share of observations outside each measure's sample, by years since entry:")
print(cps %>%
        group_by(years_since_entry) %>%
        summarise(across(all_of(measures), ~ round(weighted.mean(is.na(.x), ASECWT), 3)), .groups = "drop"),
      n = 21)

by_year <- map_dfr(measures, function(m) {
  cps %>%
    filter(!is.na(.data[[m]])) %>%
    group_by(income_year, years_since_entry) %>%
    summarise(mean_annual_earnings = weighted.mean(.data[[m]], ASECWT),
              n_obs = n(),
              .groups = "drop") %>%
    mutate(measure = m)
})

profile <- by_year %>%
  group_by(measure, years_since_entry) %>%
  summarise(mean_annual_earnings = mean(mean_annual_earnings),   # equal weight per income year
            n_obs = sum(n_obs),
            n_income_years = n(),
            .groups = "drop") %>%
  arrange(measure, years_since_entry) %>%
  group_by(measure) %>%
  mutate(cum_mean_earnings = cumsum(mean_annual_earnings)) %>%
  ungroup()

assert_that(all(profile$n_income_years == 24))
assert_that(nrow(profile) == 21 * length(measures))

wide <- profile %>%
  select(measure, years_since_entry, mean_annual_earnings) %>%
  pivot_wider(names_from = measure, values_from = mean_annual_earnings)
print(wide %>% mutate(across(-years_since_entry, round)), n = 21)

# Check: the unrestricted measure against lifetime_earnings_cps.R's output from cps_00021
base <- read.csv(file.path(dir, "cps_earnings_by_experience.csv"))
check <- wide %>% select(years_since_entry, all) %>%
  left_join(base %>% select(years_since_entry, base = mean_annual_earnings), by = "years_since_entry")
print(paste0("Max abs. difference vs cps_earnings_by_experience.csv: $",
             round(max(abs(check$all - check$base)))))

write.csv(profile, file.path(dir, "cps_earnings_by_experience_fulltime.csv"), row.names = FALSE)
