# Early-career earnings profile (CPS ASEC)

Validation target for the lifetime earnings of new entrants in
`calibration_remote/exposure_entry_policy_exercise.py` (Section 16).

Run from the repository root: `Rscript data/occ_macro_vars/lifetime_earnings_cps/lifetime_earnings_cps.R`
(about 2 minutes; reads the IPUMS extract `data/occ_macro_vars/ipums_vars/cps_00021.dat`).

## Construction (`lifetime_earnings_cps.R`)
- **Sample:** CPS ASEC 2001-2024 (income years 2000-2023), civilian labour force (employed or unemployed),
  since every agent in the model is a labour-force participant.
- **Years since entry:** age minus an education-based entry age, using the same mapping as
  `../BLS_entry_exit_reqs/entry_exit_indicators_full_omn.R` (18 for high school or less, 21 for some college or a
  bachelor's degree, 23 for an associate's or master's degree, 30 for a professional or doctoral degree).
- **Earnings:** annual wage and salary income (`INCWAGE`), zeros included, weighted by `ASECWT`.
- **Dollars:** nominal, computed for each income year and then averaged across 2000-2023 with equal weight.
  This mirrors the model's occupation wages, which are the unweighted mean of nominal OEWS wages over
  2000-2023 (`../OEWS/oews_wage_distributions.R`), so no price index is applied to either side.
- **Statistic:** the mean. For a synthetic cohort the cumulative mean equals the sum of annual means, so it is
  directly comparable to the model's mean cumulative earnings; cumulative medians are not.

## Output
- `cps_earnings_by_experience.csv`: by years since entry (0-20): mean and median annual earnings, share with
  positive earnings, observations, and cumulative mean earnings.
- `cps_earnings_by_experience_by_year.csv`: the same, for each income year.

## Caveats
- A synthetic cohort (different people at each experience level), not a panel.
- `INCWAGE` is actual annual earnings, including part-year work; the model records the annual wage of the job
  held at each yearly snapshot, or zero if unemployed then.
- Potential experience is approximate: it ignores time out of the labour force and delayed schooling.
