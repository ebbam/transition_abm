# Calibration targets for the sticky-belief rule in the optimizing-search benchmark
# (calibration_remote/job_search_notebook.ipynb): Mueller, Spinnewijn & Topa (2021, AER), Figure 3 --
# elicited vs realized 3-month job-finding probability by duration of unemployment (SCE).
#
# Writes (this folder):
#   mueller_fig3_targets.csv           the four duration bins: elicited and realized 3-month job-finding,
#                                      standard errors and observations, from MST's own output
#                                      (Appendix Table D6, Panel A = the data plotted in Figure 3)
#   mueller_fig3_duration_weights.csv  within each bin, the weighted share of observations at each month of
#                                      unemployment (tau = floor(udur) + 1, the model's time_unemployed at
#                                      search; capped at 60), from the SCE microdata with the Figure 3 sample
#                                      definition used in mueller_repl_analysis.R -- used to average the
#                                      model's month-by-month beliefs within each bin the way MST average
#                                      over the stock of unemployed
library(tidyverse)
library(here)
library(readxl)
library(assertthat)

base <- here("data/behav_params/Mueller_Replication/")

targets <- read_excel(paste0(base, "120501-V1/MST/EMPIRICAL_ANALYSIS/Output/Appendix/sce_tabD6panelA.xlsx"),
                      sheet = 2, col_names = c("label", "pjob_find", "se_pjob_find", "rjob_find", "se_rjob_find", "nobs")) %>%
  filter(label != "Full Sample") %>%
  mutate(udur_bin = row_number())

assert_that(nrow(targets) == 4)

data <- readRDS(paste0(base, "sce_datafile_em_2025.RDS"))
sample_fig3 <- data %>%
  filter(age >= 20 & age <= 65 & in_sample_2 == 1, !is.na(udur_bins), !is.na(weight), !is.na(find_job_3mon))

# Check that this sample reproduces the official elicited probabilities
check <- sample_fig3 %>%
  group_by(udur_bins) %>%
  summarise(pjob_find_own = weighted.mean(find_job_3mon, weight), .groups = "drop")
assert_that(isTRUE(all.equal(round(check$pjob_find_own, 3), round(targets$pjob_find, 3))))

weights <- sample_fig3 %>%
  mutate(tau = pmin(floor(udur) + 1, 60)) %>%
  group_by(udur_bin = udur_bins, tau) %>%
  summarise(w = sum(weight), n = n(), .groups = "drop") %>%
  group_by(udur_bin) %>%
  mutate(share = w / sum(w)) %>%
  ungroup() %>%
  select(udur_bin, tau, share, n)

print(targets)
print(weights %>% group_by(udur_bin) %>% summarise(tau_min = min(tau), tau_max = max(tau),
                                                   mean_tau = sum(share * tau), n = sum(n)))

write.csv(targets, paste0(base, "mueller_fig3_targets.csv"), row.names = FALSE)
write.csv(weights, paste0(base, "mueller_fig3_duration_weights.csv"), row.names = FALSE)
