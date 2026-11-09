# AI occupational exposure matched to the full_omn occupations (ACS 2010 codes)
#
# Sources (raw copies in raw/, provenance in README.md):
#   - Eloundou, Manning, Mishkin & Rock (2024, Science): dv_rating_beta (E2: LLMs + software),
#     keyed by O*NET-SOC 2019 codes
#   - Anthropic Economic Index: observed_exposure, keyed by SOC 2018 detailed codes
#
# Matching follows BLS_entry_exit_reqs/entry_exit_indicators_full_omn.R: hierarchical, each step only
# for occupations unmatched at all prior steps -- exact SOC 2010, then SOC broad (6-digit), then SOC
# minor (4-digit). The ACS-code and CPS-code steps of that script are not available here (neither
# source carries ACS codes). A final SOC major (2-digit) step is added for occupations whose whole
# minor group is absent from a source (only Tour and Travel Guides, 39-7010, in the Anthropic data).
#
# Both sources use the SOC 2018 / O*NET-SOC 2019 taxonomy while our crosswalk is SOC 2010, so each
# source is first translated to SOC 2010 with O*NET's 2010-to-2019 crosswalk. Without this step,
# codes renumbered in 2018 (e.g. 15-1132 software developers -> 15-1252/15-1256) only match at the
# coarse SOC-minor level.
#
# Output (this folder):
#   ai_exposure_by_acs_full_omn.csv        one row per network occupation (network order), raw exposure
#                                          score and match level per source -- read by
#                                          calibration_remote/exposure_entry_policy_exercise.py
#   ai_exposure_match_summary_full_omn.tex  matching summary table (same layout as the entry/exit table)
library(tidyverse)
library(here)
library(assertthat)
library(knitr)
library(kableExtra)

dir <- here("data/occ_macro_vars/eloundou_exposure")

# Network occupations: ACS 2010 codes with their SOC 2010 code (same crosswalk and cleaning as
# entry_exit_indicators_full_omn.R); row order = network occupation_id
cw <- read.csv(here("data/crosswalk_occ_soc_cps_codes_full_omn.csv")) %>%
  tibble %>%
  mutate(SOC2010 = gsub("X", "0", SOC2010),
         # some labels in the crosswalk are just the code; use the OCC2010 description there
         acs_occ_code_label = if_else(grepl("^[0-9]+$", acs_occ_code_label), OCC2010_desc, acs_occ_code_label),
         SOC_broad = paste0(substr(SOC2010, 1, 6), 0),
         SOC_minor = paste0(substr(SOC2010, 1, 4), "000"),
         SOC_major = substr(SOC2010, 1, 2)) %>%
  select(id = X, acs_occ_code, acs_occ_code_label, SOC2010, SOC_broad, SOC_minor, SOC_major)

assert_that(nrow(cw) == n_distinct(cw$acs_occ_code))

# O*NET-SOC 2010 <-> 2019 crosswalk, collapsed to 6-digit SOC codes
soc_19_to_10 <- read.csv(file.path(dir, "raw/onet_2010_to_2019_crosswalk.csv"), check.names = FALSE) %>%
  tibble %>%
  transmute(soc2010 = substr(`O*NET-SOC 2010 Code`, 1, 7),
            soc2019 = substr(`O*NET-SOC 2019 Code`, 1, 7)) %>%
  distinct()

# Source scores at the 6-digit SOC 2018 level (Eloundou: mean over O*NET sub-occupations)
eloundou <- read.csv(file.path(dir, "raw/eloundou_2024_occ_level.csv"), check.names = FALSE) %>%
  tibble %>%
  transmute(soc2019 = substr(`O*NET-SOC Code`, 1, 7), exposure = dv_rating_beta) %>%
  group_by(soc2019) %>%
  summarise(exposure = mean(exposure), .groups = "drop")

anthropic <- read.csv(file.path(dir, "raw/anthropic_economic_index_job_exposure.csv")) %>%
  tibble %>%
  transmute(soc2019 = occ_code, exposure = observed_exposure)

# Translate a source to SOC 2010. A SOC 2010 code that maps to several 2018 codes gets their mean;
# 2018 codes absent from the crosswalk are kept as they are (assumed unchanged).
to_soc2010 <- function(src, name) {
  translated <- src %>%
    left_join(soc_19_to_10, by = "soc2019", relationship = "many-to-many") %>%
    mutate(translated = !is.na(soc2010),
           soc2010 = coalesce(soc2010, soc2019))
  print(paste0(name, ": ", sum(!translated$translated), " of ", nrow(src),
               " source codes not in the O*NET crosswalk (kept as is)"))
  translated %>%
    group_by(soc2010) %>%
    summarise(exposure = mean(exposure), .groups = "drop") %>%
    mutate(SOC_broad = paste0(substr(soc2010, 1, 6), 0),
           SOC_minor = paste0(substr(soc2010, 1, 4), "000"),
           SOC_major = substr(soc2010, 1, 2))
}

test_missing <- function(df, name, level) {
  paste0(name, " -- ", level, ": matched ", sum(!is.na(df$exposure)),
         ", still missing ", sum(is.na(df$exposure))) %>% print(.)
}

# Hierarchical matching: exact SOC 2010 -> SOC broad -> SOC minor -> SOC major
match_hierarchy <- function(src, name) {
  by_broad <- src %>% group_by(SOC_broad) %>% summarise(exposure = mean(exposure), .groups = "drop")
  by_minor <- src %>% group_by(SOC_minor) %>% summarise(exposure = mean(exposure), .groups = "drop")
  by_major <- src %>% group_by(SOC_major) %>% summarise(exposure = mean(exposure), .groups = "drop")

  soc2010_matches <- cw %>%
    left_join(select(src, soc2010, exposure), by = c("SOC2010" = "soc2010")) %>%
    mutate(match_level = if_else(!is.na(exposure), "SOC 2010 (exact)", NA_character_))
  test_missing(soc2010_matches, name, "SOC 2010 (exact)")

  soc_broad_matches <- soc2010_matches %>%
    filter(is.na(exposure)) %>%
    select(-exposure, -match_level) %>%
    left_join(by_broad, by = "SOC_broad") %>%
    mutate(match_level = if_else(!is.na(exposure), "SOC broad (6-digit)", NA_character_))
  test_missing(soc_broad_matches, name, "SOC broad")

  soc_minor_matches <- soc_broad_matches %>%
    filter(is.na(exposure)) %>%
    select(-exposure, -match_level) %>%
    left_join(by_minor, by = "SOC_minor") %>%
    mutate(match_level = if_else(!is.na(exposure), "SOC minor (4-digit)", NA_character_))
  test_missing(soc_minor_matches, name, "SOC minor")

  soc_major_matches <- soc_minor_matches %>%
    filter(is.na(exposure)) %>%
    select(-exposure, -match_level) %>%
    left_join(by_major, by = "SOC_major") %>%
    mutate(match_level = if_else(!is.na(exposure), "SOC major (2-digit)", NA_character_))
  test_missing(soc_major_matches, name, "SOC major")

  out <- bind_rows(filter(soc2010_matches, !is.na(exposure)),
                   filter(soc_broad_matches, !is.na(exposure)),
                   filter(soc_minor_matches, !is.na(exposure)),
                   filter(soc_major_matches, !is.na(exposure))) %>%
    select(acs_occ_code, exposure, match_level)

  assert_that(nrow(out) == nrow(cw))                     # every occupation matched exactly once
  assert_that(!any(duplicated(out$acs_occ_code)))
  out
}

eloundou_acs  <- match_hierarchy(to_soc2010(eloundou, "Eloundou"), "Eloundou")
anthropic_acs <- match_hierarchy(to_soc2010(anthropic, "Anthropic"), "Anthropic")

final <- cw %>%
  select(id, acs_occ_code, acs_occ_code_label, SOC2010) %>%
  left_join(rename(eloundou_acs, eloundou_exposure = exposure, eloundou_match_level = match_level),
            by = "acs_occ_code") %>%
  left_join(rename(anthropic_acs, anthropic_exposure = exposure, anthropic_match_level = match_level),
            by = "acs_occ_code")

assert_that(identical(final$acs_occ_code, cw$acs_occ_code))   # network order preserved
assert_that(!anyNA(final$eloundou_exposure), !anyNA(final$anthropic_exposure))

write.csv(final, file.path(dir, "ai_exposure_by_acs_full_omn.csv"), row.names = FALSE)

# Matching summary (same layout as the entry/exit indicators table)
levels_order <- c("SOC 2010 (exact)", "SOC broad (6-digit)", "SOC minor (4-digit)", "SOC major (2-digit)")
summarise_matches <- function(df, name) {
  df %>%
    count(match_level) %>%
    right_join(tibble(match_level = levels_order), by = "match_level") %>%
    mutate(n = replace_na(n, 0L),
           match_level = factor(match_level, levels = levels_order)) %>%
    arrange(match_level) %>%
    transmute(`Matching Level` = as.character(match_level),
              !!paste0(name, " matched") := n,
              !!paste0(name, " cumulative %") := round(100 * cumsum(n) / nrow(cw), 1))
}
match_summary <- summarise_matches(eloundou_acs, "Eloundou") %>%
  left_join(summarise_matches(anthropic_acs, "Anthropic"), by = "Matching Level")
print(match_summary)

tbl_body <- match_summary %>%
  kable(format = "latex", booktabs = TRUE, linesep = "",
        col.names = c("\\makecell{Matching\\\\Level}",
                      "\\makecell{Eloundou\\\\Matched}", "\\makecell{Eloundou\\\\Cumulative \\%}",
                      "\\makecell{Anthropic\\\\Matched}", "\\makecell{Anthropic\\\\Cumulative \\%}"),
        escape = FALSE)

writeLines(c(
  "\\begin{table}[H]",
  "\\centering",
  "\\caption{Occupational Code Matching Summary: AI Exposure Scores}",
  "\\label{si_tbl:ai_exposure_match_summary}",
  "\\begin{threeparttable}",
  tbl_body,
  "\\begin{tablenotes}[flushleft]",
  "\\small",
  paste0("\\item \\textit{Note: }Occupational Mobility Network (N = ", nrow(cw), "). Both sources are translated ",
         "from SOC 2018 (O*NET-SOC 2019) to SOC 2010 with the O*NET 2010--2019 crosswalk, then matched ",
         "hierarchically; each step includes only occupations unmatched at all prior steps. Broad and minor ",
         "matches take the unweighted mean of the source scores in that group; the SOC major step is needed only for ",
         "Tour and Travel Guides in the Anthropic data."),
  "\\item \\textit{Sources:} Eloundou et al. (2024), E2 exposure; Anthropic Economic Index, observed exposure.",
  "\\end{tablenotes}",
  "\\end{threeparttable}",
  "\\end{table}"
), file.path(dir, "ai_exposure_match_summary_full_omn.tex"))
