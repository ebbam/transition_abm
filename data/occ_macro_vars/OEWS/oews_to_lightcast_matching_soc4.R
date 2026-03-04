# =============================================================================
# Build Annual Employment Series Matched to Vacancy SOC-4 Codes (2010-2023)
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   Constructs a complete annual employment series (2010-2023) for each
#   SOC-4 "broad" occupational code present in the Lightcast vacancy data,
#   sourced from the OEWS employment dataset.
#
# SOC HIERARCHY NOTE:
#   SOC-4 codes (e.g. 11-1010) correspond to the "broad" group level in OEWS.
#   Their sub-codes are SOC-5 detailed codes (e.g. 11-1011). Construction
#   uses a 6-character prefix match (e.g. "11-101" from "11-1010") which
#   reliably isolates the correct detailed sub-codes because the SOC numbering
#   convention ensures sub-codes share their parent's first 6 characters.
#
# KNOWN DATA ISSUES:
#
#   1. 2010-2011: The OEWS did not report "broad" group aggregates in 2010
#      or 2011 — only detailed codes (with occ_group = NA) exist. These are
#      reconstructed by summing detailed sub-codes via 6-character prefix
#      matching. This affects all 443 matched codes.
#
#   2. 15-12XX codes (15-1210, 15-1220, 15-1230, 15-1240, 15-1250, 15-1290):
#      Created by the 2018 SOC revision reorganisation of Computer Occupations.
#      No broad or detailed sub-codes exist under these prefixes before 2019,
#      and the old 15-11XX broad groups do not map cleanly 1-to-1 to the new
#      structure. Direct match available for 2019-2023 only.
#      STATUS: UNRESOLVABLE for 2010-2018.
#
#   3. 31-1120 (Home Health & Personal Care Aides) and
#      31-1130 (Nursing & Psychiatric Aides):
#      Created by the 2018 SOC revision split of broad group 31-1010. No
#      detailed sub-codes exist under "31-112X" or "31-113X" prefixes before
#      2019, and the predecessor 31-1010 cannot be split between the two
#      without an official crosswalk. Direct match available for 2019-2023.
#      STATUS: UNRESOLVABLE for 2010-2018.
#
#   4. 19-5010 (Occupational Health & Safety): The 2018 SOC revision moved
#      this broad group from 29-9010 (under Healthcare) to 19-5010 (under
#      Life, Physical & Social Science). The predecessor broad-group code
#      29-9010 is used directly for 2012-2018. For 2010-2011, 29-9010 also
#      does not exist as a broad-group aggregate, so the detailed sub-codes
#      29-9011 and 29-9012 are summed directly via a dedicated crosswalk
#      (Step 2b). The general sub-code construction in Step 4 cannot handle
#      this because the old sub-codes (29-90XX) have a different prefix
#      than the new code (19-5010), so prefix matching would find nothing.
#
#   5. Codes with no OEWS coverage at all (completely unresolvable):
#      25-1090: Not a valid OEWS broad group
#      45-3030: Fishing & Hunting — not in OEWS (excluded industry)
#      55-9990: Military — not covered by OEWS
#      99-9990: Unclassified — not covered by OEWS
#
#   6. Additional codes with gaps beyond 2010-2011 due to SOC revisions
#      where no sub-codes exist for the missing years (unresolvable gaps):
#      11-9170  missing 2010-2020 | 15-2050  missing 2010-2020
#      25-3040  missing 2010-2020 | 29-1210  missing 2010-2020
#      29-1240  missing 2010-2020 | 29-9020  missing 2010-2020
#      53-1040  missing 2010-2016 | 53-3050  missing 2010-2018
#      (Sub-code construction will return NA for these and they will be
#      excluded from the output — flagged in the coverage report.)
#
# SOURCE HIERARCHY (applied in order of preference):
#   "direct_match"    -> OEWS broad-group row matches vacancy code exactly
#   "predecessor_code"-> Pre-2018 SOC code used for 19-5010 (2012-2018 and
#                        2010-2011 via detailed sub-codes)
#   "constructed"     -> Summed from detailed sub-codes via 6-char prefix
#                        (2010-2011 and any other gaps not covered above)
#
# OUTPUT:
#   emp_series_soc4_matched_to_vacancies.RDS / .csv
#   Columns: soc_4, soc_4_name, year, tot_emp, source
# =============================================================================

library(tidyverse, quietly = TRUE)
library(here, quietly = TRUE)
library(ggrepel, quietly = TRUE)
source(here('code/formatting/plot_dicts.R'))
new_data = FALSE

# --- File paths — adjust as needed -------------------------------------------
emp_path <- here("data/occ_macro_vars/OEWS/occ_employment_levels_oews.csv")
vac_path <- here("data/lightcast_vacancies/lightcast_soc_4_2010_2024.RDS")
out_path <- "data/lightcast_vacancies/emp_series_soc4_matched_to_vacancies"
if(new_data){
# --- Load data ----------------------------------------------------------------
emp <- read.csv(emp_path) %>%
  filter(year >= 2010, year <= 2023)

vac <- readRDS(vac_path)

vac_codes  <- sort(unique(vac$soc_4))
soc4_names <- vac %>% select(soc_4, soc_4_name) %>% distinct()

cat("Vacancy SOC-4 codes to match:", length(vac_codes), "\n")

# =============================================================================
# STEP 1: Direct match
# Pull rows where the OEWS broad-group code exactly matches a vacancy SOC-4 code
# =============================================================================
emp_broad <- emp %>% filter(occ_group == "broad")

emp_direct <- emp_broad %>%
  filter(occ_code %in% vac_codes) %>%
  select(soc_4 = occ_code, year, tot_emp) %>%
  mutate(source = "direct_match")

cat("Direct match rows:", nrow(emp_direct), "\n")

# =============================================================================
# STEP 2: SOC 2018 reclassification crosswalk
# 19-5010 is the only SOC-4 vacancy code with a clean 1-to-1 predecessor
# broad-group code (29-9010) that can be used directly for 2012-2018.
# 2010-2011 handled by Step 2b below.
# =============================================================================
crosswalk <- c(
  "19-5010" = "29-9010"  # Occupational Health & Safety (29-9010 exists 2012-2018)
)

predecessor_rows <- do.call(rbind, lapply(names(crosswalk), function(new_code) {
  old_code <- crosswalk[new_code]
  emp_broad %>%
    filter(occ_code == old_code) %>%
    select(year, tot_emp) %>%
    mutate(soc_4 = new_code, source = "predecessor_code")
}))

cat("Predecessor crosswalk rows:", nrow(predecessor_rows), "\n")

# Combine direct and predecessor rows so Step 2b knows which years are covered
emp_so_far <- bind_rows(emp_direct, predecessor_rows)

# =============================================================================
# STEP 2b: 2010-2011 fix for cross-major-group reclassifications
#
# 19-5010 in 2010-2011: its predecessor broad code 29-9010 also does not
# exist as a broad aggregate in those years. The detailed codes 29-9011 and
# 29-9012 do exist and sum cleanly. The general construction in Step 4
# cannot find these because they have prefix "29-9" not "19-5".
# =============================================================================
crosswalk_detailed <- list(
  "19-5010" = c("29-9011", "29-9012")  # Occ Health & Safety specialists + technicians
)

early_years_rows <- do.call(rbind, lapply(names(crosswalk_detailed), function(new_code) {
  old_codes     <- crosswalk_detailed[[new_code]]
  years_covered <- unique(emp_so_far$year[emp_so_far$soc_4 == new_code])
  years_needed  <- setdiff(2010:2023, years_covered)
  
  emp %>%
    filter(occ_code %in% old_codes, year %in% years_needed) %>%
    group_by(year) %>%
    summarise(tot_emp = sum(tot_emp, na.rm = TRUE), .groups = "drop") %>%
    mutate(soc_4 = new_code, source = "predecessor_code")
}))

cat("Early-year predecessor rows:", nrow(early_years_rows), "\n")

# Update emp_so_far before Step 3 identifies gaps
emp_so_far <- bind_rows(emp_so_far, early_years_rows)

# =============================================================================
# STEP 3: Identify remaining missing code x year slots
# =============================================================================

# Completely unresolvable codes (no OEWS coverage at all)
unresolvable_no_data <- c("25-1090", "45-3030", "55-9990", "99-9990")

# SOC 2018 reclassifications with no sub-codes pre-2019 — construction would
# return NA anyway, but excluding explicitly avoids unnecessary rowwise calls
unresolvable_pre2019 <- c(
  "15-1210", "15-1220", "15-1230", "15-1240", "15-1250", "15-1290",
  "31-1120", "31-1130"
)

full_grid <- expand.grid(
  soc_4 = vac_codes,
  year  = 2010:2023,
  stringsAsFactors = FALSE
)

missing_slots <- full_grid %>%
  anti_join(emp_so_far, by = c("soc_4", "year")) %>%
  filter(!soc_4 %in% unresolvable_no_data) %>%
  filter(!(soc_4 %in% unresolvable_pre2019 & year < 2019))

cat("Remaining slots to construct from sub-codes:", nrow(missing_slots), "\n")

# =============================================================================
# STEP 4: Construct remaining missing slots by summing detailed sub-codes
#
# At the broad group level (SOC-4), sub-codes are detailed occupations (SOC-5).
# The 6-character prefix of a broad code uniquely identifies its sub-codes:
#   Broad 11-1010 -> prefix "11-101" -> detailed 11-1011, 11-1012, ...
#   Broad 11-1020 -> prefix "11-102" -> detailed 11-1021, 11-1022, ...
# In 2010-2011, detailed codes exist but with occ_group = NA; these are
# included by filtering for occ_group == "detailed" OR is.na(occ_group).
# =============================================================================
emp_detailed <- emp %>%
  filter(occ_group == "detailed" | is.na(occ_group))

construct_from_detailed <- function(broad_code, yr, emp_data) {
  prefix <- substr(broad_code, 1, 6)  # e.g. "11-101" from "11-1010"
  
  sub <- emp_data %>%
    filter(
      year == yr,
      grepl(paste0("^", prefix), occ_code),
      occ_code != broad_code  # exclude target code itself (shouldn't match, but safe)
    )
  
  if (nrow(sub) == 0) return(NA_real_)
  sum(sub$tot_emp, na.rm = TRUE)
}

constructed <- missing_slots %>%
  rowwise() %>%
  mutate(
    tot_emp = construct_from_detailed(soc_4, year, emp_detailed),
    source  = "constructed"
  ) %>%
  ungroup() %>%
  filter(!is.na(tot_emp))

cat("Successfully constructed:", nrow(constructed), "slots\n")

# =============================================================================
# STEP 5: Combine all sources and finalise
# =============================================================================
emp_matched <- bind_rows(emp_direct, predecessor_rows, early_years_rows, constructed) %>%
  left_join(soc4_names, by = "soc_4") %>%
  select(soc_4, soc_4_name, year, tot_emp, source) %>%
  arrange(soc_4, year)

# =============================================================================
# STEP 6: Coverage report
# =============================================================================
cat("\n=== Coverage Report ===\n")

coverage <- emp_matched %>%
  group_by(soc_4, soc_4_name) %>%
  summarise(
    n_years         = n_distinct(year),
    missing_years   = paste(setdiff(2010:2023, unique(year)), collapse = ", "),
    pct_constructed = paste0(round(mean(source == "constructed") * 100), "%"),
    .groups         = "drop"
  )

cat("\nCodes with PARTIAL coverage (some years unresolvable):\n")
partial <- coverage %>% filter(n_years < 14 & n_years > 0)
if (nrow(partial) > 0) print(partial, n = 50) else cat("None\n")

cat("\nCodes with NO coverage (completely unresolvable):\n")
no_data <- setdiff(vac_codes, unique(emp_matched$soc_4))
if (length(no_data) > 0) {
  print(soc4_names %>% filter(soc_4 %in% no_data))
} else {
  cat("None\n")
}

cat("\nSummary:\n")
cat("  Total vacancy codes:    ", length(vac_codes), "\n")
cat("  Fully matched (14 yrs): ", sum(coverage$n_years == 14), "\n")
cat("  Partially matched:      ", nrow(partial), "\n")
cat("  Unresolvable:           ", length(no_data), "\n")
cat("  Total rows in output:   ", nrow(emp_matched), "\n")
cat("  From direct match:      ", sum(emp_matched$source == "direct_match"), "\n")
cat("  From predecessor code:  ", sum(emp_matched$source == "predecessor_code"), "\n")
cat("  From sub-code sums:     ", sum(emp_matched$source == "constructed"), "\n")

# =============================================================================
# STEP 7: Save
# =============================================================================
saveRDS(emp_matched, paste0(out_path, ".RDS"))
write.csv(emp_matched, paste0(out_path, ".csv"), row.names = FALSE)

cat("\nSaved to:", paste0(out_path, ".RDS / .csv"), "\n")
cat("\nOutput structure:\n")
print(head(emp_matched, 10))

# =============================================================================
# STEP 8: Create Vacancy Rate Data
# =============================================================================
vac_rates_monthly <- vac %>%
  left_join(emp_matched, by = c("soc_4", "soc_4_name", "year")) %>%
  filter(!(soc_4 %in% unresolvable_no_data)) %>%
  mutate(
    vac_rate = n_ft_vacancies / tot_emp,
    date     = as.Date(paste0(year, "-", month, "-01"))
  ) %>%
  filter(year < 2024)

vac_rates_monthly %>%
  saveRDS(here("data/lightcast_vacancies/soc_4_vacancy_rate.RDS"))

}else{
  vac_rates_monthly <- readRDS(here("data/lightcast_vacancies/soc_4_vacancy_rate.RDS"))
}

national_vac <- read.csv(here("calibration_remote/data/macro_vars/collated_macro_observations.csv")) %>%
  filter(!is.na(VACRATE)) %>%
  mutate(date = as.Date(DATE)) %>%
  filter(date >= min(vac_rates_monthly$date), date <= max(vac_rates_monthly$date)) %>%
  select(date, VACRATE)

vac_rates_monthly %>%
  ggplot(aes(x = date, y = vac_rate, group = soc_4_name, color = soc_4_name)) +
  geom_line() +
  theme(legend.position = "none") +
  labs(
    title = "Vacancy Rate by Occupation (SOC-4)",
    y     = "Vacancy Rate (Monthly Vacancies / Annual Employment)",
    x     = "Date"
  )

vac_rates_monthly %>%
  mutate(year = as.factor(year)) %>%
  ggplot(aes(x = vac_rate, colour = year, group = year, fill = year)) +
  geom_density(alpha = 0.15, linewidth = 0.7) +
  scale_colour_viridis_d(option = "turbo") +
  scale_fill_viridis_d(option = "turbo") +
  scale_x_log10() +
  labs(
    x      = "Vacancy Rate (log-scale)",
    y      = "Density",
    title  = "Distribution of SOC-4 Occupational Vacancy Rates by Year (log10 X-scale)",
    colour = "Year",
    fill   = "Year"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

vac_rates_monthly %>% 
  mutate(year = as.factor(year)) %>% 
  ggplot(aes(x = vac_rate, colour = year, group = year, fill = year)) +
  geom_density(alpha = 0.15, linewidth = 0.7) +
  scale_colour_viridis_d(option = "turbo") +
  scale_fill_viridis_d(option = "turbo") +
  labs(
    x     = "Vacancy Rate (log-scale)",
    y     = "Density",
    title = "Distribution of Occupational Vacancy Rates by Year (log10 X-scale)",
    colour = "Year",
    fill   = "Year"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right") + scale_x_log10()

vac_rates_monthly%>% 
  mutate(year = as.factor(year)) %>% 
  ggplot(aes(x = vac_rate, colour = year, group = year, fill = year)) +
  geom_density(alpha = 0.15, linewidth = 0.7) +
  scale_colour_viridis_d(option = "turbo") +
  scale_fill_viridis_d(option = "turbo") +
  labs(
    x     = "Vacancy Rate (log-scale)",
    y     = "Density",
    title = "Distribution of Occupational Vacancy Rates by Year (log10 X-scale)",
    colour = "Year",
    fill   = "Year"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right") + scale_x_log10()

# --- Identify occupations to highlight ---------------------------------------
vac_rates_monthly <- vac_rates_monthly %>% 
  mutate(soc_4_name = str_wrap(soc_4_name, 40))
highlight_stats <- vac_rates_monthly%>%
  group_by(soc_4_name) %>%
  summarise(
    mean_vac = mean(vac_rate, na.rm = TRUE),
    sd_vac   = sd(vac_rate,   na.rm = TRUE),
    cv_vac   = sd_vac / mean_vac,
    .groups  = "drop"
  )

highlighted <- bind_rows(
  highlight_stats %>% slice_max(mean_vac, n = 2) %>% mutate(reason = "Mean", extreme = "High"),
  highlight_stats %>% slice_min(mean_vac, n = 2) %>% mutate(reason = "Mean", extreme = "Low"),
  highlight_stats %>% slice_max(cv_vac,   n = 2) %>% mutate(reason = "CV",   extreme = "High"),
  highlight_stats %>% slice_min(cv_vac,   n = 2) %>% mutate(reason = "CV",   extreme = "Low")
) %>%
  group_by(soc_4_name) %>%
  slice(1) %>%
  ungroup()

# --- Build named colour vector (reds for High, blues for Low) ----------------
high_occs <- highlighted %>% filter(extreme == "High") %>% pull(soc_4_name)
low_occs  <- highlighted %>% filter(extreme == "Low")  %>% pull(soc_4_name)

red_shades  <- colorRampPalette(c("#ff9999", "#8B0000"))(length(high_occs))
blue_shades <- colorRampPalette(c("#99bbff", "#00008B"))(length(low_occs))

colour_map <- c(
  setNames(red_shades,  high_occs),
  setNames(blue_shades, low_occs)
)

# --- Build named linetype vector (one entry per occupation) ------------------
linetype_map <- setNames(
  ifelse(highlighted$reason == "Mean", "solid", "dashed"),
  highlighted$soc_4_name
)

# Shared legend title — identical string in both scales triggers legend merge
legend_title <- "Occupation\nRed = high  |  Blue = low\nSolid = mean  |  Dashed = CV"

# --- Data subsets ------------------------------------------------------------
vac_grey      <- vac_rates_monthly %>% filter(!soc_4_name %in% highlighted$soc_4_name)
vac_highlight <- vac_rates_monthly %>%
  filter(soc_4_name %in% highlighted$soc_4_name) %>%
  left_join(highlighted %>% select(soc_4_name, reason, extreme), by = "soc_4_name")

label_data <- vac_highlight %>%
  group_by(soc_4_name, reason, extreme) %>%
  slice_max(date, n = 1) %>%
  ungroup()

# Choose an x anchor where each line is clearly visible for the pointer
national_x  <- as.Date("2011-05-01")
lightcast_x <- as.Date("2014-01-01")

# Look up the y values at those x positions for the segment endpoints
national_y_at_x  <- national_vac$VACRATE[national_vac$date == national_x]
lightcast_y_at_x <- vac_rate_total$vac_rate[vac_rate_total$date == lightcast_x]
# --- Plot --------------------------------------------------------------------
ggplot() +
  geom_line(
    data      = vac_grey,
    aes(x = date, y = vac_rate, group = soc_4_name),
    colour    = "grey80",
    linewidth = 0.4,
    alpha     = 0.4
  ) +
  # linetype mapped to soc_4_name (not reason) so it merges with colour legend
  geom_line(
    data = vac_highlight,
    aes(x = date, y = vac_rate, group = soc_4_name,
        colour   = soc_4_name,
        linetype = soc_4_name),
    linewidth = 0.7
  ) +
  # National vacancy rate overlaid in black
  geom_line(
    data      = national_vac,
    aes(x = date, y = VACRATE, group = 1),
    colour    = "black",
    alpha = 0.7,
    linewidth = 1,
    linetype  = "solid",
    inherit.aes = FALSE
  ) +
  # Pointer lines from label down to the relevant line
  annotate(
    "segment",
    x    = national_x,  xend = national_x,
    y    = 0.25,        yend = national_y_at_x,
    colour    = "black",
    linewidth = 1,
    linetype  = "dotted"
  ) +
  annotate(
    "segment",
    x    = lightcast_x,  xend = lightcast_x,
    y    = 0.2,         yend = lightcast_y_at_x,
    colour    = "darkgreen",
    linewidth = 1,
    linetype  = "dotted"
  ) +
  # Floating labels at y = 0.20
  annotate(
    "label",
    x      = national_x,
    y      = 0.35,
    label  = "National vacancy rate",
    hjust  = 0.5,
    vjust  = 0.5,
    size   = 4,
    colour = "black",
    fill   = "white",
    label.size    = 0.3,
    label.padding = unit(0.3, "lines")
  ) +
  annotate(
    "label",
    x      = lightcast_x,
    y      = 0.25,
    label  = "Lightcast natl. vacancy rate",
    hjust  = 0.5,
    vjust  = 0.5,
    size   = 4,
    colour = "darkgreen",
    fill   = "white",
    label.size    = 0.3,
    label.padding = unit(0.3, "lines")
  ) +
  geom_line(
    data      = vac_rate_total,
    aes(x = date, y = vac_rate, group = 1),
    colour    = "darkgreen",
    linewidth = 1,
    linetype  = "solid",
    inherit.aes = FALSE
  ) +
  # annotate(
  #   "label",
  #   x     = max(national_vac$date),
  #   y     = national_vac$VACRATE[national_vac$date == max(national_vac$date)],
  #   label = "Lightcast natl. vacancy rate",
  #   hjust = -0.05,
  #   vjust = 6,
  #   size  = 5,
  #   colour = "darkgreen"
  # ) +
  geom_text_repel(
    data           = label_data,
    aes(x = date, y = vac_rate, label = soc_4_name, colour = soc_4_name),
    size           = 4.5,
    nudge_x        = 30,
    direction      = "y",
    hjust          = 0,
    segment.size   = 0.3,
    segment.colour = "grey50",
    show.legend    = FALSE
  ) +
  # Identical name in both scales -> ggplot merges into one legend
  scale_colour_manual(
    values = colour_map,
    name   = legend_title
  ) +
  scale_linetype_manual(
    values = linetype_map,
    name   = legend_title
  ) +
  scale_x_date(expand = expansion(mult = c(0.02, 0.25))) +
  scale_y_log10(
    breaks = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5),
    labels = scales::label_percent(accuracy = 0.1)
  ) +
  guides(
    colour   = guide_legend(title.position = "top", title.hjust = 0, ncol = 2),
    linetype = guide_legend(title.position = "top", title.hjust = 0, ncol = 2)
  ) +
  labs(
    title    = "Vacancy Rate by Occupation",
    subtitle = "Solid: top/bottom 2 by mean vacancy rate\nDashed: top/bottom 2 by coefficient of variation\nNote: Advertising, Marketing, Promotions, PR, Sales Managers have rank w. high mean and low CV\nY-axis is log-scale with real-value labels.",
    y        = "Vacancy Rate (log scale)",
    x        = "Date"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "bottom",
    legend.box      = "vertical",
    legend.title    = element_text(face = "bold"),
    ncol = 3
  ) + common_theme -> vac_rates_desc

print(vac_rates_desc)

