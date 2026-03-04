# =============================================================================
# Build Annual Employment Series Matched to Vacancy SOC-3 Codes (2010-2023)
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   Constructs a complete annual employment series (2010-2023) for each
#   SOC-3 occupational code present in the Lightcast vacancy data, sourced
#   from the OEWS employment dataset.
#
# KNOWN DATA ISSUES:
#
#   1. 2010-2011: The OEWS data only contains major-group and detailed-level
#      codes — minor-group (SOC-3) aggregates are absent for ALL codes. These
#      are reconstructed by summing broad-level sub-codes (falling back to
#      detailed-level if broad codes are unavailable). This affects all 97
#      vacancy codes including 51-5100.
#
#   2. 15-1200 (Computer Occupations): The 2018 SOC revision renumbered this
#      minor group from 15-1100 to 15-1200. The predecessor code 15-1100 is
#      used directly for 2012-2018; 2010-2011 are covered by issue #1 above.
#
#   3. 31-1100 (Home Health & Personal Care Aides): The 2018 SOC revision
#      renumbered this minor group from 31-1000 to 31-1100. The predecessor
#      code 31-1000 is used directly for 2012-2018; 2010-2011 are covered
#      by issue #1 above.
#
#   4. 19-5000 (Occupational Health & Safety): The 2018 SOC revision moved
#      this minor group from 29-9010 (under Healthcare) to 19-5000 (under
#      Life, Physical & Social Science). The predecessor minor-group code
#      29-9010 is used directly for 2012-2018. For 2010-2011, 29-9010 also
#      does not exist as a minor-group aggregate, so the detailed sub-codes
#      29-9011 and 29-9012 are summed directly via a dedicated crosswalk
#      (Step 2b). Note: the general sub-code construction in Step 4 cannot
#      handle this case because the old sub-codes (29-90XX) have a different
#      major-group prefix than the new code (19-5000), so prefix matching
#      would find nothing.
#
#   5. 45-3000 (Fishing & Hunting Workers): Reclassified after 2017 with no
#      direct successor. No recoverable data for 2018-2023.
#
#   6. 55-9000 (Military): Not covered by OEWS. No recoverable data.
#
#   7. 99-9000 (Unclassified): Not covered by OEWS. No recoverable data.
#
# SOURCE HIERARCHY (applied in order of preference):
#   "direct_match"    -> OEWS minor-group row matches vacancy code exactly
#   "predecessor_code"-> Pre-2018 SOC code used for 15-1200, 31-1100, 19-5000
#   "constructed"     -> Summed from broad/detailed sub-codes (2010-2011
#                        and any other gaps not covered above)
#
# OUTPUT:
#   emp_series_matched_to_vacancies.RDS / .csv
#   Columns: soc_3, soc_3_name, year, tot_emp, source
# =============================================================================

library(tidyverse, quietly = TRUE)
library(here, quietly = TRUE)
library(conflicted, quietly = TRUE)
library(ggrepel, quietly = TRUE)
library(patchwork, quietly = TRUE)
conflict_prefer_all("dplyr", quiet = TRUE)
source(here('code/formatting/plot_dicts.R'))
new_data = FALSE

# --- File paths — adjust as needed -------------------------------------------
emp_path <- here("data/occ_macro_vars/OEWS/occ_employment_levels_oews.csv")
vac_path <- here("data/lightcast_vacancies/lightcast_soc_3_2010_2024.RDS")
out_path <- "data/lightcast_vacancies/emp_series_matched_to_vacancies"

if(new_data){
  # --- Load data ----------------------------------------------------------------
  emp <- read.csv(emp_path) %>%
    filter(year >= 2010, year <= 2023)
  
  vac <- readRDS(vac_path)
  
  vac_codes  <- sort(unique(vac$soc_3))
  soc3_names <- vac %>% select(soc_3, soc_3_name) %>% distinct()
  
  cat("Vacancy SOC-3 codes to match:", length(vac_codes), "\n")
  
  # =============================================================================
  # STEP 1: Direct match
  # Pull rows where the OEWS code exactly matches a vacancy SOC-3 code
  # =============================================================================
  emp_direct <- emp %>%
    filter(occ_code %in% vac_codes) %>%
    select(soc_3 = occ_code, year, tot_emp) %>%
    mutate(source = "direct_match")
  
  cat("Direct match rows:", nrow(emp_direct), "\n")
  
  # =============================================================================
  # STEP 2: SOC 2018 reclassification crosswalk
  # For codes that were renumbered in the 2018 SOC revision, pull the
  # predecessor minor-group code and relabel it as the new code.
  # This gives clean minor-group aggregates for 2012-2018.
  # 2010-2011 are still missing (predecessor codes also absent) and will be
  # handled either by Step 2b (19-5000) or sub-code construction in Step 4.
  # =============================================================================
  crosswalk <- c(
    "15-1200" = "15-1100",  # Computer Occupations (15-1100 exists 2012-2018)
    "31-1100" = "31-1000",  # Home Health & Personal Care Aides (31-1000 exists 2012-2018)
    "19-5000" = "29-9010"   # Occupational Health & Safety (29-9010 exists 2012-2018)
  )
  
  predecessor_rows <- do.call(rbind, lapply(names(crosswalk), function(new_code) {
    old_code <- crosswalk[new_code]
    emp %>%
      filter(occ_code == old_code) %>%
      select(year, tot_emp) %>%
      mutate(soc_3 = new_code, source = "predecessor_code")
  }))
  
  cat("Predecessor crosswalk rows:", nrow(predecessor_rows), "\n")
  
  # Combine direct and predecessor rows so Step 2b knows which years are covered
  emp_so_far <- bind_rows(emp_direct, predecessor_rows)
  
  # =============================================================================
  # STEP 2b: 2010-2011 fix for cross-major-group reclassifications
  #
  # The general sub-code construction in Step 4 works by matching a 4-character
  # prefix (e.g. "19-5" for 19-5000). This breaks for 19-5000 in 2010-2011
  # because its predecessor detailed codes (29-9011, 29-9012) have a completely
  # different prefix ("29-9"), so prefix matching finds nothing and returns NA.
  #
  # The fix is to explicitly name the old detailed codes and sum them directly
  # for any years not already covered by Steps 1 and 2.
  # =============================================================================
  crosswalk_detailed <- list(
    "19-5000" = c("29-9011", "29-9012")  # Occ Health & Safety specialists + technicians
  )
  
  early_years_rows <- do.call(rbind, lapply(names(crosswalk_detailed), function(new_code) {
    old_codes     <- crosswalk_detailed[[new_code]]
    years_covered <- unique(emp_so_far$year[emp_so_far$soc_3 == new_code])
    years_needed  <- setdiff(2010:2023, years_covered)
    
    emp %>%
      filter(occ_code %in% old_codes, year %in% years_needed) %>%
      group_by(year) %>%
      summarise(tot_emp = sum(tot_emp, na.rm = TRUE), .groups = "drop") %>%
      mutate(soc_3 = new_code, source = "predecessor_code")
  }))
  
  cat("Early-year predecessor rows:", nrow(early_years_rows), "\n")
  
  # Update emp_so_far to include early-year rows before Step 3 identifies gaps
  emp_so_far <- bind_rows(emp_so_far, early_years_rows)
  
  # =============================================================================
  # STEP 3: Identify remaining missing code x year slots
  # =============================================================================
  
  # Codes with no recoverable data anywhere in OEWS
  unresolvable <- c("55-9000", "99-9000")
  
  full_grid <- expand.grid(
    soc_3 = vac_codes,
    year  = 2010:2023,
    stringsAsFactors = FALSE
  )
  
  missing_slots <- full_grid %>%
    anti_join(emp_so_far, by = c("soc_3", "year")) %>%
    filter(!soc_3 %in% unresolvable)
  
  cat("Remaining slots to construct from sub-codes:", nrow(missing_slots), "\n")
  
  # =============================================================================
  # STEP 4: Construct remaining missing slots by summing sub-codes
  #
  # For minor group XX-X000, match all codes with prefix ^XX-X:
  #   - Prefer broad-level codes (XX-XX00) to avoid double-counting detailed
  #   - Fall back to detailed-level codes if no broad codes available
  #   - Exclude group-level codes (ending 000) and the target code itself
  # =============================================================================
  construct_from_subcodes <- function(minor_code, yr, emp_data) {
    prefix <- substr(minor_code, 1, 4)  # e.g. "15-1" from "15-1200"
    
    sub <- emp_data %>%
      filter(
        year == yr,
        grepl(paste0("^", prefix), occ_code),
        !grepl("000$", occ_code),    # exclude group-level aggregates
        occ_code != minor_code       # exclude the target code itself
      )
    
    if (nrow(sub) == 0) return(NA_real_)
    
    # Prefer broad-level codes (XX-XX00) to avoid double-counting
    broad <- sub %>% filter(grepl("00$", occ_code) & !grepl("000$", occ_code))
    
    if (nrow(broad) > 0) {
      sum(broad$tot_emp, na.rm = TRUE)
    } else {
      sum(sub$tot_emp, na.rm = TRUE)
    }
  }
  
  constructed <- missing_slots %>%
    rowwise() %>%
    mutate(
      tot_emp = construct_from_subcodes(soc_3, year, emp),
      source  = "constructed"
    ) %>%
    ungroup() %>%
    filter(!is.na(tot_emp))
  
  cat("Successfully constructed:", nrow(constructed), "slots\n")
  
  # =============================================================================
  # STEP 5: Combine all sources and finalise
  # =============================================================================
  emp_matched <- bind_rows(emp_direct, predecessor_rows, early_years_rows, constructed) %>%
    left_join(soc3_names, by = "soc_3") %>%
    select(soc_3, soc_3_name, year, tot_emp, source) %>%
    arrange(soc_3, year)
  
  # =============================================================================
  # STEP 6: Coverage report
  # =============================================================================
  cat("\n=== Coverage Report ===\n")
  
  coverage <- emp_matched %>%
    group_by(soc_3, soc_3_name) %>%
    summarise(
      n_years         = n_distinct(year),
      missing_years   = paste(setdiff(2010:2023, unique(year)), collapse = ", "),
      pct_constructed = paste0(round(mean(source == "constructed") * 100), "%"),
      .groups         = "drop"
    )
  
  cat("\nCodes with PARTIAL coverage (some years unresolvable):\n")
  partial <- coverage %>% filter(n_years < 14 & n_years > 0)
  if (nrow(partial) > 0) print(partial) else cat("None\n")
  
  cat("\nCodes with NO coverage (completely unresolvable):\n")
  no_data <- setdiff(vac_codes, unique(emp_matched$soc_3))
  if (length(no_data) > 0) {
    print(soc3_names %>% filter(soc_3 %in% no_data))
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
    left_join(., emp_matched, by = c('soc_3', 'soc_3_name', 'year')) %>% 
    # Filter out the soc codes for which we do not have employment values - "not specified" and "military"
    filter(!(soc_3 %in% unresolvable)) %>% 
    mutate(vac_rate = n_ft_vacancies/tot_emp, 
           date = as.Date(paste0(year, "-", month, "-01"))) %>% 
    filter(year < 2024)
  
  vac_rates_annual <- vac %>% 
    left_join(., emp_matched, by = c('soc_3', 'soc_3_name', 'year')) %>% 
    group_by(soc_3, soc_3_name, year) %>% 
    summarise(across(c(n_ft_vacancies, tot_emp), ~mean(., na.rm = TRUE))) %>% 
    ungroup %>% 
    # Filter out the soc codes for which we do not have employment values - "not specified" and "military"
    filter(!(soc_3 %in% unresolvable)) %>% 
    mutate(vac_rate = n_ft_vacancies/tot_emp, 
           date = year) %>% 
    filter(year < 2024) %>% 
    ungroup
  
  vac_rate_total <- vac %>% 
    left_join(., emp_matched, by = c('soc_3', 'soc_3_name', 'year')) %>% 
    filter(!(soc_3 %in% unresolvable)) %>% 
    mutate(as.Date(paste0(year, "-", month, "-01"))) %>% 
    group_by(date) %>% 
    summarise(across(c(n_ft_vacancies, tot_emp), ~sum(., na.rm = TRUE))) %>% 
    mutate(vac_rate = n_ft_vacancies/tot_emp) %>% 
    filter(date < "2024-01-01")
  
  vac_rate_total %>% 
    saveRDS(here("data/lightcast_vacancies/total_soc_3_vacancy_rate_monthly.RDS"))
  
  vac_rates_monthly %>% 
    saveRDS(here("data/lightcast_vacancies/soc_3_vacancy_rate_monthly.RDS"))
  
  vac_rates_annual %>% 
    saveRDS(here("data/lightcast_vacancies/soc_3_vacancy_rate_annual.RDS"))
}else{
  vac_rates_monthly <- readRDS(here("data/lightcast_vacancies/soc_3_vacancy_rate_monthly.RDS"))
  
  vac_rate_total <- readRDS(here("data/lightcast_vacancies/total_soc_3_vacancy_rate_monthly.RDS"))
  
  vac_rates_annual <- readRDS(here("data/lightcast_vacancies/soc_3_vacancy_rate_annual.RDS"))
}

# --- Load national vacancy rate -------------------------------------------
national_vac <- read.csv(here("calibration_remote/data/macro_vars/collated_macro_observations.csv")) %>%
  filter(!is.na(VACRATE)) %>%
  mutate(date = as.Date(DATE)) %>%
  filter(date >= min(vac_rates_monthly$date), date <= max(vac_rates_monthly$date)) %>%
  select(date, VACRATE)

vac_rates_monthly %>% 
  ggplot(aes(x = date, y = vac_rate, group = soc_3_name, color = soc_3_name)) + 
  geom_line(alpha = 0.7) + 
  geom_line(
    data      = national_vac,
    aes(x = date, y = VACRATE, group = 1),
    colour    = "black",
    linewidth = 1,
    linetype  = "dashed",
    inherit.aes = FALSE
  ) +
  annotate(
    "segment",
    x    = as.Date("2015-01-01"), xend = as.Date("2015-01-01"),
    y    = 0.1,         yend = national_vac$VACRATE[national_vac$date == as.Date("2015-01-01")],
    colour    = "black",
    linewidth = 1,
    linetype  = "dotted"
  ) +
  annotate(
    "label",
    x     = as.Date("2015-01-01"),
    y     = 0.1,#national_vac$VACRATE[national_vac$date == max(national_vac$date)],
    label = "JOTLS Natl. vacancy rate",
    #hjust = 1,
    #vjust = -5,
    size  = 5,
    colour = "black"
  ) +
  geom_line(
    data      = vac_rate_total,
    aes(x = date, y = vac_rate, group = 1),
    colour    = "darkred",
    linewidth = 1,
    linetype  = "dashed",
    inherit.aes = FALSE
  ) +
  annotate(
    "segment",
    x    = max(vac_rate_total$date), xend = max(vac_rate_total$date),
    y    = 0.16,         yend = vac_rate_total$vac_rate[vac_rate_total$date == max(vac_rate_total$date)],
    colour    = "darkred",
    linewidth = 1,
    linetype  = "dotted"
  ) +
  annotate(
    "label",
    x     = max(national_vac$date),
    y     = 0.16,
    label = "Lightcast natl. vacancy rate",
    hjust = 1,
    #vjust = 4,
    size  = 5,
    colour = "darkred"
  ) +
  theme(legend.position = "none") + 
  labs(title = "Vacancy Rate by Occupation", y = "Vacancy Rate (Monthly Vacancies/Annual Employment)", x = "Date") + common_theme -> p_monthly



vac_rates_annual %>% 
  ggplot(aes(x = as.Date(paste0(date, "-12-01")), y = vac_rate, group = soc_3_name, color = soc_3_name)) + 
  geom_line(alpha = 0.7) + 
  geom_line(
    data      = national_vac,
    aes(x = date, y = VACRATE, group = 1),
    colour    = "black",
    linewidth = 1,
    linetype  = "dashed",
    inherit.aes = FALSE
  ) +
  annotate(
    "text",
    x     = max(national_vac$date),
    y     = national_vac$VACRATE[national_vac$date == max(national_vac$date)],
    label = "National vacancy rate",
    hjust = 1,
    vjust = -10,
    size  = 5,
    colour = "black"
  ) +
  theme(legend.position = "none") + 
  labs(title = "Vacancy Rate by Occupation", y = "Vacancy Rate (Monthly Vacancies/Annual Employment", x = "Date") + common_theme -> p_annual


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
  theme(legend.position = "right") + scale_x_log10() + common_theme -> p_vac_density


vac_rates_annual %>% 
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
  theme(legend.position = "right") + scale_x_log10() + common_theme -> p_vac_density_annual

  
# --- Identify occupations to highlight ---------------------------------------

vac_rates_monthly <- vac_rates_monthly %>% 
  mutate(soc_3_name = str_wrap(soc_3_name, 40))
highlight_stats <- vac_rates_monthly %>%
  group_by(soc_3_name) %>%
  summarise(
    mean_vac = mean(vac_rate, na.rm = TRUE),
    sd_vac   = sd(vac_rate,   na.rm = TRUE),
    cv_vac   = sd_vac / mean_vac,
    .groups  = "drop"
  ) 

highlighted <- bind_rows(
  highlight_stats %>% slice_max(mean_vac, n = 2) %>% mutate(reason = "Mean", extreme = "High"),
  highlight_stats %>% slice_min(mean_vac, n = 3) %>% mutate(reason = "Mean", extreme = "Low"),
  highlight_stats %>% slice_max(cv_vac,   n = 2) %>% mutate(reason = "CV",   extreme = "High"),
  highlight_stats %>% slice_min(cv_vac,   n = 2) %>% mutate(reason = "CV",   extreme = "Low")
) %>%
  group_by(soc_3_name) %>%
  slice(1) %>%
  ungroup()

# --- Build named colour vector (reds for High, blues for Low) ----------------
high_occs <- highlighted %>% filter(extreme == "High") %>% pull(soc_3_name)
low_occs  <- highlighted %>% filter(extreme == "Low")  %>% pull(soc_3_name)

red_shades  <- colorRampPalette(c("#ff9999", "#8B0000"))(length(high_occs))
blue_shades <- colorRampPalette(c("#99bbff", "#00008B"))(length(low_occs))

colour_map <- c(
  setNames(red_shades,  high_occs),
  setNames(blue_shades, low_occs)
)

# --- Build named linetype vector (one entry per occupation) ------------------
linetype_map <- setNames(
  ifelse(highlighted$reason == "Mean", "solid", "dashed"),
  highlighted$soc_3_name
)

# Shared legend title — identical string in both scales triggers legend merge
legend_title <- "Occupation\nRed = high  |  Blue = low\nSolid = mean  |  Dashed = CV"

# --- Data subsets ------------------------------------------------------------
vac_grey      <- vac_rates_monthly %>% filter(!soc_3_name %in% highlighted$soc_3_name)
vac_highlight <- vac_rates_monthly %>%
  filter(soc_3_name %in% highlighted$soc_3_name) %>%
  left_join(highlighted %>% select(soc_3_name, reason, extreme), by = "soc_3_name")

label_data <- vac_highlight %>%
  group_by(soc_3_name, reason, extreme) %>%
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
    aes(x = date, y = vac_rate, group = soc_3_name),
    colour    = "grey80",
    linewidth = 0.4,
    alpha     = 0.4
  ) +
  # linetype mapped to soc_3_name (not reason) so it merges with colour legend
  geom_line(
    data = vac_highlight,
    aes(x = date, y = vac_rate, group = soc_3_name,
        colour   = soc_3_name,
        linetype = soc_3_name),
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
    aes(x = date, y = vac_rate, label = soc_3_name, colour = soc_3_name),
    size           = 4.5,
    nudge_x        = 1000,
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
  ) + common_theme -> vac_rates_summary

print(vac_rates_summary)
print(p_monthly /p_vac_density)
#print(p_annual)
#print(p_vac_density_annual)
