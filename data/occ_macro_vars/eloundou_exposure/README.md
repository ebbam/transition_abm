# AI occupational exposure, matched to the full_omn occupations

Produces `ai_exposure_by_acs_full_omn.csv`, read by `calibration_remote/exposure_entry_policy_exercise.py`
(`EXPOSURE_SOURCE = "eloundou_v2"` or `"anthropic_v2"`).

Run from the repository root: `Rscript data/occ_macro_vars/eloundou_exposure/ai_exposure_matching_full_omn.R`

## Raw data (`raw/`)
- `eloundou_2024_occ_level.csv`: Eloundou, Manning, Mishkin & Rock (2024), "GPTs are GPTs", *Science* 384,
  1306–1308; `data/occ_level.csv` from https://github.com/openai/GPTs-are-GPTs. O*NET-SOC 2019 codes. We use
  `dv_rating_beta` (E2: exposure to LLMs plus software built on them).
- `anthropic_economic_index_job_exposure.csv`: Anthropic Economic Index, `labor_market_impacts/job_exposure.csv`
  from https://huggingface.co/datasets/Anthropic/EconomicIndex. SOC 2018 detailed codes, `observed_exposure`.
- `onet_2010_to_2019_crosswalk.csv`: O*NET-SOC 2010 to O*NET-SOC 2019 taxonomy crosswalk (O*NET Resource Center).

Both exposure files were downloaded on 2026-08-27. The copies previously used in
`calibration_remote/data/ai_exposure/` are identical.

## Matching (`ai_exposure_matching_full_omn.R`)
Same hierarchical approach as `../BLS_entry_exit_reqs/entry_exit_indicators_full_omn.R`, using the same
crosswalk (`data/crosswalk_occ_soc_cps_codes_full_omn.csv`, ACS 2010 to SOC 2010):
1. Translate each source from SOC 2018 / O*NET-SOC 2019 to SOC 2010 using the O*NET crosswalk (at the 6-digit
   SOC level; Eloundou is first averaged over O*NET sub-occupations).
2. Match exact SOC 2010, then SOC broad (6-digit), then SOC minor (4-digit), then SOC major (2-digit), each step
   only for occupations not matched before. Broad, minor and major matches take the unweighted mean of the source
   scores in that group. The major step is needed only for Tour and Travel Guides (39-7010) in the Anthropic data.

| Matching level | Eloundou | Anthropic |
|---|---|---|
| SOC 2010 (exact) | 388 | 379 |
| SOC broad | 121 | 119 |
| SOC minor | 19 | 29 |
| SOC major | 0 | 1 |
| **Total** | **528 / 528** | **528 / 528** |

The earlier matching in the exposure exercise (ACS to SOC 2010, matched directly to the SOC 2018-coded sources)
left 167 of 528 occupations unmatched for Eloundou and set them to zero exposure, including computer programmers,
database administrators and software developers.

## Output
- `ai_exposure_by_acs_full_omn.csv`: one row per network occupation, in network order (`id`, `acs_occ_code`,
  label, `SOC2010`, raw `eloundou_exposure` and `anthropic_exposure`, and the match level for each). Scores are
  raw; the exercise min-max normalises the selected source.
- `ai_exposure_match_summary_full_omn.tex`: matching summary table (same layout as the entry/exit table).
