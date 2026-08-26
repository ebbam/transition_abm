library(dplyr)
library(tidyr)
library(ggplot2)
library(here)

# ── Load data ─────────────────────────────────────────────────────────────────
gdp_growth <- read.csv(here("data/macro_vars/GDPC1_pct_ch.csv")) %>%
  rename(gr_real_gdp = GDPC1_PC1)

realgdp <- read.csv(here("data/macro_vars/GDPC1.csv")) %>%
  rename(real_gdp = GDPC1) %>%
  mutate(log_real_gdp = log(real_gdp))

jorate <- read.csv(here("data/macro_vars/JTSJOR.csv")) %>%
  rename(vac_rate = JTSJOR) %>%
  mutate(fd_vac_rate = vac_rate - lag(vac_rate))

full <- gdp_growth %>%
  left_join(realgdp, by = "DATE") %>%
  left_join(jorate,  by = "DATE") %>%
  tibble() %>%
  mutate(date = as.Date(DATE)) %>%
  filter(date > as.Date("1995-01-01")) %>%
  arrange(date) %>%
  mutate(
    time         = row_number(),
    vac_rate_lag = lag(vac_rate),
    fd_vac_rate  = vac_rate - lag(vac_rate)   # recompute after filter/arrange
  )

# ── Fit four models ───────────────────────────────────────────────────────────
mod1 <- lm(vac_rate    ~ gr_real_gdp,                 data = full)
mod2 <- lm(vac_rate    ~ gr_real_gdp + time,          data = full)
mod3 <- lm(vac_rate    ~ gr_real_gdp + vac_rate_lag,  data = full)
mod4 <- lm(fd_vac_rate ~ gr_real_gdp,                 data = full)

# ── Attach predictions ────────────────────────────────────────────────────────
# Models 1–3 predict levels directly. Model 4 predicts first differences;
# we also reconstruct an implied level series by cumulating the predicted
# changes from the first observed vacancy rate.
full_pred <- full %>%
  mutate(
    pred_mod1    = predict(mod1, newdata = full),
    pred_mod2    = predict(mod2, newdata = full),
    pred_mod3    = predict(mod3, newdata = full),
    pred_mod4_fd = predict(mod4, newdata = full)
  )

first_idx <- which(!is.na(full_pred$vac_rate))[1]
start_val <- full_pred$vac_rate[first_idx]

full_pred <- full_pred %>%
  mutate(
    pred_mod4_level = start_val + cumsum(replace_na(pred_mod4_fd, 0))
  )

# ── Fit statistics ────────────────────────────────────────────────────────────
# Heads up: Model 4's R² and RMSE are in CHANGE space, not comparable 1-for-1
# with Models 1–3. Also report the RMSE of the reconstructed level series
# so you can compare apples to apples on levels.
rmse_mod4_level <- sqrt(mean(
  (full_pred$vac_rate - full_pred$pred_mod4_level)^2, na.rm = TRUE
))

fit_stats <- tibble(
  Model = c(
    "Model 1: vac_rate ~ gr_real_gdp",
    "Model 2: vac_rate ~ gr_real_gdp + time",
    "Model 3: vac_rate ~ gr_real_gdp + lag(vac_rate)",
    "Model 4: Δvac_rate ~ gr_real_gdp  (change-space)"
  ),
  R2    = c(summary(mod1)$r.squared,
            summary(mod2)$r.squared,
            summary(mod3)$r.squared,
            summary(mod4)$r.squared),
  AdjR2 = c(summary(mod1)$adj.r.squared,
            summary(mod2)$adj.r.squared,
            summary(mod3)$adj.r.squared,
            summary(mod4)$adj.r.squared),
  RMSE  = c(sqrt(mean(residuals(mod1)^2)),
            sqrt(mean(residuals(mod2)^2)),
            sqrt(mean(residuals(mod3)^2)),
            sqrt(mean(residuals(mod4)^2))),
  RMSE_levels = c(sqrt(mean(residuals(mod1)^2)),
                  sqrt(mean(residuals(mod2)^2)),
                  sqrt(mean(residuals(mod3)^2)),
                  rmse_mod4_level),
  AIC   = c(AIC(mod1), AIC(mod2), AIC(mod3), AIC(mod4)),
  BIC   = c(BIC(mod1), BIC(mod2), BIC(mod3), BIC(mod4))
)
print(fit_stats)

# ── Reshape for plotting (levels) ─────────────────────────────────────────────
# Model 4 appears as its reconstructed level series so all four are on the
# same y-axis. The change-space version gets its own panel below.
full_long <- full_pred %>%
  select(date, vac_rate, pred_mod1, pred_mod2, pred_mod3, pred_mod4_level) %>%
  pivot_longer(
    cols      = c(vac_rate, pred_mod1, pred_mod2, pred_mod3, pred_mod4_level),
    names_to  = "series",
    values_to = "value"
  ) %>%
  mutate(series = recode(series,
                         vac_rate         = "Actual",
                         pred_mod1        = "Model 1: gr_real_gdp",
                         pred_mod2        = "Model 2: gr_real_gdp + time",
                         pred_mod3        = "Model 3: gr_real_gdp + lag(vac_rate)",
                         pred_mod4_level  = "Model 4: Δvac_rate ~ gr_real_gdp (reconstructed)"
  )) %>%
  mutate(series = factor(series, levels = c(
    "Actual",
    "Model 1: gr_real_gdp",
    "Model 2: gr_real_gdp + time",
    "Model 3: gr_real_gdp + lag(vac_rate)",
    "Model 4: Δvac_rate ~ gr_real_gdp (reconstructed)"
  )))

# ── Subtitle with fit statistics ──────────────────────────────────────────────
fit_subtitle <- paste0(
  "Level-space RMSE — ",
  sprintf("M1: %.3f  |  M2: %.3f  |  M3: %.3f  |  M4: %.3f",
          fit_stats$RMSE_levels[1], fit_stats$RMSE_levels[2],
          fit_stats$RMSE_levels[3], fit_stats$RMSE_levels[4])
)

# ── Plot 1: all four models vs. actual (levels) ───────────────────────────────
series_colours <- c(
  "Actual"                                              = "black",
  "Model 1: gr_real_gdp"                                = "#D85A30",
  "Model 2: gr_real_gdp + time"                         = "#378ADD",
  "Model 3: gr_real_gdp + lag(vac_rate)"                = "#1D9E75",
  "Model 4: Δvac_rate ~ gr_real_gdp (reconstructed)"    = "#8A3FAB"
)

series_linetypes <- c(
  "Actual"                                              = "solid",
  "Model 1: gr_real_gdp"                                = "dashed",
  "Model 2: gr_real_gdp + time"                         = "dashed",
  "Model 3: gr_real_gdp + lag(vac_rate)"                = "dashed",
  "Model 4: Δvac_rate ~ gr_real_gdp (reconstructed)"    = "dashed"
)

series_sizes <- c(
  "Actual"                                              = 0.9,
  "Model 1: gr_real_gdp"                                = 0.6,
  "Model 2: gr_real_gdp + time"                         = 0.6,
  "Model 3: gr_real_gdp + lag(vac_rate)"                = 0.6,
  "Model 4: Δvac_rate ~ gr_real_gdp (reconstructed)"    = 0.6
)

ggplot(full_long, aes(x = date, y = value,
                      colour = series, linetype = series, linewidth = series)) +
  geom_line() +
  scale_colour_manual(values = series_colours) +
  scale_linetype_manual(values = series_linetypes) +
  scale_linewidth_manual(values = series_sizes) +
  labs(
    x = NULL, y = "Vacancy rate",
    title    = "Actual vs. predicted vacancy rate — four specifications",
    subtitle = fit_subtitle,
    colour   = NULL, linetype = NULL, linewidth = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold"),
    plot.subtitle    = element_text(size = 9, colour = "grey30"),
    legend.position  = "top",
    panel.grid.minor = element_blank()
  )

# ── Plot 2: faceted version — one panel per model (levels) ────────────────────
ggplot(
  full_long %>% filter(series != "Actual") %>%
    rename(model = series) %>%
    left_join(full_pred %>% select(date, actual = vac_rate), by = "date"),
  aes(x = date)
) +
  geom_line(aes(y = actual),  colour = "black",    linewidth = 0.8) +
  geom_line(aes(y = value),   colour = "#D85A30",  linewidth = 0.6,
            linetype = "dashed") +
  facet_wrap(~ model, ncol = 1) +
  labs(
    x = NULL, y = "Vacancy rate",
    title    = "Actual (black) vs. predicted (orange) — by model",
    subtitle = fit_subtitle
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold"),
    plot.subtitle    = element_text(size = 9, colour = "grey30"),
    strip.text       = element_text(face = "bold", hjust = 0),
    panel.grid.minor = element_blank()
  )

# ── Plot 3: Model 4 in its native change-space ────────────────────────────────
# The honest way to show the differenced model — compare predicted vs. actual
# Δvac_rate. Don't be alarmed that R² looks small; that's normal for
# differenced macro series.
ggplot(full_pred, aes(x = date)) +
  geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.3) +
  geom_line(aes(y = fd_vac_rate),  colour = "black",   linewidth = 0.8) +
  geom_line(aes(y = pred_mod4_fd), colour = "#8A3FAB", linewidth = 0.6,
            linetype = "dashed") +
  labs(
    x = NULL, y = "Δ vacancy rate",
    title    = "Model 4 in change-space: Δvac_rate ~ gr_real_gdp",
    subtitle = sprintf("Actual (black) vs. predicted (purple) first differences  |  R² = %.3f  |  RMSE = %.4f",
                       summary(mod4)$r.squared,
                       sqrt(mean(residuals(mod4)^2)))
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title       = element_text(face = "bold"),
    plot.subtitle    = element_text(size = 9, colour = "grey30"),
    panel.grid.minor = element_blank()
  )
