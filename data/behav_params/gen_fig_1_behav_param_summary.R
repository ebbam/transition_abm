library(here)
library(ggplot2)
library(dplyr)
library(tidyr)

source(here('code/formatting/plot_dicts.R'))

source(here('data/behav_params/Eeckhout_Replication/eeckhout_replication.R'))

p1 <- t2_no_title +
  labs(subtitle = "Fixed On-the-Job Search Propensity") +
  theme(plot.title = element_blank(),
        legend.title.position = "top",
        legend.title = element_text(hjust = 0.5))


new_data <- readRDS(here("data/behav_params/app_effort_brief_for_composite_fig.RDS"))

set.seed(1234)
n_sims <- 1000

pred_grid <- data.frame(PRUNEDUR_MO = 0:max_mo) %>%
  mutate(PRUNEDUR_MO2 = PRUNEDUR_MO^2)
for (var in control_vars) pred_grid[[var]] <- mean(bea_temp[[var]], na.rm = TRUE)

ev_from_probs <- function(probs) as.vector(probs %*% bin_values[colnames(probs)])

# Point estimate
ev_point <- tibble(PRUNEDUR_MO = pred_grid$PRUNEDUR_MO,
                   exp_apps    = ev_from_probs(predict(best_model, newdata = pred_grid, type = "probs")))

# Simulation draws
n_coef <- length(coef(best_model))
draws  <- MASS::mvrnorm(n_sims,
                        mu    = c(coef(best_model), best_model$zeta),
                        Sigma = vcov(best_model))

sim_ev <- map_dfr(seq_len(n_sims), function(s) {
  m <- best_model
  m$coefficients <- draws[s, 1:n_coef]
  m$zeta         <- draws[s, -(1:n_coef)]
  if (is.unsorted(m$zeta)) return(NULL)
  tibble(PRUNEDUR_MO = pred_grid$PRUNEDUR_MO,
         exp_apps    = ev_from_probs(predict(m, newdata = pred_grid, type = "probs")))
})

ev_data <- sim_ev %>%
  group_by(PRUNEDUR_MO) %>%
  summarise(lo = quantile(exp_apps, 0.025),
            hi = quantile(exp_apps, 0.975)) %>%
  left_join(ev_point, by = "PRUNEDUR_MO")

stopifnot(all(c("exp_apps", "lo", "hi") %in% names(ev_data)))
sel_months <- c(0, 10, 20, 30)
max_mo     <- max(sel_months)
bins       <- c("0", "1 to 10", "11 to 20", "21 to 80", "81 or more")
bin_values <- c("0" = 1, "1 to 10" = 5.5, "11 to 20" = 15.5,
                "21 to 80" = 50.5, "81 or more" = 90.5) / 2

# ev_data (exp_apps, lo, hi) as built in the previous step, without the offsets
peak <- slice_max(ev_data, exp_apps, n = 1)

# Survey averages (unweighted, no covariate adjustment), quarterly duration bins
obs_means <- bea_temp %>%
  filter(PRUNEDUR_MO <= max_mo) %>%
  mutate(apps = bin_values[as.character(PEA1)],
         q    = floor(PRUNEDUR_MO / 3) * 3 + 1) %>%
  group_by(q) %>%
  summarise(n = n(), mean = mean(apps), se = sd(apps) / sqrt(n)) %>%
  filter(n >= 30)


fig_font <- calc_element("text", theme_grey() + common_theme)$family
fig_size <- calc_element("axis.text", theme_grey() + common_theme)$size   # in pt


p_ev <- ggplot(ev_data, aes(PRUNEDUR_MO, exp_apps)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = "slateblue", alpha = 0.2) +
  # geom_pointrange(data = obs_means,
  #                 aes(x = q, y = mean, ymin = mean - 1.96 * se, ymax = mean + 1.96 * se),
  #                 inherit.aes = FALSE, colour = "grey55", size = 0.25) +
  geom_line(linewidth = 1, colour = "slateblue") +
  geom_point(data = peak, size = 2.5, colour = "slateblue") +
  geom_text(data = peak, vjust = -1.2,
            family = fig_font, size = fig_size, size.unit = "pt",
            aes(label = sprintf("Peak: %.1f applications at %d months",
                                exp_apps, PRUNEDUR_MO))) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.1))) +
  scale_x_continuous(breaks = seq(0, max_mo, 6), expand = c(0.01, 0)) +
  labs(y = "Expected applications sent", x = "Unemployment Duration (Months)", 
      subtitle= str_wrap("Predicted Number of Applications Sent by Unemployment Duration", width = 32))+
  common_theme +
  theme(panel.grid.minor = element_blank())

bins_long <- new_data_long %>%
  filter(PRUNEDUR_MO <= max_mo) %>%
  mutate(Category = factor(Category, levels = bins,
                           labels = paste(bins, "applications")))
bins_sel <- filter(bins_long, PRUNEDUR_MO %in% sel_months)

p_bins <- ggplot(bins_long, aes(PRUNEDUR_MO, Probability,
                                colour = Category, fill = Category)) +
  geom_col(data = bins_sel, width = 5, alpha = 0.55, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_text(data = bins_sel, aes(label = scales::percent(Probability, accuracy = 1)),
            vjust = -0.6, size = 2.3, colour = "grey20") +
  facet_wrap(~Category, nrow = 1, scales = "free_y") +
  scale_fill_viridis_d(option = "mako", begin = 0.15, end = 0.8,
                       aesthetics = c("fill", "colour"), guide = "none") +
  scale_y_continuous(labels = scales::percent, limits = c(0, NA),
                     expand = expansion(mult = c(0, 0.2))) +
  scale_x_continuous(breaks = sel_months) +
  labs(x = "Unemployment duration (months)", y = "Share of job seekers") +
  common_theme +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor   = element_blank())

p2 <- (p_ev / p_bins) +
  plot_layout(heights = c(1.4, 1)) +
  plot_annotation(tag_levels = "A",
                  title = "Application Effort by Unemployment Duration",
                  theme = plot_annotation_theme)

p2 <- new_data %>% 
  ggplot(aes(x = PRUNEDUR_MO, y = Probability, fill = Category)) +
  geom_area(stat = "identity") +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Probability of Sending X Applications", 
       x = "Unemployment Duration (Months)", 
       fill = "# of Applications",
       subtitle= str_wrap("Predicted Number of Applications Sent by Unemployment Duration", width = 32))+
  common_theme +
  theme(legend.position = "bottom",
        legend.title.position = "top",   
        legend.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(
    override.aes = list(size = 3),
    keywidth = unit(0.4, "cm"),
    keyheight = unit(0.4, "cm")
  ))


sim_all <- readRDS(here("data/behav_params/res_wage_dists_brief_for_composite_fig.RDS"))
# Plot: Violin plots of simulated predicted values
p3 <- ggplot(filter(sim_all, model == "LM"), aes(x = factor(dur_unemp), y = sim_val, fill = model)) +
  geom_violin(position = position_dodge(width = 0.8), color = NA, width = 0.7, alpha = 0.7) +
 # geom_line() +
  stat_summary(fun = mean, geom = "point", position = position_dodge(width = 0.8), 
               shape = 21, size = 1, color = "black", fill = "grey40") +
  labs(
    subtitle = str_wrap("Predicted Reservation Wage Ratio by Unemployment Duration", 32),
    #subtitle = "Violin plots from LM, GLM, and EB model predictions",
    x = "Unemployment Duration (Months)",
    y = "Reservation Wage Ratio",
    fill = "Model"
  ) +
  theme(
    axis.text.x = element_text(vjust = 0.5),
    legend.position = "none",
    panel.grid.minor = element_blank()#,
    #panel.grid.major = element_blank()
    
  )  +
  common_theme +
  scale_fill_brewer(palette = 'Dark2') +
  scale_x_discrete(
    breaks = function(x) x[as.numeric(x) %% 5 == 0]
  ) +
  theme(panel.grid.minor = element_blank(),
        #panel.grid.major = element_blank(),
        legend.title.position = "top",   
        legend.title = element_text(hjust = 0.5)) +
  guides(fill = guide_legend(
    keywidth = unit(0.4, "cm"),
    keyheight = unit(0.4, "cm")
  ))

# Search probability model: p = sigmoid(alpha + beta_c * comp) -- matches
# p_search_logit() in abm_funs.py exactly (age/cycle terms are zeroed out there)
sigmoid <- function(x) 1 / (1 + exp(-x))

alpha <- -4
comp_range   <- seq(0, 1, length.out = 100)
theta_values <- c(0.01, 1, 2, 3, 4, 5, 7.5, 10, 20, 30, 40)

plot_df <- expand_grid(comp = comp_range, beta_c = theta_values) %>%
  mutate(prob = sigmoid(alpha + beta_c * comp))
        # beta_c = factor(beta_c))

range_df <- data.frame(
  xmin = 0.033, xmax = 0.5, ymin = -Inf, ymax = Inf,
  label = "Expected range\n(2-30 Seekers per Vac.)"
)

p4 <- ggplot(plot_df, aes(x = comp, y = prob)) +
  # geom_rect(data = range_df,
  #           aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = label),
  #           inherit.aes = FALSE, alpha = 0.15) +
  geom_line(aes(color = beta_c, group = beta_c), linewidth = 0.5) +
  geom_hline(yintercept = 0.06, color = "darkblue", 
             linewidth = 0.5, linetype = "dashed") +
  annotate("text", 
           x = 0.75,        # left-align near the y-axis
           y = 0.1,        # slightly above the line
           label = "6% Target", 
           color = "darkblue",
           size = 3,
           hjust = 0,
           family = "Latin Modern Roman") +
  #scale_fill_manual(name = NULL, values = c("Expected range\n(2-30 Seekers per Vac.)" = "blue")) +
  scale_linetype_manual(name = NULL, values = c("6% target" = "dashed")) +
  scale_color_distiller(palette = "Purples", direction = 1, name = expression(beta[c]))+
  
  #scale_color_brewer(palette = "Blues", direction = 1, name = expression(beta[c]))+
  #scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  #scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(
    x = "Competition Metric",
    y = "Search Probability",
    subtitle = expression(paste("Dynamic On-the-Job Search Propensity"))
  ) +
  common_theme +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank())

composite <- (p_ev + p3)/(eeckhout_plot) & common_theme & theme(panel.grid.minor = element_blank(), 
                                                        legend.text = element_text(size = 9),
                                                          legend.key.size = unit(0.4, "cm"),
                                                        legend.title = element_text(size = 9, face = "bold", hjust = 0.5))
composite
save_plot_standard(composite, here("output/behavioural_param_summary_new.pdf"), aspect_ratio = 1.2)

