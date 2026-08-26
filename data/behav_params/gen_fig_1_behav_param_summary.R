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
p3 <- ggplot(sim_all, aes(x = factor(dur_unemp), y = sim_val, fill = model)) +
  geom_violin(position = position_dodge(width = 0.8), color = NA, width = 0.7, alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", position = position_dodge(width = 0.8), 
               shape = 21, size = 1, color = "black", fill = "grey40") +
  labs(
    subtitle = str_wrap("Predicted Reservation Wage Ratio by Unemployment Duration", 32),
    #subtitle = "Violin plots from LM, GLM, and EB model predictions",
    x = "Unemployment Duration (Months)",
    y = "Simulated Predicted Wage Ratio",
    fill = "Model"
  ) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank()
    
  )  +
  common_theme +
  scale_fill_brewer(palette = 'Dark2') +
  scale_x_discrete(
    breaks = function(x) x[as.numeric(x) %% 5 == 0]
  ) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
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

composite <- (p2 + p3)/(p4 + p1) & common_theme & theme(panel.grid.minor = element_blank(), 
                                                        legend.text = element_text(size = 9),
                                                          legend.key.size = unit(0.4, "cm"),
                                                        legend.title = element_text(size = 9, face = "bold", hjust = 0.5))

save_plot_standard(composite, here("output/behavioural_param_summary.png"), aspect_ratio = 1.2)

