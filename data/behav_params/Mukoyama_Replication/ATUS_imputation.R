library(tidyverse)
library(survey)
library(MASS)
library(here)
library(haven)
library(conflicted)
conflict_prefer_all("dplyr", quiet = TRUE)

# Load dataset
raw_ATUS <- readRDS(here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/ATUS/merged_ATUS_2023.rds"))
ref_raw_ATUS <- read_dta(here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/ATUS/merged_ATUS_2014.dta"))
data <- ref_raw_ATUS 
raw_ATUS_2014 <- readRDS(here("data/behav_params/Mukoyama_Replication/mukoyama_all/raw_data/ATUS/merged_ATUS_2023.rds")) %>% 
  filter(year <= 2014)

ref_reweight <- read_dta(here('data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/ATUS/time_method_reweight.dta'))
df_list <- list("2014_orig" = ref_raw_ATUS,  "2014_new" = raw_ATUS_2014, "2023" = raw_ATUS)
for(k in names(df_list)){

data <- df_list[[k]]
# Filter: Only non-employed
data <- data %>% filter(mlr %in% c(3, 4, 5))

# Drop outliers
data <- data %>% filter(!is.na(time_less8))

# Define variables
srch_mth <- c("empldir", "pubemkag", "PriEmpAg", "FrendRel", "SchEmpCt", "Unionpro", 
              "Resumes", "Plcdads", "Otheractve", "lkatads", "Jbtrnprg", "otherpas")
observables <- c("age", "age2", "age3", "age4", "female", "hs", "somecol", "college", "black", 
                 "married", "marriedfemale")
lfs <- c("np_other", "layoff", "nonsearchers")

# Generate new variables
data <- data %>% 
  mutate(
    dummy_search = as.integer(time_less8 > 0),
    ltime_less8 = ifelse(time_less8 > 0, log(time_less8), NA)
  ) 

# First-stage probit regression
probit_model <- glm(as.formula(paste0("dummy_search ~ ", paste(c(srch_mth, lfs, observables), collapse = "+"))), 
                    family = binomial(link = "probit"), 
                    data = data)
data$pi <- predict(probit_model, type = "link")
data$pi_scaled <- data$pi/sd(data$pi, na.rm = TRUE)
data$prob_search <- predict(probit_model, type = "response")

# Compute inverse Mills ratio
data$invmills <- (dnorm(data$pi_scaled) / pnorm(data$pi_scaled))*data$pi_scaled
# data <- data %>% 
#   mutate(invmills = ifelse(is.nan(invmills), NA_real_, invmills))

# Second-stage regression
reg_model <- lm(as.formula(paste0("ltime_less8 ~ ", paste(c(srch_mth, lfs, observables), collapse = "+"), " + invmills")), 
                data = data)

data$search_cond <- predict(reg_model, newdata = data) 
# %>% data.frame("prediction" = .) %>%
#   tibble::rownames_to_column(., "obs") %>%
#   tibble %>% mutate(obs = as.numeric(obs)) %>% complete(obs = 1:nrow(data)) %>% pull(prediction)
#data$resid <- residuals(reg_model)  
data$resid <- NA  # Initialize all residuals as NA
valid_rows <- which(!is.na(data$ltime_less8))  # Get indices used in regression
data$resid[valid_rows] <- residuals(reg_model)  # Assign residuals only to valid rows
# %>% data.frame("prediction" = .) %>%
#   tibble::rownames_to_column(., "obs") %>%
#   tibble %>% mutate(obs = as.numeric(obs)) %>% complete(obs = 1:nrow(data)) %>% pull(prediction)
sigma_hat <- var(data$resid, na.rm = TRUE)
data$search_cond <- exp(data$search_cond + sigma_hat / 2)
data$time_create <- data$prob_search * data$search_cond


# Creating weights for CPS sample
probit_coeff <- tibble(data.frame(t(sapply(coef(probit_model), c)))) %>%  # Equivalent to matrix b = e(b)
  rename(constant = "X.Intercept.") %>% 
  rename_with(~paste0("pweight_", .)) %>% 
  relocate(pweight_constant, .after = last_col())

# Linear Regression (Part 2: search | search > 0)
reg_coeff <- tibble(data.frame(t(sapply(coef(reg_model), c)))) %>% 
  rename(constant = "X.Intercept.") %>% 
  rename_with(~paste0("sweight_", .)) %>% # Equivalent to matrix b = e(b)
  relocate(sweight_constant, .after = last_col())

# Save relevant weights
reweight_data <- probit_coeff %>%
  cbind(reg_coeff) %>% 
  tibble %>% 
  mutate(sigma_hat = sigma_hat) %>% 
  relocate(sigma_hat) #%>% 
  #select(contains("weight")) #%>%
  #slice(1)  # Equivalent to Stata's `keep if _n==1`

write.csv(reweight_data, here(paste0('data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/ATUS/time_method_reweight_', k, '.csv')), row.names = FALSE)

# Create datasets for plotting
fig2a_data_new <- data %>% group_by(year) %>%
  summarize(time_create = weighted.mean(time_create, wgt, na.rm = TRUE),
            time_less8 = weighted.mean(time_less8, wgt, na.rm = TRUE), .groups = "drop")
write.csv(fig2a_data_new,
          here(paste0('data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/ATUS/Fig2a_data', k, '.csv')),
               row.names = FALSE)

fig2b_data_new <- data %>% filter(mlr %in% c(3, 4)) %>%
  group_by(year) %>%
  summarize(time_create = weighted.mean(time_create, wgt, na.rm = TRUE),
            time_less8 = weighted.mean(time_less8, wgt, na.rm = TRUE), .groups = "drop")
write.csv(fig2b_data_new,
          here(paste0('data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/ATUS/Fig2b_data', k, '.csv')),
          row.names = FALSE)

# Load data (update file paths as needed)
fig2a_data <- read.csv('data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/ATUS/Fig2a_data.csv') 
fig2b_data <- read.csv('data/behav_params/Mukoyama_Replication/mukoyama_all/int_data/ATUS/Fig2b_data.csv') 

year <- fig2a_data[[1]]
year2 <- fig2a_data_new[[1]]
nonemp_base <- fig2a_data[2:3]
unemp_base <- fig2b_data[2:3]

nonemp_base_new <- fig2a_data_new[2:3]
unemp_base_new <- fig2b_data_new[2:3]

# Helper function to add shaded recession areas
add_recession <- function(p) {
  p +
    annotate("rect", xmin = 2007 + 11/12, xmax = 2009.5, ymin = -Inf, ymax = Inf, alpha = 0.2) +
    annotate("rect", xmin = 2001 + 3/12, xmax = 2001 + 11/12, ymin = -Inf, ymax = Inf, alpha = 0.2)
}

##################################
############ Figures 2a-b ########
##################################

# Figure 2a
fig2a <- ggplot() +
  geom_line(aes(x = year, y = nonemp_base[[1]]), color = "red", size = 0.5) +
  #geom_line(aes(x = year, y = nonemp_base[[2]]), color = "blue", size = 0.5) +
  geom_line(aes(x = year2, y = nonemp_base_new[[1]]), color = "purple", linetype = "dotted", size = 1) +
  geom_line(aes(x = year2, y = nonemp_base_new[[2]]), color = "blue", size = 0.5) +
  #scale_x_continuous(breaks = 2003:2023) +
  #scale_y_continuous(limits = c(0, 60), breaks = seq(0, 10, by = 2)) +
  theme_minimal()
fig2a <- add_recession(fig2a)

# Figure 2b
fig2b <- ggplot() +
  geom_line(aes(x = year, y = unemp_base[[1]]), color = "red", size = 0.5) +
  #geom_line(aes(x = year, y = unemp_base[[2]]), color = "blue", size = 0.5) +
  geom_line(aes(x = year2, y = unemp_base_new[[1]]), color = "purple", linetype = "dotted", size = 1) +
  geom_line(aes(x = year2, y = unemp_base_new[[2]]), color = "blue", size = 0.5)+
  #scale_x_continuous(breaks = 2003:2023) +
  #scale_y_continuous(limits = c(10, 50), breaks = seq(10, 50, by = 10)) +
  theme_minimal()
fig2b <- add_recession(fig2b)

print(fig2a + fig2b + plot_annotation("Figure 2. Actual and Imputed Average Search Time (minutes per day) \nfor All Nonemployed Workers ( panel A) and Unemployed Workers ( panel B)",
                                      caption = "Notes: Regressions are estimated in the ATUS from 2003–2014. \nWhile both panels A and B plot the fitted values from the sample regression, panel A plots the actual and imputed search time for all nonemployed, while panel B plots them for just the unemployed. \nObservations are weighted by their ATUS sample weight.",
                                      theme=theme(plot.title=element_text(hjust=0.5))))
}

