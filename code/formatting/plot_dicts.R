# Formatting functions in R
library(ggplot2)
library(extrafont)
extrafont::loadfonts(quiet = TRUE)

# Global wrap_width for str_wrap
wrap_width_fixed = 60

# theme for plotting
library(ggtext)

common_theme <- theme(
  text = ggplot2::element_text(family = "Latin Modern Roman"),
  
  # Replace element_text with element_textbox_simple for wrapping
  plot.title = element_text(
    family = "LMRoman10-Bold",
    size = 11,
    face = "bold"),
  
  plot.subtitle = element_text(
    family = "Latin Modern Roman",
    size = 10
  ),
  
  plot.caption = element_text(
    family = "Latin Modern Roman",
    size = 9,
    halign = 1),
  
  legend.text = element_text(size = 6),
  legend.title = element_text(size = 8, face = "bold"),
  
  # Axis titles
  axis.title.x = element_text(size = 9, margin = margin(t = 5)),
  axis.title.y = element_text(size = 9, margin = margin(r = 5)),
  
  # Axis text
  axis.text.x = element_text(size = 7, color = "grey20"),
  axis.text.y = element_text(size = 7, color = "grey20"),
  
  # Grid styling
  panel.background = element_rect(fill = "white", color = NA),
  panel.grid.major = element_line(color = "grey85", linewidth = 0.4),
  panel.grid.minor = element_line(color = "grey92", linewidth = 0.2),
  
  strip.text = element_text(size = 10, hjust = 0),
  strip.background = element_rect(fill = "grey95", color = NA),
  legend.key = element_blank(),
  
  # Plot margins
  plot.margin = margin(5, 5, 5, 5)
)
facet_wrap_custom <- function(formula, width = 20, ...) {
  facet_wrap(formula, labeller = label_wrap_gen(width), ...)
}

plot_annotation_theme = theme(
  text = ggplot2::element_text(
    family = "Latin Modern Roman"),
  plot.title = element_text(family = "LMRoman10-Bold",  # Use the exact bold font name
                            size = 11, 
                            face = "bold"),
  plot.subtitle = element_text(size = 10))

# Formatting functions in R
library(ggplot2)

# # theme for plotting
# common_theme <- theme(
#   # Title styling
#   plot.title = element_text(size = 14, face = "bold", hjust = 0, margin = margin(b = 5)),
#   plot.subtitle = element_text(size = 12, hjust = 0, margin = margin(b = 5)),
#   plot.caption = element_text(size = 10, hjust = 1, color = "grey40", margin = margin(t = 5)),
#   
#   # Axis titles
#   axis.title.x = element_text(size = 10, margin = margin(t = 5)),
#   axis.title.y = element_text(size = 10, margin = margin(r = 5)),
#   
#   # Axis text
#   axis.text.x = element_text(size = 8, color = "grey20"),
#   axis.text.y = element_text(size = 8, color = "grey20"),
#   
#   # Grid styling
#   panel.background = element_rect(fill = "white", color = NA),
#   panel.grid.major = element_line(color = "grey85", linewidth = 0.4),
#   panel.grid.minor = element_line(color = "grey92", linewidth = 0.2),
#   
#   strip.text = element_text(size = 10, hjust = 0),
#   strip.background = element_rect(fill = "grey95", color = NA),
#   
#   # Legend styling
#   legend.position = "bottom",
#   legend.title = element_text(size = 8, face = "bold"),
#   legend.text = element_text(size = 6),
#   legend.key = element_blank(),
#   # Plot margins
#   plot.margin = margin(5, 5, 5, 5)
# )

facet_wrap_custom <- function(formula, width = 20, ...) {
  facet_wrap(formula, labeller = label_wrap_gen(width), ...)
}
