# Formatting functions in R

# theme for plotting
common_theme <- theme(
  plot.title = element_text(size = 20),
  legend.text = element_text(size = 18),
  legend.title = element_text(size = 18),
  axis.title.x = element_text(size = 16),
  axis.title.y = element_text(size = 16),
  axis.text.x = element_text(size = 14),
  axis.text.y = element_text(size = 14),
  panel.background = element_rect(fill = NA),
  panel.grid.major = element_line(color = "grey90", size = 0.5),
  panel.grid.minor = element_line(color = "grey90", size = 0.25)
)