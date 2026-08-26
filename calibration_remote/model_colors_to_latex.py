from model_colors import MODEL_COLORS

def rgb_to_latex(name, rgb_tuple):
    r, g, b = [int(x * 255) for x in rgb_tuple]
    safe_name = name.replace(" ", "").replace(".", "").replace("-", "").replace("/", "")
    return f"\\definecolor{{model{safe_name}}}{{RGB}}{{{r},{g},{b}}}"

for model_name, color in MODEL_COLORS.items():
    print(rgb_to_latex(model_name, color))