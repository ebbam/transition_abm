import pandas as pd
from pathlib import Path
import re

def params_to_latex(
    df, 
    out_tex_path="params_table.tex",
    param_col="Parameter",
    value_col="Value",
    model_col="model_cat",
    timestamp_col="Timestamp",
    prior_col_candidates=("Prior", "prior", "Prior Distribution", "prior_distribution"),
    model_name_map=None,
    desired_model_order=None,
    decimals=3
):
    """
    Takes a dataframe of parameter estimates and produces a LaTeX table.
    Keeps the latest row for each (parameter, model_cat) pair.
    """

    # ensure Timestamp is sortable (attempt parse)
    if timestamp_col in df.columns:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        except Exception:
            pass

    # Keep only latest row per (param, model)
    df_sorted = df.sort_values(by=timestamp_col, ascending=False) if timestamp_col in df.columns else df
    df_latest = df_sorted.drop_duplicates(subset=[param_col, model_col], keep="first").copy()
    #df_latest = df_sorted.drop_duplicates(subset=[param_col, model_col], keep="first")

    # Optionally remap model_cat names to pretty names
    if model_name_map:
        df_latest[model_col] = df_latest[model_col].replace(model_name_map)

    # detect prior column if present
    prior_col = None
    for cand in prior_col_candidates:
        if cand in df_latest.columns:
            prior_col = cand
            break

    # Pivot so each model is a column
    pivot = df_latest.pivot_table(index=param_col, columns=model_col, values=value_col, aggfunc='first')

    # Order models: put desired_model_order first, then any remaining
    available_models = list(pivot.columns.astype(str))
    ordered_models = []
    if desired_model_order:
        def norm(s): return "".join(str(s).lower().split()).replace(".", "")
        for want in desired_model_order:
            match = next((a for a in available_models if norm(a) == norm(want)), None)
            if match:
                ordered_models.append(match)
    for a in available_models:
        if a not in ordered_models:
            ordered_models.append(a)

    # Build DataFrame for LaTeX output
    latex_df = pd.DataFrame(index=pivot.index)
    if prior_col:
        priors = df_latest.groupby(param_col)[prior_col].first()
        latex_df["Prior Distribution"] = priors.reindex(latex_df.index).fillna("").astype(str)
    else:
        latex_df["Prior Distribution"] = f"$U(0.0001,0.9)$"

    # numeric formatting
    def fmt_val(x):
        if pd.isna(x):
            return ""
        try:
            f = float(x)
            s = f"{f:.{decimals}f}"
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s
        except Exception:
            return str(x)

    for col in ordered_models:
        latex_df[col] = pivot[col].apply(fmt_val)

    # escape underscores in parameters for LaTeX
    def escape_param(p):
        return str(p).replace("_", "\\_")

    # ------------------------------
    # NEW: split model names before every "w."
    # ------------------------------
    def split_model_name(m):
        """
        Splits model name into multiple lines before each 'w.' (case-insensitive),
        keeping 'w.' at the start of each continuation line.
        Example:
            "Non-behavioural" → one line
            "Non-behavioural w. OTJ" → two lines
            "Behavioural w. Cyc. OTJ w. RW" → three lines
        """
        s = str(m).strip()
        # split at whitespace that precedes "w." (case insensitive)
        parts = re.split(r'\s(?=(?i:w\.))', s)
        if len(parts) == 1:
            return s
        return "\\shortstack{" + " \\\\ ".join(parts) + "}"
    # ------------------------------

    # Build LaTeX string
    n_models = len(ordered_models)
    col_align = "|l|l|" + "c|" * n_models
    lines = []
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{width=\\textwidth}")
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append("\\hline")

    # Multi-line group header for Model
    lines.append("\\textbf{Parameter} & \\textbf{Prior Distribution} & "
                "\\multicolumn{" + str(n_models) + "}{c|}{\\textbf{Model Category}} \\\\")

    lines.append("\\cline{3-" + str(2 + n_models) + "}")

    # Header row for model names
    lines.append("& & " +
                 " & ".join([f"\\textbf{{{split_model_name(m)}}}" for m in ordered_models]) +
                 " \\\\")
    lines.append("\\hline")

    # data rows
    for param in latex_df.index:
        p_escaped = escape_param(param)
        prior = latex_df.loc[param, "Prior Distribution"].replace("%", "\\%")
        values = [latex_df.loc[param, col] for col in ordered_models]
        row = f"{p_escaped} & {prior} & " + " & ".join(values) + " \\\\ \\hline"
        lines.append(row)

    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\caption{Prior distribution and parameter estimates for all models. $U(a, b)$ denotes a uniform distribution on $[a,b]$.}")
    lines.append("\\label{tab:priors_posteriors}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    Path(out_tex_path).write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX table to {out_tex_path}")
    return out_tex_path
