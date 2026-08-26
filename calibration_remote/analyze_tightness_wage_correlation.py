"""
Analyze correlation between simulated market tightness and real-world wage growth.

For each occupation:
1. Extract simulated market tightness (V/U) time series
2. Match with real wage growth data
3. Run regression: wage_growth ~ tightness
4. Plot regression coefficients across occupations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

# Paths
data_path = "/Users/ebbamark/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/"
path = "/Users/ebbamark/Documents/Documents - Nuff-Malham/GitHub/transition_abm/calibration_remote/output/cos_calib_smart_hire_4/full_omn/"

# Load wage growth data
wage_data = pd.read_csv(data_path + "data/occ_wages_ts_gr.csv", index_col=0)
print(f"Wage data shape: {wage_data.shape}")
print(f"Occupations in wage data: {len(wage_data)}")

# Load model results (assuming you have a cached result with filtered_sim_results)
# Adjust this path to your actual cached results file
cache_file = path + "figures/cached_results/model_results.pkl"  # modify as needed
try:
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)

    # Extract the relevant results dictionary
    # Assuming structure: cached['filtered_sim_results'] contains record_df by model
    if 'filtered_sim_results' in cached:
        sim_results = cached['filtered_sim_results']
    else:
        # Try to find record_df in the cache
        sim_results = cached

    print(f"Loaded cached results")
    print(f"Available keys: {list(sim_results.keys()) if isinstance(sim_results, dict) else 'Not a dict'}")

except FileNotFoundError:
    print(f"Cache file not found: {cache_file}")
    print("Please provide the path to your model results (record_df with Vacancies and Unemployment by occupation)")
    exit(1)

# For demonstration, let's assume sim_results is a dict with model names as keys
# Each contains a DataFrame with columns: ['Time Step', 'Occupation', 'Vacancies', 'Unemployment', ...]
# If your structure is different, adjust accordingly

# Reshape wage data from wide to long format (do this once before the loop)
wage_long = wage_data.melt(id_vars=['acs_occ_code'],
                            var_name='Year',
                            value_name='WageGrowth')
wage_long['Year'] = wage_long['Year'].astype(int)
wage_long = wage_long.dropna(subset=['WageGrowth'])  # drop missing values

print(f"\nWage data: {len(wage_long)} occupation-year observations")
print(f"Years in wage data: {sorted(wage_long['Year'].unique())}")

print(f"\nFound {len(sim_results)} models to analyze: {list(sim_results.keys())}")

# Loop through all models
for model_key in sim_results.keys():
    print(f"\n{'='*60}")
    print(f"Analyzing model: {model_key}")
    print(f"{'='*60}")

    record_df = sim_results[model_key]

    print(f"Record shape: {record_df.shape}")
    print(f"Record columns: {record_df.columns.tolist()}")

    # Compute market tightness (V/U) by occupation and time
    # Assuming record_df has columns: 'Time Step', 'Occupation', 'Vacancies', 'Unemployment'
    if 'Vacancies' not in record_df.columns or 'Unemployment' not in record_df.columns:
        print("ERROR: record_df must have 'Vacancies' and 'Unemployment' columns")
        print(f"Available columns: {record_df.columns.tolist()}")
        continue  # skip this model

    # Calculate tightness
    record_df['Tightness'] = record_df['Vacancies'] / (record_df['Unemployment'] + 1)  # +1 to avoid division by zero

    # Map time steps to years (assuming monthly data, adjust as needed)
    # If delay=0 and data starts in 2001-01, then time_step 0 = 2001-01
    # Adjust these parameters based on your actual calibration window
    calib_start_year = 2001  # adjust based on your calib_date
    months_per_year = 12

    record_df['Year'] = calib_start_year + (record_df['Time Step'] // months_per_year)

    # Aggregate tightness to annual level (mean within each year-occupation)
    annual_tightness = record_df.groupby(['Occupation', 'Year'])['Tightness'].mean().reset_index()

    print(f"\nAnnual tightness computed for {annual_tightness['Occupation'].nunique()} occupations")
    print(f"Years in simulation: {sorted(annual_tightness['Year'].unique())}")

    # Diagnostic: Check occupation ID matching
    print("\n=== Occupation ID Diagnostic ===")
    model_occs = sorted(annual_tightness['Occupation'].unique())
    wage_occs = sorted(wage_long['acs_occ_code'].unique())
    print(f"Model occupations: {len(model_occs)} total")
    print(f"  Sample: {model_occs[:10]}")
    print(f"Wage occupations: {len(wage_occs)} total")
    print(f"  Sample: {wage_occs[:10]}")

    # Check if IDs match directly
    direct_matches = set(model_occs) & set(wage_occs)
    print(f"Direct ID matches: {len(direct_matches)}")

    # If no direct matches, we need a mapping
    # Load occupation mapping file if it exists
    if len(direct_matches) == 0:
        print("\nWARNING: No direct occupation ID matches!")
        print("Attempting to load occupation mapping...")

        # Try to load occ_ids mapping (adjust path as needed)
        try:
            occ_ids = pd.read_csv(data_path + "dRC_Replication/data/ipums_variables_full_omn_w_exp.csv")
            print(f"Loaded occupation mapping with columns: {occ_ids.columns.tolist()}")

            # Create mapping: model_occ_id -> acs_occ_code
            # Adjust column names based on actual structure
            if 'occ_id' in occ_ids.columns and 'acs_occ_code' in occ_ids.columns:
                occ_mapping = dict(zip(occ_ids['occ_id'], occ_ids['acs_occ_code']))
                annual_tightness['acs_occ_code'] = annual_tightness['Occupation'].map(occ_mapping)
                annual_tightness = annual_tightness.dropna(subset=['acs_occ_code'])
                print(f"Mapped {len(annual_tightness['acs_occ_code'].unique())} occupations")
            else:
                print(f"ERROR: Expected columns 'occ_id' and 'acs_occ_code', found: {occ_ids.columns.tolist()}")
                print("Please create a mapping manually or adjust column names")
                continue  # skip this model
        except FileNotFoundError as e:
            print(f"ERROR: Could not load occupation mapping file: {e}")
            print("Please provide a CSV mapping model occupation IDs to ACS codes")
            continue  # skip this model
    else:
        # Direct match works - just use Occupation as acs_occ_code
        annual_tightness['acs_occ_code'] = annual_tightness['Occupation']

    # Merge tightness and wage data
    merged = annual_tightness.merge(
        wage_long,
        on=['acs_occ_code', 'Year'],
        how='inner'
    )

    print(f"\nMerged data: {len(merged)} observations across {merged['Occupation'].nunique()} occupations")

    if len(merged) == 0:
        print("ERROR: No matching observations after merge")
        continue  # skip this model

    # Run regression for each occupation: WageGrowth ~ Tightness
    regression_results = []

    for occ_code in merged['acs_occ_code'].unique():
        occ_data = merged[merged['acs_occ_code'] == occ_code]

        # Need at least 3 observations for meaningful regression
        if len(occ_data) < 3:
            continue

        X = occ_data['Tightness'].values
        y = occ_data['WageGrowth'].values

        # Run OLS regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

        regression_results.append({
            'Occupation': occ_code,
            'N_obs': len(occ_data),
            'Beta': slope,
            'R_squared': r_value**2,
            'P_value': p_value,
            'Std_Error': std_err,
            'Mean_Tightness': X.mean(),
            'Mean_WageGrowth': y.mean()
        })

    reg_df = pd.DataFrame(regression_results)
    print(f"\nRan regressions for {len(reg_df)} occupations")
    print(f"Significant (p<0.05): {(reg_df['P_value'] < 0.05).sum()}")
    print(f"Positive coefficients: {(reg_df['Beta'] > 0).sum()}")

    # Summary statistics
    print(f"\nRegression coefficient summary:")
    print(reg_df['Beta'].describe())

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Add model name to the overall figure title
    fig.suptitle(f'Tightness-Wage Correlation Analysis: {model_key}', fontsize=14, fontweight='bold', y=0.995)

    # Plot 1: Distribution of regression coefficients
    ax = axes[0, 0]
    ax.hist(reg_df['Beta'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero effect')
    ax.axvline(x=reg_df['Beta'].median(), color='blue', linestyle='--', linewidth=2,
               label=f'Median = {reg_df["Beta"].median():.4f}')
    ax.set_xlabel('Regression Coefficient (β)')
    ax.set_ylabel('Number of Occupations')
    ax.set_title('Distribution of Tightness → Wage Growth Coefficients')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Coefficient vs significance
    ax = axes[0, 1]
    colors = ['red' if p < 0.05 else 'gray' for p in reg_df['P_value']]
    ax.scatter(reg_df['Beta'], -np.log10(reg_df['P_value']), c=colors, alpha=0.6, s=50)
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Regression Coefficient (β)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Coefficient Significance')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 3: Coefficient by occupation (sorted)
    ax = axes[1, 0]
    sorted_reg = reg_df.sort_values('Beta')
    colors_sig = ['red' if p < 0.05 else 'lightblue' for p in sorted_reg['P_value']]
    ax.barh(range(len(sorted_reg)), sorted_reg['Beta'], color=colors_sig)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Regression Coefficient (β)')
    ax.set_ylabel('Occupation (sorted by β)')
    ax.set_title('Coefficients by Occupation (red = p<0.05)')
    ax.set_yticks([])  # too many to label
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 4: R-squared distribution
    ax = axes[1, 1]
    ax.hist(reg_df['R_squared'], bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('R²')
    ax.set_ylabel('Number of Occupations')
    ax.set_title('Model Fit Distribution')
    ax.axvline(x=reg_df['R_squared'].median(), color='blue', linestyle='--',
               label=f'Median R² = {reg_df["R_squared"].median():.3f}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with model-specific filename
    model_filename = model_key.replace(' ', '_').replace('/', '_')
    plot_path = path + f'tightness_wage_regression_coefficients_{model_filename}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")

    # Save regression results with model-specific filename
    csv_path = path + f'tightness_wage_regressions_{model_filename}.csv'
    reg_df.to_csv(csv_path, index=False)
    print(f"Regression results saved to: {csv_path}")

    plt.close(fig)  # close figure to free memory

    # Print top 10 occupations by coefficient
    print("\n=== Top 10 Occupations (Strongest Positive Effect) ===")
    print(reg_df.nlargest(10, 'Beta')[['Occupation', 'Beta', 'P_value', 'R_squared', 'N_obs']])

    print("\n=== Top 10 Occupations (Strongest Negative Effect) ===")
    print(reg_df.nsmallest(10, 'Beta')[['Occupation', 'Beta', 'P_value', 'R_squared', 'N_obs']])

print("\n" + "="*60)
print("Analysis complete for all models!")
print("="*60)
