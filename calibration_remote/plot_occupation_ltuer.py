import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_occupation_ltuer(sim_results_dict, save=False, path=None):
    """
    Creates scatterplots of mean LTUER by occupation for each element in the dictionary.
    
    Parameters:
    - sim_results_dict: Dictionary containing simulation results DataFrames
    - save: Boolean indicating whether to save the plots
    - path: Path to save the plots if save=True
    """
    # First, calculate global y-axis limits
    all_ltuer_rates = []
    for sim_results in sim_results_dict.values():
        occ_data = sim_results.loc[:, ['Time Step', 'Occupation_ID', 'Workers', 'LT Unemployed Persons']]
        occ_data['LTUE Rate'] = occ_data['LT Unemployed Persons'] / occ_data['Workers']
        mean_ltuer = occ_data.groupby('Occupation_ID')['LTUE Rate'].mean()
        all_ltuer_rates.extend(mean_ltuer.values)
    
    # Add some padding to the y-axis limits
    y_min = min(all_ltuer_rates) * 0.95  # 5% padding below
    y_max = max(all_ltuer_rates) * 1.05  # 5% padding above
    
    for name, sim_results in sim_results_dict.items():
        # Extract occupation-specific data
        occ_data = sim_results.loc[:, ['Time Step', 'Occupation_ID', 'Workers', 'LT Unemployed Persons']]
        
        # Calculate LTUER for each occupation at each time step
        occ_data['LTUE Rate'] = occ_data['LT Unemployed Persons'] / occ_data['Workers']
        
        # Calculate mean LTUER for each occupation
        mean_ltuer = occ_data.groupby('Occupation_ID')['LTUE Rate'].mean().reset_index()
        
        # Sort occupations by mean LTUER and create ordered x-axis positions
        mean_ltuer = mean_ltuer.sort_values('LTUE Rate')
        mean_ltuer['x_pos'] = range(len(mean_ltuer))
        
        # Create the scatterplot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=mean_ltuer, x='x_pos', y='LTUE Rate', s=100)
        
        # Customize the plot
        plt.title(f'Mean Long-term Unemployment Rate by Occupation ({name})', fontweight='bold')
        plt.xlabel('Occupation (ordered by mean LTUER)')
        plt.ylabel('Mean Long-term Unemployment Rate')
        
        # Set x-axis ticks to show occupation IDs in the sorted order
        plt.xticks(mean_ltuer['x_pos'], mean_ltuer['Occupation_ID'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Set consistent y-axis limits
        plt.ylim(y_min, y_max)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Import simulation results
    from calibration_us_gdp import filtered_sim_results
    
    # Plot for each element in the dictionary
    plot_occupation_ltuer(filtered_sim_results, save=False) 