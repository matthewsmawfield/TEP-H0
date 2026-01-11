
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Apply TEP Style
try:
    from scripts.utils.plot_style import apply_tep_style
    colors = apply_tep_style()
except ImportError:
    colors = {'blue': '#395d85', 'accent': '#b43b4e', 'dark': '#301E30', 'light_blue': '#4b6785', 'green': '#4a2650'}

DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "site/public/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load Cepheid, Host, and SN data."""
    # 1. Cepheids
    cepheids = pd.read_csv(DATA_DIR / "interim/reconstructed_shoes_cepheids.csv")
    # Recover LogP. Assuming L_col_bW is (logP - 1)
    # Check if L_col_bW corresponds to parameter bW.
    # In step_1, we extracted 'L_col_bW' from the column corresponding to 'bW' parameter.
    # The term is bW * (logP - 1). So the column contains (logP - 1).
    cepheids['logP'] = cepheids['L_col_bW'] + 1.0
    cepheids['Period'] = 10**cepheids['logP']
    
    # 2. Hosts (with Sigma)
    hosts = pd.read_csv(DATA_DIR / "processed/hosts_processed.csv")
    
    # 3. SN Data (Pantheon+) for Color (c)
    # We need to parse this again or grab it if saved. 
    # It wasn't fully saved in processed, so let's reload raw.
    pantheon_path = DATA_DIR / "raw/Pantheon+SH0ES.dat"
    if pantheon_path.exists():
        pan_df = pd.read_csv(pantheon_path, sep=r'\s+')
        # Filter for the SH0ES subset? Or just use those matched to our hosts.
        # We can merge on pantheon_id (CID)
        # First we need to know which CID belongs to which host.
        # hosts_processed has 'pantheon_id'
        pass
    else:
        pan_df = None
        
    return cepheids, hosts, pan_df

def analyze_period_age(cepheids, hosts):
    """Analyze dependency of Period (Age proxy) on Host Sigma."""
    print("--- Analyzing Period-Age vs Sigma ---")
    
    # Group by Host
    host_periods = cepheids.groupby('Source')['logP'].agg(['mean', 'std', 'count']).reset_index()
    host_periods.rename(columns={'mean': 'mean_logP', 'std': 'std_logP', 'count': 'n_cepheids'}, inplace=True)
    
    # Merge with Sigma
    # Need to handle host name matching. 'Source' in cepheids vs 'source_id'/'normalized_name' in hosts.
    # Step 1 used a normalize_name function. We should probably use that or fuzzy match.
    # However, Step 1 saved 'reconstructed_shoes_cepheids.csv' with 'Source' column from y_R22.
    # And 'hosts_processed.csv' has 'source_id' which should match 'Source'.
    
    merged = pd.merge(host_periods, hosts, left_on='Source', right_on='source_id', how='inner')
    
    # Filter valid sigma
    data = merged.dropna(subset=['sigma_inferred'])
    
    # Correlation
    r_val, p_val = stats.pearsonr(data['sigma_inferred'], data['mean_logP'])
    rho_val, sp_val = stats.spearmanr(data['sigma_inferred'], data['mean_logP'])
    
    print(f"N Hosts: {len(data)}")
    print(f"Pearson r(Sigma, Mean LogP): {r_val:.3f} (p={p_val:.3f})")
    print(f"Spearman rho(Sigma, Mean LogP): {rho_val:.3f} (p={sp_val:.3f})")
    
    # Plot
    plt.figure(figsize=(14, 9))
    # sns.scatterplot(data=data, x='sigma_inferred', y='mean_logP', size='n_cepheids', sizes=(20, 200), alpha=0.7)
    plt.scatter(data['sigma_inferred'], data['mean_logP'], s=data['n_cepheids']*2, alpha=0.7, color=colors['blue'], edgecolor='white', linewidth=0.5, label='Hosts (Size ~ N_cep)')
    
    # Fit line
    m, b = np.polyfit(data['sigma_inferred'], data['mean_logP'], 1)
    x_range = np.linspace(data['sigma_inferred'].min(), data['sigma_inferred'].max(), 100)
    plt.plot(x_range, m*x_range + b, color=colors['accent'], linestyle='--', alpha=0.9, linewidth=2.5, label=f'Fit (r={r_val:.2f})')
    
    plt.xlabel(r'Velocity Dispersion $\sigma$ (km/s)')
    plt.ylabel(r'Mean Cepheid $\log P$ (Days)')
    plt.title('Cepheid Period Distribution vs Host Potential')
    plt.legend()
    
    out_path = FIGURES_DIR / "check_period_sigma.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    plt.close()
    
    return data

def analyze_dust_color(hosts, pan_df):
    """Analyze dependency of SN Color (c) on Host Sigma."""
    print("\n--- Analyzing SN Color (Dust) vs Sigma ---")
    
    if pan_df is None:
        print("Pantheon data not found, skipping dust analysis.")
        return
    
    # Merge hosts with Pantheon data to get 'c'
    # hosts_processed has 'pantheon_id' which maps to 'CID' in pan_df
    
    merged = pd.merge(hosts, pan_df[['CID', 'c', 'cERR', 'x1', 'x1ERR']], left_on='pantheon_id', right_on='CID', how='inner')
    
    # Filter valid sigma
    data = merged.dropna(subset=['sigma_inferred', 'c'])
    
    # Correlation
    r_val, p_val = stats.pearsonr(data['sigma_inferred'], data['c'])
    rho_val, sp_val = stats.spearmanr(data['sigma_inferred'], data['c'])
    
    print(f"N Hosts: {len(data)}")
    print(f"Pearson r(Sigma, SN Color c): {r_val:.3f} (p={p_val:.3f})")
    print(f"Spearman rho(Sigma, SN Color c): {rho_val:.3f} (p={sp_val:.3f})")
    
    # Plot
    plt.figure(figsize=(14, 9))
    plt.errorbar(data['sigma_inferred'], data['c'], yerr=data['cERR'], fmt='o', alpha=0.7, color=colors['green'], ecolor=colors['purple'], capsize=3, label='Pantheon+ SN')
    
    # Fit line
    m, b = np.polyfit(data['sigma_inferred'], data['c'], 1)
    x_range = np.linspace(data['sigma_inferred'].min(), data['sigma_inferred'].max(), 100)
    plt.plot(x_range, m*x_range + b, color=colors['accent'], linestyle='--', alpha=0.9, linewidth=2.5, label=f'Fit (r={r_val:.2f})')
    
    plt.xlabel(r'Velocity Dispersion $\sigma$ (km/s)')
    plt.ylabel(r'SN Color parameter $c$ (Dust Proxy)')
    plt.title('SN Color vs Host Potential')
    plt.legend()
    
    out_path = FIGURES_DIR / "check_dust_sigma.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    plt.close()

def main():
    cepheids, hosts, pan_df = load_data()
    
    # 1. Period (Age) Analysis
    analyze_period_age(cepheids, hosts)
    
    # 2. Dust (SN Color) Analysis
    analyze_dust_color(hosts, pan_df)

if __name__ == "__main__":
    main()
