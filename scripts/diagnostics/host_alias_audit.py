#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import sys

# Import normalize_name from step 1
base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))
from scripts.steps.step_1_data_ingestion import Step1DataIngestion

def run_audit():
    print("--- Host-Alias Join Audit ---")
    step1 = Step1DataIngestion()
    
    # 1. Get all SH0ES hosts from R22 distances
    r22_df = pd.read_csv(base_dir / "data" / "interim" / "r22_distances.csv")
    shoes_hosts = r22_df['source_id'].tolist()
    
    # 2. Get literature velocity dispersions
    lit_df = pd.read_csv(base_dir / "data" / "raw" / "external" / "velocity_dispersions_literature.csv", comment='#')
    lit_hosts = lit_df['galaxy'].tolist()
    
    # 3. Audit the join
    matches = 0
    missing = []
    
    print(f"{'SH0ES ID':<15} {'Normalized':<15} {'Matched in Literature CSV?':<25}")
    print("-" * 55)
    for host in shoes_hosts:
        norm_name = step1.normalize_name(host)
        
        # Check if norm_name or alternative is in lit_hosts
        matched = False
        if norm_name in lit_hosts:
            matched = True
        elif norm_name.startswith("NGC "):
            # Try without leading zero
            alt = "NGC " + norm_name[4:].lstrip("0")
            if alt in lit_hosts:
                matched = True
                
        if matched:
            matches += 1
            print(f"{host:<15} {norm_name:<15} {'YES':<25}")
        else:
            missing.append((host, norm_name))
            print(f"{host:<15} {norm_name:<15} {'NO':<25}")
            
    print("-" * 55)
    print(f"Total SH0ES hosts: {len(shoes_hosts)}")
    print(f"Matched to Literature: {matches}")
    print(f"Missing: {len(missing)}")
    
    if missing:
        print("\nMissing Hosts Details:")
        for h, n in missing:
            print(f" - {h} (normalized to {n})")
            
    # Check specific targets requested by user
    print("\n--- Specific Target Check ---")
    targets = ['M101', 'M31', 'N4258', 'LMC', 'SMC']
    for t in targets:
        n = step1.normalize_name(t)
        found = n in lit_hosts
        print(f"{t:<10} -> {n:<15} in Literature? {found}")

if __name__ == '__main__':
    run_audit()
