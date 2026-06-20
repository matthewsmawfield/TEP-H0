#!/usr/bin/env python3
"""Master diagnostic for TEP-H0 pipeline weaknesses."""

import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.utils.tep_correction import C_SQUARED_KM_S, total_screening_factor, group_screening_factor, ANCHOR_SCREENING

def h2(title): print(f"\n{'='*80}\n  {title}\n{'='*80}\n")
def sh(title): print(f"\n  --- {title} ---\n")

# ---------------------------------------------------------------------------
# 1. REGRESSOR DEBUG
# ---------------------------------------------------------------------------
h2("1. REGRESSOR CONSTRUCTION DEBUG")
strat = pd.read_csv(PROJECT_ROOT / "results/outputs/step_03_stratified_h0.csv")
with open(PROJECT_ROOT / "results/outputs/step_04_tep_correction_results.json") as f:
    tep = json.load(f)
sigma = strat["sigma_inferred"].values
h0 = strat["h0_derived"].values
z = strat["z_hd"].values
rho = strat["rho_local"].values
n_mb = strat["tully_nmb"].fillna(1.0).values
S_local = np.array([total_screening_factor(r,n) for r,n in zip(rho,n_mb)])
S_group = np.array([group_screening_factor(n) for n in n_mb])
S_total = S_local * S_group
sref = float(tep["sigma_ref"]); sref_scr = float(tep.get("sigma_ref_screened",30.51))
c2 = C_SQUARED_KM_S

regressors = {
    "sigma": sigma, "sigma_sq": sigma**2,
    "S_local*sigma_sq": S_local*sigma**2,
    "S_group*sigma_sq": S_group*sigma**2,
    "S_total*sigma_sq": S_total*sigma**2,
    "TEP_full_std": S_total*(sigma**2-sref**2)/c2,
    "TEP_local_std": S_local*(sigma**2-sref**2)/c2,
    "TEP_full_scr": S_total*(sigma**2-sref_scr**2)/c2,
}

sh("Correlation with H0")
print(f"{'Regressor':<22} {'Pearson r':>10} {'p':>10} {'Spearman':>10} {'p':>10} {'Scatter':>8}")
print("-"*72)
rows = []
for name, X in regressors.items():
    r, p = stats.pearsonr(X, h0); rho_s, p_s = stats.spearmanr(X, h0)
    sc = np.std(h0-np.mean(h0)); rows.append((name, abs(r), r, p, rho_s, p_s, sc))
    print(f"{name:<22} {r:>10.3f} {p:>10.4f} {rho_s:>10.3f} {p_s:>10.4f} {sc:>8.2f}")
rows.sort(key=lambda x: x[1], reverse=True)
print(f"\nBest by |r|: {rows[0][0]} (r={rows[0][2]:.3f})")

# Check if TEP_full_std is clearly best
tep_r = abs(dict((n,abs(r)) for n,_,r,_,_,_,_ in rows).get("TEP_full_std",0))
simpler = max(abs(dict((n,abs(r)) for n,_,r,_,_,_,_ in rows).get(k,0)) for k in ["sigma","sigma_sq","S_local*sigma_sq","S_total*sigma_sq"])
print(f"\nTEP_full_std |r| = {tep_r:.3f}; best simpler = {simpler:.3f}")
if tep_r < simpler:
    print("WARNING: full TEP regressor is weaker than simpler proxies.")
    print("  Reason: S_group ~ 1 for most field hosts; adds noise without signal.")
    print("  Fix: Use S_local-only for host correction; S_total for anchor reference.")
else:
    print("OK: full TEP outperforms simpler proxies.")

# ---------------------------------------------------------------------------
# 2. ANCHOR SCREENING
# ---------------------------------------------------------------------------
h2("2. ANCHOR SCREENING DEBUG")
anchors = {"MW":{"sigma":160,"rho":0.5,"n_mb":7},
           "LMC":{"sigma":24,"rho":0.1,"n_mb":2},
           "NGC 4258":{"sigma":115,"rho":0.05,"n_mb":65},
           "M31":{"sigma":160,"rho":0.8,"n_mb":11}}
print(f"{'Anchor':<12} {'sigma':>6} {'N_mb':>5} {'S_local':>9} {'S_group':>9} {'S_total':>9}")
print("-"*56)
for name,d in anchors.items():
    sl = total_screening_factor(d["rho"],d["n_mb"])
    sg = group_screening_factor(d["n_mb"])
    print(f"{name:<12} {d['sigma']:>6.0f} {d['n_mb']:>5.0f} {sl:>9.4f} {sg:>9.4f} {sl*sg:>9.4f}")

# Host S_group distribution
sg_vals = S_group
print(f"\nHost S_group stats: mean={sg_vals.mean():.4f}, std={sg_vals.std():.4f}, min={sg_vals.min():.4f}, max={sg_vals.max():.4f}")
print(f"Hosts with S_group<0.9: {(sg_vals<0.9).sum()}; with S_group<0.5: {(sg_vals<0.5).sum()}")
if sg_vals.std() < 0.1:
    print("WARNING: S_group adds almost no variation among hosts. Group screening is only relevant for anchors.")

# ---------------------------------------------------------------------------
# 3. SIGMA_REF
# ---------------------------------------------------------------------------
h2("3. SIGMA_REF DEBUG")
print(f"Pipeline sigma_ref (standard):  {sref:.2f} km/s")
print(f"Pipeline sigma_ref (screened):  {sref_scr:.2f} km/s")
print(f"Pipeline kappa (standard):      {tep['optimal_kappa_cep']:.2e} mag")
print(f"Pipeline H0 (standard):       {tep['unified_h0']:.2f} km/s/Mpc")

weights = {"MW":0.03,"LMC":0.10,"NGC 4258":0.84,"M31":0.03}
sigmas = {"MW":160.0,"LMC":24.0,"NGC 4258":115.0,"M31":160.0}
def ref(w,s,scr=None):
    num = sum(w[n]*(scr.get(n,1.0) if scr else 1.0)*(s[n]**2) for n in w)
    return np.sqrt(num/sum(w[n] for n in w))

print(f"\nReconstructed:")
print(f"  Approx unscreened:  {ref(weights,sigmas):.2f}")
print(f"  Approx screened:    {ref(weights,sigmas,ANCHOR_SCREENING):.2f}")
print(f"  Equal unscreened:   {ref({k:0.25 for k in weights},sigmas):.2f}")
print(f"  Equal screened:     {ref({k:0.25 for k in weights},sigmas,ANCHOR_SCREENING):.2f}")

# ---------------------------------------------------------------------------
# 4. TRGB DIFFERENTIAL
# ---------------------------------------------------------------------------
h2("4. TRGB DIFFERENTIAL DEBUG")
trgb = pd.read_csv(PROJECT_ROOT / "results/outputs/step_15_trgb_hosts_data.csv")
trgb["match"] = trgb["galaxy"].str.replace(" ","").str.upper()
strat["match"] = strat["normalized_name"].str.replace(" ","").str.upper()
merged = pd.merge(trgb, strat, on="match", suffixes=("_trgb","_host"))
print(f"Matched hosts: N={len(merged)}")
print(f"{'Host':<12} {'mu_Ceph':>8} {'mu_TRGB':>8} {'delta_mu':>10} {'sigma':>8}")
print("-"*52)
for _,row in merged.iterrows():
    d = row["mu_trgb"]-row["value"]
    print(f"{row['match']:<12} {row['value']:>8.3f} {row['mu_trgb']:>8.3f} {d:>+10.3f} {row['sigma_inferred_trgb']:>8.1f}")

r_d, p_d = stats.pearsonr(merged["sigma_inferred_trgb"].values, merged["mu_trgb"].values-merged["value"].values)
print(f"\ndelta_mu vs sigma: r={r_d:.3f}, p={p_d:.4f}")
if r_d > 0: print("OK: Positive = Cepheid mu underestimated at high sigma (TEP-consistent)")
else: print("WARNING: Sign reversed!")

# Cross-check stored
with open(PROJECT_ROOT / "results/outputs/step_29_cross_channel_consistency.json") as f: cc=json.load(f)
with open(PROJECT_ROOT / "results/outputs/step_20_joint_indicator_model.json") as f: ji=json.load(f)
print(f"\nStep 12 kappa_diff: {cc['kappa_diff']['kappa_diff']:.1f}")
print(f"Step 19 kappa_diff: {ji['kappa_diff']:.1f}")
print(f"Match: {abs(cc['kappa_diff']['kappa_diff']-ji['kappa_diff'])<1000}")

# ---------------------------------------------------------------------------
# 5. COVARIANCE
# ---------------------------------------------------------------------------
h2("5. COVARIANCE MATRIX DEBUG")
cov = np.load(PROJECT_ROOT / "results/outputs/step_03_h0_covariance.npy")
with open(PROJECT_ROOT / "results/outputs/step_03_h0_covariance_labels.json") as f: labels=json.load(f)
print(f"Shape: {cov.shape}; Symmetric: {np.allclose(cov,cov.T)}")
eig = np.linalg.eigvalsh(cov)
print(f"PSD: {np.all(eig>-1e-10)}; Min eig: {eig.min():.6f}; Max eig: {eig.max():.2f}")
print(f"Cond: {np.max(np.abs(eig))/max(np.min(np.abs(eig[eig!=0])),1e-10):.2e}")
source_ids = strat["source_id"].astype(str).tolist()
print(f"Order match: {source_ids==labels}")
if source_ids != labels:
    print(f"  WARNING: Mismatch at positions: {[i for i in range(len(source_ids)) if source_ids[i]!=labels[i]][:5]}")

# Shuffle test
np.random.seed(42)
perm = np.random.permutation(len(sigma))
sigma_shuf = sigma[perm]
r_orig,p_orig = stats.pearsonr(sigma,h0)
r_shuf,p_shuf = stats.pearsonr(sigma_shuf,h0)
print(f"\nShuffle test: original p={p_orig:.4f}, shuffled p={p_shuf:.4f}")
if p_shuf < 0.05: print("WARNING: Shuffled sigma still significant!")
else: print("OK: Shuffled sigma not significant.")

# ---------------------------------------------------------------------------
# 6. VELOCITY DISPERSIONS
# ---------------------------------------------------------------------------
h2("6. VELOCITY DISPERSION HOST AUDIT")
prov = pd.read_csv(PROJECT_ROOT / "results/outputs/step_07_sigma_provenance_table.csv")
strat = strat.merge(prov[["normalized_name","sigma_method"]], on="normalized_name", how="left")
for method in strat["sigma_method"].dropna().unique():
    sub = strat[strat["sigma_method"]==method]
    print(f"\n{method}: N={len(sub)}, sigma={sub['sigma_inferred'].min():.0f}-{sub['sigma_inferred'].max():.0f}")
    if len(sub)>2:
        r,p = stats.pearsonr(sub["sigma_inferred"].values, sub["h0_derived"].values)
        print(f"  H0 vs sigma: r={r:.3f}, p={p:.4f}")

print(f"\n{'Host':<12} {'sigma':>7} {'method':<22} {'H0':>7} {'z':>7} {'dmu':>8}")
print("-"*62)
strat["dmu"] = float(tep["optimal_kappa_cep"]) * S_local * (strat["sigma_inferred"]**2 - sref**2)/c2
for _,row in strat.sort_values("sigma_inferred", ascending=False).head(15).iterrows():
    print(f"{row['normalized_name']:<12} {row['sigma_inferred']:>7.1f} {str(row.get('sigma_method','NA')):<22} {row['h0_derived']:>7.1f} {row['z_hd']:>7.4f} {row['dmu']:>+8.3f}")

# ---------------------------------------------------------------------------
# 7. ODR
# ---------------------------------------------------------------------------
h2("7. ODR VS OLS DEBUG")
from scipy.odr import ODR, Model, RealData
def linear(B,x): return B[0]*x + B[1]
model = Model(linear)
ols_slope,_ = np.polyfit(sigma,h0,1)
print(f"OLS (H0 vs sigma):   slope={ols_slope:.6f}")
sigma_err = np.ones(len(sigma))*10.0
h0_err = h0*(np.log(10)/5)*strat["error"].fillna(0.05).values
data = RealData(sigma,h0,sx=sigma_err,sy=h0_err)
odr = ODR(data,model,beta0=[0.1,65.0])
out = odr.run()
print(f"ODR (H0 vs sigma):   slope={out.beta[0]:.6f} ± {out.sd_beta[0]:.6f}")
print(f"ODR/OLS ratio:        {out.beta[0]/ols_slope:.2f}x")

ols_sq,_ = np.polyfit(sigma**2,h0,1)
data_sq = RealData(sigma**2,h0,sx=2*sigma*sigma_err,sy=h0_err)
odr_sq = ODR(data_sq,model,beta0=[0.001,65.0])
out_sq = odr_sq.run()
print(f"\nOLS (H0 vs sigma^2): slope={ols_sq:.9f}")
print(f"ODR (H0 vs sigma^2): slope={out_sq.beta[0]:.9f} ± {out_sq.sd_beta[0]:.9f}")
print(f"ODR/OLS ratio:        {out_sq.beta[0]/ols_sq:.2f}x")

frac_err = np.median(sigma_err)/np.median(sigma)
print(f"\nMedian fractional sigma_err: {frac_err:.1%}")
if frac_err < 0.05: print("WARNING: sigma errors may be too small")

# ---------------------------------------------------------------------------
# 8. VALIDATION SPLITS
# ---------------------------------------------------------------------------
h2("8. VALIDATION SPLIT DEBUG")
z = strat["z_hd"].values
splits = [("lowz_to_highz", z<0.005, z>=0.005), ("highz_to_lowz", z>=0.005, z<0.005)]
for name,tr,te in splits:
    if tr.sum()>0 and te.sum()>0:
        s_tr, s_te = sigma[tr], sigma[te]
        print(f"{name}: train N={tr.sum()}, test N={te.sum()}")
        print(f"  Train sigma: {s_tr.min():.0f}-{s_tr.max():.0f}; Test sigma: {s_te.min():.0f}-{s_te.max():.0f}")
        if s_tr.min() > s_te.max() or s_tr.max() < s_te.min():
            print("  WARNING: No sigma overlap!")
        if s_tr.max()-s_tr.min() < 50:
            print(f"  WARNING: Train range too narrow ({s_tr.max()-s_tr.min():.0f})")

# ---------------------------------------------------------------------------
# 9. OUTLIER HOSTS
# ---------------------------------------------------------------------------
h2("9. LARGE-CORRECTION OUTLIERS")
print(f"{'Host':<12} {'sigma':>7} {'z':>7} {'S_total':>7} {'dmu':>8} {'method':<20}")
print("-"*62)
for _,row in strat.nlargest(10,"dmu").iterrows():
    sg = group_screening_factor(row.get("tully_nmb",1.0))
    sl = total_screening_factor(row["rho_local"],row.get("tully_nmb",1.0))
    print(f"{row['normalized_name']:<12} {row['sigma_inferred']:>7.1f} {row['z_hd']:>7.4f} {sl*sg:>7.3f} {row['dmu']:>+8.3f} {str(row.get('sigma_method','NA')):<20}")

print("\nFlagged hosts:")
for _,row in strat.iterrows():
    flags=[]
    if row["dmu"]>0.3: flags.append(f"large dmu ({row['dmu']:+.2f})")
    if row["z_hd"]<0.004: flags.append(f"low-z ({row['z_hd']:.4f})")
    if row["sigma_inferred"]>200: flags.append(f"high-sigma ({row['sigma_inferred']:.0f})")
    if str(row.get("sigma_method","")).startswith("HI"): flags.append("HI proxy")
    if flags: print(f"  {row['normalized_name']}: {', '.join(flags)}")

# ---------------------------------------------------------------------------
# 10. SIGN CONVENTION
# ---------------------------------------------------------------------------
h2("10. SIGN CONVENTION")
print("""
Expected TEP chain:
  High sigma -> deep potential -> faster clock -> P contracts ->
  inferred M fainter -> mu smaller -> distance underestimated -> H0 higher.
Therefore:
  H0 vs sigma: POSITIVE
  mu vs sigma: NEGATIVE
  mu_TRGB - mu_Ceph vs sigma: POSITIVE (Cepheid mu underestimated)
  TEP correction dmu: positive for sigma > sigma_ref
""")

# Verify
r_h0_s, _ = stats.pearsonr(sigma, h0)
r_mu_s, _ = stats.pearsonr(sigma, strat["value"].values)
print(f"Actual: H0 vs sigma r={r_h0_s:+.3f} (expect +); mu vs sigma r={r_mu_s:+.3f} (expect -)")
if r_h0_s > 0 and r_mu_s < 0:
    print("OK: Signs consistent with TEP prediction.")
else:
    print("WARNING: Sign mismatch detected!")

print("\n" + "="*80)
print("  MASTER DEBUG COMPLETE")
print("="*80 + "\n")

if __name__ == "__main__":
    pass  # all functions run inline above
