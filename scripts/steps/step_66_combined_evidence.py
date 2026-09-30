#!/usr/bin/env python3
"""
step_66_combined_evidence.py

Cross-channel combined-evidence quantification.

No single H0 channel reaches 3 sigma, but several channels that draw on
independent noise sources all return the TEP direction.  This step computes
the honest joint significance with explicit independence assumptions and a
sensitivity analysis over inter-channel correlations:

Primary channel set (independent noise sources):
  1. velocity_endpoint (step_39): Gamma_X, redshift-distance residuals.
     Noise: peculiar velocities + redshift scatter.
  2. within_host_period (step_52): kappa_P in the joint kappa_P + kappa_Z
     fit, period-coupled structure in the SH0ES design matrix. Noise:
     Cepheid photometric residuals.
  3. within_host_metallicity (step_52): kappa_Z, fitted jointly with
     kappa_P (fit correlation -0.15 carried explicitly).
  4. trgb_differential (step_41): Cepheid - TRGB distance-scale
     differential. Noise: TRGB photometry, shares host set and anchors
     with (1).

Consistency channel (not independent evidence; derived propagation):
  5. hfsn_contrast (step_65): the calibrator-vs-HF environment contrast,
     reported one-sided in the predicted direction.

Combination: Stouffer's Z = sum(z_i w_i) / sqrt(W + 2 sum_{i<j} rho_ij
w_i w_j), with equal weights, two-sided channel p-values converted to z.
Reported at rho = 0 (baseline), and as a sensitivity over a shared
correlation rho_s applied to all non-fitted pairs.

Outputs:
  results/outputs/step_66_combined_evidence.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "results" / "outputs"


def print_status(msg, level="INFO"):
    prefix = {"SECTION": "=" * 60, "INFO": "[INFO]", "SUCCESS": "[SUCCESS]",
              "WARNING": "[WARNING]"}.get(level, "[INFO]")
    if level == "SECTION":
        print(f"\n{prefix}\n{msg}\n{prefix}")
    else:
        print(f"{prefix} {msg}")


def load_channels():
    """Collect (estimate, error, z) per channel from current outputs."""
    ch = []

    # 1. velocity-space endpoint, canonical sigma_v = 250 row
    r39 = json.load(open(OUT_DIR / "step_39_environment_slope_decomposition.json"))
    row = [m for m in r39
           if m.get("sigma_v") == 250 and m.get("sample") == "all_r22_hosts"
           and m.get("z_cut") == 0.0][0]
    ch.append({
        "channel": "velocity_endpoint_GammaX",
        "source": "step_39", "estimate": row["Gamma_X"],
        "error": row["Gamma_X_err"], "z": row["Gamma_X_sig"],
        "independent_noise": "peculiar velocity + redshift residuals",
        "primary": True})

    # 2-3. within-host period and metallicity terms, joint fit,
    # anchor_screened_physical convention
    r52 = json.load(open(OUT_DIR / "step_52_bounded_response_comparison.json"))
    jrow = [r for r in r52["kappaP_kappaZ_disentanglement"]
            if r["model"] == "kappaP_plus_kappaZ"
            and r["anchor_convention"] == "anchor_screened_physical"][0]
    ch.append({
        "channel": "within_host_period_kappaP",
        "source": "step_52", "estimate": jrow["kappaP_6"],
        "error": jrow["kappaP_6_err"], "z": jrow["kappaP_6_sig"],
        "independent_noise": "Cepheid photometric residuals (design matrix)",
        "fit_corr_with": {"within_host_metallicity_kappaZ":
                          jrow["corr_kappaP_kappaZ"]},
        "primary": True})
    ch.append({
        "channel": "within_host_metallicity_kappaZ",
        "source": "step_52", "estimate": jrow["kappaZ_6"],
        "error": jrow["kappaZ_6_err"], "z": jrow["kappaZ_6_sig"],
        "independent_noise": "same fit as kappa_P (corr carried explicitly)",
        "primary": True})

    # 4. Cepheid-TRGB differential
    r41 = json.load(open(OUT_DIR / "step_41_external_distance_breakers.json"))
    dk = r41["differential_kappa"]
    ch.append({
        "channel": "cepheid_trgb_differential",
        "source": "step_41", "estimate": dk["kappa_Cep"],
        "error": dk["kappa_Cep_err"], "z": dk["kappa_Cep_sig"],
        "independent_noise": "TRGB photometry; shared anchors/hosts",
        "primary": True})

    # 5. HF-SN contrast (consistency, one-sided)
    r65 = json.load(open(OUT_DIR / "step_65_hfsn_environment_contrast.json"))
    z65 = -r65["delta_X"] / r65["delta_X_err_boot"]  # sign: contrast <0 predicted
    ch.append({
        "channel": "hfsn_environment_contrast",
        "source": "step_65", "estimate": r65["delta_X"],
        "error": r65["delta_X_err_boot"], "z": float(z65),
        "independent_noise": "HF host mass distribution + map bootstrap",
        "primary": False,
        "note": "derived propagation under common-mode response; "
                "one-sided consistency check, not independent evidence"})
    return ch


def stouffer(zs, corr_pairs=None, shared_rho=0.0):
    """Stouffer Z with optional pairwise correlations.

    corr_pairs: dict {(i,j): rho}. shared_rho is applied to all pairs not
    otherwise specified.
    """
    n = len(zs)
    w = np.ones(n)
    var = float(np.sum(w ** 2))
    for i in range(n):
        for j in range(i + 1, n):
            rho = shared_rho
            if corr_pairs and (i, j) in corr_pairs:
                rho = corr_pairs[(i, j)]
            var += 2.0 * rho * w[i] * w[j]
    return float(np.sum(zs * w) / np.sqrt(max(var, 1e-12)))


def run():
    print_status("Step 66: Combined cross-channel evidence", "SECTION")
    ch = load_channels()
    for c in ch:
        print_status(f"{c['channel']:34s} {c['source']:8s} "
                     f"est={c['estimate']:+.3e} +/- {c['error']:.3e} "
                     f"z={c['z']:+.2f}", "INFO")

    # signed z: all channel estimates are tabulated in their TEP-predicted
    # direction as reported by the source steps (Gamma_X > 0, kappa_P < 0,
    # kappa_Z < 0, differential > 0, contrast < 0).  Convert to signed z in
    # the predicted direction for the directional test, and |z| for the
    # two-sided combination.
    pred_sign = {"velocity_endpoint_GammaX": 1,
                 "within_host_period_kappaP": -1,
                 "within_host_metallicity_kappaZ": -1,
                 "cepheid_trgb_differential": 1,
                 "hfsn_environment_contrast": -1}
    zs_dir = np.array([c["z"] * np.sign(c["estimate"]) * pred_sign[c["channel"]]
                       if c["channel"] != "hfsn_environment_contrast"
                       else c["z"] * np.sign(-c["estimate"]) * 1
                       for c in ch])

    prim = [i for i, c in enumerate(ch) if c["primary"]]
    name_idx = {c["channel"]: i for i, c in enumerate(ch)}

    # fitted correlation between kappa_P and kappa_Z
    corr_pairs = {}
    kp, kz = name_idx["within_host_period_kappaP"], name_idx["within_host_metallicity_kappaZ"]
    cp = ch[kp].get("fit_corr_with", {})
    corr_pairs[(kp, kz)] = float(cp.get("within_host_metallicity_kappaZ", 0.0))

    sensitivity = {}
    for tag, idxs, rho in [
        ("primary_rho0", prim, 0.0),
        ("primary_rho_shared_0.1", prim, 0.1),
        ("primary_rho_shared_0.3", prim, 0.3),
        ("velocity_plus_period_only",
         [name_idx["velocity_endpoint_GammaX"],
          name_idx["within_host_period_kappaP"]], 0.0),
        ("drop_metallicity",
         [i for i in prim if i != kz], 0.0),
        ("with_hfsn_consistency", prim + [name_idx["hfsn_environment_contrast"]], 0.0),
    ]:
        Z = stouffer(zs_dir[idxs], corr_pairs=corr_pairs, shared_rho=rho)
        p2 = 2.0 * (1.0 - stats.norm.cdf(abs(Z)))
        p1 = 1.0 - stats.norm.cdf(Z)
        sensitivity[tag] = {
            "channels": [ch[i]["channel"] for i in idxs],
            "shared_rho": rho,
            "stouffer_Z": Z,
            "p_two_sided": float(p2), "p_one_sided": float(p1)}
        print_status(f"{tag:28s} Z={Z:+.2f}  p1={p1:.4f}  p2={p2:.4f}",
                     "SUCCESS")

    # Sign concordance: fraction of channels in the predicted direction
    n_pred = int(np.sum(zs_dir[prim] > 0))
    result = {
        "channels": ch,
        "n_primary_in_predicted_direction": n_pred,
        "sign_concordance_p": float(stats.binomtest(
            n_pred, len(prim), 0.5).pvalue),
        "fitted_correlations": {f"{ch[i]['channel']}--{ch[j]['channel']}": v
                                for (i, j), v in corr_pairs.items()},
        "sensitivity": sensitivity,
        "headline": {
            "combination": "primary_rho0",
            "Z": sensitivity["primary_rho0"]["stouffer_Z"],
            "note": "Stouffer combination of the four independent-noise "
                    "primary channels; two-sided z per channel in the "
                    "predicted direction; fitted kappa_P-kappa_Z "
                    "correlation (-0.15) carried explicitly. The HF-SN "
                    "contrast is a derived consistency channel and is "
                    "excluded from the headline."},
    }
    with open(OUT_DIR / "step_66_combined_evidence.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print_status(f"Saved {OUT_DIR / 'step_66_combined_evidence.json'}", "SUCCESS")
    return result


if __name__ == "__main__":
    run()
