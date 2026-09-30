#!/usr/bin/env python3
"""
step_65_hfsn_environment_contrast.py

Hubble-flow SN environment contrast: the common-mode propagation channel.

Under a universal clock response, a host-level environmental shift applies to
both distance indicators in a calibrator galaxy (Cepheids and the hosted SN).
The shift is then absorbed into the latent host modulus mu_i and is invisible
to the raw ladder matrix; it re-emerges only through the environment
difference between the calibrator hosts and the Hubble-flow SN hosts:

    mu_j^meas = mu_j^true - kappa * (X_j - <X_cal>)
    Delta H0  = -H0 * (ln10/5) * kappa * (<X_HF> - <X_cal>)

This step builds the missing ingredient: X for the Hubble-flow hosts. Direct
kinematic proxies are not available for the 277 Pantheon+/SH0ES Hubble-flow
hosts, so X is transferred from the calibrator hosts through their stellar
masses (HOST_LOGMASS, present for all rows of Pantheon+SH0ES.dat):

    X = a + b * log10(M_host)

fitted on the 37 calibrator hosts, and applied per HF SN with bootstrap
propagation of (a) the map coefficients, (b) the map residual scatter, and
(c) the host-mass uncertainties. The product kappa * Delta_X is reported for
the canonical benchmark and for the endpoint likelihood's kappa_equiv, as the
bounded H0 bias the common-mode interpretation permits.

Inputs:
  data/raw/Pantheon+SH0ES.dat
  data/processed/hosts_processed.csv
  results/outputs/step_04_tep_correction_results.json   (sigma_ref)

Outputs:
  results/outputs/step_65_hfsn_environment_contrast.json
  results/outputs/step_65_hf_sn_X_table.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C2 = 299792.458 ** 2
LN10_OVER_5 = np.log(10.0) / 5.0
N_BOOT = 2000
SEED = 65010


def print_status(msg, level="INFO"):
    prefix = {"SECTION": "=" * 60, "INFO": "[INFO]", "SUCCESS": "[SUCCESS]",
              "WARNING": "[WARNING]"}.get(level, "[INFO]")
    if level == "SECTION":
        print(f"\n{prefix}\n{msg}\n{prefix}")
    else:
        print(f"{prefix} {msg}")


def run():
    print_status("Step 65: Hubble-flow SN environment contrast", "SECTION")

    from scripts.utils.tep_correction import build_host_screening_map
    pan = pd.read_csv(DATA_DIR / "raw" / "Pantheon+SH0ES.dat", sep=r"\s+")
    hosts = pd.read_csv(DATA_DIR / "processed" / "hosts_processed.csv")
    smap = build_host_screening_map(hosts, BASE_DIR)
    with open(OUT_DIR / "step_04_tep_correction_results.json") as f:
        sigma_ref = float(json.load(f)["sigma_ref"])

    # --- Calibrator X and mass -------------------------------------------------
    h = hosts.dropna(subset=["host_logmass", "sigma_inferred"]).copy()
    h = h[h["sigma_inferred"] > 0]
    h["S"] = [smap.get(n, 1.0) for n in h["normalized_name"]]
    X_cal = (h["S"] * h["sigma_inferred"] ** 2 - sigma_ref ** 2) / C2
    lm_cal = h["host_logmass"].values

    hf = pan[(pan["USED_IN_SH0ES_HF"] == 1) & (pan["IS_CALIBRATOR"] == 0)].copy()
    hf = hf.dropna(subset=["HOST_LOGMASS"])
    lm_hf = hf["HOST_LOGMASS"].values
    lm_hf_err = np.abs(hf["HOST_LOGMASS_ERR"].fillna(0.1).values)

    # --- Point estimate --------------------------------------------------------
    b, a = np.polyfit(lm_cal, X_cal.values, 1)
    resid = X_cal.values - (a + b * lm_cal)
    r_map = float(np.corrcoef(lm_cal, X_cal.values)[0, 1])
    X_hf_pt = a + b * lm_hf
    dX_pt = float(X_hf_pt.mean() - X_cal.mean())
    print_status(
        f"Map X = a + b*logM: a={a:.3e}, b={b:.3e}, r={r_map:.3f}, "
        f"resid rms={resid.std():.3e}", "INFO")
    print_status(
        f"<X_cal>={X_cal.mean():.3e}  <X_HF>={X_hf_pt.mean():.3e}  "
        f"contrast={dX_pt:.3e}", "INFO")

    # --- Bootstrap propagation --------------------------------------------------
    rng = np.random.default_rng(SEED)
    n_cal, n_hf = len(lm_cal), len(lm_hf)
    dX_boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n_cal, n_cal)
        try:
            bb, aa = np.polyfit(lm_cal[idx], X_cal.values[idx], 1)
        except np.linalg.LinAlgError:
            dX_boot[i] = np.nan
            continue
        jdx = rng.integers(0, n_hf, n_hf)
        lm_draw = lm_hf[jdx] + rng.normal(0, lm_hf_err[jdx])
        X_hf = aa + bb * lm_draw + rng.normal(0, resid.std(), n_hf)
        X_cal_draw = X_cal.values[idx].mean()
        dX_boot[i] = X_hf.mean() - X_cal_draw
    dX_boot = dX_boot[np.isfinite(dX_boot)]
    dX_err = float(np.std(dX_boot))
    dX_lo, dX_hi = np.percentile(dX_boot, [16, 84])

    print_status(f"Delta_X = {dX_pt:.3e} + {dX_hi - dX_pt:.3e}/- {dX_pt - dX_lo:.3e} "
                 f"(bootstrap sd {dX_err:.3e})", "SUCCESS")

    # --- Alternative map forms: is the contrast robust to the map shape? ----
    # The linear map is weak (r~0.23), so the reported contrast must not hinge
    # on the linear functional form.  Three non-equivalent alternatives are
    # evaluated: a robust (Theil-Sen) line, a quadratic fit, and a
    # distribution-free quantile (rank) map that transfers each HF host's mass
    # quantile to the corresponding quantile of the calibrator X distribution.
    from scipy.stats import theilslopes
    alt_maps = {}

    ts_b, ts_a, _, _ = theilslopes(X_cal.values, lm_cal)
    alt_maps["theil_sen"] = {
        "delta_X": float((ts_a + ts_b * lm_hf).mean() - X_cal.mean()),
        "slope": float(ts_b), "intercept": float(ts_a)}

    q2 = np.polyfit(lm_cal, X_cal.values, 2)
    alt_maps["quadratic"] = {
        "delta_X": float(np.polyval(q2, lm_hf).mean() - X_cal.mean()),
        "coef": [float(c) for c in q2]}

    lm_sorted = np.sort(lm_cal)
    X_sorted = np.sort(X_cal.values)
    ranks = np.clip(np.searchsorted(lm_sorted, lm_hf) / n_cal, 0.0, 1.0)
    X_rank = np.quantile(X_sorted, ranks)
    alt_maps["quantile_rank"] = {
        "delta_X": float(X_rank.mean() - X_cal.mean())}

    for name, am in alt_maps.items():
        print_status(f"alt map {name}: Delta_X = {am['delta_X']:+.3e}", "INFO")

    # --- H0 bias under the canonical benchmark ----------------------------------
    from core.constants import KAPPA_GAL
    H0 = 70.0
    rows = []
    for label, kap in [("kappa_canonical", KAPPA_GAL),
                       ("kappa_endpoint_0.39e6", 3.86e5),
                       ("kappa_wls_0.23e6", 2.34e5)]:
        dH0 = -H0 * LN10_OVER_5 * kap * dX_pt
        dH0_lo = -H0 * LN10_OVER_5 * kap * dX_lo
        dH0_hi = -H0 * LN10_OVER_5 * kap * dX_hi
        rows.append({"kappa_case": label, "kappa": float(kap),
                     "delta_H0_kms": float(dH0),
                     "delta_H0_lo": float(dH0_lo), "delta_H0_hi": float(dH0_hi)})
        print_status(f"{label}: Delta_H0 = {dH0:+.2f} km/s/Mpc "
                     f"[{dH0_lo:+.2f}, {dH0_hi:+.2f}]", "INFO")

    # Canonical-amplitude H0 shift under each alternative map
    for name, am in alt_maps.items():
        am["delta_H0_canonical"] = float(
            -H0 * LN10_OVER_5 * KAPPA_GAL * am["delta_X"])

    # --- Per-SN X table -----------------------------------------------------------
    out_hf = hf[["CID", "zHD", "HOST_LOGMASS", "HOST_LOGMASS_ERR"]].copy()
    out_hf["X_pred"] = X_hf_pt
    out_hf["X_pred_err"] = np.sqrt(resid.var() + (b * lm_hf_err) ** 2)
    out_hf.to_csv(OUT_DIR / "step_65_hf_sn_X_table.csv", index=False)

    result = {
        "map": {"a": float(a), "b": float(b), "r": r_map,
                "resid_rms": float(resid.std()), "n_calibrators": int(n_cal)},
        "mean_X_cal": float(X_cal.mean()),
        "mean_X_hf": float(X_hf_pt.mean()),
        "delta_X": dX_pt,
        "delta_X_err_boot": dX_err,
        "delta_X_p16": float(dX_lo),
        "delta_X_p84": float(dX_hi),
        "n_hf_sn": int(n_hf),
        "n_hf_below_cal_mass_range": int((lm_hf < lm_cal.min()).sum()),
        "n_hf_above_cal_mass_range": int((lm_hf > lm_cal.max()).sum()),
        "cal_logmass_range": [float(lm_cal.min()), float(lm_cal.max())],
        "hf_logmass_range": [float(lm_hf.min()), float(lm_hf.max())],
        "delta_H0_cases": rows,
        "alt_maps": alt_maps,
        "interpretation": (
            "Under a common-mode clock response the calibrator-internal shift "
            "is absorbed into the latent host moduli and the ladder-level "
            "kappa_Cep is unidentified; the H0 bias propagates only through "
            "the environment contrast between calibrator and Hubble-flow "
            "hosts. The mass map is weak (r~0.23); the reported contrast "
            "uncertainty includes map, residual-scatter, and mass-error terms."
        ),
    }
    with open(OUT_DIR / "step_65_hfsn_environment_contrast.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    print_status("Step 65 complete", "SUCCESS")
    return result


if __name__ == "__main__":
    run()
