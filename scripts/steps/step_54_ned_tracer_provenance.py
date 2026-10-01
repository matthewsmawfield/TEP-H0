"""
Step 54: NED Tracer Provenance Compilation for Host Galaxies

Resolves issue 11-4 by querying the NASA/IPAC Extragalactic Database (NED)
for the redshift provenance of the 41 host/anchor galaxies. For each host it
records the preferred redshift, the refcode of NED's preferred measurement
(the first row of the redshifts table, which equals the catalog preferred
value in every case checked), and the verbatim NED measurement metadata
(Frequency Targeted, Spectral Range, Measurement Mode Technique, Spatial
Mode, Spectrograph, Features, Comments, Qualifiers) aggregated across all
redshift-table rows bearing that preferred refcode. Where the preferred
refcode's own rows carry no method fields, the modal method among all rows
consistent with the preferred redshift (|dz| < 2e-5) is recorded as the
classification basis instead; if neither exists the host is marked
'unclassified' rather than guessed.

The tracer class is then:
  HI               : 21-cm / radio / heterodyne evidence (disk-integrated)
  optical-nuclear  : optical with explicit nucleus/fiber spatial mode
  optical          : optical spectral range without spatial qualifier
  unclassified     : no method evidence in NED records

The mechanism under test requires the spectroscopic aperture to sit inside
the Cepheid field (r_spec < r_Cep). Disk-integrated HI and nuclear-fiber
optical redshifts weight the host potential differently, so the measured
environmental slope Gamma_X is refit per class with the same
(H_app, Gamma_X, sigma_int) likelihood used in Step 39 at the canonical
sigma_v = 250 km/s convention, on the primary sigma_* coordinate.

Inputs (all real, no simulated labels):
  data/processed/ned_redshift_provenance_raw.csv  -- per-host preferred z
                                                   and refcode (NED query)
  data/processed/ned_tracer_fields.json           -- verbatim NED field
                                                   aggregates per host
  data/processed/hosts_processed.csv              -- host sigma_*, z, mu

Outputs:
  results/outputs/step_54_ned_tracer_provenance.json
  results/outputs/step_54_tracer_class_table.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_JSON = ROOT / "results" / "outputs" / "step_54_ned_tracer_provenance.json"
OUT_CSV = ROOT / "results" / "outputs" / "step_54_tracer_class_table.csv"
PROV_CSV = ROOT / "data" / "processed" / "ned_redshift_provenance_raw.csv"
FIELDS_JSON = ROOT / "data" / "processed" / "ned_tracer_fields.json"
HOSTS_CSV = ROOT / "data" / "processed" / "hosts_processed.csv"

C_KM_S = 299792.458
LN10_OVER_5 = np.log(10.0) / 5.0
SIGMA_REF = 87.16507328052906  # standard weighted anchor reference (km/s)
SIGMA_V = 250.0  # canonical peculiar-velocity convention (km/s)

# hosts_processed.csv maps SH0ES source_id -> catalog display name directly.

ANCHORS = {"N4258", "M31", "MW", "SMC", "LMC"}


def _text(v):
    s = str(v).strip()
    return "" if s in ("", "nan", "None", "--") else s


def _classify_blob(blob):
    """Return an aperture class for a field blob, or None if uninformative."""
    radio = any(
        t in blob
        for t in ("21-cm", "hi line", "radio", "heterodyne", "21 cm")
    )
    optical = "optical" in blob
    nuclear = any(
        t in blob for t in ("nucleus", "nuclear", "fiber", "central", "core")
    )
    systemic = "systemic" in blob

    if radio:
        return "HI"
    if optical and nuclear:
        return "optical-nuclear"
    if optical and systemic:
        return "optical-integrated"
    if optical:
        return "optical"
    if nuclear:
        return "optical-nuclear"
    if systemic:
        return "optical-integrated"
    return None


def classify_host(pref_fields, zmatch_modal):
    """Aperture class from verbatim NED fields. Never guesses."""
    blob = " | ".join(
        _text(v)
        for f, vals in pref_fields.items()
        for v in (vals if isinstance(vals, list) else [vals])
    ).lower()
    cls = _classify_blob(blob)
    if cls is not None:
        return cls, "ned_refcode_fields"

    # Secondary: modal fields over rows whose redshift equals the preferred z.
    modal_blob = " | ".join(
        _text(v[0]) for v in zmatch_modal.values() if v
    ).lower()
    cls = _classify_blob(modal_blob)
    if cls is not None:
        return cls, "ned_zmatch_modal"
    return "unclassified", "none"


def fit_gamma_X(cz, d, X, mu_err, sigma_v=SIGMA_V):
    """Mirror of step_39 fit_multivariate_gamma for the single Gamma_X
    regressor: cz = d*(H_app + Gamma_X*X) with fitted intrinsic scatter."""
    X = np.asarray(X, dtype=float)
    scale = 1e7

    def neg_logL(p):
        H_app, g, sig_int = p[0], p[1], max(p[2], 0.01)
        cz_mod = d * (H_app + g * scale * X)
        resid = cz - cz_mod
        sig_cz = LN10_OVER_5 * np.abs(cz_mod) * mu_err
        var = np.maximum(sigma_v**2 + sig_cz**2 + sig_int**2, 0.01)
        return 0.5 * np.sum(resid**2 / var + np.log(var))

    A = np.column_stack([np.ones(len(cz)), scale * X])
    c0 = np.linalg.lstsq(A, cz / d, rcond=None)[0]
    x0 = np.array([c0[0], c0[1], 5.0])
    bounds = [(30.0, 90.0), (-100.0, 100.0), (0.01, 250.0)]
    res = optimize.minimize(neg_logL, x0, method="L-BFGS-B", bounds=bounds)
    for si in (50.0, 120.0, 200.0):
        xt = res.x.copy()
        xt[-1] = si
        rt = optimize.minimize(neg_logL, xt, method="L-BFGS-B", bounds=bounds)
        if rt.fun < res.fun - 1e-9:
            res = rt
    H_app, g = float(res.x[0]), float(res.x[1])
    # curvature errors on (H_app, Gamma_X) at fixed sigma_int
    try:
        hess = optimize.approx_fprime(
            res.x, lambda x: optimize.approx_fprime(x, neg_logL, 1e-5), 1e-5
        )
        hess = 0.5 * (hess + hess.T)
        active = res.x[-1] > bounds[2][0] + 1e-3 and res.x[-1] < bounds[2][1] - 1e-3
        sub = hess[:2, :2] if not active else hess
        cov = np.linalg.pinv(sub, rcond=1e-12)
        g_err = float(np.sqrt(max(cov[1, 1], 0.0))) * scale
        h_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    except Exception:
        g_err = h_err = float("nan")
    Gamma = g * scale
    return {
        "H_app": H_app,
        "H_app_err": h_err,
        "Gamma_X": Gamma,
        "Gamma_X_err": g_err,
        "Gamma_X_sig": (Gamma / g_err if g_err and np.isfinite(g_err) and g_err > 0 else float("nan")),
        "n_hosts": int(len(cz)),
    }


def main():
    prov = pd.read_csv(PROV_CSV)
    fields = json.loads(FIELDS_JSON.read_text())
    hosts = pd.read_csv(HOSTS_CSV)

    # Map SH0ES host id -> catalog display name -> NED row
    prov_by_galaxy = {r["galaxy"]: r for _, r in prov.iterrows()}
    sh0es_to_display = {}
    for _, r in hosts.iterrows():
        sh0es_to_display[str(r["source_id"])] = str(r["normalized_name"])
        sh0es_to_display.setdefault(str(r["normalized_name"]), str(r["normalized_name"]))

    # Load R22 design-matrix fits to get mu / mu_err per host exactly as
    # step_39 does.
    sys.path.insert(0, str(ROOT))
    from scripts.steps.step_39_environment_slope_decomposition import (
        load_sh0es_data,
        load_host_metadata,
        compute_host_covariates,
    )
    L, y, C, q = load_sh0es_data()
    host_sigma, host_z, host_z_cmb, host_S, host_mass = load_host_metadata()
    cov = compute_host_covariates(L, y, C, q, host_sigma, host_z, SIGMA_REF, host_z_cmb, host_mass)

    # Attach NED provenance per host row
    rows = []
    for _, r in cov.iterrows():
        sh0es = r["host"]
        display = sh0es_to_display.get(sh0es, sh0es)
        prov_row = prov_by_galaxy.get(display)

        fld = fields.get(display) or fields.get(sh0es) or {}
        cls, basis = classify_host(
            fld.get("refcode_fields", {}), fld.get("zmatch_modal", {})
        )
        row = {
            "host": sh0es,
            "ned_query_name": display,
            "ned_object_name": _text(prov_row["ned_name"]) if prov_row is not None else "",
            "preferred_z": float(prov_row["preferred_z"]) if prov_row is not None and prov_row["preferred_z"] == prov_row["preferred_z"] else np.nan,
            "z_refcode": _text(prov_row["z_refcode"]) if prov_row is not None else "",
            "tracer_class": cls,
            "class_basis": basis,
            "n_pref_refcode_rows": fld.get("n_pref_refcode_rows", 0),
            "sigma_kms": float(r["sigma"]),
            "z_hd": float(r["z_hd"]) if pd.notna(r["z_hd"]) else np.nan,
            "mu": float(r["mu"]),
            "mu_err": float(r["mu_err"]) if pd.notna(r["mu_err"]) else 0.05,
            "is_anchor": bool(r["is_anchor"]),
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)

    # Per-class Gamma_X split on the primary (non-anchor, z_hd>0) sample
    primary = table[(~table["is_anchor"]) & table["z_hd"].notna() & (table["z_hd"] > 0)]
    S_map = dict(host_S)  # all SH0ES/source_id/display aliases already keyed

    def host_X(row):
        s2 = row["sigma_kms"] ** 2
        S = S_map.get(row["host"], 1.0)
        return (S * s2 - SIGMA_REF**2) / C_KM_S**2

    primary = primary.copy()
    primary["X"] = primary.apply(host_X, axis=1)
    primary["cz"] = C_KM_S * primary["z_hd"]
    primary["d"] = 10.0 ** ((primary["mu"] - 25.0) / 5.0)

    splits = {}
    for cls, grp in primary.groupby("tracer_class"):
        if len(grp) >= 4:
            splits[cls] = fit_gamma_X(
                grp["cz"].values, grp["d"].values, grp["X"].values,
                grp["mu_err"].values,
            )
        else:
            splits[cls] = {"n_hosts": int(len(grp)), "status": "insufficient"}

    all_fit = fit_gamma_X(primary["cz"], primary["d"], primary["X"], primary["mu_err"])

    out = {
        "step": "step_54_ned_tracer_provenance",
        "status": "EXECUTED",
        "method": (
            "Preferred redshift and refcode from NED query_object + redshifts "
            "table row 0 per host; aperture class from verbatim NED "
            "measurement fields aggregated over rows bearing the preferred "
            "refcode, else modal method among z-matching rows; 'unclassified' "
            "if no method evidence exists. No simulated labels."
        ),
        "class_counts": table["tracer_class"].value_counts().to_dict(),
        "per_host_table": str(OUT_CSV.relative_to(ROOT)),
        "gamma_X_by_class": splits,
        "gamma_X_all": all_fit,
        "sigma_v": SIGMA_V,
        "interpretation": (
            "The observing-chain mechanism requires the spectroscopic aperture "
            "to sit within the Cepheid field (r_spec < r_Cep). HI 21-cm "
            "systemic redshifts are disk-integrated and weight the outer "
            "potential differently from nuclear/inner optical spectra; the "
            "per-class Gamma_X fits test whether the endpoint response "
            "survives conditioning on real tracer geometry."
        ),
        "related_issue": "11-4",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("step_54: EXECUTED --", out["class_counts"])
    print("  Gamma_X by class:", {k: v.get("Gamma_X_sig") for k, v in splits.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
