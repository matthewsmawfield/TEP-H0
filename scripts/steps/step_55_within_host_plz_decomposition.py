"""
Step 55: Within-Host PLZ Decomposition (Metallicity Channel Separation)

Addresses the remaining degeneracy flagged under issue 11-8 item (b):
whether the environment-coupled metallicity term
    kappa_Z = -X_i * [O/H]
separates from a conventional period-metallicity interaction
    (log10 P - 1) * [O/H].

The step_53 battery fitted the raw product t * z as the conventional
channel.  That column is a mixture.  Writing t = t_bar_h + t~ and
z = z_bar_h + z~ within each host h (bars denote host means, tildes the
within-host deviations),

    t * z  =  t_bar z_bar                      (host-constant, mu_i absorbed)
           +  t_bar z~ + z_bar t~              (host-mean-weighted pieces)
           +  t~ z~                            (pure within-host product)

Only the last piece is free of host-level ordering by construction: a
genuine per-Cepheid PLZ interaction must appear in t~ z~, whereas any
signal in the cross pieces is driven by the ordering of the host means
t_bar_h, z_bar_h on the environmental coordinate.

This step therefore refits the SH0ES GLS system (identical data path,
screening map and sample conventions as Steps 34/52/53) with the
decomposed columns and asks:

  Part A. Does the environment-coupled period coefficient kappa_P
          survive the full PLZ decomposition, and which decomposition
          piece carries the conventional signal?
  Part B. Do the per-host metallicity slopes order on the potential
          coordinate X_i -- the parallel of the per-host period-slope
          test (step_53 Part D) for the ZW channel?
  Part C. Is the mechanical entanglement itself environment-ordered?
          The per-host covariance cov(t~, z~) is the column overlap
          driver between X_i t~ and t~ z~; if it is flat in X_i the
          aliasing is unspecific.

Outputs:
  step_55_plz_decomposition_battery.csv   -- GLS battery results
  step_55_per_host_metallicity_slopes.csv -- per-host Z-slope table
  step_55_within_host_plz_decomposition.json -- full record
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from scripts.utils.logger import print_status
from scripts.steps.step_53_within_host_structure_battery import (
    WhitenedGLS, load_and_assign, host_x_map, X_SCALE, ANCHOR_HOSTS)

OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def demean_within_host(col, cep_rows, host_of_row):
    """Within-host deviation of a column on Cepheid rows (0 elsewhere)."""
    d = np.zeros(len(col))
    hosts = {h for h in host_of_row[cep_rows] if h}
    for h in hosts:
        m = cep_rows[host_of_row[cep_rows] == h]
        d[m] = col[m] - col[m].mean()
    return d


def host_mean_row(col, cep_rows, host_of_row, hosts):
    """Host-mean value of a column mapped onto each host's Cepheid rows."""
    out = np.zeros(len(col))
    for h in hosts:
        m = cep_rows[host_of_row[cep_rows] == h]
        out[m] = col[m].mean()
    return out


def fit_with(wg, L, cols):
    """Append named columns to the design matrix; return per-term fits."""
    names = [n for n, _ in cols]
    A = np.column_stack([L] + [c for _, c in cols]) if cols else L
    theta, cov, chi2, _ = wg.fit(A)
    off = L.shape[1]
    res = {"chi2": chi2, "n_params": int(A.shape[1])}
    for j, n in enumerate(names):
        res[n] = {"coef": float(theta[off + j]),
                  "err": float(np.sqrt(cov[off + j, off + j])),
                  "sig": float(abs(theta[off + j])
                               / np.sqrt(cov[off + j, off + j]))}
    if len(names) >= 2:
        i0, i1 = off + len(names) - 2, off + len(names) - 1
        res["_last_pair_corr"] = float(
            cov[i0, i1] / np.sqrt(cov[i0, i0] * cov[i1, i1]))
    return res


# ---------------------------------------------------------------------------
# Part A: decomposed PLZ battery
# ---------------------------------------------------------------------------
def part_a(L, wg, cep_rows, t_col, z_col, X_row_scaled,
           t_dm, z_dm, tbar_row, zbar_row):
    colP = -X_row_scaled * t_col            # kappa_P (X in 1e-6 units)
    colZ = -X_row_scaled * z_col            # kappa_Z
    colPW = t_dm * z_dm                     # pure within-host PLZ
    colPC = tbar_row * z_dm + zbar_row * t_dm   # host-mean-weighted PLZ

    models = [
        ("baseline", []),
        ("kappaP", [("kappaP", colP)]),
        ("plz_within", [("c_plz_w", colPW)]),
        ("plz_cross", [("c_plz_c", colPC)]),
        ("kappaP+plz_within", [("kappaP", colP), ("c_plz_w", colPW)]),
        ("kappaP+kappaZ", [("kappaP", colP), ("kappaZ", colZ)]),
        ("kappaP+kappaZ+plz_decomp",
         [("kappaP", colP), ("kappaZ", colZ),
          ("c_plz_w", colPW), ("c_plz_c", colPC)]),
        ("kappaZ+plz_decomp",
         [("kappaZ", colZ), ("c_plz_w", colPW), ("c_plz_c", colPC)]),
        ("plz_decomp_only",
         [("c_plz_w", colPW), ("c_plz_c", colPC)]),
    ]
    _, _, chi2_base, _ = wg.fit(L)
    rows = []
    for name, cols in models:
        r = fit_with(wg, L, cols)
        row = {"model": name, "delta_chi2": chi2_base - r["chi2"],
               "chi2": r["chi2"], "n_params": r["n_params"]}
        for k, v in r.items():
            if k.startswith("_") or k in ("chi2", "n_params"):
                if k == "_last_pair_corr":
                    row["last_pair_corr"] = v
                continue
            row[f"{k}"] = v["coef"]
            row[f"{k}_err"] = v["err"]
            row[f"{k}_sig"] = v["sig"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B: per-host metallicity-slope decomposition
# ---------------------------------------------------------------------------
def part_b(L, y, wg, q, cep_rows, host_of_row, X_host, t_col, z_col,
           theta_base):
    """Replace ZW with per-host metallicity slopes; regress on X_i."""
    z_idx = np.where(q == "ZW")[0][0]
    hosts = sorted(X_host.keys())
    host_rows = {h: cep_rows[host_of_row[cep_rows] == h] for h in hosts}
    slope_hosts = [h for h in hosts
                   if len(host_rows[h]) >= 3
                   and np.ptp(z_col[host_rows[h]]) > 0.02]
    small_hosts = [h for h in hosts if h not in slope_hosts]

    small_rows = cep_rows[np.isin(host_of_row[cep_rows], small_hosts)]
    L0 = np.delete(L, z_idx, axis=1)
    cols = [np.where(np.isin(np.arange(len(y)), small_rows),
                     z_col, 0.0)]
    for h in slope_hosts:
        c = np.zeros(len(y))
        c[host_rows[h]] = z_col[host_rows[h]]
        cols.append(c)
    A = np.column_stack([L0] + cols)
    theta, cov, chi2, _ = wg.fit(A)

    off = L0.shape[1]
    rec = []
    for j, h in enumerate(slope_hosts):
        jdx = off + 1 + j
        rec.append({"host": h, "n_cep": int(len(host_rows[h])),
                    "z_slope": float(theta[jdx]),
                    "z_slope_err": float(np.sqrt(cov[jdx, jdx])),
                    "X": float(X_host[h]),
                    "is_anchor": h in ANCHOR_HOSTS})
    df = pd.DataFrame(rec)
    return df, slope_hosts


def wls_on_x(df, val_col, err_col):
    """Inverse-variance weighted regression of a per-host column on X."""
    from scipy import stats as st
    xv = df["X"].to_numpy(dtype=float) * X_SCALE
    yv = df[val_col].to_numpy(dtype=float)
    wv = 1.0 / np.maximum(df[err_col].to_numpy(dtype=float), 1e-12) ** 2
    ok = np.isfinite(xv) & np.isfinite(yv) & (wv > 0)
    xv, yv, wv = xv[ok], yv[ok], wv[ok]
    if len(xv) < 3:
        return {"n": int(len(xv))}
    W = np.diag(wv)
    Xd = np.column_stack([np.ones(len(xv)), xv])
    covb = np.linalg.inv(Xd.T @ W @ Xd)
    beta = covb @ (Xd.T @ W @ yv)
    resid = yv - Xd @ beta
    chi2 = float(resid.T @ W @ resid)
    dof = max(len(xv) - 2, 1)
    r = np.corrcoef(xv, yv)[0, 1]
    tstat = r * np.sqrt(max(len(xv) - 2, 1) / max(1e-12, 1 - r ** 2))
    p = 2 * st.t.sf(abs(tstat), len(xv) - 2)
    return {"n": int(len(xv)), "slope": float(beta[1]),
            "slope_err": float(np.sqrt(covb[1, 1])),
            "sig": float(abs(beta[1]) / np.sqrt(covb[1, 1])),
            "r": float(r), "p_value": float(p),
            "chi2_per_dof": float(chi2 / dof)}


# ---------------------------------------------------------------------------
# Part C: mechanical entanglement diagnostics
# ---------------------------------------------------------------------------
def part_c(t_col, z_col, cep_rows, host_of_row, X_host, X_row_scaled):
    """Within-host cov(t~,z~) ordering on X and design-column overlap."""
    hosts = sorted(X_host.keys())
    rec = []
    for h in hosts:
        m = cep_rows[host_of_row[cep_rows] == h]
        if len(m) < 3:
            continue
        tc = t_col[m] - t_col[m].mean()
        zc = z_col[m] - z_col[m].mean()
        rec.append({"host": h, "n_cep": int(len(m)),
                    "X": float(X_host[h]),
                    "cov_tz": float(np.cov(tc, zc)[0, 1]),
                    "corr_tz": float(np.corrcoef(tc, zc)[0, 1]),
                    "std_t": float(tc.std()), "std_z": float(zc.std())})
    df = pd.DataFrame(rec)
    df = df[np.isfinite(df["corr_tz"])].reset_index(drop=True)

    def plain_wls(xv, yv):
        xv = np.asarray(xv, float) * X_SCALE
        yv = np.asarray(yv, float)
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[ok], yv[ok]
        Xd = np.column_stack([np.ones(len(xv)), xv])
        beta, res, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        n = len(xv)
        dof = max(n - 2, 1)
        s2 = float((yv - Xd @ beta) @ (yv - Xd @ beta)) / dof
        covb = s2 * np.linalg.inv(Xd.T @ Xd)
        r = np.corrcoef(xv, yv)[0, 1]
        return {"n": n, "slope": float(beta[1]),
                "slope_err": float(np.sqrt(covb[1, 1])),
                "r": float(r)}

    # host-mean covariate ordering on X
    tbar = np.array([t_col[cep_rows[host_of_row[cep_rows] == h]].mean()
                     for h in df["host"]])
    zbar = np.array([z_col[cep_rows[host_of_row[cep_rows] == h]].mean()
                     for h in df["host"]])
    Xv = df["X"].to_numpy()

    # design-column overlap: corr(X_i t~, t~ z~) on whitened cepheid rows
    t_dm = demean_within_host(t_col, cep_rows, host_of_row)
    z_dm = demean_within_host(z_col, cep_rows, host_of_row)
    col_xt = (-X_row_scaled * t_col)[cep_rows]
    col_pw = (t_dm * z_dm)[cep_rows]
    overlap = float(np.corrcoef(col_xt, col_pw)[0, 1])
    col_xz = (-X_row_scaled * z_col)[cep_rows]
    overlap_xz = float(np.corrcoef(col_xz, col_pw)[0, 1])
    overlap_pp = float(np.corrcoef(col_xt, col_xz)[0, 1])

    return {"cov_tz_vs_X": plain_wls(Xv, df["cov_tz"]),
            "corr_tz_vs_X": plain_wls(Xv, df["corr_tz"]),
            "tbar_vs_X": plain_wls(Xv, tbar),
            "zbar_vs_X": plain_wls(Xv, zbar),
            "median_within_corr_tz": float(df["corr_tz"].median()),
            "design_overlap_kappaP_plz_within": overlap,
            "design_overlap_kappaZ_plz_within": overlap_xz,
            "design_overlap_kappaP_kappaZ": overlap_pp,
            "per_host": df.to_dict("records")}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run():
    print_status("Step 55: within-host PLZ decomposition", "SECTION")

    (fl, L, y, C, q, y_source, cep_rows, host_of_row, t_col, z_col,
     host_sigma, host_screening, sigma_ref) = load_and_assign()

    wg = WhitenedGLS(C, y)
    theta_base, cov_base, chi2_base, rank_base = wg.fit(L)

    X_host = host_x_map(host_of_row, cep_rows, host_sigma, host_screening,
                        sigma_ref, "anchor_screened_physical")
    X_row = np.zeros(len(y))
    for h, X in X_host.items():
        X_row[cep_rows[host_of_row[cep_rows] == h]] = X
    X_row_scaled = X_row * X_SCALE

    hosts = sorted(X_host.keys())
    t_dm = demean_within_host(t_col, cep_rows, host_of_row)
    z_dm = demean_within_host(z_col, cep_rows, host_of_row)
    tbar_row = host_mean_row(t_col, cep_rows, host_of_row, hosts)
    zbar_row = host_mean_row(z_col, cep_rows, host_of_row, hosts)

    print_status("Part A: decomposed PLZ battery", "INFO")
    dfA = part_a(L, wg, cep_rows, t_col, z_col, X_row_scaled,
                 t_dm, z_dm, tbar_row, zbar_row)
    dfA.to_csv(OUT_DIR / "step_55_plz_decomposition_battery.csv",
               index=False)
    for _, r in dfA.iterrows():
        line = f"  {r['model']:32s} dChi2={r['delta_chi2']:8.2f}"
        for k in ("kappaP", "kappaZ", "c_plz_w", "c_plz_c"):
            if k in r and pd.notna(r[k]):
                line += f"  {k}={r[k]:+.3e} ({r[k + '_sig']:.2f}s)"
        print_status(line, "INFO")

    print_status("Part B: per-host metallicity-slope ordering", "INFO")
    dfB, slope_hosts = part_b(L, y, wg, q, cep_rows, host_of_row, X_host,
                              t_col, z_col, theta_base)
    dfB.to_csv(OUT_DIR / "step_55_per_host_metallicity_slopes.csv",
               index=False)
    z_ord = wls_on_x(dfB, "z_slope", "z_slope_err")
    anchor_z = dfB[dfB["is_anchor"]]
    non_anchor_z = dfB[~dfB["is_anchor"]]
    z_ord["anchor_slope_mean"] = float(anchor_z["z_slope"].mean()) \
        if len(anchor_z) else None
    z_ord["nonanchor_slope_mean"] = float(non_anchor_z["z_slope"].mean())
    z_ord["n_slope_hosts"] = len(slope_hosts)
    z_ord["n_hosts_total"] = len(hosts)
    print_status(f"  per-host Z-slope on X: slope={z_ord['slope']:+.3e} "
                 f"({z_ord['sig']:.2f}s, n={z_ord['n']})", "INFO")

    print_status("Part C: entanglement diagnostics", "INFO")
    diag = part_c(t_col, z_col, cep_rows, host_of_row, X_host,
                  X_row_scaled)
    print_status(f"  design overlap kappaP vs plz_within: "
                 f"{diag['design_overlap_kappaP_plz_within']:+.3f}", "INFO")
    print_status(f"  cov(t~,z~) vs X slope: "
                 f"{diag['cov_tz_vs_X']['slope']:+.3e} "
                 f"({diag['cov_tz_vs_X']['slope_err']:.2e})", "INFO")

    out = {
        "step": "step_55_within_host_plz_decomposition",
        "purpose": ("separate the conventional per-Cepheid PLZ "
                    "interaction from the environment-coupled "
                    "metallicity term kappa_Z by within-host "
                    "demeaned decomposition"),
        "anchor_convention": "anchor_screened_physical",
        "n_rows": int(len(y)), "n_cepheid_rows": int(len(cep_rows)),
        "n_hosts": len(hosts),
        "battery": dfA.to_dict("records"),
        "per_host_z_slope_ordering": z_ord,
        "entanglement": {k: v for k, v in diag.items()
                         if k != "per_host"},
    }
    def _jsonable(o):
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_jsonable(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    with open(OUT_DIR / "step_55_within_host_plz_decomposition.json",
              "w") as f:
        json.dump(_jsonable(out), f, indent=2, default=str)

    kp_full = dfA[dfA["model"] == "kappaP+kappaZ+plz_decomp"].iloc[0]
    print_status(
        f"Step 55 done: kappaP under full PLZ decomposition = "
        f"{kp_full['kappaP']:+.3e} ({kp_full['kappaP_sig']:.2f}s); "
        f"kappaZ = {kp_full['kappaZ']:+.3e} ({kp_full['kappaZ_sig']:.2f}s)",
        "SUCCESS")


if __name__ == "__main__":
    run()
