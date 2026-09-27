#!/usr/bin/env python3
"""
step_53_within_host_structure_battery.py

Within-Host Period-Structure Discriminating Battery

Motivation
----------
Step 52 measured an environment-coupled within-host period term,
kappa_P (X_i x (logP-1) on Cepheid rows), at 2.6--3.1 sigma jointly with
the metallicity-coupled term kappa_Z.  Two classes of conventional
explanation must be excluded before the term can be read as
environment-ordered structure:

  1. PL-shape systematics.  A global period-luminosity curvature or the
     canonical 10-day slope break, if present but unmodelled, projects
     onto any regressor that correlates with the host period
     distribution.  Controls: free global quadratic (logP-1)^2, free
     10-day break (logP-1)*Theta(logP-1), and the global PLZ
     interaction (logP-1)*[O/H].

  2. Environmental-coordinate confounding.  X_i correlates with host
     mass, distance, and mean metallicity.  The environmental reading
     requires X to be the best single carrier of the period structure
     among the measured host covariates (fitted modulus mu_i -- the
     crowding/distance proxy -- sigma^2 unscreened, S_i, log M_host,
     host mean [O/H], host mean logP), and to survive joint fits with
     the strongest alternative.

  3. Assignment specificity.  Permuting the host->X assignment must
     collapse the signal if the structure is genuinely ordered by the
     measured environment coordinate.

  4. Anchor-segment null.  Under the anchor_reference_zero convention,
     anchor Cepheids (LMC, SMC, M31, NGC 4258 rows) carry X = 0 and
     cannot contribute to kappa_P; a free anchor-segment period slope
     then asks whether the zero-X segment exhibits the same within-host
     structure (global systematic) or is flat (environmental
     ordering).

  5. Per-host slope decomposition.  Replacing the global PL slope bW by
     per-host slopes s_i gives direct within-host observables.  Under
     the environment-coupled mechanism s_i = bW - kappa_P X_i, so the
     weighted regression of s_i on X_i must recover -kappa_P with an
     intercept at X = 0 consistent with the undisturbed PL slope.

  6. Leave-one-host-out stability of kappa_P: the coefficient must not
     rest on a single host.

All fits reuse the Step 34 FullLadderLikelihood machinery (identical
design matrix, covariance, host assignment, and anchor conventions).
No simulated data are introduced; permutation nulls are built from the
measured design alone.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from scripts.utils.logger import print_status
from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood

OUT_DIR = BASE_DIR / "results" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_SCALE = 1.0e6
ANCHOR_HOSTS = {"N4258", "LMC", "M31", "MW", "SMC"}
N_PERM = 400
PERM_SEED = 20260927


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------
def load_and_assign():
    """Load the SH0ES system and assign each Cepheid row a host label."""
    fl = FullLadderLikelihood()
    L, y, C, q, y_source = fl.load_sh0es_data()
    host_sigma, host_screening = fl.load_host_metadata()
    sigma_ref = fl.calculate_effective_sigma_ref()

    b_idx = np.where(q == "bW")[0][0]
    z_idx = np.where(q == "ZW")[0][0]
    mu_idx = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_names = [q[i] for i in mu_idx]

    n_rows = len(y)
    t_col = L[:, b_idx]          # (log10 P - 1) on Cepheid rows
    z_col = L[:, z_idx]          # [O/H] on Cepheid rows
    cep_rows = np.where(np.abs(t_col) > 0.01)[0]

    host_of_row = np.empty(n_rows, dtype=object)
    for i in cep_rows:
        source = str(y_source[i])
        if source.startswith("LMC_"):
            host = "LMC"
        elif source.startswith("MHW1_"):
            host = "MW"
        elif source and source not in {"nan", "None"}:
            host = source
        else:
            host = None
        if host is None:
            for idx, name in zip(mu_idx, mu_names):
                if abs(L[i, idx]) > 0.01:
                    host = name.replace("mu_", "")
                    break
        host_of_row[i] = host

    return fl, L, y, C, q, y_source, cep_rows, host_of_row, t_col, z_col, \
        host_sigma, host_screening, sigma_ref


def host_x_map(host_of_row, cep_rows, host_sigma, host_screening, sigma_ref,
               anchor_convention):
    """Per-host environmental coordinate under the stated convention."""
    hosts = sorted({h for h in host_of_row[cep_rows] if h})
    out = {}
    for h in hosts:
        X = FullLadderLikelihood.build_host_x(
            h, host_sigma, host_screening, sigma_ref, mode="centered")
        if anchor_convention == "anchor_reference_zero" and h in ANCHOR_HOSTS:
            X = 0.0
        out[h] = X
    return out


# ---------------------------------------------------------------------------
# GLS machinery with cached whitening
# ---------------------------------------------------------------------------
class WhitenedGLS:
    """Cache the Cholesky factor of C; every subsequent fit is a lstsq."""

    def __init__(self, C, y):
        self.Lc = np.linalg.cholesky(C)
        self.y_w = linalg.solve_triangular(
            self.Lc, y, lower=True, check_finite=False)

    def whiten(self, A):
        return linalg.solve_triangular(
            self.Lc, A, lower=True, check_finite=False)

    def fit(self, A):
        A_w = self.whiten(A)
        theta, res, rank, _ = np.linalg.lstsq(A_w, self.y_w, rcond=1e-12)
        resid = self.y_w - A_w @ theta
        chi2 = float(resid @ resid)
        # parameter covariance: (A^T C^-1 A)^-1
        AtA = A_w.T @ A_w
        try:
            cov = np.linalg.inv(AtA)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(AtA)
        return theta, cov, chi2, int(rank)


# ---------------------------------------------------------------------------
# Part A: PL-shape control battery
# ---------------------------------------------------------------------------
def part_a(L, y, wg, q, cep_rows, X_row, t_col, z_col):
    """kappa_P under global PL-shape freedom."""
    colP = (-X_row * t_col)            # environment-coupled period term
    colZ = (-X_row * z_col)            # environment-coupled metallicity term
    colQ = t_col**2                    # global PL curvature
    colB = np.maximum(t_col, 0.0)      # 10-day slope break (logP > 1)
    colPZ = t_col * z_col              # global PLZ interaction
    colPQ = (-X_row * t_col**2)        # environment-coupled curvature

    models = [
        ("baseline", []),
        ("kappaP", [("kappaP", colP)]),
        ("quad_PL", [("c_quad", colQ)]),
        ("break_PL", [("c_break", colB)]),
        ("plz_PL", [("c_plz", colPZ)]),
        ("kappaP+quad", [("kappaP", colP), ("c_quad", colQ)]),
        ("kappaP+break", [("kappaP", colP), ("c_break", colB)]),
        ("kappaP+quad+break",
         [("kappaP", colP), ("c_quad", colQ), ("c_break", colB)]),
        ("kappaP+kappaZ",
         [("kappaP", colP), ("kappaZ", colZ)]),
        ("kappaP+kappaZ+quad+break",
         [("kappaP", colP), ("kappaZ", colZ),
          ("c_quad", colQ), ("c_break", colB)]),
        ("kappaP+kappaZ+plz",
         [("kappaP", colP), ("kappaZ", colZ), ("c_plz", colPZ)]),
        ("kappaZ+plz", [("kappaZ", colZ), ("c_plz", colPZ)]),
        ("env_quad_only", [("kappaPQ", colPQ)]),
        ("kappaP+envquad",
         [("kappaP", colP), ("kappaPQ", colPQ)]),
    ]

    theta_base, _, chi2_base, _ = wg.fit(L)
    n_rows = len(y)
    rows = []
    for tag, cols in models:
        A = np.column_stack([L] + [c for _, c in cols]) if cols else L
        theta, cov, chi2, rank = wg.fit(A)
        row = {"model": tag, "n_params": A.shape[1], "chi2": chi2,
               "delta_chi2": chi2_base - chi2,
               "AIC": chi2 + 2 * A.shape[1],
               "BIC": chi2 + A.shape[1] * np.log(n_rows),
               "rank": rank}
        for j, (nm, _) in enumerate(cols):
            jdx = L.shape[1] + j
            sc = X_SCALE if nm.startswith("kappa") else 1.0
            row[nm] = theta[jdx] * sc
            sd = np.sqrt(cov[jdx, jdx])
            row[nm + "_err"] = sd * sc
            row[nm + "_sig"] = abs(theta[jdx]) / sd
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B: modulator discrimination
# ---------------------------------------------------------------------------
def part_b(L, y, wg, q, cep_rows, host_of_row, X_host,
           host_sigma, host_screening, sigma_ref, theta_base):
    """Which per-host coordinate carries the period structure?"""
    mu_idx = [i for i, p in enumerate(q) if p.startswith("mu_")]
    mu_fit = {q[i].replace("mu_", ""): theta_base[i] for i in mu_idx}

    hosts = sorted(X_host.keys())
    hosts_meta = pd.read_csv(BASE_DIR / "data" / "processed" /
                             "hosts_processed.csv")

    def norm_name(s):
        """Normalize host labels: NGC 4424/N4424 -> N4424."""
        s = str(s).upper().replace(" ", "")
        for pre in ("NGC", "UGC", "MRK"):
            if s.startswith(pre):
                s = {"NGC": "N", "UGC": "U", "MRK": "M"}[pre] + \
                    s[len(pre):]
                break
        # strip leading zeros in the digit part (N0691 -> N691)
        if s and s[0] in "NUM":
            i = 1
            while i < len(s) and s[i] == "0":
                i += 1
            s = s[0] + s[i:]
        return s

    logmass = {}
    for _, r in hosts_meta.iterrows():
        logmass[norm_name(r["normalized_name"])] = \
            r.get("host_logmass", np.nan)

    def nm_lookup(h):
        v = logmass.get(norm_name(h), np.nan)
        if pd.isna(v) and norm_name(h) and norm_name(h)[-1].isalpha():
            v = logmass.get(norm_name(h)[:-1], np.nan)
        return float(v) if pd.notna(v) else np.nan

    # host mean period/metallicity from the Cepheid rows
    t_host = {}
    z_host = {}
    for h in hosts:
        m = cep_rows[host_of_row[cep_rows] == h]
        t_host[h] = float(np.mean(L[m, np.where(q == "bW")[0][0]]))
        z_host[h] = float(np.nanmean(L[m, np.where(q == "ZW")[0][0]]))

    def centered(vals):
        vals = np.asarray(vals, dtype=float)
        m = np.nanmean(vals)
        return vals - m

    modulators = {
        "X_env": np.array([X_host[h] for h in hosts]),
        "mu_fitted": centered([mu_fit.get(h, np.nan) for h in hosts]),
        "sigma2_unscreened": centered(
            [host_sigma.get(h, np.nan) ** 2 / 9e10
             if host_sigma.get(h) else np.nan for h in hosts]),
        "S_screening": centered(
            [host_screening.get(h, np.nan) for h in hosts]),
        "logM_host": centered([nm_lookup(h) for h in hosts]),
        "mean_OH_host": centered([z_host[h] for h in hosts]),
        "mean_logP_host": centered([t_host[h] for h in hosts]),
    }
    # X_env in 1e6-scaled units for comparability of coefficients;
    # hosts missing a covariate sit at the centered mean (0).
    modulators["X_env"] = modulators["X_env"] * X_SCALE
    for k, v in modulators.items():
        modulators[k] = np.nan_to_num(v, nan=0.0)

    _, _, chi2_base, _ = wg.fit(L)
    t_col = L[:, np.where(q == "bW")[0][0]]
    rows = []
    host_index = {h: i for i, h in enumerate(hosts)}
    for name, mvec in modulators.items():
        col = np.zeros(len(y))
        mrow = mvec[[host_index[h] for h in host_of_row[cep_rows]]]
        col[cep_rows] = -mrow * t_col[cep_rows]
        A = np.column_stack([L, col])
        theta, cov, chi2, rank = wg.fit(A)
        sd = np.sqrt(cov[-1, -1])
        rows.append({"modulator": name, "delta_chi2": chi2_base - chi2,
                     "coef": theta[-1], "coef_err": sd,
                     "sig": abs(theta[-1]) / sd,
                     "n_nan_hosts": int(np.isnan(mvec).sum())})
    df = pd.DataFrame(rows).sort_values("delta_chi2", ascending=False)

    # joint fit: X_env plus the best alternative
    best_alt = df[df["modulator"] != "X_env"].iloc[0]["modulator"]
    def col_of(name):
        mvec = modulators[name]
        col = np.zeros(len(y))
        mrow = mvec[[host_index[h] for h in host_of_row[cep_rows]]]
        col[cep_rows] = -mrow * t_col[cep_rows]
        return col
    A = np.column_stack([L, col_of("X_env"), col_of(best_alt)])
    theta, cov, chi2, _ = wg.fit(A)
    joint = {"modulator_B": best_alt,
             "kappaP_coef": theta[-2], "kappaP_err": np.sqrt(cov[-2, -2]),
             "kappaP_sig": abs(theta[-2]) / np.sqrt(cov[-2, -2]),
             "alt_coef": theta[-1], "alt_err": np.sqrt(cov[-1, -1]),
             "alt_sig": abs(theta[-1]) / np.sqrt(cov[-1, -1]),
             "delta_chi2_joint": chi2_base - chi2,
             "corr_XP_alt": float(
                 cov[-2, -1] / np.sqrt(cov[-2, -2] * cov[-1, -1]))}
    return df, joint, modulators, hosts


# ---------------------------------------------------------------------------
# Part C: anchor-segment null
# ---------------------------------------------------------------------------
def part_c(L, y, wg, q, cep_rows, host_of_row, t_col,
           host_sigma, host_screening, sigma_ref):
    """Anchor-only period-slope term under both anchor conventions."""
    is_anchor = np.isin(host_of_row[cep_rows], list(ANCHOR_HOSTS))
    anchor_rows = cep_rows[is_anchor]
    colA = np.zeros(len(y))
    colA[anchor_rows] = t_col[anchor_rows]

    out = {}
    for conv in ["anchor_screened_physical", "anchor_reference_zero"]:
        X_host = host_x_map(host_of_row, cep_rows, host_sigma,
                            host_screening, sigma_ref, conv)
        X_row = np.zeros(len(y))
        for h, X in X_host.items():
            X_row[cep_rows[host_of_row[cep_rows] == h]] = X
        colP = -X_row * t_col * X_SCALE

        _, _, chi2_base, _ = wg.fit(L)

        # anchor-only slope alone
        A = np.column_stack([L, colA])
        th, cv, c2, _ = wg.fit(A)
        anchor_only = {"coef": th[-1], "err": np.sqrt(cv[-1, -1]),
                       "sig": abs(th[-1]) / np.sqrt(cv[-1, -1]),
                       "delta_chi2": chi2_base - c2}
        # kappaP + anchor slope jointly
        A2 = np.column_stack([L, colP, colA])
        th, cv, c2, _ = wg.fit(A2)
        joint = {"kappaP": th[-2] * X_SCALE,
                 "kappaP_err": np.sqrt(cv[-2, -2]) * X_SCALE,
                 "kappaP_sig": abs(th[-2]) / np.sqrt(cv[-2, -2]),
                 "anchor_slope": th[-1], "anchor_err": np.sqrt(cv[-1, -1]),
                 "anchor_sig": abs(th[-1]) / np.sqrt(cv[-1, -1]),
                 "delta_chi2_2par": chi2_base - c2}
        out[conv] = {"anchor_only": anchor_only, "joint": joint,
                     "n_anchor_rows": int(len(anchor_rows))}
    return out


# ---------------------------------------------------------------------------
# Part D: per-host slope decomposition
# ---------------------------------------------------------------------------
def part_d(L, y, wg, q, cep_rows, host_of_row, X_host, t_col,
           mu_fit_names, theta_base, hosts_meta_df):
    """Replace bW with per-host slopes; regress deviations on X_i."""
    b_idx = np.where(q == "bW")[0][0]
    hosts = sorted(X_host.keys())

    # host -> Cepheid rows; require >=3 rows and some t variance
    host_rows = {h: cep_rows[host_of_row[cep_rows] == h] for h in hosts}
    slope_hosts = [h for h in hosts
                   if len(host_rows[h]) >= 3
                   and np.ptp(t_col[host_rows[h]]) > 0.05]
    small_hosts = [h for h in hosts if h not in slope_hosts]

    # Design: L without bW, plus per-host slope columns, plus a shared
    # bW column restricted to small hosts (rows outside slope_hosts).
    small_rows = cep_rows[np.isin(host_of_row[cep_rows], small_hosts)]
    L0 = np.delete(L, b_idx, axis=1)
    cols = [np.where(np.isin(np.arange(len(y)), small_rows),
                     t_col, 0.0)]
    names = ["bW_small_hosts"]
    for h in slope_hosts:
        c = np.zeros(len(y))
        c[host_rows[h]] = t_col[host_rows[h]]
        cols.append(c)
        names.append(h)
    A = np.column_stack([L0] + cols)
    theta, cov, chi2, rank = wg.fit(A)

    off = L0.shape[1]
    rec = []
    for j, h in enumerate(slope_hosts):
        jdx = off + 1 + j
        s = theta[jdx]
        sd = np.sqrt(cov[jdx, jdx])
        rec.append({"host": h, "n_cep": int(len(host_rows[h])),
                    "slope": s, "slope_err": sd,
                    "X": X_host[h],
                    "is_anchor": h in ANCHOR_HOSTS})
    df = pd.DataFrame(rec)

    # add confounder covariates
    mu_map = {q[i].replace("mu_", ""): theta_base[i]
              for i, p in enumerate(q) if p.startswith("mu_")}

    def norm_name(s):
        s = str(s).upper().replace(" ", "")
        for pre in ("NGC", "UGC", "MRK"):
            if s.startswith(pre):
                s = {"NGC": "N", "UGC": "U", "MRK": "M"}[pre] + \
                    s[len(pre):]
                break
        if s and s[0] in "NUM":
            i = 1
            while i < len(s) and s[i] == "0":
                i += 1
            s = s[0] + s[i:]
        return s

    logmass = {}
    for _, r in hosts_meta_df.iterrows():
        logmass[norm_name(r["normalized_name"])] = \
            r.get("host_logmass", np.nan)

    def lookup_mass(h):
        v = logmass.get(norm_name(h), np.nan)
        if pd.isna(v) and norm_name(h) and norm_name(h)[-1].isalpha():
            v = logmass.get(norm_name(h)[:-1], np.nan)
        return float(v) if pd.notna(v) else np.nan

    df["mu_fitted"] = df["host"].map(mu_map)
    df["logM"] = df["host"].map(lookup_mass)

    def wls(xv, yv, wv):
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
        # correlation p-value via t
        tstat = r * np.sqrt(max(len(xv) - 2, 1) / max(1e-12, 1 - r**2))
        from scipy import stats as st
        p = 2 * st.t.sf(abs(tstat), len(xv) - 2)
        return {"n": int(len(xv)), "slope": float(beta[1]),
                "slope_err": float(np.sqrt(covb[1, 1])),
                "intercept": float(beta[0]),
                "intercept_err": float(np.sqrt(covb[0, 0])),
                "r": float(r), "p_value": float(p),
                "chi2": chi2, "dof": dof,
                "chi2_per_dof": chi2 / dof}

    w = 1.0 / df["slope_err"].values ** 2
    # anchor hosts: MW absent; report anchor slopes separately
    regs = {
        "on_X": wls(df["X"].values, df["slope"].values, w),
        "on_mu": wls(df["mu_fitted"].values, df["slope"].values, w),
        "on_logM": wls(df["logM"].values, df["slope"].values, w),
    }
    anchor_rows_df = df[df["is_anchor"]]
    return df, regs, {
        "slope_hosts": slope_hosts, "small_hosts": small_hosts,
        "n_fitted": int(len(df)), "chi2": float(chi2), "rank": int(rank),
        "anchor_slopes": anchor_rows_df.to_dict("records")}


# ---------------------------------------------------------------------------
# Part E: leave-one-host-out kappa_P
# ---------------------------------------------------------------------------
def part_e(L, y, C, q, cep_rows, host_of_row, X_row, t_col, z_col):
    """Drop each host's Cepheid rows; refit kappa_P + kappa_Z."""
    colP = (-X_row * t_col) * X_SCALE
    colZ = (-X_row * z_col) * X_SCALE
    hosts = sorted({h for h in host_of_row[cep_rows] if h})
    out = []
    for h in hosts:
        drop = cep_rows[host_of_row[cep_rows] == h]
        keep = np.ones(len(y), dtype=bool)
        keep[drop] = False
        # a host whose only rows are Cepheids also leaves a dead mu column;
        # keep it (prior/SN rows may still reference it) -- GLS handles rank.
        Ls, ys, Cs = L[keep], y[keep], C[np.ix_(keep, keep)]
        cPs, cZs = colP[keep], colZ[keep]
        try:
            Lc = np.linalg.cholesky(Cs)
            yw = linalg.solve_triangular(Lc, ys, lower=True,
                                         check_finite=False)
            A = np.column_stack([Ls, cPs, cZs])
            Aw = linalg.solve_triangular(Lc, A, lower=True,
                                         check_finite=False)
            th, res, rank, _ = np.linalg.lstsq(Aw, yw, rcond=1e-12)
            resid = yw - Aw @ th
            AtA = Aw.T @ Aw
            cov = np.linalg.inv(AtA)
            kP, sP = th[-2] * X_SCALE, np.sqrt(cov[-2, -2]) * X_SCALE
            kZ, sZ = th[-1] * X_SCALE, np.sqrt(cov[-1, -1]) * X_SCALE
            out.append({"dropped_host": h, "n_dropped": int(len(drop)),
                        "kappaP": kP, "kappaP_err": sP,
                        "kappaP_sig": abs(kP) / sP,
                        "kappaZ": kZ, "kappaZ_err": sZ,
                        "kappaZ_sig": abs(kZ) / sZ,
                        "chi2": float(resid @ resid)})
        except Exception as e:
            out.append({"dropped_host": h, "error": str(e)})
    df = pd.DataFrame(out)
    ok = df.dropna(subset=["kappaP"])
    summary = {
        "n_dropped": int(len(ok)),
        "sign_stable_negative": int((ok["kappaP"] < 0).sum()),
        "kappaP_min": float(ok["kappaP"].min()),
        "kappaP_max": float(ok["kappaP"].max()),
        "kappaP_sig_min": float(ok["kappaP_sig"].min()),
        "kappaP_sig_med": float(ok["kappaP_sig"].median()),
        "worst_host": ok.loc[ok["kappaP_sig"].idxmin(), "dropped_host"]
        if len(ok) else None,
        "lever_host": ok.loc[ok["kappaP_sig"].idxmax(), "dropped_host"]
        if len(ok) else None,
    }
    return df, summary


# ---------------------------------------------------------------------------
# Part F: host-assignment permutation null
# ---------------------------------------------------------------------------
def part_f(L, y, wg, q, cep_rows, host_of_row, X_host, t_col,
           kappaP_obs):
    """Permute host X labels; rebuild the period-coupled column."""
    hosts = sorted(X_host.keys())
    X_vals = np.array([X_host[h] for h in hosts])
    rng = np.random.default_rng(PERM_SEED)

    # whiten base once
    Lw = wg.whiten(L)
    yw = wg.y_w

    nulls = []
    for _ in range(N_PERM):
        perm = rng.permutation(len(hosts))
        X_row = np.zeros(len(y))
        for j, h in enumerate(hosts):
            X_row[cep_rows[host_of_row[cep_rows] == h]] = X_vals[perm[j]]
        colP = (-X_row * t_col) * X_SCALE
        cp_w = linalg.solve_triangular(wg.Lc, colP, lower=True,
                                       check_finite=False)
        Aw = np.column_stack([Lw, cp_w])
        th, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=1e-12)
        nulls.append(th[-1] * X_SCALE)
    nulls = np.asarray(nulls)
    p = (np.sum(np.abs(nulls) >= abs(kappaP_obs)) + 1) / (N_PERM + 1)
    return {"n_perm": N_PERM, "perm_mean": float(nulls.mean()),
            "perm_std": float(nulls.std()),
            "perm_p_two_sided": float(p),
            "frac_abs_ge_obs": float(np.mean(np.abs(nulls) >=
                                             abs(kappaP_obs))),
            "null_min": float(nulls.min()), "null_max": float(nulls.max())}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run():
    print_status("Step 53: within-host period-structure battery", "SECTION")

    (fl, L, y, C, q, y_source, cep_rows, host_of_row, t_col, z_col,
     host_sigma, host_screening, sigma_ref) = load_and_assign()

    wg = WhitenedGLS(C, y)
    theta_base, cov_base, chi2_base, rank_base = wg.fit(L)

    # per-row X under primary convention
    X_host = host_x_map(host_of_row, cep_rows, host_sigma, host_screening,
                        sigma_ref, "anchor_screened_physical")
    X_row = np.zeros(len(y))
    for h, X in X_host.items():
        X_row[cep_rows[host_of_row[cep_rows] == h]] = X
    X_row_scaled = X_row * X_SCALE

    print_status("Part A: PL-shape controls", "INFO")
    dfA = part_a(L, y, wg, q, cep_rows, X_row_scaled, t_col, z_col)
    dfA.to_csv(OUT_DIR / "step_53_model_battery.csv", index=False)
    for _, r in dfA.iterrows():
        print_status(
            f"  {r['model']:32s} dchi2={r['delta_chi2']:8.2f} "
            + (f"kP={r['kappaP']:.2e} ({r['kappaP_sig']:.2f}sig)"
               if pd.notna(r.get('kappaP')) else "")
            + (f" kZ={r['kappaZ']:.2e} ({r['kappaZ_sig']:.2f}sig)"
               if pd.notna(r.get('kappaZ')) else "")
            + (f" cq={r['c_quad']:.3f} ({r['c_quad_sig']:.2f}sig)"
               if pd.notna(r.get('c_quad')) else "")
            + (f" cb={r['c_break']:.3f} ({r['c_break_sig']:.2f}sig)"
               if pd.notna(r.get('c_break')) else ""), "INFO")

    print_status("Part B: modulator discrimination", "INFO")
    dfB, jointB, modulators, hostsB = part_b(
        L, y, wg, q, cep_rows, host_of_row, X_host,
        host_sigma, host_screening, sigma_ref, theta_base)
    dfB.to_csv(OUT_DIR / "step_53_modulator_comparison.csv", index=False)
    for _, r in dfB.iterrows():
        print_status(f"  {r['modulator']:20s} dchi2={r['delta_chi2']:7.2f} "
                     f"sig={r['sig']:.2f}", "INFO")
    print_status(f"  joint X + {jointB['modulator_B']}: "
                 f"kP sig={jointB['kappaP_sig']:.2f}, "
                 f"alt sig={jointB['alt_sig']:.2f}, "
                 f"corr={jointB['corr_XP_alt']:.2f}", "INFO")

    print_status("Part C: anchor-segment null", "INFO")
    resC = part_c(L, y, wg, q, cep_rows, host_of_row, t_col,
                  host_sigma, host_screening, sigma_ref)
    for conv, d in resC.items():
        print_status(
            f"  {conv}: anchor-only slope={d['anchor_only']['coef']:.4f} "
            f"({d['anchor_only']['sig']:.2f}sig); joint: "
            f"kP={d['joint']['kappaP']:.3e} ({d['joint']['kappaP_sig']:.2f}sig), "
            f"anchor={d['joint']['anchor_slope']:.4f} "
            f"({d['joint']['anchor_sig']:.2f}sig)", "INFO")

    print_status("Part D: per-host slope decomposition", "INFO")
    hosts_meta_df = pd.read_csv(BASE_DIR / "data" / "processed" /
                                "hosts_processed.csv")
    dfD, regsD, metaD = part_d(L, y, wg, q, cep_rows, host_of_row,
                              X_host, t_col, q, theta_base, hosts_meta_df)
    dfD.to_csv(OUT_DIR / "step_53_per_host_period_slopes.csv", index=False)
    for k, v in regsD.items():
        if "slope" in v:
            print_status(
                f"  s_i on {k}: slope={v['slope']:.3e}+/-"
                f"{v['slope_err']:.3e} (r={v['r']:.2f}, p={v['p_value']:.3g}, "
                f"n={v['n']}, chi2/dof={v['chi2_per_dof']:.2f})", "INFO")

    print_status("Part E: leave-one-host-out kappa_P", "INFO")
    dfE, sumE = part_e(L, y, C, q, cep_rows, host_of_row, X_row,
                       t_col, z_col)
    dfE.to_csv(OUT_DIR / "step_53_kappaP_loo.csv", index=False)
    print_status(
        f"  LOO: {sumE['n_dropped']} hosts, negative {sumE['sign_stable_negative']}, "
        f"sig range [{sumE['kappaP_sig_min']:.2f}, med {sumE['kappaP_sig_med']:.2f}], "
        f"worst={sumE['worst_host']}, lever={sumE['lever_host']}", "INFO")

    print_status("Part F: host-assignment permutation null", "INFO")
    kappaP_obs = float(dfA[dfA["model"] == "kappaP"]["kappaP"].iloc[0])
    resF = part_f(L, y, wg, q, cep_rows, host_of_row, X_host, t_col,
                  kappaP_obs)
    print_status(
        f"  permutations: {resF['n_perm']}, p={resF['perm_p_two_sided']:.4f}, "
        f"null std={resF['perm_std']:.3e}, obs={kappaP_obs:.3e}", "INFO")

    # ------------------------------------------------------------------
    out = {
        "primary_anchor_convention": "anchor_screened_physical",
        "baseline_chi2": float(chi2_base),
        "model_battery": dfA.to_dict("records"),
        "modulator_comparison": dfB.to_dict("records"),
        "modulator_joint": jointB,
        "anchor_null": resC,
        "per_host_regressions": regsD,
        "per_host_meta": metaD,
        "loo_summary": sumE,
        "permutation_null": resF,
        "kappaP_observed": kappaP_obs,
    }
    with open(OUT_DIR / "step_53_within_host_structure_battery.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print_status("Step 53 complete: within-host structure battery",
                 "SUCCESS")
    return out


if __name__ == "__main__":
    run()
