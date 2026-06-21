#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Make direct execution behave the same as module execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import print_status
from scripts.steps.step_34_full_ladder_likelihood import FullLadderLikelihood
from scripts.steps.step_39_environment_slope_decomposition import LN10_OVER_5, fit_gamma


def _build_host_z_map(df: pd.DataFrame) -> dict[str, float]:
    host_z: dict[str, float] = {}

    def add_variants(name: str, z: float) -> None:
        if not name:
            return
        host_z[name] = z
        sh0es_name = name.replace(" ", "").replace("NGC", "N").replace("UGC", "U")
        host_z[sh0es_name] = z

        if sh0es_name.startswith(("N", "U")):
            parts = sh0es_name[1:]
            if parts.isdigit():
                padded = sh0es_name[0] + parts.zfill(4)
                host_z[padded] = z
                unpadded = sh0es_name[0] + parts.lstrip("0")
                if unpadded != padded:
                    host_z[unpadded] = z

        if sh0es_name.startswith("N"):
            host_z["NGC" + sh0es_name[1:]] = z

    for _, row in df.iterrows():
        name = str(row.get("normalized_name", ""))
        z_hd = row.get("z_hd", np.nan)
        if pd.isna(z_hd) or z_hd <= 0:
            continue
        add_variants(name, float(z_hd))

    explicit = {"M1337": "N1337", "N105A": "N105", "N976A": "N976"}
    for sh0es_name, csv_name in explicit.items():
        if csv_name in host_z and sh0es_name not in host_z:
            host_z[sh0es_name] = host_z[csv_name]

    return host_z


def run():
    print_status("Step 43: Toy Recovery Experiment (Step 34 vs Velocity-Space)", "SECTION")

    out_dir = PROJECT_ROOT / "results" / "outputs"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_name = "step_43_figure_01_toy_recovery_experiment.png"
    fig_path = fig_dir / fig_name
    out_path = out_dir / "step_43_toy_recovery_experiment.json"

    seed = 43
    rng = np.random.default_rng(seed)

    PRIMARY_Z_CUT = 0.0035
    anchor_hosts = {"MW", "LMC", "SMC", "M31", "N4258"}

    H_app_true = 73.04
    Gamma_inj = 2.35e7
    kappa_inj = float(Gamma_inj / (LN10_OVER_5 * H_app_true))

    # ------------------------------------------------------------------
    # Method (1): SH0ES-style design matrix with free host moduli
    # Inject an environment-dependent Cepheid bias that is algebraically
    # equivalent to shifting the latent host moduli mu_i.
    # ------------------------------------------------------------------
    fll = FullLadderLikelihood()
    L, y, C, q, _ = fll.load_sh0es_data()
    host_sigma, host_screening = fll.load_host_metadata()
    host_screening = {
        k: (float(v) if v is not None and np.isfinite(v) else 1.0)
        for k, v in host_screening.items()
    }
    sigma_ref = fll.calculate_effective_sigma_ref()

    theta_base, cov_base, _, _, _ = fll.fit_gls(L, y, C)

    mu_indices = [i for i, p in enumerate(q) if str(p).startswith("mu_")]
    mu_names = [str(q[i]) for i in mu_indices]

    # Host redshift lookup (for selecting the primary N=29)
    df_hosts = pd.read_csv(fll.hosts_path)
    host_z = _build_host_z_map(df_hosts)

    # Build per-host X for each mu_i
    host_X = {}
    for mu_param in mu_names:
        host = mu_param.replace("mu_", "")
        X = fll.build_host_x(host, host_sigma, host_screening, sigma_ref, mode="centered")
        X = float(X)
        host_X[host] = X if np.isfinite(X) else 0.0

    theta_inj = theta_base.copy()
    for idx, mu_param in zip(mu_indices, mu_names):
        host = mu_param.replace("mu_", "")
        X = host_X.get(host, 0.0)
        theta_inj[idx] = theta_inj[idx] - kappa_inj * X

    # Synthetic SH0ES data vector consistent with the biased latent moduli
    y_mock = L @ theta_inj

    # Fit augmented model with an explicit kappa column (as in Step 34)
    x_cepheid, x_sn, _, _ = fll.build_tep_columns(
        L,
        q,
        host_sigma,
        host_screening,
        sigma_ref,
        x_mode="centered",
        anchor_convention="anchor_screened_physical",
    )
    X_SCALE = 1e6
    L_aug, q_aug = fll.build_model_matrix(L, q, x_cepheid, x_sn, "cepheid_offset", X_SCALE=X_SCALE)

    theta_aug, cov_aug, _, _, _ = fll.fit_gls(L_aug, y_mock, C)
    kappa6_hat = float(theta_aug[len(q)])
    kappa_hat = kappa6_hat * X_SCALE
    kappa_err = float(np.sqrt(cov_aug[len(q), len(q)])) * X_SCALE

    print_status(f"Injected kappa_Cep (latent-modulus-equivalent): {kappa_inj:.3e} mag", "INFO")
    print_status(f"Recovered Step-34 kappa_Cep: {kappa_hat:.3e} +/- {kappa_err:.3e} mag", "INFO")

    # ------------------------------------------------------------------
    # Method (2): velocity-space generative likelihood (fit Gamma_X)
    # Build a synthetic cz sample from true distances but biased observed mu.
    # ------------------------------------------------------------------
    primary_hosts = []
    mu_true = []
    mu_obs = []
    mu_err = []
    X_vec = []

    for idx, mu_param in zip(mu_indices, mu_names):
        host = mu_param.replace("mu_", "")
        if host in anchor_hosts:
            continue
        z = host_z.get(host)
        if z is None or z < PRIMARY_Z_CUT:
            continue
        X = host_X.get(host, 0.0)
        primary_hosts.append(host)
        mu_t = float(theta_base[idx])
        mu_o = float(theta_base[idx] - kappa_inj * X)
        mu_true.append(mu_t)
        mu_obs.append(mu_o)
        mu_err.append(float(np.sqrt(cov_base[idx, idx])))
        X_vec.append(float(X))

    mu_true = np.array(mu_true)
    mu_obs = np.array(mu_obs)
    mu_err = np.array(mu_err)
    X_vec = np.array(X_vec)

    if len(primary_hosts) == 0:
        raise RuntimeError("No primary hosts selected for toy recovery experiment")

    d_true = 10 ** ((mu_true - 25.0) / 5.0)
    d_obs = 10 ** ((mu_obs - 25.0) / 5.0)

    sigma_v_true = 250.0
    cz_obs = d_true * H_app_true

    X_centered = X_vec - np.mean(X_vec)

    gamma_fit = fit_gamma(cz_obs, d_obs, X_centered, mu_err, sigma_v=sigma_v_true, sigma_int_guess=5.0)

    print_status(f"Injected Gamma_X: {Gamma_inj:.3e}", "INFO")
    print_status(
        f"Recovered Gamma_X: {gamma_fit['Gamma_X']:.3e} +/- {gamma_fit['Gamma_X_err']:.3e}",
        "INFO",
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    payload = {
        "seed": seed,
        "primary_z_cut": PRIMARY_Z_CUT,
        "N_primary": int(len(primary_hosts)),
        "injection": {
            "H_app_true": float(H_app_true),
            "kappa_Cep_injected": float(kappa_inj),
            "Gamma_X_injected": float(Gamma_inj),
            "sigma_v_true": float(sigma_v_true),
        },
        "design_matrix": {
            "kappa_Cep_recovered": float(kappa_hat),
            "kappa_Cep_err": float(kappa_err),
            "kappa_Cep_sig": float(abs(kappa_hat) / kappa_err) if kappa_err > 0 else np.nan,
        },
        "velocity_space": {
            "Gamma_X_recovered": float(gamma_fit.get("Gamma_X", np.nan)),
            "Gamma_X_err": float(gamma_fit.get("Gamma_X_err", np.nan)),
            "Gamma_X_sig": float(gamma_fit.get("Gamma_X_sig", np.nan)),
        },
        "outputs": {
            "json": str(out_path.relative_to(PROJECT_ROOT)),
            "figure": str(fig_path.relative_to(PROJECT_ROOT)),
        },
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print_status(f"Saved toy recovery JSON: {out_path}", "SUCCESS")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].set_title("Design-matrix fit (Step 34 style)")
    ax[0].axhline(0.0, color="black", linewidth=1.0)
    ax[0].errorbar([0], [kappa_hat], yerr=[kappa_err], fmt="o", capsize=4)
    ax[0].set_xticks([0])
    ax[0].set_xticklabels(["kappa_Cep"])
    ax[0].set_ylabel("Recovered coefficient (mag)")

    ax[1].set_title("Velocity-space fit")
    ax[1].axhline(Gamma_inj, color="black", linewidth=1.0, linestyle="--", label="Injected")
    ax[1].errorbar([0], [gamma_fit["Gamma_X"]], yerr=[gamma_fit["Gamma_X_err"]], fmt="o", capsize=4, label="Recovered")
    ax[1].set_xticks([0])
    ax[1].set_xticklabels(["Gamma_X"])
    ax[1].set_ylabel("Coefficient (km/s/Mpc)")
    ax[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print_status(f"Saved toy recovery figure: {fig_path}", "SUCCESS")

    # Copy to site/public/figures if available
    public_fig_dir = PROJECT_ROOT / "site" / "public" / "figures"
    if public_fig_dir.exists():
        try:
            public_fig_path = public_fig_dir / fig_name
            public_fig_path.write_bytes(fig_path.read_bytes())
            print_status(f"Copied figure to: {public_fig_path}", "SUCCESS")
        except Exception as e:
            print_status(f"Figure copy failed: {e}", "WARNING")

    return payload


if __name__ == "__main__":
    run()
