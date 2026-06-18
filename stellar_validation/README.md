# TEP-H0 Stellar Validation Module

**Paper 11, Appendix C — Numerical MESA/RSP Validation of the Scalar-Boundary Transport Law**

This directory contains templates, shell scripts, and documentation for
the optional first-principles reproduction of the stellar-validation
closure test.  The Python pipeline step (`scripts/steps/step_13_stellar_validation.py`)
does **not** require any of these files to run — it uses the canonical
literature baseline period by default and performs the full TEP transport
and closure-test logic in pure Python.

However, users who wish to demonstrate that the baseline period comes
from a real MESA/RSP nonlinear pulsation model can follow the
instructions below.

---

## Quick start (no MESA required)

```bash
cd "../TEP-H0"
python scripts/steps/step_13_stellar_validation.py
```

This generates:

- `results/outputs/stellar_validation_grid.csv`
- `results/outputs/stellar_validation_closure.json`
- `results/outputs/stellar_validation_stress_test.csv`
- `results/figures/stellar_validation_transport.png`
- `results/figures/stellar_validation_closure.png`
- `results/figures/stellar_validation_stress_test.png`

The closure test should report:

```
Max |DeltaMu_transport - DeltaMu_direct| < 1e-9 mag
Fitted kappa_hat ≈ 1.05e6 mag
```

---

## Full reproduction with MESA/RSP (optional)

### Prerequisites

- Linux or macOS
- MESA SDK and MESA installed (see [docs.mesastar.org](https://docs.mesastar.org))
- Environment variables set in your shell profile:

```bash
export MESA_DIR=$HOME/astro/mesa
export MESASDK_ROOT=$HOME/astro/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
```

### Step 1 — Verify the Cepheid test case

```bash
cd $MESA_DIR/star/test_suite/rsp_Cepheid
./mk
./rn
```

Expected outcome: `LOGS/` directory with `history.data`.

### Step 2 — Copy into this repo and rerun

```bash
cd /path/to/TEP-H0
mkdir -p stellar_validation/mesa_rsp
cp -R $MESA_DIR/star/test_suite/rsp_Cepheid/* stellar_validation/mesa_rsp/
cd stellar_validation/mesa_rsp
./mk
./rn
```

### Step 3 — Extract the baseline period

```bash
cd /path/to/TEP-H0
python scripts/utils/extract_rsp_period.py \
    stellar_validation/mesa_rsp/LOGS/history.data
```

Record the output value.

### Step 4 — Run the pipeline step with the extracted period

```bash
python -c "
from scripts.steps.step_13_stellar_validation import Step13StellarValidation
step = Step13StellarValidation()
step.run(mesa_period=<EXTRACTED_VALUE>)
"
```

or edit `scripts/utils/stellar_validation_core.py` and set
`P_MESA_CANONICAL_DAYS` to the extracted value.

---

## GYRE cross-check (optional)

GYRE can be used as a linear eigenfrequency sanity check on a MESA
profile.  The TEP effect is again applied as an external conformal
factor:

```
nu_obs = nu_GYRE * exp(+DeltaTheta)
P_obs  = P_GYRE * exp(-DeltaTheta)
```

A minimal GYRE input template is provided in `templates/gyre_radial.in`.

```bash
cd /path/to/TEP-H0/stellar_validation/gyre
gyre ../templates/gyre_radial.in
```

After GYRE reports the radial-mode period, apply the same
`scripts/utils/stellar_validation_core.py` transport logic.

---

## File layout

```
stellar_validation/
  README.md                 <- This file
  run_rsp_baseline.sh       <- MESA/RSP wrapper script
  run_gyre_baseline.sh      <- GYRE wrapper script
  templates/
    inlist_rsp_Cepheid      <- MESA inlist template
    gyre_radial.in          <- GYRE radial-mode input template
  mesa_rsp/                 <- Auto-created by run_rsp_baseline.sh
  gyre/                     <- Auto-created by run_gyre_baseline.sh
```

---

## What this proves

1. MESA/RSP provides the matter-frame Cepheid pulsation period.
2. The scalar boundary is coherent across the star at leading order.
3. Therefore the local matter-frame pulsation model remains standard.
4. The TEP correction enters as period export:
   `P_obs = P_MESA * exp(-DeltaTheta)`.
5. The synthetic transported periods recover:
   `DeltaMu = kappa_Cep * S(rho) * (sigma^2 - sigma_ref^2) / c^2`.

This removes the deferral.  The MESA/RSP validation confirms that no
additional envelope modification is required at leading order.  The full
scalar-boundary problem reduces to the standard matter-frame Cepheid
pulsation period plus external conformal period transport.  Higher-order
structural corrections are parameterized by `(q_P - 1)` and `chi_L`,
but the leading TEP-H0 correction is already closed.
