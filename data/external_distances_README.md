# External Distance Database

## Purpose

This file catalogs non-Cepheid distance measurements for TEP-H0 host galaxies. The goal is to break the κ_Cep–β_X degeneracy by providing quasi-independent constraints on the true distance modulus μ_true per host.

## Schema

| Column | Description |
|--------|-------------|
| `host` | Original literature name (e.g., "NGC1309") |
| `host_normalized` | Normalized name matching `hosts_processed.csv` (e.g., "N1309") |
| `mu_ext` | External distance modulus (mag) |
| `mu_ext_err` | External distance modulus error (mag) |
| `method` | Distance method: TRGB, SBF, maser, EB, JWST-Cepheid, SNIa-cal, etc. |
| `method_detail` | Instrument/bandpass details |
| `calibration_family` | Calibration source (e.g., Freedman2024, Riess2022, Cantiello2018) |
| `is_independent_of_cepheids` | True if the method does not use Cepheid calibration |
| `reference` | ADS bibcode or arXiv identifier |
| `N_hosts_in_sample` | Number of hosts in the reference's sample |
| `notes` | Additional context |

## Current Status

### TRGB (Freedman+2024 / CCHP)

- 18 hosts with HST/ACS F814W TRGB distances from the Chicago–Carnegie Hubble Program
- 13 overlap with the primary Hubble-flow Cepheid sample (N=29, z ≥ 0.0035)
- Differential κ (Step 41): +3.19×10^5 ± 3.90×10^5 (0.82σ) — underpowered to break degeneracy at current overlap

### Geometric anchor

- **NGC 4258 maser** (Reid et al. 2019): μ = 29.397 ± 0.032 mag — truly geometric, independent of stellar clocks
- Included as a reference entry; NGC 4258 is an anchor, not a Hubble-flow host

### Coverage Gaps

The following 16 primary Hubble-flow hosts (z ≥ 0.0035) have **no** published independent external distance in this database:

N0691, N105A, N2525, N2608, N3147, N3254, N3447, N3583, N4680, N5468, N5728, N7329, N7541, N7678, N976A, U9391

These are mostly faint-to-moderate luminosity spiral galaxies. Published TRGB/SBF distances for this specific sample are scarce because:
- TRGB requires resolved stellar populations; many of these hosts are at D ≳ 25 Mpc where HST resolution becomes limiting
- SBF applies primarily to early-type galaxies; nearly all SH0ES hosts are late-type spirals
- JWST programs (GO-1995, GO-3055) are beginning to target some of these, but published distances are not yet available for most

### Realistic expansion targets

| Method | Expected new overlap | Key references | Notes |
|--------|---------------------|--------------|-------|
| JWST TRGB | ~3-5 hosts | Anand+2024, 2025 | NGC 1559, NGC 5584 already covered; NGC 5468 too distant for current depth |
| JWST JAGB | ~2-3 hosts | Lee+2024 (ApJ 961 132) | NGC 7250, NGC 4536, NGC 3972 — method demonstration, final distances pending |
| SBF | ~0-1 host | Jensen+2021, 2025 | Most SH0ES hosts are spirals, not early-types |
| Masers | 0 hosts | — | No other megamaser hosts in the SH0ES sample |
| Eclipsing binaries | 0 hosts | — | No DEB systems known in these hosts |

## Method-specific susceptibility

Different distance methods may have different κ_m values:

| Method | Expected κ_m relative to Cepheids |
|--------|----------------------------------|
| TRGB | ~0 (different stellar population, but may share environment systematics) |
| SBF | ~0 (stellar population dependent, different physics) |
| JWST Cepheids | ~κ_Cep (same method, less crowding) |
| Masers | ~0 (geometric, independent of stellar clocks) |
| EB | ~0 (geometric, independent of stellar clocks) |

## How to add data

1. Add a new row to `external_distances.csv` with all columns filled.
2. Set `is_independent_of_cepheids` appropriately.
3. Re-run Step 41 (`step_41_external_distance_breakers.py`) to update overlap statistics.
4. Re-run Step 42 (`step_42_tep_native_ladder.py`) to update gauge interpretations.
5. Verify regression gates still pass.
