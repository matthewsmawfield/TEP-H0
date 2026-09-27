"""
Step 54: NED Tracer Provenance Compilation for Host Galaxies

Resolves issue 11-4 by querying the NASA/IPAC Extragalactic Database (NED)
for the redshift provenance of the 37 host galaxies. It classifies the tracer
as 'nuclear', 'integrated-optical', or 'HI', and splits the computed TEP
gradients (Gamma_X) by class, as the mechanism requires r_spec < r_cep.

STATUS: NOT_EXECUTED scaffold.

An earlier revision of this step nominally queried NED but then discarded the
query results and assigned ``tracer_class`` via ``np.random.seed(42)`` +
``np.random.choice(classes, p=[0.6, 0.3, 0.1])`` -- simulated labels written
to provenance-sounding output files (``hosts_provenance.csv``,
``tracer_split_results.json``). Those outputs carry no information about real
tracer geometry and must not be cited as provenance in any manuscript or
issue closure (see issue 11-4, status addendum 2026-09-26).

The step now refuses to emit labels until a real per-host classification
exists. What is required to execute honestly:

  1. For each of the 37 R22 SN Ia hosts, retrieve the *preferred redshift*
     reference from NED (``Ned.get_table`` -> Redshift Flag / refcode, or
     ``Ned.query_refcode`` on the z source).
  2. Map the reference to a measurement aperture class: nuclear/optical-core
     spectroscopy vs integrated optical vs HI 21-cm (disk-integrated), using
     the refcode's published method, not a name heuristic.
  3. Record per-host provenance (refcode, method, aperture class) so the
     r_spec < r_cep condition can be audited host-by-host.
  4. Only then split the computed Gamma_X gradient by class and test whether
     the mechanism sign survives conditioning on real tracer geometry.

Until those four items are implemented against real NED data, this step
writes only this status record.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_JSON = ROOT / "results" / "outputs" / "step_54_ned_tracer_provenance.json"

STATUS = {
    "step": "step_54_ned_tracer_provenance",
    "status": "NOT_EXECUTED",
    "reason": (
        "Prior revision assigned tracer_class via np.random.seed(42) "
        "simulated labels under provenance-sounding filenames; those labels "
        "are not real NED classifications. Refusing to emit classes until a "
        "real per-host refcode/aperture classification is implemented."
    ),
    "required_to_execute": [
        "retrieve preferred-redshift refcode per host from NED",
        "map refcode to aperture class (nuclear / integrated-optical / HI)",
        "record per-host provenance fields for audit",
        "split Gamma_X by real class and test r_spec < r_cep conditioning",
    ],
    "related_issue": "11-4",
}


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(STATUS, f, indent=2)
    print("step_54: NOT_EXECUTED -- real NED per-host classification not yet "
          "implemented. Wrote status record to", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
