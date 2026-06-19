import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "results" / "outputs"


def test_primary_corrected_csv_matches_headline_json():
    corrected = pd.read_csv(OUTPUTS / "tep_corrected_h0.csv")
    with open(OUTPUTS / "tep_correction_results.json") as f:
        tep = json.load(f)

    assert corrected["h0_corrected"].mean() == pytest.approx(tep["unified_h0"], abs=1e-6)


def test_projected_bic_remains_strong():
    with open(OUTPUTS / "covariance_robustness.json") as f:
        cov = json.load(f)

    projected = cov["bayesian_comparison"]["projected"]
    assert projected["delta_bic_matched"] > 10.0


def test_screened_reference_uses_ngc4258_anchor_screening():
    with open(OUTPUTS / "tep_correction_results.json") as f:
        tep = json.load(f)

    assert tep["anchor_screening"]["NGC 4258"] < 0.2
    assert tep["sigma_ref_screened"] < 40.0
