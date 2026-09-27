from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.steps.step_39_environment_slope_decomposition import bootstrap_gamma


class ThresholdBootstrapTests(unittest.TestCase):
    def test_missing_group_is_recorded_as_unidentified(self):
        x = np.r_[0., np.ones(15)]
        d = np.linspace(10, 40, len(x))
        cz = d*(65+8*x) + np.random.default_rng(19).normal(0, 10, len(x))
        args = (cz, d, x, np.full(len(x), .01), 20.)
        pairs = bootstrap_gamma(*args, n_boot=80)
        conditional = bootstrap_gamma(*args, n_boot=80, strata=x)
        self.assertGreater(pairs['n_unidentifiable'], 0)
        self.assertEqual(pairs['n_attempted'], pairs['n_identifiable']+pairs['n_unidentifiable'])
        self.assertEqual(conditional['n_unidentifiable'], 0)
        self.assertEqual(conditional['group_sizes'], [1, 15])
        self.assertGreater(conditional['Gamma_X_ci_low'], 0)
        self.assertLess(conditional['Gamma_X_ci_high'], 20)


if __name__ == '__main__':
    unittest.main()
