"""Physical predictions must not depend on the numerical unit of a regressor."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.steps.step_39_environment_slope_decomposition import fit_gamma, LN10_OVER_5


class GammaScalingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(-1, 1, 37)
        self.d = np.linspace(10, 60, 37)
        self.muerr = np.full(37, .05)
        self.cz = self.d*(71+6*self.x) + np.random.default_rng(142).normal(0, 100, 37)

    def test_rescaled_regressor_preserves_fit_and_profile(self):
        fits = [fit_gamma(self.cz, self.d, self.x*s, self.muerr, 100,
                          compute_profile=True) for s in (1., 1e-3, 1e-7)]
        reference = fits[0]
        for s, fit in zip((1., 1e-3, 1e-7), fits):
            self.assertEqual(fit['status'], 'converged')
            self.assertAlmostEqual(fit['logL'], reference['logL'], places=6)
            self.assertAlmostEqual(fit['H_app'], reference['H_app'], places=4)
            self.assertAlmostEqual(fit['Gamma_X']*s, reference['Gamma_X'], places=4)
            for side in ('low', 'high'):
                self.assertAlmostEqual(fit[f'Gamma_X_profile_{side}']*s,
                                       reference[f'Gamma_X_profile_{side}'], places=3)

    def test_agrees_with_independent_derivative_free_optimization(self):
        def nll(p):
            model = self.d*(p[0]+p[1]*self.x)
            variance = 100**2 + (LN10_OVER_5*model*self.muerr)**2 + p[2]**2
            return .5*np.sum((self.cz-model)**2/variance + np.log(variance))
        reference = minimize(nll, [70, 6, 5], method='Powell',
                             bounds=[(30, 90), (-100, 100), (.01, 50)],
                             options={'xtol': 1e-9, 'ftol': 1e-12})
        fit = fit_gamma(self.cz, self.d, self.x, self.muerr, 100,
                        compute_uncertainty=False)
        self.assertTrue(reference.success)
        self.assertAlmostEqual(-fit['logL'], reference.fun, places=5)


if __name__ == '__main__':
    unittest.main()
