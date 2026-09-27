from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.steps.step_39_environment_slope_decomposition import (
    LN10_OVER_5, fit_gamma, velocity_likelihood_hessian,
)


class VelocityCurvatureTests(unittest.TestCase):
    def test_analytic_curvature_matches_central_differences(self):
        d = np.linspace(10., 60., 37)
        x = np.linspace(-1., 1., 37)
        muerr = np.full(37, .08)
        cz = d*(70+5*x)+np.random.default_rng(8).normal(0, 100, 37)
        p = np.array([69., 4., 15.])
        def objective(q):
            model = d*(q[0]+q[1]*x)
            var = 100**2+(LN10_OVER_5*model*muerr)**2+q[2]**2
            return .5*np.sum((cz-model)**2/var+np.log(var))
        h = .02
        basis = np.eye(3)*h
        numerical = np.array([
            [(objective(p+a+b)-objective(p+a-b)-objective(p-a+b)+objective(p-a-b))/(4*h*h)
             for b in basis] for a in basis])
        exact = velocity_likelihood_hessian(p, cz, d, x, muerr, 100.)
        np.testing.assert_allclose(exact, numerical, rtol=1e-4, atol=2e-8)

    def test_bound_scatter_does_not_create_zero_amplitude_error(self):
        d = np.linspace(10., 60., 37)
        x = np.linspace(-1., 1., 37)
        cz = d*(70+5*x)+np.random.default_rng(8).normal(0, 100, 37)
        fits = [fit_gamma(cz, d, x*s, np.full(37, .05), 250.) for s in (1., 1e-7)]
        for scale, fit in zip((1., 1e-7), fits):
            self.assertIn('conditional_on_active_bounds', fit['uncertainty_method'])
            self.assertGreater(fit['Gamma_X_err'], 0)
            self.assertTrue(np.isnan(fit['sigma_int_v_err']))
            self.assertAlmostEqual(fit['Gamma_X_err']*scale, fits[0]['Gamma_X_err'], places=4)


if __name__ == '__main__':
    unittest.main()
