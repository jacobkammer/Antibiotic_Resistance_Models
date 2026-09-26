"""Regression tests for solver convergence and hard-LOD extinction."""

import unittest

import numpy as np

import model_Bacteremia as model
from parameters import (
    BASE_PARAMS, LOD, PK_PARAMS, TOTAL_H, VANCO_START, Y0,
    bacteremia_outcome_masks,
)
from solver import IntegrationError, integrate_model


class SolverConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pk = model.PharmacokineticModel()
        cls.immune = model.ImmuneResponse()
        cls.van = pk.concentration_function("vancomycin", TOTAL_H, VANCO_START)
        cls.lzd = pk.concentration_function(
            "linezolid", TOTAL_H, VANCO_START + pk.van_duration
        )
        cls.args = (BASE_PARAMS, cls.van, cls.lzd, cls.immune)

    def solve(self, points, method="LSODA"):
        return integrate_model(
            model.dual_reservoir_model, Y0,
            np.linspace(0.0, TOTAL_H, points), self.args, method=method,
        )

    def test_nonnegative_and_grid_independent(self):
        coarse = self.solve(1150)
        dense = self.solve(2300)
        self.assertTrue(np.all(coarse >= 0))
        self.assertTrue(np.all(dense >= 0))
        self.assertFalse(np.any((coarse > 0) & (coarse < LOD)))
        self.assertFalse(np.any((dense > 0) & (dense < LOD)))
        np.testing.assert_allclose(coarse[-1], dense[-1], rtol=2e-6, atol=1e-40)

    def test_independent_stiff_solver_agrees(self):
        lsoda = self.solve(1150, "LSODA")[-1]
        bdf = self.solve(1150, "BDF")[-1]
        np.testing.assert_allclose(lsoda, bdf, rtol=2e-5, atol=1e-40)

    def test_invalid_initial_state_rejected(self):
        with self.assertRaises(ValueError):
            integrate_model(
                model.dual_reservoir_model, [0, -1, 100, 100],
                np.linspace(0.0, 2.0, 3), self.args,
            )

    def test_strong_clearance_becomes_exact_extinction(self):
        strong_immune = model.ImmuneResponse(k_immune=0.65)
        args = (BASE_PARAMS, self.args[1], self.args[2], strong_immune)
        solution = integrate_model(
            model.dual_reservoir_model, Y0,
            np.linspace(0.0, TOTAL_H, 1150), args,
        )
        self.assertTrue(np.all(np.isfinite(solution)))
        self.assertTrue(np.all(solution >= 0))
        self.assertTrue(np.all(solution[-1] == 0.0))

    def test_sub_lod_population_cannot_regrow_autonomously(self):
        solution = integrate_model(
            model.dual_reservoir_model, [0.0, 0.0, 0.0, LOD - 1.0],
            np.linspace(0.0, 48.0, 97), self.args,
        )
        self.assertTrue(np.all(solution == 0.0))

    def test_detectable_reservoir_can_reseed_cleared_blood(self):
        solution = integrate_model(
            model.dual_reservoir_model, [0.0, 0.0, 0.0, 100.0],
            np.linspace(0.0, 48.0, 193), self.args,
        )
        self.assertGreater(np.max(solution[:, 1]), LOD)

    def test_relapse_classifier_restores_plotting_nans_to_extinction(self):
        lzd_end = VANCO_START + PK_PARAMS["van_duration"] + PK_PARAMS["lzd_duration"]
        times = np.array([0.0, lzd_end, TOTAL_H])
        histories = np.array([
            [100.0, np.nan, LOD],  # clears, then relapses
            [100.0, LOD, LOD],     # never clears
            [100.0, np.nan, np.nan],  # resolves
        ])
        relapse, persistent, resolved = bacteremia_outcome_masks(histories, times)
        np.testing.assert_array_equal(relapse, [True, False, False])
        np.testing.assert_array_equal(persistent, [False, True, False])
        np.testing.assert_array_equal(resolved, [False, False, True])


if __name__ == "__main__":
    unittest.main()
