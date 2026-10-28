"""Pruebas reproducibles del modelo Monte Carlo."""

from __future__ import annotations

import unittest
from dataclasses import replace

from loopback_qkd import SimulationConfig, simulate


class LoopBackQKDTests(unittest.TestCase):
    def test_ideal_has_zero_qber_and_half_sifting(self) -> None:
        result = simulate(
            SimulationConfig(
                pulses=200_000,
                seed=11,
                source_model="single_photon",
                loss_ab_db=0.0,
                bob_insertion_loss_db=0.0,
                loss_ba_db=0.0,
                detector_efficiency=1.0,
                dark_count_probability=0.0,
                bob_encoding_error_std_deg=0.0,
            )
        )
        self.assertEqual(result.qber, 0.0)
        self.assertEqual(result.detection_probability, 1.0)
        self.assertAlmostEqual(result.sifted_fraction_of_pulses, 0.5, delta=0.01)

    def test_realistic(self) -> None:
        result = simulate(
            SimulationConfig(
                pulses=1_000_000,
                seed=104,
                source_model="weak_coherent",
                mean_photon_number=0.1,
            )
        )
        self.assertAlmostEqual(
            result.detection_probability,
            result.expected_total_click_probability,
            delta=0.003,
        )
        self.assertAlmostEqual(
            result.sifted_fraction_of_pulses,
            result.detection_probability / 2.0,
            delta=0.002,
        )
        self.assertLess(result.qber, 0.001)

    def test_full_intercept_resend_gives_quarter_qber(self) -> None:
        result = simulate(
            SimulationConfig(
                pulses=500_000,
                seed=12,
                source_model="single_photon",
                attack_fraction=1.0,
            )
        )
        self.assertAlmostEqual(result.qber, 0.25, delta=0.01)

    def test_weak_coherent_click_probability_matches_poisson_model(self) -> None:
        result = simulate(
            SimulationConfig(
                pulses=1_000_000,
                seed=13,
                source_model="weak_coherent",
                mean_photon_number=0.1,
                loss_ab_db=2.0,
                bob_insertion_loss_db=1.0,
                loss_ba_db=2.0,
                detector_efficiency=0.6,
                dark_count_probability=1e-5,
            )
        )
        self.assertAlmostEqual(
            result.detection_probability,
            result.expected_total_click_probability,
            delta=0.001,
        )

    def test_reproducibility(self) -> None:
        config = SimulationConfig(pulses=10_000, seed=14)
        first = simulate(config)
        second = simulate(replace(config))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
