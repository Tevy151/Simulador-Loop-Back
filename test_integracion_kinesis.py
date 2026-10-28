"""Tests de la integración con el actuador (K10CR2) y la sincronización (MFF101).

No requieren Kinesis: usan los modos ideales y un actuador falso rápido.
Ejecutar:  python -m unittest -v test_integracion_kinesis.py
"""

import contextlib
import io
import subprocess
import sys
import unittest
from dataclasses import asdict

from actuator import IdealActuator, MoveReport, make_actuator, plate2_angle_deg
from sequence_loopback import run_sequence_demo
from sync import IdealMirror, SimulatedPhotodiode, match_pattern, run_preamble

# Salida del sequence_loopback.py ORIGINAL (antes de la integración), 2000 pulsos, seed por defecto.
ORIGINAL_IDEAL = {"sifted_bits": 979, "errors": 0, "detections": 2000}
ORIGINAL_EVE = {"sifted_bits": 1008, "errors": 261, "detections": 2000}


class FastFakeActuator:
    """Actuador con tiempos pero instantáneo: activa el modo por rondas sin Kinesis."""

    name = "fake"

    def __init__(self):
        self.moves = 0

    def set_sign(self, sign):
        self.moves += 1
        t = plate2_angle_deg(sign)
        return MoveReport(t, t, 0.0, True, False)


class ActuatorTests(unittest.TestCase):
    def test_plate2_angles(self):
        self.assertEqual(plate2_angle_deg(+1), 11.25)
        self.assertEqual(plate2_angle_deg(-1), -11.25)
        self.assertEqual(plate2_angle_deg(+1, bob_rotation_deg=45.0), 22.5)
        with self.assertRaises(ValueError):
            plate2_angle_deg(0)

    def test_ideal_reports_no_kinesis(self):
        act = make_actuator("ideal")
        self.assertIsInstance(act, IdealActuator)
        act.set_sign(-1)
        status = act.status()
        self.assertFalse(status["kinesis_active"])
        self.assertEqual(status["actuator_backend"], "ideal")


class IdealUnchangedTests(unittest.TestCase):
    def test_default_matches_original(self):
        for attack, expected in ((0.0, ORIGINAL_IDEAL), (1.0, ORIGINAL_EVE)):
            r = asdict(run_sequence_demo(pulses=2000, attack_fraction=attack))
            for key, value in expected.items():
                self.assertEqual(r[key], value, f"attack={attack} {key}")

    def test_ideal_actuator_matches_original(self):
        r = asdict(run_sequence_demo(pulses=2000, actuator=IdealActuator()))
        self.assertEqual(r["sifted_bits"], ORIGINAL_IDEAL["sifted_bits"])

    def test_timed_rounds_same_physics(self):
        # Sin pérdidas, el modo por rondas saca los aleatorios en el mismo orden que el original.
        with contextlib.redirect_stderr(io.StringIO()):  # silencia las líneas de ronda
            timed = run_sequence_demo(pulses=200, actuator=FastFakeActuator(),
                                      round_period_s=0.001, start_delay_s=0.0)
        ideal = run_sequence_demo(pulses=200)
        self.assertEqual(asdict(timed), asdict(ideal))
        self.assertEqual(timed.same_photon_objects_at_alice, 200)


class KeyBitsTests(unittest.TestCase):
    def test_stops_at_key_bits(self):
        r = run_sequence_demo(key_bits=64)
        self.assertEqual(r.sifted_bits, 64)
        self.assertEqual(r.errors, 0)
        self.assertLess(r.pulses, 200)

    def test_eve_detected_with_key_bits(self):
        r = run_sequence_demo(key_bits=256, attack_fraction=1.0)
        self.assertEqual(r.sifted_bits, 256)
        self.assertGreater(r.qber, 0.15)
        self.assertLess(r.qber, 0.35)


class SyncTests(unittest.TestCase):
    PATTERN = (0.0, 0.10, 0.25)

    def test_match_exact(self):
        self.assertEqual(match_pattern([10.0, 10.1, 10.25], self.PATTERN, 0.02), 10.25)

    def test_match_with_spurious_pulse_before(self):
        self.assertEqual(match_pattern([9.3, 10.0, 10.1, 10.25], self.PATTERN, 0.02), 10.25)

    def test_match_with_jitter(self):
        self.assertEqual(match_pattern([10.0, 10.11, 10.24], self.PATTERN, 0.02), 10.24)

    def test_no_match(self):
        self.assertIsNone(match_pattern([10.0, 10.2, 10.4], self.PATTERN, 0.02))
        self.assertIsNone(match_pattern([10.0, 10.1], self.PATTERN, 0.02))

    def test_run_preamble_ideal(self):
        report, t0_alice, t0_bob = run_preamble(
            IdealMirror(), SimulatedPhotodiode(latency_s=0.001), start_delay_s=1.0,
            pattern=self.PATTERN)
        self.assertTrue(report.pattern_detected)
        self.assertGreaterEqual(report.offset_s, 0.0)
        self.assertLess(report.offset_s, 0.05)
        self.assertGreater(report.margin_s, 0.0)
        self.assertAlmostEqual(t0_bob - t0_alice, report.offset_s)


class CliTests(unittest.TestCase):
    def test_vectorized_rejects_kinesis(self):
        proc = subprocess.run(
            [sys.executable, "loopback_qkd.py", "--actuator", "kinesis", "--pulses", "10"],
            capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("sequence_loopback.py", proc.stderr)


if __name__ == "__main__":
    unittest.main()
