"""Simulador Monte Carlo de Loop-Back QKD transparente A_TX -> B -> A_RX.

Bob no mide, no refleja y no regenera el pulso. En cada ranura temporal,
selecciona un bit y una base BB84, fija la polarización del pulso que lo
atraviesa y lo reenvía a la interfaz receptora de Alice.

El modelo incluye:
  * fuente de fotón único o pulsos coherentes débiles;
  * pérdidas independientes A->B, inserción de Bob y B->A;
  * errores angulares de codificación y del canal de retorno;
  * eficiencia y cuentas oscuras de dos detectores umbral;
  * dobles clics resueltos aleatoriamente;
  * ataque interceptar--reenviar en una fracción configurable del canal B->A.

La estimación de clave usa 1-2 h2(Q) únicamente como diagnóstico asintótico.
No constituye una prueba de seguridad para pulsos coherentes sin estados señuelo.




A SimulationConfig se le agrega el parametro atmospheric_error_e, que representa la probabilidad de error 'e' del canal despolarizante isotrópico introducidos por la atmósfera en el canal. 

Tambien se ve presente la dispersion de Rayleigh (proporcional a 1/lambda^4), la cual considerando un lamnda de 850 nm se encontraron los siguientes articulos que presentan los siguientes QBER a pocos centimetros de distancia:
arXiv:2207.12424 50cm 2.4%
Chun, H., et al. (2017) — "Handheld free space quantum key distribution with dynamic motion compensation" (Optics Express, Vol. 25, Issue 6) 50cm 3%
"Stable, low-error, and calibration-free polarization encoder for free-space quantum communication"  0.5%


"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from math import log2
from typing import Any

import numpy as np


BB84_ANGLES_DEG = np.array(
    [
        [0.0, 90.0],   # Base Z: H, V
        [45.0, 135.0], # Base X: D, A
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Parámetros de una corrida Monte Carlo."""

    pulses: int = 200_000
    seed: int = 20260730
    source_model: str = "weak_coherent"
    mean_photon_number: float = 0.1
    loss_ab_db: float = 3.0
    bob_insertion_loss_db: float = 2.0
    loss_ba_db: float = 3.0
    detector_efficiency: float = 0.55 ## Revisar bien la grafica https://www.excelitas.com/product/spcm-aqrh file:///C:/Users/vicen/Downloads/Excelitas%20SPCM-AQRH%20Family%20datasheet.pdf
    dark_count_probability: float = 4e-8 ## https://opg.optica.org/ol/fulltext.cfm?uri=ol-49-18-5252
    bob_encoding_error_std_deg: float = 0.14 #Absolute accuracy https://media.thorlabs.com/globalassets/items/k/k1/k10/k10cr2_m/etn064249-d02.pdf?v=0116115646
    return_channel_error_std_deg: float = 0.14
    attack_fraction: float = 0.0

    # Reservado para el futuro modelo de despolarización atmosférica.
    atmospheric_error_e: float = 0.0

    def validate(self) -> None:
        if self.pulses <= 0:
            raise ValueError("pulses debe ser positivo")
        if self.source_model not in {"single_photon", "weak_coherent"}:
            raise ValueError(
                "source_model debe ser 'single_photon' o 'weak_coherent'"
            )
        if self.mean_photon_number < 0:
            raise ValueError("mean_photon_number no puede ser negativo")
        for name in ("loss_ab_db", "bob_insertion_loss_db", "loss_ba_db"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} no puede ser negativo")
        for name in (
            "detector_efficiency",
            "dark_count_probability",
            "attack_fraction",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} debe estar entre 0 y 1")
        for name in (
            "bob_encoding_error_std_deg",
            "return_channel_error_std_deg",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} no puede ser negativo")

    @property
    def optical_transmission(self) -> float:
        """Transmisión óptica A->B->A antes de la eficiencia del detector."""

        total_loss_db = (
            self.loss_ab_db
            + self.bob_insertion_loss_db
            + self.loss_ba_db
        )
        return 10.0 ** (-total_loss_db / 10.0)

    @property
    def total_detection_efficiency(self) -> float:
        """Transmisión óptica total multiplicada por la eficiencia del detector."""

        return self.optical_transmission * self.detector_efficiency


@dataclass(frozen=True)
class SimulationResult:
    """Métricas observadas y expectativas analíticas elementales."""

    pulses: int
    detections: int
    sifted_bits: int
    errors: int
    detection_probability: float
    sifted_fraction_of_pulses: float
    basis_match_fraction_among_detections: float
    qber: float
    asymptotic_secret_fraction: float
    diagnostic_secret_bits: float
    optical_transmission: float
    total_detection_efficiency: float
    expected_signal_click_probability: float
    expected_total_click_probability: float
    attack_fraction: float
    source_model: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def binary_entropy(probability: float) -> float:
    """Entropía binaria h2(p), con tratamiento de los extremos."""

    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * log2(probability) - (1.0 - probability) * log2(
        1.0 - probability
    )


def _signal_photons_detected(
    config: SimulationConfig, rng: np.random.Generator
) -> np.ndarray:
    """Número de fotones señal detectables por ranura antes del PBS."""

    eta = config.total_detection_efficiency
    if config.source_model == "single_photon":
        return (rng.random(config.pulses) < eta).astype(np.int16)

    # El adelgazamiento de una distribución de Poisson vuelve a ser Poisson.
    return rng.poisson(
        config.mean_photon_number * eta,
        size=config.pulses,
    ).astype(np.int16)


def _expected_signal_click_probability(config: SimulationConfig) -> float:
    eta = config.total_detection_efficiency
    if config.source_model == "single_photon":
        return eta
    return 1.0 - np.exp(-config.mean_photon_number * eta)


def simulate(config: SimulationConfig) -> SimulationResult:
    """Ejecuta una simulación vectorizada del enlace transparente A-B-A."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    n = config.pulses

    # Bob elige el símbolo BB84 que imprime sobre el pulso que lo atraviesa.
    bob_bits = rng.integers(0, 2, size=n, dtype=np.int8)
    bob_bases = rng.integers(0, 2, size=n, dtype=np.int8)
    alice_bases = rng.integers(0, 2, size=n, dtype=np.int8)
    polarization_angle = BB84_ANGLES_DEG[bob_bases, bob_bits].copy()

    if config.bob_encoding_error_std_deg:
        polarization_angle += rng.normal(
            0.0,
            config.bob_encoding_error_std_deg,
            size=n,
        )

    # Eve actúa sólo sobre el tramo B->A y reenvía el estado que obtuvo.
    attacked = rng.random(n) < config.attack_fraction
    if np.any(attacked):
        eve_bases = rng.integers(0, 2, size=n, dtype=np.int8)
        eve_zero_angles = 45.0 * eve_bases
        eve_p_zero = np.cos(
            np.deg2rad(polarization_angle - eve_zero_angles)
        ) ** 2
        eve_bits = (rng.random(n) >= eve_p_zero).astype(np.int8)
        eve_angles = BB84_ANGLES_DEG[eve_bases, eve_bits]
        polarization_angle = np.where(attacked, eve_angles, polarization_angle)

    if config.return_channel_error_std_deg:
        polarization_angle += rng.normal(
            0.0,
            config.return_channel_error_std_deg,
            size=n,
        )

    ##########################################################
    #                                                        #
    # Implementación de un canal despolarizante isotrópico   #
    #                                                        #
    ##########################################################

    # atmospheric_error_e queda reservado hasta definir el modelo atmosférico.

    detected_photons = _signal_photons_detected(config, rng)

    # Alice mide en Z (eje 0 grados) o X (eje 45 grados).
    alice_zero_angles = 45.0 * alice_bases
    p_detector_zero = np.cos(
        np.deg2rad(polarization_angle - alice_zero_angles)
    ) ** 2
    photons_at_zero = rng.binomial(detected_photons, p_detector_zero)
    photons_at_one = detected_photons - photons_at_zero

    click_zero = photons_at_zero > 0
    click_one = photons_at_one > 0

    if config.dark_count_probability:
        click_zero |= rng.random(n) < config.dark_count_probability
        click_one |= rng.random(n) < config.dark_count_probability

    detections_mask = click_zero | click_one
    double_clicks = click_zero & click_one

    alice_bits = np.zeros(n, dtype=np.int8)
    alice_bits[click_one & ~click_zero] = 1
    if np.any(double_clicks):
        alice_bits[double_clicks] = rng.integers(
            0,
            2,
            size=int(np.count_nonzero(double_clicks)),
            dtype=np.int8,
        )

    sifted_mask = detections_mask & (alice_bases == bob_bases)
    error_mask = sifted_mask & (alice_bits != bob_bits)

    detections = int(np.count_nonzero(detections_mask))
    sifted_bits = int(np.count_nonzero(sifted_mask))
    errors = int(np.count_nonzero(error_mask))
    qber = errors / sifted_bits if sifted_bits else float("nan")
    matched_fraction = sifted_bits / detections if detections else float("nan")

    if np.isnan(qber) or qber > 0.5:
        secret_fraction = 0.0
    else:
        secret_fraction = max(0.0, 1.0 - 2.0 * binary_entropy(qber))

    p_signal = float(_expected_signal_click_probability(config))
    p_dark_any = 1.0 - (1.0 - config.dark_count_probability) ** 2
    p_total_expected = 1.0 - (1.0 - p_signal) * (1.0 - p_dark_any)

    return SimulationResult(
        pulses=n,
        detections=detections,
        sifted_bits=sifted_bits,
        errors=errors,
        detection_probability=detections / n,
        sifted_fraction_of_pulses=sifted_bits / n,
        basis_match_fraction_among_detections=matched_fraction,
        qber=qber,
        asymptotic_secret_fraction=secret_fraction,
        diagnostic_secret_bits=sifted_bits * secret_fraction,
        optical_transmission=config.optical_transmission,
        total_detection_efficiency=config.total_detection_efficiency,
        expected_signal_click_probability=p_signal,
        expected_total_click_probability=p_total_expected,
        attack_fraction=config.attack_fraction,
        source_model=config.source_model,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulación Monte Carlo de Loop-Back QKD transparente A-B-A"
    )
    parser.add_argument("--pulses", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--source",
        dest="source_model",
        choices=("single_photon", "weak_coherent"),
        default="weak_coherent",
    )
    parser.add_argument("--mu", dest="mean_photon_number", type=float, default=0.1)
    parser.add_argument("--loss-ab-db", type=float, default=0.0)
    parser.add_argument("--bob-loss-db", dest="bob_insertion_loss_db", type=float, default=0.0)
    parser.add_argument("--loss-ba-db", type=float, default=0.0)
    parser.add_argument("--detector-efficiency", type=float, default=0.55)
    parser.add_argument("--dark-probability", dest="dark_count_probability", type=float, default=4e-8)
    parser.add_argument("--bob-error-deg", dest="bob_encoding_error_std_deg", type=float, default=0.14)
    parser.add_argument("--channel-error-deg", dest="return_channel_error_std_deg", type=float, default=0.0)
    parser.add_argument("--attack-fraction", type=float, default=0.0)
    parser.add_argument(
        "--actuator",
        default="ideal",
        help="solo 'ideal'; para Kinesis usar sequence_loopback.py",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    actuator = vars(args).pop("actuator")
    if actuator != "ideal":
        raise SystemExit(
            f"loopback_qkd.py no tiene eje de tiempo y solo admite --actuator ideal "
            f"(se pidió '{actuator}'). Para probar tiempos con Kinesis usa:\n"
            f"  python sequence_loopback.py --pulses 10 --actuator kinesis"
        )
    print("[ACTUADOR] Modo IDEAL: Kinesis NO se usa (simulación vectorizada sin tiempos).",
          file=sys.stderr)
    config = SimulationConfig(**vars(args))
    result = simulate(config)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


