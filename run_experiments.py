"""Ejecuta los casos de validación y genera tablas y gráficas."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "loopback-qkd-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loopback_qkd import SimulationConfig, simulate


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    ideal_cfg = SimulationConfig(
        pulses=200_000,
        seed=101,
        source_model="single_photon",
        loss_ab_db=0.0,
        bob_insertion_loss_db=0.0,
        loss_ba_db=0.0,
        detector_efficiency=1.0,
        dark_count_probability=0.0,
        bob_encoding_error_std_deg=0.0,
    )
    eve_cfg = replace(
        ideal_cfg,
        pulses=500_000,
        seed=102,
        attack_fraction=1.0,
    )
    realistic_cfg = SimulationConfig(
        pulses=1_000_000,
        seed=103,
        source_model="weak_coherent",
        mean_photon_number=0.1,
        loss_ab_db=3.0,
        bob_insertion_loss_db=2.0,
        loss_ba_db=3.0,
        detector_efficiency=0.25,
        dark_count_probability=1e-5,
        bob_encoding_error_std_deg=1.0,
        return_channel_error_std_deg=1.0,
    )
    realistic_test_cfg = SimulationConfig(
        pulses=1_000_000,
        seed=104,
        source_model="weak_coherent",
        mean_photon_number=0.1,
    )

    scenarios = [
        ("ideal", ideal_cfg, simulate(ideal_cfg)),
        ("eve_intercepta_100", eve_cfg, simulate(eve_cfg)),
        ("realista_ejemplo", realistic_cfg, simulate(realistic_cfg)),
        ("realista_test", realistic_test_cfg, simulate(realistic_test_cfg)),
    ]
    summary_rows = []
    for name, config, result in scenarios:
        row = {"scenario": name}
        row.update(asdict(config))
        row.update({f"result_{key}": value for key, value in asdict(result).items()})
        summary_rows.append(row)
    write_csv(RESULTS / "summary.csv", summary_rows)

    attack_rows = []
    for index, fraction in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        config = replace(
            ideal_cfg,
            pulses=300_000,
            seed=200 + index,
            attack_fraction=fraction,
        )
        result = simulate(config)
        attack_rows.append(
            {
                "attack_fraction": fraction,
                "qber_simulated": result.qber,
                "qber_ideal_expected": 0.25 * fraction,
                "sifted_bits": result.sifted_bits,
            }
        )
    write_csv(RESULTS / "attack_sweep.csv", attack_rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fractions = [row["attack_fraction"] for row in attack_rows]
    ax.plot(
        fractions,
        [row["qber_simulated"] for row in attack_rows],
        "o-",
        label="Monte Carlo",
    )
    ax.plot(
        fractions,
        [row["qber_ideal_expected"] for row in attack_rows],
        "--",
        label=r"Esperado: $Q=0.25f_E$",
    )
    ax.set(
        xlabel="Fracción de pulsos interceptados por Eve",
        ylabel="QBER",
        title="Validación del ataque interceptar–reenviar",
        xlim=(-0.02, 1.02),
        ylim=(-0.01, 0.28),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "qber_attack_sweep.png", dpi=180)
    plt.close(fig)

    loss_base = SimulationConfig(
        pulses=700_000,
        seed=300,
        source_model="weak_coherent",
        mean_photon_number=0.1,
        detector_efficiency=0.6,
    )
    loss_rows = []
    for index, segment_loss_db in enumerate((0, 1, 2, 3, 4, 5)):
        config = replace(
            loss_base,
            seed=300 + index,
            loss_ab_db=float(segment_loss_db),
            loss_ba_db=float(segment_loss_db),
        )
        result = simulate(config)
        loss_rows.append(
            {
                "loss_per_segment_db": segment_loss_db,
                "total_path_loss_db": 2 * segment_loss_db,
                "click_probability_simulated": result.detection_probability,
                "click_probability_expected": result.expected_total_click_probability,
                "sifted_bits": result.sifted_bits,
                "qber": result.qber,
            }
        )
    write_csv(RESULTS / "loss_sweep.csv", loss_rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    path_loss = [row["total_path_loss_db"] for row in loss_rows]
    ax.semilogy(
        path_loss,
        [row["click_probability_simulated"] for row in loss_rows],
        "o-",
        label="Monte Carlo",
    )
    ax.semilogy(
        path_loss,
        [row["click_probability_expected"] for row in loss_rows],
        "--",
        label="Modelo Poisson",
    )
    ax.set(
        xlabel="Pérdida óptica total A→B→A (dB)",
        ylabel="Probabilidad de clic por pulso",
        title="Efecto de las pérdidas del lazo transparente",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "click_loss_sweep.png", dpi=180)
    plt.close(fig)

    ideal = scenarios[0][2]
    eve = scenarios[1][2]
    realistic = scenarios[2][2]
    checks = {
        "QBER ideal exactamente cero": ideal.qber == 0.0,
        "Sifting ideal dentro de 0.49--0.51": 0.49
        <= ideal.sifted_fraction_of_pulses
        <= 0.51,
        "QBER con Eve dentro de 0.24--0.26": 0.24 <= eve.qber <= 0.26,
        "Barrido de Eve concuerda dentro de 0.01": all(
            abs(row["qber_simulated"] - row["qber_ideal_expected"]) < 0.01
            for row in attack_rows
        ),
        "Barrido de pérdida concuerda dentro de 0.001": all(
            abs(
                row["click_probability_simulated"]
                - row["click_probability_expected"]
            )
            < 0.001
            for row in loss_rows
        ),
    }

    lines = [
        "VALIDACIÓN DE LA SIMULACIÓN LOOP-BACK QKD TRANSPARENTE",
        "=" * 60,
        "",
        "Arquitectura: Alice_TX -> Bob_transparente -> Alice_RX",
        "Bob modifica la polarización y reenvía el pulso; no refleja ni mide.",
        "",
        f"Ideal: detecciones={ideal.detections}, sifted={ideal.sifted_bits}, "
        f"QBER={ideal.qber:.6f}",
        f"Eve 100%: detecciones={eve.detections}, sifted={eve.sifted_bits}, "
        f"QBER={eve.qber:.6f}",
        f"Ejemplo realista: detecciones={realistic.detections}, "
        f"sifted={realistic.sifted_bits}, QBER={realistic.qber:.6f}",
        "",
        "Comprobaciones:",
    ]
    lines.extend(
        f"  [{'OK' if passed else 'FALLO'}] {description}"
        for description, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "Nota de seguridad:",
            "La fracción 1-2 h2(Q) es sólo un diagnóstico asintótico.",
            "No prueba seguridad de pulsos coherentes sin estados señuelo.",
        ]
    )
    (RESULTS / "validation_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    if not all(checks.values()):
        raise SystemExit("Una o más comprobaciones fallaron")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
