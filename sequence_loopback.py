"""Demostración A_TX -> B_transparente -> A_RX usando el núcleo de SeQUeNCe.

Esta versión confirma explícitamente que Bob reenvía el mismo objeto Photon:
no hay espejo, medición, fuente secundaria ni regeneración del pulso.

El modelo de SeQUeNCe de este archivo usa fotones individuales. Para pulsos
coherentes débiles, cuentas oscuras, dobles clics y barridos rápidos, véase
loopback_qkd.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

from sequence.components.optical_channel import QuantumChannel
from sequence.components.photon import Photon
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node
from sequence.utils.encoding import polarization

from actuator import DEFAULT_ROUND_PERIOD_S, banner, make_actuator
from sync import (DEFAULT_MIRROR_SERIAL, DEFAULT_PATTERN_S, IdealMirror, KinesisMirror,
                  SimulatedPhotodiode, run_preamble)


@dataclass(frozen=True)
class SequenceResult:
    pulses: int
    photons_received_by_bob: int
    photons_forwarded_by_bob: int
    photons_received_by_alice: int
    same_photon_objects_at_alice: int
    detections: int
    sifted_bits: int
    errors: int
    qber: float
    sifted_fraction_of_detections: float
    attack_fraction: float


class AliceTransmitter(Node):
    def __init__(self, name, timeline, destination, seed):
        super().__init__(name, timeline, seed=seed)
        self.destination = destination
        self.emitted = 0

    def emit(self, pulse_index: int) -> None:
        photon = Photon(
            name=f"pulse_{pulse_index}",
            timeline=self.timeline,
            wavelength=1550,
            location=self,
            encoding_type=polarization,
            quantum_state=(complex(1), complex(0)),
            use_qm=False,
        )
        photon.pulse_index = pulse_index
        photon.source_object_id = id(photon)
        self.emitted += 1
        self.send_qubit(self.destination, photon)


class TransparentBob(Node):
    """Codifica y reenvía; nunca mide ni sustituye el objeto Photon."""

    def __init__(self, name, timeline, destination, insertion_transmission, seed,
                 actuator=None):
        super().__init__(name, timeline, seed=seed)
        self.destination = destination
        self.insertion_transmission = insertion_transmission
        self.received = 0
        self.forwarded = 0
        self.actuator = actuator
        self._pending = None  # (base, bit) elegidos antes de que llegue el pulso

    def choose_basis(self):
        """Modo con tiempos: Bob elige y gira la Placa 2 ANTES de que llegue el pulso."""
        rng = self.get_generator()
        bob_basis = int(rng.integers(0, 2))
        bob_bit = int(rng.integers(0, 2))
        self._pending = (bob_basis, bob_bit)
        return self.actuator.set_sign(+1 if bob_basis == 0 else -1)

    def receive_qubit(self, src: str, qubit: Photon) -> None:
        self.received += 1
        rng = self.get_generator()
        if self._pending is None:  # modo ideal: igual que el original
            bob_basis = int(rng.integers(0, 2))
            bob_bit = int(rng.integers(0, 2))
        else:
            bob_basis, bob_bit = self._pending
            self._pending = None

        # Polarización transparente del pulso recibido.
        qubit.set_state(polarization["bases"][bob_basis][bob_bit])
        qubit.bob_basis = bob_basis
        qubit.bob_bit = bob_bit
        qubit.bob_object_id = id(qubit)

        if rng.random() <= self.insertion_transmission:
            self.forwarded += 1
            self.send_qubit(self.destination, qubit)


class AliceReceiver(Node):
    def __init__(self, name, timeline, detector_efficiency, attack_fraction, seed):
        super().__init__(name, timeline, seed=seed)
        self.detector_efficiency = detector_efficiency
        self.attack_fraction = attack_fraction
        self.received = 0
        self.same_object = 0
        self.detected = 0
        self.sifted = 0
        self.errors = 0

    def receive_qubit(self, src: str, qubit: Photon) -> None:
        self.received += 1
        if (
            id(qubit) == qubit.source_object_id
            and id(qubit) == qubit.bob_object_id
        ):
            self.same_object += 1

        rng = self.get_generator()

        # Eve intercepta y reenvía en el tramo B->A, cuando está habilitada.
        if rng.random() < self.attack_fraction:
            eve_basis = int(rng.integers(0, 2))
            eve_bit = Photon.measure(
                polarization["bases"][eve_basis],
                qubit,
                rng,
            )
            qubit.set_state(polarization["bases"][eve_basis][eve_bit])

        if rng.random() > self.detector_efficiency:
            return

        alice_basis = int(rng.integers(0, 2))
        alice_bit = Photon.measure(
            polarization["bases"][alice_basis],
            qubit,
            rng,
        )
        self.detected += 1

        if alice_basis == qubit.bob_basis:
            self.sifted += 1
            self.errors += int(alice_bit != qubit.bob_bit)


def _sleep_until(t: float) -> None:
    remaining = t - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _run_rounds(timeline, alice_tx, bob, alice_rx, max_pulses, pulse_period_ps,
                timed, round_period_s, t0_alice, t0_bob, key_bits) -> None:
    """Pulso a pulso. Con tiempos: rondas fijas, Bob gira al inicio y Alice dispara al final,
    cada uno según su propio inicio (t0_bob, t0_alice).
    Con key_bits: se detiene cuando Alice junta esa cantidad de bits filtrados."""
    if timed:
        n_est = 2 * key_bits if key_bits is not None else max_pulses
        goal = f"hasta {key_bits} bits de clave" if key_bits is not None else f"{max_pulses} rondas"
        wait = max(0.0, t0_bob - time.perf_counter())
        print(f"[RONDAS] Inicio en {wait:.1f} s; {goal}, rondas de {round_period_s:.2f} s "
              f"(~{(wait + n_est * round_period_s) / 60:.1f} min).", file=sys.stderr)
    for k in range(max_pulses):
        if key_bits is not None and alice_rx.sifted >= key_bits:
            break
        if timed:
            _sleep_until(t0_bob + k * round_period_s)            # inicio de la ronda k (Bob)
            move = bob.choose_basis()                            # Bob gira (o no) la Placa 2
            _sleep_until(t0_alice + (k + 1) * round_period_s)    # fin de la ronda: Alice dispara
        timeline.schedule(Event(k * pulse_period_ps, Process(alice_tx, "emit", [k])))
        timeline.run()
        if timed:
            bits = f"{alice_rx.sifted}/{key_bits}" if key_bits is not None else f"{alice_rx.sifted}"
            print(f"  ronda {k + 1}: Placa 2 a {move.target_deg:+.2f}°, motor {move.elapsed_s:.2f} s, "
                  f"bits {bits}{'  [AVISO] ATRASADO' if move.late else ''}", file=sys.stderr)
    if key_bits is not None and alice_rx.sifted < key_bits:
        print(f"[AVISO] Se alcanzó el máximo de {max_pulses} pulsos con solo "
              f"{alice_rx.sifted}/{key_bits} bits.", file=sys.stderr)


def run_sequence_demo(
    pulses: int = 50_000,
    seed: int = 20260730,
    loss_ab_db: float = 0.0,
    bob_insertion_loss_db: float = 0.0,
    loss_ba_db: float = 0.0,
    detector_efficiency: float = 1.0,
    attack_fraction: float = 0.0,
    segment_distance_m: float = 1_000.0,
    pulse_rate_hz: float = 1e6,
    actuator=None,
    round_period_s: float = DEFAULT_ROUND_PERIOD_S,
    start_delay_s: float = 5.0,
    key_bits: int | None = None,
    start_times: tuple[float, float] | None = None,
) -> SequenceResult:
    """start_times = (t0_alice, t0_bob) del preámbulo; si es None, ambos parten en
    ahora + start_delay_s."""
    if pulses <= 0:
        raise ValueError("pulses debe ser positivo")
    if key_bits is not None and key_bits <= 0:
        raise ValueError("key_bits debe ser positivo")
    if segment_distance_m <= 0 or pulse_rate_hz <= 0:
        raise ValueError("distancia y frecuencia deben ser positivas")
    if not 0 <= detector_efficiency <= 1:
        raise ValueError("detector_efficiency debe estar entre 0 y 1")
    if not 0 <= attack_fraction <= 1:
        raise ValueError("attack_fraction debe estar entre 0 y 1")

    timed = actuator is not None and actuator.name != "ideal"
    # Con tiempos, el tiempo virtual de SeQUeNCe también avanza un plazo de ronda por pulso.
    pulse_period_ps = round(round_period_s * 1e12) if timed else round(1e12 / pulse_rate_hz)
    if key_bits is not None:
        # Pulso a pulso: cada pulso debe terminar su viaje A->B->A antes del siguiente.
        pulse_period_ps = max(pulse_period_ps, int(2 * segment_distance_m / 0.0002) + 1)
    # Margen amplio para los dos tiempos de propagación.
    stop_time = int(
        (pulses + 2) * pulse_period_ps
        + 2 * segment_distance_m / 0.0002
    )
    timeline = Timeline(stop_time=stop_time)

    alice_tx = AliceTransmitter("Alice_TX", timeline, "Bob", seed)
    bob = TransparentBob(
        "Bob",
        timeline,
        "Alice_RX",
        insertion_transmission=10 ** (-bob_insertion_loss_db / 10),
        seed=seed + 1,
        actuator=actuator,
    )
    alice_rx = AliceReceiver(
        "Alice_RX",
        timeline,
        detector_efficiency=detector_efficiency,
        attack_fraction=attack_fraction,
        seed=seed + 2,
    )

    channel_ab = QuantumChannel(
        "Canal_A_B",
        timeline,
        attenuation=loss_ab_db / segment_distance_m,
        distance=segment_distance_m,
        polarization_fidelity=1.0,
        frequency=pulse_rate_hz,
    )
    channel_ba = QuantumChannel(
        "Canal_B_A",
        timeline,
        attenuation=loss_ba_db / segment_distance_m,
        distance=segment_distance_m,
        polarization_fidelity=1.0,
        frequency=pulse_rate_hz,
    )
    channel_ab.set_ends(alice_tx, bob.name)
    channel_ba.set_ends(bob, alice_rx.name)

    timeline.init()
    if timed or key_bits is not None:
        if start_times is None:
            t0 = time.perf_counter() + start_delay_s
            start_times = (t0, t0)
        _run_rounds(timeline, alice_tx, bob, alice_rx, pulses, pulse_period_ps,
                    timed, round_period_s, start_times[0], start_times[1], key_bits)
    else:  # original, sin cambios
        for pulse_index in range(pulses):
            timeline.schedule(
                Event(
                    pulse_index * pulse_period_ps,
                    Process(alice_tx, "emit", [pulse_index]),
                )
            )
        timeline.run()

    qber = alice_rx.errors / alice_rx.sifted if alice_rx.sifted else float("nan")
    sift_fraction = (
        alice_rx.sifted / alice_rx.detected
        if alice_rx.detected
        else float("nan")
    )
    return SequenceResult(
        pulses=alice_tx.emitted,
        photons_received_by_bob=bob.received,
        photons_forwarded_by_bob=bob.forwarded,
        photons_received_by_alice=alice_rx.received,
        same_photon_objects_at_alice=alice_rx.same_object,
        detections=alice_rx.detected,
        sifted_bits=alice_rx.sifted,
        errors=alice_rx.errors,
        qber=qber,
        sifted_fraction_of_detections=sift_fraction,
        attack_fraction=attack_fraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demostración SeQUeNCe del Bob transparente"
    )
    parser.add_argument("--pulses", type=int, default=None,
                        help="pulsos a enviar (50000 por defecto); con --key-bits es el máximo")
    parser.add_argument("--key-bits", type=int, default=None,
                        help="seguir hasta que Alice tenga N bits filtrados (p. ej. 128 o 256)")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--loss-ab-db", type=float, default=0.0)
    parser.add_argument("--bob-loss-db", type=float, default=0.0)
    parser.add_argument("--loss-ba-db", type=float, default=0.0)
    parser.add_argument("--detector-efficiency", type=float, default=1.0)
    parser.add_argument("--attack-fraction", type=float, default=0.0)
    parser.add_argument("--actuator", choices=["ideal", "kinesis"], default="ideal")
    parser.add_argument("--round-period-s", type=float, default=DEFAULT_ROUND_PERIOD_S)
    parser.add_argument("--start-delay-s", type=float, default=5.0)
    parser.add_argument("--mirror-serial", default=DEFAULT_MIRROR_SERIAL,
                        help="serie del MFF101 (espejo del preámbulo)")
    parser.add_argument("--preamble-pattern", default=",".join(str(x) for x in DEFAULT_PATTERN_S),
                        help="tiempos del patrón del preámbulo en s, p. ej. 0,0.1,0.25")
    parser.add_argument("--photodiode-latency-s", type=float, default=0.001)
    parser.add_argument("--real", action="store_true",
                        help="usar el K10CR2 físico, no el simulador de Kinesis")
    args = parser.parse_args()
    if args.pulses is None:
        args.pulses = 20 * args.key_bits if args.key_bits is not None else 50_000

    kw = {"use_simulation": not args.real} if args.actuator == "kinesis" else {}
    actuator = make_actuator(args.actuator, args.round_period_s, **kw)
    print(banner(actuator), file=sys.stderr)
    timed = actuator.name != "ideal"
    mirror = (KinesisMirror(args.mirror_serial, use_simulation=not args.real)
              if timed else IdealMirror())
    sync_report = None
    start_times = None
    actuator.open()
    try:
        if timed:
            mirror.open()
            pattern = tuple(float(x) for x in args.preamble_pattern.split(","))
            print(f"[SYNC] Espejo MFF101 {args.mirror_serial} puesto; Alice envía el preámbulo "
                  f"{list(pattern)} s.", file=sys.stderr)
            sync_report, t0_alice, t0_bob = run_preamble(
                mirror, SimulatedPhotodiode(args.photodiode_latency_s),
                args.start_delay_s, pattern)
            start_times = (t0_alice, t0_bob)
            mirror.close()  # el espejo ya no se usa; si sigue conectado frena al K10CR2
            print(f"[SYNC] Patrón reconocido; desfase Bob-Alice "
                  f"{sync_report.offset_s * 1e3:.2f} ms; espejo quitado en "
                  f"{sync_report.mirror_out_s:.2f} s; margen {sync_report.margin_s:.2f} s.",
                  file=sys.stderr)
        result = run_sequence_demo(
            pulses=args.pulses,
            seed=args.seed,
            loss_ab_db=args.loss_ab_db,
            bob_insertion_loss_db=args.bob_loss_db,
            loss_ba_db=args.loss_ba_db,
            detector_efficiency=args.detector_efficiency,
            attack_fraction=args.attack_fraction,
            actuator=actuator,
            round_period_s=args.round_period_s,
            start_delay_s=args.start_delay_s,
            key_bits=args.key_bits,
            start_times=start_times,
        )
    finally:
        mirror.close()      # primero el espejo: el actuador es dueño del simulador de Kinesis
        actuator.close()
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    if timed:
        print(json.dumps({"actuador": actuator.status(),
                          "sincronizacion": sync_report.as_dict()},
                         indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
