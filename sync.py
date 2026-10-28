"""Arranque inicial Alice-Bob: preámbulo óptico + espejo abatible MFF101.

Alice envía un patrón de pulsos brillantes (clásicos). Bob tiene el MFF101
puesto, que desvía la luz al fotodiodo PDA36A2. Al reconocer el patrón fija
t_ref, quita el espejo, y ambos empiezan las rondas en t_ref + T_inicio
(valor acordado de antemano, no se transmite).

Espejo:     ideal (instantáneo, SIN Kinesis) | kinesis (MFF101 vía Kinesis)
Fotodiodo:  simulado por software (en la Raspberry sería un ADC/GPIO).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from actuator import KINESIS_PATH

DEFAULT_PATTERN_S = (0.0, 0.10, 0.25)   # separaciones distintas: difícil de imitar
DEFAULT_MIRROR_SERIAL = "37000001"
MIRROR_IN_POSITION = 1                  # posición del MFF101 con el espejo en el haz
MIRROR_OUT_POSITION = 2
TIMEOUT_FLIP_MS = 10000


class IdealMirror:
    name = "ideal"

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def put_in(self) -> float:
        return 0.0

    def take_out(self) -> float:
        return 0.0


class KinesisMirror:
    """MFF101 vía Kinesis. Si owns_simulation=False, otro objeto (el actuador)
    ya inicializó el simulador de Kinesis y es quien lo cierra."""

    name = "kinesis"

    def __init__(self, serial: str = DEFAULT_MIRROR_SERIAL, use_simulation: bool = True,
                 owns_simulation: bool = False):
        self.serial = serial
        self.use_simulation = use_simulation
        self.owns_simulation = owns_simulation
        self.device = None
        self._sim = None
        self._UInt32 = None

    def open(self) -> None:
        try:
            import clr  # pythonnet
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo importar pythonnet ('clr') para el MFF101: {exc}") from exc
        sys.path.append(KINESIS_PATH)
        clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
        clr.AddReference("Thorlabs.MotionControl.FilterFlipperCLI")
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI, SimulationManager
        from Thorlabs.MotionControl.FilterFlipperCLI import FilterFlipper
        from System import UInt32
        self._UInt32 = UInt32
        if self.use_simulation and self.owns_simulation:
            SimulationManager.Instance.InitializeSimulations()
            self._sim = SimulationManager
        DeviceManagerCLI.BuildDeviceList()
        found = [str(d) for d in DeviceManagerCLI.GetDeviceList()]
        if self.serial not in found:
            self.close()
            raise RuntimeError(
                f"El MFF101 {self.serial} no aparece en la lista de dispositivos: {found}. "
                "¿Está creado en Kinesis Simulator?")
        dev = FilterFlipper.CreateFilterFlipper(self.serial)
        dev.Connect(self.serial)
        dev.WaitForSettingsInitialized(5000)
        dev.StartPolling(250)  # con 50 ms, junto al K10CR2, el simulador se satura (cambios de ~5 s)
        time.sleep(0.5)
        dev.EnableDevice()
        time.sleep(0.5)
        self.device = dev

    def _move(self, position: int) -> float:
        # Si ya está ahí, Kinesis espera un movimiento que no ocurre (~5 s): no mandar nada.
        if int(self.device.Position) == position:
            return 0.0
        t0 = time.perf_counter()
        self.device.SetPosition(self._UInt32(position), TIMEOUT_FLIP_MS)
        elapsed = time.perf_counter() - t0
        if int(self.device.Position) != position:
            print(f"[AVISO] MFF101: se pidió la posición {position} pero informa "
                  f"{self.device.Position}.", file=sys.stderr)
        return elapsed

    def put_in(self) -> float:
        return self._move(MIRROR_IN_POSITION)

    def take_out(self) -> float:
        return self._move(MIRROR_OUT_POSITION)

    def close(self) -> None:
        try:
            if self.device is not None:
                self.device.StopPolling()
                self.device.Disconnect(True)
        finally:
            self.device = None
            if self._sim is not None:
                self._sim.Instance.UninitializeSimulations()
                self._sim = None


class SimulatedPhotodiode:
    """PDA36A2 simulado: entrega la hora en que Bob 've' cada pulso del preámbulo."""

    def __init__(self, latency_s: float = 0.001):
        self.latency_s = latency_s

    def receive(self, send_times: Sequence[float]) -> list[float]:
        seen = []
        for t in send_times:
            remaining = t + self.latency_s - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            seen.append(time.perf_counter())
        return seen


def match_pattern(times: Sequence[float], pattern: Sequence[float],
                  tol_s: float) -> Optional[float]:
    """Busca el patrón (separaciones) en los tiempos detectados. Devuelve la hora del
    último pulso del patrón (t_ref), o None. Tolera pulsos sueltos antes del patrón."""
    n = len(pattern)
    for i in range(len(times) - n + 1):
        if all(abs((times[i + j] - times[i]) - (pattern[j] - pattern[0])) <= tol_s
               for j in range(n)):
            return times[i + n - 1]
    return None


@dataclass
class SyncReport:
    pattern_detected: bool
    pattern_s: tuple
    alice_t_ref: float        # hora en que Alice envió el último pulso del patrón
    bob_t_ref: float          # hora en que Bob lo detectó
    offset_s: float           # desfase Bob - Alice (error de sincronización)
    mirror_out_s: float       # lo que tardó en quitarse el espejo
    margin_s: float           # holgura hasta la ronda 1 (negativo = atrasado)

    def as_dict(self) -> dict:
        return {
            "pattern_detected": self.pattern_detected,
            "pattern_s": list(self.pattern_s),
            "offset_bob_minus_alice_ms": round(self.offset_s * 1e3, 3),
            "mirror_out_s": round(self.mirror_out_s, 3),
            "margin_before_round_1_s": round(self.margin_s, 3),
        }


def run_preamble(mirror, photodiode: SimulatedPhotodiode, start_delay_s: float,
                 pattern: Sequence[float] = DEFAULT_PATTERN_S,
                 tol_s: float = 0.02) -> tuple[SyncReport, float, float]:
    """Ejecuta el arranque. Devuelve (reporte, t0_alice, t0_bob): inicio de la ronda 1
    según el reloj de cada uno."""
    mirror.put_in()
    t_first = time.perf_counter() + 0.5            # Alice empieza a enviar en 0.5 s
    alice_times = [t_first + p for p in pattern]
    bob_times = photodiode.receive(alice_times)
    bob_t_ref = match_pattern(bob_times, pattern, tol_s)
    if bob_t_ref is None:
        raise RuntimeError("Bob no reconoció el patrón del preámbulo.")
    alice_t_ref = alice_times[-1]
    mirror_s = mirror.take_out()
    margin = (bob_t_ref + start_delay_s) - time.perf_counter()
    if margin < 0:
        print(f"[AVISO] Bob no alcanzó a quitar el espejo: atraso de {-margin:.2f} s.",
              file=sys.stderr)
    report = SyncReport(True, tuple(pattern), alice_t_ref, bob_t_ref,
                        bob_t_ref - alice_t_ref, mirror_s, margin)
    return report, alice_t_ref + start_delay_s, bob_t_ref + start_delay_s
