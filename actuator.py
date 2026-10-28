"""Actuador de Bob: Placa 2 (λ/2) montada en el K10CR2.

Modos:
  ideal   : instantáneo, SIN Kinesis (rápido; para barridos y pruebas).
  kinesis : simulador o dispositivo real de Kinesis vía pythonnet (lento).

Mapeo: base 0 -> signo +1 -> Placa 2 en +11.25°
       base 1 -> signo -1 -> Placa 2 en -11.25°
Si se pide kinesis y no se puede, falla con un mensaje claro (no cae a ideal).

Prueba aislada:  python actuator.py --actuator kinesis --moves 5
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from dataclasses import dataclass
from typing import Optional

KINESIS_PATH = r"C:\Program Files\Thorlabs\Kinesis"
DEFAULT_SERIAL = "55000002"
TIMEOUT_MOVE_MS = 60000
DEFAULT_ROUND_PERIOD_S = 3.0      # plazo fijo por ronda (peor caso medido ~2.5 s a 20 °/s)
DEFAULT_BOB_ROTATION_DEG = 22.5   # rotación de Bob; Placa 2 a ±θ/2 = ±11.25°
DEFAULT_MAX_VELOCITY = 20.0       # °/s   (verificar en datasheet K10CR2)
DEFAULT_ACCELERATION = 20.0       # °/s²


def plate2_angle_deg(sign: int, bob_rotation_deg: float = DEFAULT_BOB_ROTATION_DEG) -> float:
    """θ = 2(φ2 - φ1) con Placa 1 en 0°  =>  φ2 = ±θ/2."""
    if sign not in (1, -1):
        raise ValueError("sign debe ser +1 o -1")
    return sign * bob_rotation_deg / 2.0


@dataclass
class MoveReport:
    target_deg: float
    reached_deg: Optional[float]   # None si no se pudo leer
    elapsed_s: float               # tiempo real del movimiento
    position_ok: Optional[bool]    # None si no verificable
    late: bool                     # superó el plazo de la ronda


class IdealActuator:
    """Sin Kinesis: Placa 2 salta instantáneamente."""

    name = "ideal"

    def __init__(self, round_period_s: float = DEFAULT_ROUND_PERIOD_S,
                 bob_rotation_deg: float = DEFAULT_BOB_ROTATION_DEG):
        self.round_period_s = round_period_s
        self.bob_rotation_deg = bob_rotation_deg
        self.moves = 0

    def open(self) -> None:
        pass

    def set_sign(self, sign: int) -> MoveReport:
        target = plate2_angle_deg(sign, self.bob_rotation_deg)
        self.moves += 1
        return MoveReport(target, target, 0.0, True, False)

    def close(self) -> None:
        pass

    def status(self) -> dict:
        return {
            "actuator_backend": "ideal",
            "kinesis_active": False,
            "bob_rotation_deg": self.bob_rotation_deg,
            "moves": self.moves,
            "late_moves": 0,
        }


class KinesisActuator:
    """K10CR2 vía Kinesis (simulado o real). Requiere Windows + Kinesis + pythonnet."""

    name = "kinesis"

    def __init__(self, round_period_s: float = DEFAULT_ROUND_PERIOD_S, serial: str = DEFAULT_SERIAL,
                 use_simulation: bool = True, tol_deg: float = 0.05,
                 bob_rotation_deg: float = DEFAULT_BOB_ROTATION_DEG,
                 max_velocity: float = DEFAULT_MAX_VELOCITY,
                 acceleration: float = DEFAULT_ACCELERATION):
        self.round_period_s = round_period_s
        self.serial = serial
        self.use_simulation = use_simulation
        self.tol_deg = tol_deg
        self.bob_rotation_deg = bob_rotation_deg
        self.max_velocity = max_velocity
        self.acceleration = acceleration
        self.device = None
        self._sim = None
        self._Decimal = None
        self.simulation_initialized = False
        self.moves = 0
        self.late_moves = 0
        self.position_errors = 0
        self.max_elapsed_s = 0.0

    def open(self) -> None:
        try:
            import clr  # pythonnet
        except Exception as exc:
            raise RuntimeError(
                "Se pidió kinesis pero no se pudo importar pythonnet ('clr'): "
                f"{exc}. Instala pythonnet o usa --actuator ideal.") from exc
        sys.path.append(KINESIS_PATH)
        try:
            clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
            clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
            clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
        except Exception as exc:
            raise RuntimeError(
                f"No se encontraron las DLL de Kinesis en {KINESIS_PATH}: {exc}") from exc

        from Thorlabs.MotionControl.DeviceManagerCLI import (
            DeviceManagerCLI, SimulationManager, DeviceConfiguration)
        from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import CageRotator
        from System import Decimal
        self._Decimal = Decimal
        self._sim = SimulationManager

        if self.use_simulation:
            SimulationManager.Instance.InitializeSimulations()
            self.simulation_initialized = True

        DeviceManagerCLI.BuildDeviceList()
        found = [str(d) for d in DeviceManagerCLI.GetDeviceList()]
        if self.serial not in found:
            self.close()
            raise RuntimeError(
                f"El K10CR2 {self.serial} no aparece en la lista de dispositivos: {found}. "
                "¿Está creado en Kinesis Simulator?")

        dev = CageRotator.CreateCageRotator(self.serial)
        dev.Connect(self.serial)
        time.sleep(0.5)
        dev.StartPolling(250)
        time.sleep(0.5)
        dev.EnableDevice()
        time.sleep(0.5)
        cfg = dev.LoadMotorConfiguration(
            self.serial, DeviceConfiguration.DeviceSettingsUseOptionType.UseDeviceSettings)
        # Sin esto Kinesis usa "Default" (escala ×3 respecto del K10CR2 real).
        cfg.DeviceSettingsName = "K10CR2 Rotation Stage"
        cfg.UpdateCurrentConfiguration()
        settings = dev.MotorDeviceSettings
        # Permite ángulos negativos (Placa 2 a ±θ/2) sin cruzar por 360°.
        settings.Physical.MinPosUnit = Decimal(-180)
        settings.Rotation.RotationMode = type(settings.Rotation.RotationMode).LinearRange
        dev.SetSettings(settings, True, False)
        time.sleep(0.25)
        dev.SetVelocityParams(Decimal(self.max_velocity), Decimal(self.acceleration))
        dev.Home(TIMEOUT_MOVE_MS)  # Placa 2 en 0° (referencia)
        self.device = dev

    def _read_position(self) -> Optional[float]:
        try:
            return float(str(self.device.Position).replace(",", "."))  # Windows en español
        except Exception:
            return None

    def set_sign(self, sign: int) -> MoveReport:
        target = plate2_angle_deg(sign, self.bob_rotation_deg)
        t0 = time.perf_counter()
        self.device.MoveTo(self._Decimal(target), TIMEOUT_MOVE_MS)  # bloquea hasta llegar
        elapsed = time.perf_counter() - t0
        self.moves += 1
        self.max_elapsed_s = max(self.max_elapsed_s, elapsed)

        reached = self._read_position()
        ok = None
        if reached is not None:
            diff = abs((reached - target + 180.0) % 360.0 - 180.0)  # módulo 360
            ok = diff <= self.tol_deg
            if not ok:
                self.position_errors += 1
                print(f"[AVISO] Placa 2 pedida a {target:+.2f}° pero quedó en {reached:.2f}°",
                      file=sys.stderr)
        late = elapsed > self.round_period_s  # el aviso lo muestra quien llama (línea de ronda)
        if late:
            self.late_moves += 1
        return MoveReport(target, reached, elapsed, ok, late)

    def close(self) -> None:
        try:
            if self.device is not None:
                self.device.StopPolling()
                self.device.Disconnect(True)
        finally:
            self.device = None
            if self.simulation_initialized and self._sim is not None:
                self._sim.Instance.UninitializeSimulations()
                self.simulation_initialized = False

    def status(self) -> dict:
        return {
            "actuator_backend": "kinesis",
            "kinesis_active": self.moves > 0,
            "kinesis_mode": "simulador de Kinesis" if self.use_simulation else "dispositivo físico",
            "bob_rotation_deg": self.bob_rotation_deg,
            "max_velocity_deg_s": self.max_velocity,
            "acceleration_deg_s2": self.acceleration,
            "moves": self.moves,
            "late_moves": self.late_moves,
            "position_errors": self.position_errors,
            "max_move_elapsed_s": round(self.max_elapsed_s, 3),
        }


def make_actuator(mode: str, round_period_s: float = DEFAULT_ROUND_PERIOD_S, **kw):
    if mode == "ideal":
        return IdealActuator(round_period_s, kw.get("bob_rotation_deg", DEFAULT_BOB_ROTATION_DEG))
    if mode == "kinesis":
        return KinesisActuator(round_period_s, **kw)
    raise ValueError("actuator debe ser 'ideal' o 'kinesis'")


def banner(actuator) -> str:
    if actuator.name != "kinesis":
        return "[ACTUADOR] Modo IDEAL: Kinesis NO se usa. Placa 2 cambia instantáneamente."
    modo = "simulador de Kinesis" if actuator.use_simulation else "dispositivo físico"
    return (f"[ACTUADOR] Modo KINESIS ({modo}), K10CR2 serie {actuator.serial}: "
            "cada ronda espera el movimiento real (lento).")


def main() -> None:
    p = argparse.ArgumentParser(description="Prueba aislada del actuador de Bob.")
    p.add_argument("--actuator", choices=["ideal", "kinesis"], default="ideal")
    p.add_argument("--moves", type=int, default=5)
    p.add_argument("--round-period-s", type=float, default=DEFAULT_ROUND_PERIOD_S)
    p.add_argument("--real", action="store_true", help="usar el K10CR2 físico, no el simulador")
    p.add_argument("--bob-rotation-deg", type=float, default=DEFAULT_BOB_ROTATION_DEG)
    args = p.parse_args()

    kw = {"bob_rotation_deg": args.bob_rotation_deg}
    if args.actuator == "kinesis":
        kw["use_simulation"] = not args.real
    act = make_actuator(args.actuator, args.round_period_s, **kw)
    print(banner(act))
    act.open()
    try:
        for i in range(args.moves):
            sign = 1 if secrets.randbits(1) == 0 else -1
            r = act.set_sign(sign)
            reached = "?" if r.reached_deg is None else f"{r.reached_deg:+.2f}°"
            print(f"  mov {i + 1}: signo {sign:+d} -> objetivo {r.target_deg:+.2f}°, "
                  f"llegó {reached}, {r.elapsed_s:.2f} s{'  (TARDE)' if r.late else ''}")
    finally:
        act.close()
    print(act.status())


if __name__ == "__main__":
    main()
