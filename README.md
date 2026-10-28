# Simulación de Loop-Back QKD transparente

Este paquete implementa la arquitectura lineal:

$$
\mathrm{Alice}_{\mathrm{TX}}\longrightarrow
\mathrm{Bob}_{\mathrm{transparente}}\longrightarrow
\mathrm{Alice}_{\mathrm{RX}}.
$$

Bob no refleja, no mide y no regenera el pulso. Selecciona un bit y una base
BB84, modifica la polarización del mismo pulso que recibe y lo reenvía por su
puerto de salida hacia la interfaz receptora de Alice.

## Contenido

- `loopback_qkd.py`: modelo Monte Carlo completo y rápido.
- `sequence_loopback.py`: demostración sobre el núcleo de SeQUeNCe.
- `run_experiments.py`: genera los casos ideal, realista y con Eve.
- `test_loopback_qkd.py`: pruebas automatizadas.
- `results/`: tablas, gráficas y reporte de validación generado.
- `actuator.py`: actuador de Bob (Placa 2 en el K10CR2), modo ideal o Kinesis.
- `sync.py`: arranque inicial Alice–Bob (preámbulo óptico, espejo MFF101,
  fotodiodo PDA36A2 simulado).
- `test_integracion_kinesis.py`: pruebas del actuador y la sincronización
  (no requieren Kinesis).

La versión entregada fue probada con Python 3.12.13, NumPy 2.3.5,
Matplotlib 3.10.8 y SeQUeNCe 1.0.0.

## Modelo físico

Bob selecciona aleatoriamente:

- base $Z$: $H$ o $V$;
- base $X$: $D$ o $A$.

La eficiencia señal-detector es

$$
\eta_{\mathrm{tot}}=
10^{-(L_{AB}+L_B+L_{BA})/10}\eta_D.
$$

Para una fuente coherente débil con número medio de fotones $\mu$, la
probabilidad de al menos un clic producido por la señal es

$$
p_{\mathrm{sig}}=1-\exp(-\mu\eta_{\mathrm{tot}}).
$$

Se simulan dos detectores umbral. Los dobles clics se asignan aleatoriamente,
y las cuentas oscuras se introducen como una probabilidad por detector y por
ventana de detección.

## Ejecución básica

Requiere Python 3.10 o posterior y NumPy:

```bash
python -m pip install -r requirements.txt
python loopback_qkd.py --pulses 200000 --source single_photon
python run_experiments.py
python -m unittest -v test_loopback_qkd.py
```

Ejemplo con pulsos coherentes débiles y pérdidas:

```bash
python loopback_qkd.py \
  --pulses 1000000 \
  --source weak_coherent \
  --mu 0.1 \
  --loss-ab-db 3 \
  --bob-loss-db 2 \
  --loss-ba-db 3 \
  --detector-efficiency 0.25 \
  --dark-probability 0.00001 \
  --bob-error-deg 1 \
  --channel-error-deg 1
```

Ataque interceptar--reenviar sobre todos los pulsos:

```bash
python loopback_qkd.py \
  --pulses 500000 \
  --source single_photon \
  --attack-fraction 1
```

En condiciones ideales, el resultado esperado es QBER $=0$. Con
interceptación y reenvío completos, el QBER esperado es aproximadamente
$25\%$.

## Demostración con SeQUeNCe

Instalar SeQUeNCe:

```bash
python -m pip install -r requirements-sequence.txt
python sequence_loopback.py --pulses 50000
python sequence_loopback.py --pulses 100000 --attack-fraction 1
```

La salida incluye:

- fotones recibidos y reenviados por Bob;
- fotones recibidos por Alice;
- número de ocasiones en que Alice recibió exactamente el mismo objeto
  `Photon` creado por Alice y atravesado por Bob;
- bits filtrados, errores y QBER.

En el caso sin pérdidas debe cumplirse:

```text
photons_received_by_alice == same_photon_objects_at_alice
```

Esto verifica computacionalmente que Bob no crea un fotón sustituto.

Para correr hasta obtener una clave de largo fijo, en lugar de un número fijo
de pulsos:

```bash
python sequence_loopback.py --key-bits 256
python sequence_loopback.py --key-bits 256 --attack-fraction 1
```

Con `--key-bits N` la simulación se detiene cuando Alice reúne N bits
filtrados; `--pulses` pasa a ser solo un tope (por defecto 20·N).

## Tiempos del cambio de base de Bob (Kinesis)

`sequence_loopback.py` puede mover el actuador real de Bob en cada ronda para
medir los tiempos del cambio de base. Solo cambia el lado de Bob; Alice y la
lógica del protocolo no se modifican.

### Modos

| `--actuator`          | Qué hace                                                       | Uso                    |
| ----------------------- | --------------------------------------------------------------- | ---------------------- |
| `ideal` (por defecto) | Kinesis **no** se usa; sin tiempos                         | Validación y barridos |
| `kinesis`             | Mueve el K10CR2 y el MFF101 vía Kinesis; rondas en tiempo real | Medir tiempos (lento)  |

En modo `ideal` los resultados son **idénticos** a la versión original (lo
verifica `test_integracion_kinesis.py`). El programa siempre indica en pantalla
si Kinesis se está usando. `loopback_qkd.py` no tiene eje de tiempo y solo
acepta `--actuator ideal`.

### Requisitos (modo `kinesis`)

- Windows con Thorlabs Kinesis instalado en `C:\Program Files\Thorlabs\Kinesis`.
- `python -m pip install pythonnet`.
- Kinesis Simulator abierto con dos dispositivos creados:
  K10CR2 serie `55000002` y MFF101 serie `37000001`.
  Con equipos físicos, agregar `--real`.

### Comandos

```bash
# Prueba aislada del motor (6 movimientos)
python actuator.py --actuator kinesis --moves 6

# Clave corta de prueba (~1 min)
python sequence_loopback.py --key-bits 4 --actuator kinesis

# Clave de 128 bits (~14 min), sin y con Eve
python sequence_loopback.py --key-bits 128 --actuator kinesis
python sequence_loopback.py --key-bits 128 --actuator kinesis --attack-fraction 1
```

### Secuencia en modo `kinesis`

1. Bob hace *home* de la Placa 2 y pone el espejo MFF101 en el haz.
2. Alice envía el preámbulo (patrón de pulsos, por defecto a 0; 0,10 y
   0,25 s). El fotodiodo de Bob lo reconoce y fija la referencia $t_{\mathrm{ref}}$.
3. Bob quita el espejo y lo desconecta (si queda conectado, sus consultas
   frenan al K10CR2).
4. Las rondas empiezan en $t_{\mathrm{ref}} + T_{\mathrm{inicio}}$ (5 s, acordado de
   antemano). Cada ronda dura un plazo fijo: Bob gira (o no) la Placa 2 al
   inicio y Alice dispara al final.
5. Si el motor no termina dentro del plazo se muestra `[AVISO] ATRASADO`.

Mapeo de Bob: base 0 → Placa 2 a +11,25°; base 1 → Placa 2 a −11,25°
(rotación de polarización de ±22,5° con la Placa 1 fija en 0°).

### Parámetros

| Opción                    | Por defecto    | Descripción                                            |
| -------------------------- | -------------- | ------------------------------------------------------- |
| `--round-period-s`       | 3.0            | Duración fija de cada ronda                            |
| `--start-delay-s`        | 5.0            | $T_{\mathrm{inicio}}$: espera entre el preámbulo y la ronda 1 |
| `--preamble-pattern`     | `0,0.1,0.25` | Tiempos del patrón del preámbulo (s)                  |
| `--photodiode-latency-s` | 0.001          | Latencia del fotodiodo simulado                         |
| `--mirror-serial`        | `37000001`   | Serie del MFF101                                        |
| `--real`                 | —             | Usar equipos físicos en vez del simulador              |

Velocidad del K10CR2: 20 °/s y 20 °/s² (fijada por el código al conectar).

### Salida adicional

Durante la corrida se imprime una línea por ronda (ángulo, tiempo del motor,
bits acumulados). Al final, además del resultado habitual, se imprime:

- `actuador`: `kinesis_active`, movimientos, rondas atrasadas (`late_moves`),
  errores de posición y tiempo máximo de movimiento;
- `sincronizacion`: patrón detectado, desfase Bob − Alice (ms), tiempo de
  retiro del espejo y margen antes de la ronda 1.

### Pruebas

```bash
python -m unittest -v test_integracion_kinesis.py
```
