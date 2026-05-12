# Dino Project - Parte 1

Este módulo implementa el generador de nombres de dinosaurios a nivel carácter usando una arquitectura autoregresiva tipo decoder-only (RNN/GRU/LSTM), y deja artefactos consumibles por la web del entregable final.

## Estructura

- `src/dino_gen/`: entrenamiento, modelo, datos, sampling y experimentos
- `scripts/fetch_data.sh`: descarga de `dinos.csv`
- `artifacts/models/`: pesos y vocabulario
- `artifacts/reports/`: métricas e historial
- `artifacts/samples/`: nombres generados y grid de muestreo
- `web_api/app.py`: endpoint de generación para integración

## 1) Instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Descargar datos

```bash
./scripts/fetch_data.sh
```

## 3) Entrenar

```bash
python -m src.dino_gen.train --model-type gru --epochs 60 --max-len 32 --temperature 1.0 --top-k 0 --top-p 1.0 --num-samples 10
```

## 4) Experimentos de muestreo (temperatura, top-k, top-p)

```bash
python -m src.dino_gen.experiments --temperatures 0.7,1.0,2.5,4.0 --top-ks 0,10,20 --top-ps 1.0,0.95,0.9 --n-per-setup 10
```

## 5) Curvas de aprendizaje 

```bash
MPLCONFIGDIR=/tmp/mpl python -m src.dino_gen.plot_curves
```

## 6) Levantar API local (para futura web)

```bash
uvicorn web_api.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:
- `GET /health`
- `POST /generate`

```json
{
  "n": 10,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.9,
  "max_len": 32
}
```

## Salidas clave 

- `artifacts/models/char_decoder.pt`
- `artifacts/models/vocab.json`
- `artifacts/reports/metrics.json`
- `artifacts/samples/baseline_samples.json`
- `artifacts/samples/sampling_experiments.json`
- `artifacts/reports/learning_curves.png` (si ejecutas `plot_curves`)
