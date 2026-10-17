# Migration inventory

Old paths → new paths after the public-first restructure.
Local binary assets stay on disk; they are gitignored / untracked, not deleted.

## Notebooks

| Old path | New path |
| -------- | -------- |
| `codes/Chart_img_creation.ipynb` | `notebooks/Chart_img_creation.ipynb` |
| `codes/ensemble.ipynb` | `notebooks/ensemble.ipynb` |
| `codes/ensemble_newtrial.ipynb` | `notebooks/ensemble_newtrial.ipynb` |
| `codes/pidl.ipynb` | `notebooks/pidl.ipynb` |
| `codes/webscrapeing_tradingview.ipynb` | `notebooks/webscrapeing_tradingview.ipynb` |
| `codes/xrpusdt_forecasting.ipynb` | `notebooks/xrpusdt_forecasting.ipynb` |
| `codes/image_gen-train/new chart gen.ipynb` | `notebooks/new_chart_gen.ipynb` |

## Python scripts → package modules

| Old path | New path |
| -------- | -------- |
| `outputs/binance_data_scrape.py` | `src/timeseries/ingest/binance.py` (+ thin CLI at `src/timeseries/ingest/__main__.py` usage via module) |
| `outputs/process_dataframe_OHLC.py` | `src/timeseries/features/ohlc.py` |
| `outputs/ensemble.py` | `src/timeseries/forecast/ensemble.py` |
| `codes/train_chart_yolo.py` | `src/timeseries/vision/train_yolo.py` |
| `codes/chart_dataset/train_chart_yolo.py` | duplicate of above; kept as local copy under `codes/chart_dataset/` (gitignored dataset tree) — use `src/timeseries/vision/train_yolo.py` |
| `codes/webscrapeing_tradingview.ipynb` (creds cell) | env-based helpers in `src/timeseries/ingest/tradingview.py` |

Original scripts under `outputs/` are moved to `notebooks/legacy_scripts/` for reference (not deleted).

## Dataset directories (local-only / gitignored)

| Old path | New / status |
| -------- | ------------ |
| `data/chart_dataset*` | stay under `data/` (gitignored) |
| `data/images_with_boxes` | stay under `data/` (gitignored) |
| `data/img_*` | stay under `data/` (gitignored) |
| `data/sr_charts*` | stay under `data/` (gitignored) |
| `data/*.csv` | stay under `data/` (gitignored); sample excerpt in `examples/` |
| `codes/chart_dataset/` | stay on disk (gitignored); YOLO `data.yaml` path documented in vision module |
| `codes/chart_images/` | stay on disk (gitignored) |
| `codes/image_gen-train/` | stay on disk (gitignored) |
| `codes/sr_charts/` | stay on disk (gitignored) |
| `codes/chart_yolo/` | stay on disk (gitignored) |

## Weights / caches (local-only → prefer `artifacts/`)

| Old path | New / status |
| -------- | ------------ |
| `outputs/*.h5`, `outputs/*.keras`, `outputs/*.weights.h5` | remain under `outputs/` for now; `outputs/` gitignored; prefer writing new runs to `artifacts/` |
| `outputs/scalers.pkl` | same |
| `codes/**/yolov8n.pt`, YOLO runs | stay on disk (gitignored) |

## Config / docs (new)

| Path | Role |
| ---- | ---- |
| `src/timeseries/paths.py` | `DATA_DIR`, `ARTIFACTS_DIR`, `EXAMPLES_DIR`, repo root |
| `pyproject.toml` | installable package |
| `AGENTS.md` + `.agent/context/` | Optimal Brain agent config |
| `CONTEXT.md` | domain stub |
| `data/README.md` | how to obtain / regenerate local data |
| `examples/` | tiny tracked samples for demos |
