# Local data (gitignored)

Full OHLC CSVs, chart-image datasets, and YOLO label trees live here (and under legacy `codes/` paths). They are **not** committed so clones stay small.

## What you already have locally

If you previously cloned or worked in this repo, your files are still on disk under:

- `data/*.csv` — XRP/BTC daily & hourly frames and feature-enriched variants
- `data/chart_dataset*`, `data/images_with_boxes`, `data/sr_charts*` — chart vision sets
- `codes/chart_dataset/`, `codes/image_gen-train/`, `codes/chart_yolo/` — YOLO training trees

See [MIGRATION.md](../MIGRATION.md) for the full inventory.

## How to regenerate

1. Copy [`.env.example`](../.env.example) → `.env` and set Binance (and optional TradingView) credentials. **Rotate any keys that were ever hardcoded.**
2. Install the package: `pip install -e ".[all]"`
3. Fetch OHLC:

```bash
python -m timeseries.ingest.binance
```

4. Engineer features (Python):

```python
import pandas as pd
from timeseries.features import process_dataframe_daily

df = pd.read_csv("data/xrpusdt_daily.csv")
process_dataframe_daily(df).to_csv("data/xrpusdt_daily_dataset_with_features.csv", index=False)
```

5. Chart / YOLO assets: re-run notebooks under `notebooks/Chart_img_creation.ipynb` and `notebooks/new_chart_gen.ipynb`, then `python -m timeseries.vision.train_yolo`.

## Demo without full data

Use [`examples/sample_xrp_daily.csv`](../examples/sample_xrp_daily.csv) and `python examples/quickstart_features.py`.
