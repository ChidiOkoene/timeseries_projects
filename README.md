# timeseries_projects

Research toolkit for **crypto OHLC forecasting** — feature engineering, stacked CNN-RNN / XGBoost ensembles, physics-informed experiments (PIDL), and YOLO-based chart-pattern detection. Primary focus: **XRPUSDT** (with some BTC).

> **Not financial advice.** This is an exploratory R&D sandbox, not a trading product.

## Why this layout

| Path | Role |
| ---- | ---- |
| `src/timeseries/` | Installable package (`ingest`, `features`, `forecast`, `vision`, `pidl`) |
| `notebooks/` | Exploratory notebooks (thin demos; source of truth is `src/`) |
| `examples/` | Tiny tracked sample CSV + quickstart script |
| `data/` | Local CSVs / datasets (**gitignored** — see `data/README.md`) |
| `artifacts/` | Local weights, plots, YOLO runs (**gitignored**) |
| `MIGRATION.md` | Old → new path map so nothing gets lost |

Large chart PNGs and model weights stay on your machine but are no longer in git, so clones stay usable.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Feature-engineering demo (no API keys needed)
python examples/quickstart_features.py
```

Optional extras: `pip install -e ".[forecast]"` (TensorFlow + XGBoost), `".[vision]"` (Ultralytics), `".[tradingview]"` (GitHub; needs `git`), or `".[all]"`.

### Credentials

Copy `.env.example` → `.env` and set:

- `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- `TV_USERNAME` / `TV_PASSWORD` (optional TradingView)

**If this repo was ever public with hardcoded keys, rotate them now.** Secrets are no longer in source; old git history may still contain them until a history rewrite (not done by default).

### Fetch OHLC (Binance)

```bash
python -m timeseries.ingest.binance
```

### Package map

```text
src/timeseries/
  ingest/     Binance + TradingView helpers
  features/   Daily / hourly OHLC indicators
  forecast/   TimeSeriesPipeline + TemporalEnsemble
  vision/     YOLOv8 chart-pattern training
  pidl/       Physics-informed experiments (see notebooks/pidl.ipynb)
  paths.py    DATA_DIR / ARTIFACTS_DIR
```

## Themes

1. **Ingest** — Binance klines; optional TradingView via `tvDatafeed`
2. **Features** — SMA/EMA, RSI, MACD, Bollinger, lags
3. **Forecast** — Hybrid CNN-RNN + XGBoost + meta-model stack
4. **PIDL** — Physics-informed nets (notebook-first)
5. **Vision** — Candlestick image gen → YOLO pattern classes

## Agent / contributor config

See `AGENTS.md` and `.agent/context/` for issue tracker, triage labels, and domain-doc locations. Domain glossary: `CONTEXT.md`.

## License

MIT (see `pyproject.toml`).
