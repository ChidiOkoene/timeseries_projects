# Domain context

Single-context glossary for this research repo. Expand via `/domain-modeling` as terms harden.

## Project

**timeseries_projects** — exploratory models for crypto (esp. XRPUSDT) price series: classical indicators, deep/ensemble forecasts, physics-informed nets, and vision over candlestick charts.

## Ubiquitous language

| Term | Meaning |
| ---- | ------- |
| OHLC | Open / high / low / close bar for a symbol and interval |
| Feature frame | OHLC plus engineered indicators (SMA/EMA, RSI, MACD, Bollinger, lags); built by `timeseries.features` |
| Feature frame builder | Deep module seam: OHLC → Feature frame (`process_dataframe_daily` / `_hourly`) |
| Sequence window | Fixed-length lookback (e.g. 30 bars) fed to sequence models |
| Temporal ensemble | Stack of Hybrid CNN-RNN, XGBoost, naive, and meta-model for next-bar close |
| Chart pattern class | YOLO label on a rendered candlestick window (indicator / structure events) |
| Artifact | Local weight, scaler, or plot under `artifacts/` (not published) |

## Avoid

- Calling this a "trading bot" or production execution system
- Synonyms that fight the glossary (e.g. "service" for a notebook experiment)

## Layout seams

- `timeseries.ingest` — market data pull
- `timeseries.features` — indicator engineering
- `timeseries.forecast` — sequence / ensemble models
- `timeseries.vision` — chart images + YOLO
- `timeseries.pidl` — physics-informed experiments
