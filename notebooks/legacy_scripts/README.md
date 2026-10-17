# Legacy scripts

These files were moved from `outputs/` and `codes/` during the package restructure.
Prefer importing from `timeseries.*` instead.

| File | Prefer |
| ---- | ------ |
| `binance_data_scrape.py` | `timeseries.ingest.binance` |
| `process_dataframe_OHLC.py` | `timeseries.features.ohlc` |
| `ensemble.py` | `timeseries.forecast.ensemble` |
| `train_chart_yolo.py` | `timeseries.vision.train_yolo` |

See `/MIGRATION.md` for the full path map.
