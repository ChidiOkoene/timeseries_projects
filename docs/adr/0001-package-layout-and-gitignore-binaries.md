# ADR-0001: Package layout and gitignore for binaries

## Status

Accepted

## Context

The repo mixed notebooks, scripts, multi-GB chart PNGs, and Keras weights in git. Visitors could not clone or navigate it. After restructuring, engineering skills need a stable layout decision.

## Decision

1. Publishable code lives under `src/timeseries/` with seams: `ingest`, `features`, `forecast`, `vision`, `pidl`.
2. Exploratory work lives under `notebooks/`; legacy scripts under `notebooks/legacy_scripts/`.
3. Local datasets and Artifacts (`data/`, `artifacts/`, chart trees, weights) stay on disk but are gitignored; tiny demos live in `examples/`.
4. Credentials come only from the environment (`.env` gitignored).

## Consequences

- Fresh clones are small and readable; local experiment assets are preserved via `git rm --cached`, not deletion.
- Domain docs: single-context `CONTEXT.md` + `docs/adr/`.
- History may still contain old binaries/secrets until an optional `git filter-repo` (out of scope for this ADR).
