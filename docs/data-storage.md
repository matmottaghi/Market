# Data contract and persistence

## Directory layout

`python -m market_dss init` creates `data/raw`, `historical`, `processed`, `features`, `forecasts`, `options`, `backtests`, `exports`, `logs` and `models`. These directories and local secrets are excluded from Git. No sample market prices are installed.

`data/historical/market.xlsx` is the canonical database. `market.xlsx.bak` contains the preceding successful snapshot. All writers take a cross-process `filelock`; file bytes are written and fsynced to a temporary file in the destination directory, then atomically replaced. A failed fetch or validation before commit does not remove prior data. Partial optional-endpoint failures can add valid rows while leaving missing endpoint history untouched.

Each dataset uses sheets named `<table>__0000`, `__0001`, etc., with at most 1,000,000 data rows per sheet. A workbook replacement commits all affected datasets together. Repeated observations are merged by stable keys; missing optional fields do not erase earlier known values. Unknown fields survive normalization and the original source workbook is retained unchanged in a SHA-256-addressed archive. Oversized Excel cells fail explicitly instead of silently truncating original raw data.

| Dataset | Key | Purpose |
|---|---|---|
| instruments | instrument_id, source | Discovered instrument identity; symbols are labels |
| prices | instrument_id, trading_date, source | Latest canonical daily observation |
| price_versions | instrument_id, trading_date, source, retrieved_at | Archived vintages for point-in-time reconstruction |
| snapshots | instrument_id, retrieved_at, source | Fund quotes including rows with unknown observation date |
| nav | instrument_id, trading_date, source | Redemption NAV; unknown-date rows are quarantined in raw input |
| clients | instrument_id, trading_date, source | Individual/institutional activity, preserving provider columns |
| options | instrument_id, retrieved_at, source | Historical option snapshots, not just today's chain |

Byte-identical imports with the same mapping/source are idempotent through `data/processed/imports.json`. A corrected workbook has a different hash and is processed again. Canonical prices can have revised values; `price_versions` preserves earlier values and their availability timestamps. Identifiers must be original text cells, especially TSETMC IDs with more than 15 digits. Excel cannot recover an identifier already rounded by another application.

## Date semantics

- `trading_date`: Gregorian ISO date of the provider's actual observation, converted from its supplied Jalali/Gregorian value.
- `original_date`: unmodified source date representation before conversion.
- `retrieved_at`: timezone-aware UTC timestamp of collection/import.
- `available_at`: timestamp when this exact observation was available. Defaults conservatively to retrieval time. A separately supplied earlier timestamp is a source assertion and must be supported by an original archive/publication record.
- `snapshot_date`: a raw collection-date field; never treated as a trading date.

Missing dates do not become today's date. The supplied collector supplies a valid fund market date only if its detail endpoint succeeds. `Option.php` documentation provides an update **time**, not a verified daily observation date; the adapter deliberately retains its quotes with no trading date. Such rows cannot be executed/backtested automatically.

Historical price backfills collected today are valid for retrospective forecasting studies, but their publication history is not automatically known. Strict option backtests require observations whose `available_at` satisfies the decision cutoff. To reconstruct an old decision, use `load_historical_data(..., as_of=...)` or the backtest engine's `price_versions` input. The current latest price view alone is not a historical vintage database.

## Supplied collector mapping

| Sheet | Mapping / treatment |
|---|---|
| Funds | `id→instrument_id`, `symbol/l18→symbol`, `market_date→trading_date`, `snapshot_ts→retrieved_at`; discovery and snapshots |
| PriceHistory | `date→trading_date`; stable ID joined from Funds/Instruments by normalized symbol |
| NAV | `predtran→nav`, date preserved; same-date NAV premiums are recomputed in the UI |
| Options | `base_id→underlying_id`, `base_l18→underlying_symbol`, `type→option_type`, `price_strike→strike`, `date_end→expiration`, `size_contract→multiplier`, `interest_open→open_interest` |
| ClientHistory | Stable ID joined by symbol; date and all original activity fields retained |
| Details, Metadata, Assemblies, Errors | Retained in the immutable original workbook; inspection reports schema, update reports surface optional endpoint errors |

Shared mappings: `pc→close`, `pl→last`, `pf→open`, `pmin→low`, `pmax→high`, `tvol→volume`, `tval→value`, `tno→transactions`, `pcc→price_change`, `pd1→bid`, `po1→ask`, `qd1→bid_quantity`, `qo1→ask_quantity`. Additional depth levels and all provider-specific fields remain available as raw columns. Canonical numeric columns are converted without creating missing values.

For another workbook use `Instruments`, `PriceHistory`, `NAV`, `ClientHistory`, `Options` with canonical columns, or supply a per-sheet column-mapping JSON. The dashboard downloads an empty schema template. No provider-specific contract style, multiplier or date is supplied by a guessed default.

## Contract enrichment

Add verified `trading_date`, `exercise_style` (`european`/`american`), `settlement` (`cash`/`physical`), `terms_verified` and `terms_source` to an options workbook. Preserve `instrument_id`, `underlying_id`, strike, expiration, multiplier and all quote timestamps. For actual execution assumptions, supply both appropriate bid/ask and quoted quantities in **contracts**; verify the provider quantity unit before mapping it. A URL or document reference belongs in `terms_source`. A user assumption alone is not exchange verification.

Reimport the enriched file. The UI reports unsupported rows and excludes them where required. A numeric margin is not inferred from contract names; strategy construction accepts separately verified committed capital and its source. Cash-settled expiry backtests also require an explicit `settlement_price` in the underlying historical observation, because its daily close may not be the exchange settlement value.

Historical execution additionally requires `execution_at`: a timezone-aware timestamp at which that exact bid/ask/depth was executable, supported by the source archive. It is not an alias for `retrieved_at`. Entries must occur strictly after the underlying information used by the signal became available and no earlier than other chain information seen by the selector. Multi-leg entries must use synchronized snapshots. Daily after-close collector files usually cannot establish these facts; their historical options backtests therefore remain disabled rather than assuming an earlier fill. Intraday timing, execution-window verification and next-session signal handling need a suitable source before those archives can be used for executed-performance claims.

## Recovery and export

Close Excel if it holds an exclusive lock on the database. A locked destination causes the update to fail while preserving the prior file. `ExcelRepository.restore_backup()` validates the backup and restores it under the repository lock; the original backup remains. Maintain independent off-machine backups for disaster recovery. Exports disable spreadsheet formula and URL interpretation for untrusted text.
