# Provider verification record — 2026-10-21

Primary BrsApi documentation was inspected when implementing the adapter. No personal API key was supplied; authenticated responses and account entitlements were not tested. The public sample values in the documentation were used to understand field names only, never imported as observations.

| Capability | Primary source | Implemented status |
|---|---|---|
| Symbol universe and market quotes | [BrsApi main documentation](https://brsapi.ir/) | Supplied collector uses `https://Api.BrsApi.ir/Tsetmc/AllSymbols.php`, `key`, `type`; free service still requires a key |
| Historical prices and clients | [History documentation](https://brsapi.ir/bourse-api-history-webservice/) | `History.php`, `l18`, `type=0/1`; runtime schema validation and endpoint errors preserved |
| Listed options | [Option documentation](https://brsapi.ir/bourse-api-option-webservice/) | `Option.php`; documented strike/expiry/multiplier/open-interest/quote fields mapped |
| Fund NAV | [NAV documentation](https://brsapi.ir/bourse-api-etf-funds-nav-webservice/) | `Nav.php`; redemption NAV `predtran`, issuance NAV `psubtran` retained |
| Codal | [Codal documentation](https://brsapi.ir/bourse-api-codal-webservice/) | Documentation reviewed; financial-report extraction is not implemented; collector Assemblies raw sheet retained |
| Direct TSETMC | [TSETMC](https://www.tsetmc.com/) | Direct CDN history/search probes were inaccessible from the browsing environment; no undocumented fallback enabled |

The BrsApi public pages describe authentication, request routes and provider field meanings. Service tiers and actual API response envelopes may differ by account; the collector validates record/envelope shape at runtime. Credentials belong in local environment variables, not source files, logs or public examples. Endpoint errors include only sanitized endpoint/status information.

The existing collector's `Symbol.php` field assumptions (`date_update`/`date`, assemblies) are preserved for compatibility but have not been independently confirmed against an authenticated response here. A missing detail date stays missing. No claim is made that market dates, NAV timestamps, option terms, broker margin, quote-size units or exchange settlement rules have been independently verified.

The default DSS source is `BrsApi`; canonical imports can use another configured source. `DataProvider.fetch()` and `CollectorAdapter` are the extension boundaries. A direct TSETMC/Codal provider should only be enabled after its live schema, authentication/entitlements and date semantics have been verified and its mapping tests added.
