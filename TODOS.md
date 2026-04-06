# TODOS

## Benchmark

### Automate benchmark artifact generation in GitHub Actions

**What:** Add a GitHub Actions workflow that runs the NOAA benchmark harness and uploads Markdown plus CSV/JSON artifacts.

**Why:** Once the manual benchmark flow is stable, CI-generated artifacts make the validation reproducible at the commit level and easier to inspect or link from docs.

**Context:** The approved v1 plan intentionally keeps benchmark execution manual to avoid expanding scope into CI flakiness and publishing concerns too early. After the repo-only runner, checked-in NOAA snapshots, and pinned `UTide` setup are stable, the next obvious step is `.github/workflows/benchmark.yml` with pinned dependencies and artifact upload.

**Effort:** M
**Priority:** P2
**Depends on:** Stable v1 benchmark runner, finalized artifact output format, pinned `UTide` version

### Add seasonal benchmark breakdowns

**What:** Extend the NOAA validation harness to report benchmark results by seasonal segments after the baseline multi-station benchmark is stable.

**Why:** Seasonal slices can reveal accuracy shifts that average metrics hide, especially when station difficulty already varies by coastal setting.

**Context:** The approved v1 benchmark deliberately stops at a stable multi-station comparison against `UTide` with Markdown plus CSV/JSON artifacts. Seasonal analysis is the next research-oriented layer on top of that same harness, likely by adding benchmark windows to the manifest or by teaching the report generator to emit segmented summaries.

**Effort:** M
**Priority:** P3
**Depends on:** Stable v1 benchmark harness, baseline station set, finalized artifact format
