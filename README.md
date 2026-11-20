# token-radar-01

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/diimzyy/token-radar-01/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/diimzyy/token-radar-01/reward.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

A token launch radar for the make.fun **token-radar-01** competition: it reads
a DefiLlama protocol snapshot (758 protocols, 40 watchlist entries), ranks
every watchlist slug by a calibrated implied-value model (per-category
mcap/TVL curves with shrinkage toward the category median), token-odds
estimates, and costed farm plans, and serves an interactive dashboard.

## Run it

```sh
npm install
node main.ts --input input/snapshot.json   # writes out/board.json + out/dossier.json + out/dashboard/
npx serve out/dashboard                    # or: python3 -m http.server -d out/dashboard
```

## Model

- Value: each protocol's implied token value is fitted per category (CEX,
  DEX, Lending, …) as a calibrated function of mcap/TVL, with shrinkage
  toward the category median so thin-liquidity outliers do not dominate.
- Token odds: launch-probability heuristics from watchlist fields (age,
  parent-chain, TVL trend, stablecoin ratio).
- Farm plans: for each top-40 row, a 6-step farm plan with estimated cost.

## Layout

- `main.ts` — pipeline entry (snapshot → board.json → dossier.json → dashboard)
- `lib/model.ts` — the value model, token odds, farm plans
- `dashboard/` — static dashboard (index.html, app.js, app.css) fed by out/dashboard/
- `out/` — build output (board.json, dossier.json, screenshots/)

---

<sub>Entered on [make.fun](https://make.fun/?ref=98bg858x&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
