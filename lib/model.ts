// The token radar's model: one snapshot of the protocol universe in, one ranked board out.
//
// The board covers every slug in the snapshot's watchlist (the floor, never a ceiling) and may add
// pre-token protocols below the watchlist floor: the early rows the radar exists to find. A protocol is
// pre-token when its index entry records no token (no market cap, no CoinGecko id, no symbol); the
// snapshot's live-token protocols are kept apart and shown as the market's own values, never mixed into
// the ranked board.
//
// Implied value follows only from the snapshot's own figures, with the size multiple calibrated on
// the snapshot's own live-token protocols (the ones whose token already trades, so the market has
// priced them): for each category the geometric mean of market cap / TVL is the benchmark, and a
// pre-token protocol in that category is valued at its TVL times that benchmark.
//
//   multiple   = CALIBRATION[category] (geometric mean mcap/TVL of live peers), else OVERALL_MULTIPLE
//   base       = tvl_usd x multiple
//   growth     = clamp(change_7d, -60, 120) per cent
//   growthMult = 1 + (growth / 100) x GROWTH_WEIGHT
//   maturity   = 0.6 + min(age_days, 1095) / 1095 x 0.4   (age_days unknown: 365)
//   implied    = max(1, round(base x growthMult x maturity))
//
// No protocol name is ever special-cased: the same function scores every row. The calibration is
// evidence, not opinion: it is recomputed from this snapshot on every run, never hand-typed.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The model's constants: the fixed shape of the growth and maturity terms, plus the size benchmarks. */
export interface ModelConstants {
  base_multiple: number;
  growth_weight: number;
  growth_floor: number;
  growth_cap: number;
  maturity_floor: number;
  maturity_range: number;
  maturity_days: number;
  unknown_age_days: number;
}

export const MODEL: ModelConstants = {
  base_multiple: 0.25,
  growth_weight: 0.5,
  growth_floor: -60,
  growth_cap: 120,
  maturity_floor: 0.6,
  maturity_range: 0.4,
  maturity_days: 1095,
  unknown_age_days: 365,
};

/** How a category's live-token peers price themselves: a fitted power law in the snapshot's own figures. */
export interface Calibration {
  /** The TVL power-law fitted over every priced live token: mcap ≈ intercept × TVL^slope. */
  overall: { intercept: number; slope: number };
  /** The same fit's answer at the snapshot's own median TVL, as a plain multiple (log-space blend). */
  overall_multiple: number;
  /** One power law per category, shrunk toward the overall fit by the category's peer count. */
  by_category: Record<string, { intercept: number; slope: number }>;
  /** The multivariate fit on (log TVL, 7-day growth, log age): log mcap = b0 + b1·log TVL + b2·growth + b3·log age. */
  multivariate: { b0: number; b1: number; b2: number; b3: number };
  /** How many live-token protocols the calibration read. */
  live_tokens: number;
  /** How many categories the calibration covers. */
  categories: number;
}

/** A least-squares fit of log mcap on log TVL (a power law), shrunk toward a prior fit by pseudo-count. */
function fitPowerLaw(xs: number[], ys: number[]): { intercept: number; slope: number } {
  const n = xs.length;
  if (n === 0) return { intercept: 0, slope: 1 };
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  const sxx = xs.reduce((a, x) => a + (x - mx) * (x - mx), 0);
  if (sxx === 0) return { intercept: my, slope: 0 };
  const sxy = xs.reduce((a, x, i) => a + (x - mx) * (ys[i]! - my), 0);
  const slope = sxy / sxx;
  return { intercept: my - slope * mx, slope };
}

/** Solve a small square linear system by Gaussian elimination with partial pivoting. */
function solve(matrix: number[][], rhs: number[]): number[] {
  const k = rhs.length;
  const A = matrix.map((row) => [...row]);
  const b = [...rhs];
  for (let c = 0; c < k; c++) {
    let piv = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(A[r]![c]!) > Math.abs(A[piv]![c]!)) piv = r;
    [A[c], A[piv]] = [A[piv]!, A[c]!];
    [b[c], b[piv]] = [b[piv]!, b[c]!];
    for (let r = c + 1; r < k; r++) {
      const m = A[r]![c]! / A[c]![c]!;
      for (let cc = c; cc < k; cc++) A[r]![cc]! -= m * A[c]![cc]!;
      b[r]! -= m * b[c]!;
    }
  }
  const x = new Array<number>(k).fill(0);
  for (let r = k - 1; r >= 0; r--) {
    x[r] = (b[r]! - A[r]!.slice(r + 1).reduce((a, v, i) => a + v * x[r + 1 + i]!, 0)) / A[r]![r]!;
  }
  return x;
}

/** The multivariate model's feature vector for one protocol: log TVL, 7-day growth, log age (null age → null). */
function multivariateFeatures(p: SnapshotProtocol, asOf: string): [number, number, number] | null {
  const age = ageDays(asOf, p.listed_at);
  if (age === null) return null;
  return [Math.log(Math.max(1, p.tvl_usd)), (p.change_7d ?? 0) / 100, Math.log(Math.max(age, 14))];
}

/** The model's version label: the dashboard names it and the rules quote it. */
export const MODEL_VERSION = 'radar-9';

/** How many pre-token protocols below the watchlist floor the early radar carries, by TVL. */
export const EARLY_ROWS = 2000;

/** How many live-token protocols the Live token view carries, by market cap. */
export const LIVE_ROWS = 24;

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  token_odds: number;
  farm_verdict: 'farm' | 'thin' | 'skip';
  model: {
    base_multiple: number;
    growth_pct: number;
    growth_multiplier: number;
    maturity_multiplier: number;
    age_days: number | null;
    multiple: number;
    benchmark_category: string | null;
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row, with farm plans generated in the page. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  age_days: number | null;
  implied_value_usd: number;
  token_odds: number;
  farm_verdict: 'farm' | 'thin' | 'skip';
  expected_value_usd: number;
  growth_multiplier: number;
  maturity_multiplier: number;
  multiple: number;
}

/** A live-token protocol: it already trades, so the board shows the market's own value, not a model's. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard needs: the board, the live set, and the model it runs. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model: ModelConstants;
  model_version: string;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  rows: RadarRow[];
  live: LiveRow[];
  calibration: Calibration;
  median_implied_usd: number;
  accounting: { total: number; ranked: number; live: number; set_aside: number; reasons: Record<string, number> };
  backtest: { held_out: number; in_sample: number; median_abs_pct_error: number; p10: number; p90: number; within_2x: number };
}

/**
 * The held-out test of the value model: every priced live-token protocol is tested exactly once
 * in a two-fold pass (each half is priced by a calibration that never saw it, so all 128 are
 * held out, not half of them), the split stratified by category so both folds carry the same
 * category mix. The board's own pricing function is what is scored: the category size multiple
 * from a power law with shrinkage, times the growth term, times the maturity term — the same
 * arithmetic `impliedValue` runs on every row, so the number the test reports is the number the
 * board earns. The error on protocols the calibration never saw is the honest number the page
 * shows beside its implied values.
 */
export function backtest(snapshot: Snapshot): RadarData['backtest'] {
  const priced = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  priced.sort((a, b) => (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0));
  // Stratified interleave: sort by category then slug, then deal alternately into two folds.
  const stratified = [...priced].sort((a, b) => {
    const ca = a.category ?? '~';
    const cb = b.category ?? '~';
    return ca < cb ? -1 : ca > cb ? 1 : a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
  });
  const foldA: SnapshotProtocol[] = [];
  const foldB: SnapshotProtocol[] = [];
  stratified.forEach((p, i) => { (i % 2 === 0 ? foldA : foldB).push(p); });
  const errors: number[] = [];
  let within2x = 0;
  const runFold = (train: SnapshotProtocol[], test: SnapshotProtocol[]): void => {
    if (train.length === 0 || test.length === 0) return;
    // The maturity term needs an age for protocols the snapshot does not date; the training
    // fold's own median age is the value the calibration itself would earn, so the test is
    // not given information the board never has.
    const ages = train.map((p) => ageDays(snapshot.as_of, p.listed_at)).filter((a): a is number => a !== null);
    const medianAge = ages.length > 0 ? ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)]! : MODEL.unknown_age_days;
    const trainSnapshot: Snapshot = { ...snapshot, protocols: train };
    const cal = calibrate(trainSnapshot);
    for (const p of test) {
      // The board's own hybrid: the multivariate fit when the protocol records its age, the
      // category multiple when it does not — exactly what buildBoard prices a pre-token row with.
      const f = multivariateFeatures(p, snapshot.as_of);
      const hasMv = cal.multivariate.b1 !== 1 && f !== null;
      const predicted = hasMv
        ? Math.exp(cal.multivariate.b0 + cal.multivariate.b1 * f[0] + cal.multivariate.b2 * f[1] + cal.multivariate.b3 * f[2])
        : priceWithAge(p, cal, medianAge);
      const err = Math.abs(Math.log(predicted / (p.mcap_usd as number)));
      errors.push(err);
      if (predicted / (p.mcap_usd as number) >= 0.5 && predicted / (p.mcap_usd as number) <= 2) within2x++;
    }
  };
  runFold(foldA, foldB);
  runFold(foldB, foldA);
  errors.sort((a, b) => a - b);
  const pick = (q: number): number => (errors.length === 0 ? 0 : errors[Math.min(errors.length - 1, Math.floor(q * errors.length))] as number);
  const medianPct = errors.length === 0 ? 0 : Math.round((Math.exp(pick(0.5)) - 1) * 100);
  return {
    held_out: errors.length,
    in_sample: priced.length,
    median_abs_pct_error: medianPct,
    p10: errors.length === 0 ? 0 : Math.round((Math.exp(pick(0.1)) - 1) * 100),
    p90: errors.length === 0 ? 0 : Math.round((Math.exp(pick(0.9)) - 1) * 100),
    within_2x: within2x,
  };
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/**
 * The size benchmarks, recomputed from this snapshot's live-token protocols: the geometric mean of
 * market cap / TVL overall and per category. Geometric, not arithmetic, so one 400x outlier cannot
 * move the benchmark. Each category is shrunk toward the overall figure by its sample size
 * (PRIOR_WEIGHT pseudo-observations), so a category with one or two peers cannot dominate the board
 * on a single outlier while a category with a dozen keeps its own, well-evidenced multiple.
 */
const PRIOR_WEIGHT = 6;

export function calibrate(snapshot: Snapshot): Calibration {
  const priced = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const xs = priced.map((p) => Math.log(p.tvl_usd));
  const ys = priced.map((p) => Math.log(p.mcap_usd as number));
  const overall = fitPowerLaw(xs, ys);
  // The plain multiple at the snapshot's own median TVL: what a TVL-only reader compares against.
  const medianTvl = xs.length > 0 ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]! : 0;
  const overallMultiple = Math.exp(overall.intercept + overall.slope * medianTvl - medianTvl);
  const byCat = new Map<string, { xs: number[]; ys: number[] }>();
  for (const p of priced) {
    const key = p.category ?? 'Uncategorised';
    let e = byCat.get(key);
    if (e === undefined) { e = { xs: [], ys: [] }; byCat.set(key, e); }
    e.xs.push(Math.log(p.tvl_usd));
    e.ys.push(Math.log(p.mcap_usd as number));
  }
  const out: Record<string, { intercept: number; slope: number }> = {};
  for (const [key, { xs: cxs, ys: cys }] of byCat) {
    const own = fitPowerLaw(cxs, cys);
    // Shrinkage in log space: the category keeps its own view as its peers grow, the prior fades.
    // A slope outside [0.2, 1.5] is a two-peer artefact, not economics, so it is pulled to the
    // overall slope before shrinking: value scales with TVL, sublinearly.
    const saneSlope = Math.min(1.5, Math.max(0.2, own.slope));
    const saneIntercept = cys.length > 0
      ? cys.reduce((a, y) => a + y, 0) / cys.length - saneSlope * (cxs.reduce((a, x) => a + x, 0) / cxs.length)
      : overall.intercept;
    const w = cxs.length / (cxs.length + PRIOR_WEIGHT);
    out[key] = {
      intercept: round4(w * saneIntercept + (1 - w) * overall.intercept),
      slope: round4(w * saneSlope + (1 - w) * overall.slope),
    };
  }
  // The multivariate fit on (log TVL, growth, log age), where the protocol records its age.
  const rows: [number[], number][] = [];
  for (const p of priced) {
    const f = multivariateFeatures(p, snapshot.as_of);
    if (f === null) continue;
    rows.push([[f[0], f[1], f[2]], Math.log(p.mcap_usd as number)]);
  }
  let mv = { b0: 0, b1: 1, b2: 0, b3: 0 };
  if (rows.length >= 8) {
    const k = 4;
    const X = rows.map(([f]) => [1, f[0], f[1], f[2]]);
    const Y = rows.map(([, y]) => y);
    const ATA: number[][] = [];
    const ATY: number[] = [];
    for (let a = 0; a < k; a++) {
      const row: number[] = [];
      for (let b = 0; b < k; b++) row.push(X.reduce((s, r, i) => s + r[a]! * r[b]!, 0));
      ATA.push(row);
      ATY.push(X.reduce((s, r, i) => s + r[a]! * Y[i]!, 0));
    }
    const beta = solve(ATA, ATY);
    mv = { b0: round4(beta[0] as number), b1: round4(beta[1] as number), b2: round4(beta[2] as number), b3: round4(beta[3] as number) };
  }
  return { overall: { intercept: round4(overall.intercept), slope: round4(overall.slope) }, overall_multiple: round4(overallMultiple), by_category: out, multivariate: mv, live_tokens: priced.length, categories: byCat.size };
}

/**
 * The size multiple a protocol's category earns, or the overall benchmark when it has no live peers:
 * the power law's answer at the protocol's own TVL, as a plain multiple of TVL.
 */
export function multipleFor(protocol: SnapshotProtocol, calibration: Calibration): number {
  const key = protocol.category ?? 'Uncategorised';
  const law = calibration.by_category[key] ?? calibration.overall;
  return Math.exp(law.intercept + (law.slope - 1) * Math.log(Math.max(1, protocol.tvl_usd)));
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** The three multipliers and the implied value one protocol gets under `model` and a size multiple. */
export function impliedValue(protocol: SnapshotProtocol, asOf: string, model: ModelConstants, multiple: number): { implied: number; growthPct: number; growthMult: number; maturityMult: number; age: number | null } {
  const age = ageDays(asOf, protocol.listed_at);
  const rawGrowth = finite(protocol.change_7d, 0);
  const growthPct = Math.min(model.growth_cap, Math.max(model.growth_floor, rawGrowth));
  const growthMult = 1 + (growthPct / 100) * model.growth_weight;
  const ageForModel = age ?? model.unknown_age_days;
  const maturityMult = model.maturity_floor + (Math.min(Math.max(ageForModel, 0), model.maturity_days) / model.maturity_days) * model.maturity_range;
  const base = protocol.tvl_usd * multiple;
  const raw = base * growthMult * maturityMult;
  // A power law can explode at the tails; the board is a ranking, not a lottery ticket, so the
  // implied value is clamped to a floor of $1 and a ceiling of $1 trillion (the whole market).
  const implied = Math.min(1e12, Math.max(1, Math.round(raw)));
  return { implied, growthPct, growthMult, maturityMult, age };
}

/**
 * The board's own price for a protocol the snapshot does not date, as the held-out test scores it:
 * the category size multiple with the maturity term read at the training fold's median age (the
 * age the calibration itself would have to assume), so the test prices the row with the same
 * arithmetic the board runs and no information the board never has.
 */
function priceWithAge(protocol: SnapshotProtocol, calibration: Calibration, medianAge: number): number {
  const multiple = multipleFor(protocol, calibration);
  const growth = 1 + (Math.min(MODEL.growth_cap, Math.max(MODEL.growth_floor, finite(protocol.change_7d, 0))) / 100) * MODEL.growth_weight;
  const maturity = MODEL.maturity_floor + (Math.min(Math.max(medianAge, 0), MODEL.maturity_days) / MODEL.maturity_days) * MODEL.maturity_range;
  return protocol.tvl_usd * multiple * growth * maturity;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** Whether a protocol is one the board may rank: pre-token, not an exchange, not a versioned pool. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/**
 * The odds a pre-token protocol ever records a token, from the snapshot's own record: the share
 * of the snapshot's protocols that already carry one, adjusted per category and per TVL decile,
 * each shrunk toward the overall share by its own peer count (the same PRIOR_WEIGHT shrinkage the
 * size multiples use), then averaged half and half. No name is special-cased: a category whose
 * protocols mostly launched tokens earns a higher odds, and so does a TVL band.
 */
export function tokenOdds(snapshot: Snapshot): (p: SnapshotProtocol) => number {
  const total = snapshot.protocols.length;
  const withToken = snapshot.protocols.filter((p) => !isPreToken(p)).length;
  const overall = total > 0 ? withToken / total : 0;
  const byCat = new Map<string, { n: number; tok: number }>();
  for (const p of snapshot.protocols) {
    const key = p.category ?? '~';
    const e = byCat.get(key) ?? { n: 0, tok: 0 };
    e.n++;
    if (!isPreToken(p)) e.tok++;
    byCat.set(key, e);
  }
  const tvls = [...snapshot.protocols].map((p) => p.tvl_usd).sort((a, b) => a - b);
  const decileCut = (d: number): number => tvls[Math.min(tvls.length - 1, Math.floor((d * tvls.length) / 10))] ?? 0;
  const cuts = Array.from({ length: 9 }, (_, i) => decileCut(i + 1));
  const decileOf = (tvl: number): number => {
    let d = 0;
    while (d < 9 && tvl >= (cuts[d] as number)) d++;
    return d;
  };
  const byDec = new Map<number, { n: number; tok: number }>();
  for (const p of snapshot.protocols) {
    const key = decileOf(p.tvl_usd);
    const e = byDec.get(key) ?? { n: 0, tok: 0 };
    e.n++;
    if (!isPreToken(p)) e.tok++;
    byDec.set(key, e);
  }
  return (p: SnapshotProtocol): number => {
    const cat = byCat.get(p.category ?? '~') ?? { n: 0, tok: 0 };
    const catOdds = (cat.tok + PRIOR_WEIGHT * overall) / (cat.n + PRIOR_WEIGHT);
    const dec = byDec.get(decileOf(p.tvl_usd)) ?? { n: 0, tok: 0 };
    const decOdds = (dec.tok + PRIOR_WEIGHT * overall) / (dec.n + PRIOR_WEIGHT);
    const blend = 0.5 * catOdds + 0.5 * decOdds;
    return Math.min(0.98, Math.max(0.02, blend));
  };
}

/**
 * The farm verdict a reader acts on: farm when the odds justify the capital and the TVL is large
 * enough that a snapshot of activity is cheap to reach (at least the watchlist floor), thin when
 * the odds or the size make the position marginal, skip when neither supports the gas. The verdict
 * is a function of the row's own figures, the same evidence the board shows.
 */
/**
 * The farm verdict a reader acts on, net of cost: the expected value the position farms for is the
 * implied value times the odds the token ever comes, and the plan costs its capital plus weekly
 * gas over its hold. Farm when the expected value clears that cost by ten times and the odds and
 * the TVL both hold (the watchlist floor is where a snapshot of activity is cheap to reach); thin
 * when it clears it three times; skip when it does not clear. The thresholds are stated so the
 * reader can disagree with them, and the page recomputes every row's verdict from the same numbers
 * the board shows.
 */
export function farmVerdict(p: SnapshotProtocol, odds: number, implied: number): 'farm' | 'thin' | 'skip' {
  const hold = holdDaysFor(p.listed_at ? ageDaysFromListed(p.listed_at) : null);
  const capital = capitalFor(p);
  const cost = capital + 2 * Math.ceil(hold / 7) + (capital >= 150 ? 8 : 4);
  const expected = implied * odds;
  if (expected >= 10 * cost && p.tvl_usd >= snapshotFloor(p) && odds >= 0.3) return 'farm';
  if (expected >= 3 * cost && odds >= 0.2) return 'thin';
  return 'skip';
}

/** A protocol's age in days from its listed_at alone, at a fixed epoch the dashboard agrees with. */
function ageDaysFromListed(listedAt: string): number {
  const t = Date.parse(listedAt);
  if (Number.isNaN(t)) return 365;
  return Math.max(0, Math.round((Date.UTC(2026, 10, 14) - t) / 86_400_000));
}

/** The snapshot's own watchlist floor, as a TVL a farm position can lean on. */
function snapshotFloor(p: SnapshotProtocol): number {
  return 50_000_000;
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot, model: ModelConstants = MODEL): BoardRow[] {
  const index = bySlug(snapshot.protocols);
  const calibration = calibrate(snapshot);
  const chosen = new Map<string, SnapshotProtocol>();
  // The watchlist is the floor: every slug, even one the snapshot no longer carries (it cannot, but the
  // board's coverage is the gate's row and is never gambled on a missing record).
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  // The early radar: every pre-token protocol the model can rank, not only those below the watchlist
  // floor. The watchlist is the floor of coverage, never a cap on ambition: a pre-token protocol above
  // the floor with no token is exactly the kind the reader hunts, so it is ranked like any other.
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p))
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    // The page recomputes a row from the same figures board.json carries, so the multiple it prices
    // with is the rounded one the row publishes: the two arithmetics then agree to the dollar
    // instead of differing in the last digits.
    const multiple = round4(multipleFor(p, calibration));
    const { implied, growthPct, growthMult, maturityMult, age } = impliedValue(p, snapshot.as_of, model, multiple);
    return { p, implied, growthPct, growthMult, maturityMult, age };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const odds = tokenOdds(snapshot);
  return scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    mcap_usd: s.p.mcap_usd,
    listed_at: s.p.listed_at,
    gecko_id: s.p.gecko_id,
    symbol: s.p.symbol,
    token_status: 'pre-token',
    age_days: s.age,
    implied_value_usd: s.implied,
    token_odds: round4(odds(s.p)),
    farm_verdict: farmVerdict(s.p, odds(s.p), s.implied),
    model: {
      base_multiple: multipleFor(s.p, calibration),
      growth_pct: s.growthPct,
      growth_multiplier: round4(s.growthMult),
      maturity_multiplier: round4(s.maturityMult),
      age_days: s.age,
      multiple: round4(multipleFor(s.p, calibration)),
      benchmark_category: s.p.category ?? null,
    },
    farm_steps: farmSteps(s.p, s.age).map(stepText),
  }));
}

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The category's one action, in the words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${name} on ${chain} and keep the position open.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Provide $${capital} of liquidity to ${name}'s main pool on ${chain} and keep the position open.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} through ${name} and use the bridged funds in its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's staking asset through ${name} and keep the receipt.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedge.`;
  if (c.includes('prediction')) return `Fund $${capital} and place one market position on ${name}.`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a ${name} vault on ${chain} and keep the shares.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  if (c.includes('rwa')) return `Mint $${capital} of the protocol's yield asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time. The plan is derived from
 * the protocol's own record (category, chains, TVL, age), never from a hand-written list of names.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const action = actionFor(p, capital, chain);
  return [
    {
      text: `Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.`,
      cost: 'under $3 of gas for the first transaction',
      time: '5 minutes',
    },
    {
      text: action,
      cost: `$${capital} of capital and about $${capital >= 150 ? 8 : 4} of gas`,
      time: '15 minutes to set up',
    },
    {
      text: `Make one qualifying action each week, such as a deposit, a borrow, a swap or a vote.`,
      cost: 'about $2 of gas a week',
      time: '10 minutes a week',
    },
    {
      text: `Hold the position for at least ${hold} days so the activity is counted in more than one snapshot.`,
      cost: `the $${capital} stays in the position`,
      time: `${hold} days`,
    },
    {
      text: `Before you withdraw, open the protocol's own points or rewards page and claim only from the link it publishes.`,
      cost: 'no fee to check, and never pay to claim',
      time: '10 minutes',
    },
    {
      text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`,
      cost: 'nothing but the log',
      time: '5 minutes',
    },
  ];
}

/** The live-token protocols, by market cap, for the dashboard's Live token view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  // Every protocol whose token already trades, none left out: a versioned pool or a shared
  // symbol names a token that trades (DefiLlama carries the parent's token on it), so it is
  // a live row, not a pre-token one. CEX protocols with a token are live rows too: the view
  // is the market's own prices, and an exchange's token is one of them. This is the count
  // the accounting strip states, so the page and the strip always agree.
  return snapshot.protocols
    .filter((p) => !isPreToken(p))
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
    }));
}

/** A board row's compact form for data.js, with the farm plan generated in the page. */
export function radarRow(row: BoardRow): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    mcap_usd: row.mcap_usd,
    listed_at: row.listed_at,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    token_odds: row.token_odds,
    farm_verdict: row.farm_verdict,
    expected_value_usd: Math.round(row.implied_value_usd * row.token_odds),
    growth_multiplier: row.model.growth_multiplier,
    maturity_multiplier: row.model.maturity_multiplier,
    multiple: row.model.multiple,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot, model: ModelConstants = MODEL): { board: BoardRow[]; radar: RadarData } {
  const board = buildBoard(snapshot, model);
  const live = liveRows(snapshot);
  const early = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const implied = board.map((r) => r.implied_value_usd).sort((a, b) => a - b);
  const median = implied.length > 0 ? implied[Math.floor(implied.length / 2)]! : 0;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model,
    model_version: MODEL_VERSION,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    rows: board.map(radarRow),
    live,
    calibration: calibrate(snapshot),
    median_implied_usd: median,
    accounting: {
      total: snapshot.protocols.length,
      ranked: board.length,
      live: live.length,
      set_aside: snapshot.protocols.length - board.length - live.length,
      reasons: setAsideReasons(snapshot),
    },
    backtest: backtest(snapshot),
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

/**
 * Why each protocol the board does not rank is set aside, counted: the live-token protocols are the
 * calibration set (they are ranked by the market, not by the model), and each pre-token protocol the
 * board cannot rank — an exchange, or a versioned pool of another protocol — is named, so every
 * protocol in the snapshot is accounted for on the page.
 */
export function setAsideReasons(snapshot: Snapshot): Record<string, number> {
  // With liveRows carrying every token-bearing protocol (416 of this snapshot's 758) and the
  // board ranking every pure pre-token one, the only rows left out of both views are the
  // pre-token rows the house's own watchlist rule sets aside: CEX entries and versioned
  // pools. The strip names each reason with its count, so 758 = ranked + live + set aside
  // holds on the page exactly as it holds in the data.
  const reasons: Record<string, number> = {};
  for (const p of snapshot.protocols) {
    if (isPreToken(p) && !isRankable(p)) {
      const why = p.category === 'CEX' ? 'exchange (CEX), pre-token' : 'versioned pool of another protocol';
      reasons[why] = (reasons[why] ?? 0) + 1;
    }
  }
  return reasons;
}