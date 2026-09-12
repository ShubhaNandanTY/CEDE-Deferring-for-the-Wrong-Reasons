"""Full recomputation of CEDE-Bench results from the scored parquet.
Every number that appears in the paper is emitted here, into stats.json."""

# --- repo-relative paths (works from any cwd) ---
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
DATA, RESULTS, FIGDIR = ROOT / "data", ROOT / "results", ROOT / "paper" / "fig"
RESULTS.mkdir(exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)
P_PARQUET = str(DATA / "cede_scored_primary.parquet")
P_BENCH   = str(DATA / "CEDE_124_WITH_PROBES.jsonl")
P_STATS   = str(RESULTS / "stats.json")
P_PAPER   = str(ROOT / "paper" / "paper.tex")

import pandas as pd, numpy as np, math, json
from scipy import stats as sps

SEED = 0
BOOT = 5000
rng = np.random.default_rng(SEED)

df = pd.read_parquet(P_PARQUET)
items = [json.loads(l) for l in open(P_BENCH) if l.strip()]
meta = {it['id']: it for it in items}
df['probe'] = df.id.map(lambda i: bool(meta[i].get('probe', False)))

dec = df[~df.is_control].copy()
ctrl = df[df.is_control].copy()
std = dec[~dec.probe].copy()
probe = dec[dec.probe].copy()

ETA_GRID = [0.125, 0.25, 0.32, 0.5, 1, 2, 3, 4, 5, 6, 8]
S = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rate(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "p": p, "lo": lo, "hi": hi}


# ---------- basic composition ----------
S['n_total'] = len(df)
S['n_decision'] = len(dec)
S['n_control'] = len(ctrl)
S['n_probe'] = len(probe)
S['n_standard'] = len(std)
S['category_counts'] = df.category.value_counts().to_dict()
S['mean_tokens_dec'] = float(dec.n_tokens.mean())
S['median_tokens_dec'] = float(dec.n_tokens.median())
S['mean_tokens_ctrl'] = float(ctrl.n_tokens.mean())

# ---------- M1 over-reach ----------
S['overreach'] = {}
for tag, eta in [('sim_eta0.32', 0.32), ('user_eta4', 4.0)]:
    S['overreach'][tag] = {
        'eta': eta,
        'all': rate(dec[f'overreach_{tag}'].sum(), len(dec)),
        'standard': rate(std[f'overreach_{tag}'].sum(), len(std)),
        'probe': rate(probe[f'overreach_{tag}'].sum(), len(probe)),
        'by_category': {c: rate(g[f'overreach_{tag}'].sum(), len(g))
                        for c, g in dec.groupby('category')},
    }
S['overreach_ids_eta4'] = dec[dec.overreach_user_eta4.astype(bool)].id.tolist()

# ---------- M2 control over-deferral ----------
S['controls'] = {}
for tag in ['sim_eta0.32', 'user_eta4']:
    early = (ctrl[f'stopfrac_{tag}'] < 1.0)
    S['controls'][tag] = {
        'early_stop': rate(early.sum(), len(ctrl)),
        'mean_stopfrac': float(ctrl[f'stopfrac_{tag}'].mean()),
        'median_stopfrac': float(ctrl[f'stopfrac_{tag}'].median()),
    }

# ---------- M4 predictability ----------
d = dec.dropna(subset=['dec_logprob']).copy()
d['bits'] = -d.dec_logprob / math.log(2)
S['predictability'] = {
    'n': len(d),
    'mean_logprob': float(d.dec_logprob.mean()),
    'median_logprob': float(d.dec_logprob.median()),
    'mean_prob_implied': float(math.exp(d.dec_logprob.mean())),
    'median_prob_implied': float(math.exp(d.dec_logprob.median())),
    'mean_prob': float(d.dec_prob.mean()),
    'median_prob': float(d.dec_prob.median()),
    'mean_bits': float(d.bits.mean()),
    'median_bits': float(d.bits.median()),
    'frac_above_half': float((d.dec_prob > 0.5).mean()),
    'n_above_half': int((d.dec_prob > 0.5).sum()),
    'frac_above_ninetenths': float((d.dec_prob > 0.9).mean()),
    'n_above_ninetenths': int((d.dec_prob > 0.9).sum()),
    'frac_under_4bits': float((d.bits < 4).mean()),
    'by_category': {c: {'n': len(g), 'mean_prob': float(g.dec_prob.mean()),
                        'median_prob': float(g.dec_prob.median()),
                        'mean_bits': float(g.bits.mean()),
                        'frac_above_half': float((g.dec_prob > 0.5).mean())}
                    for c, g in d.groupby('category')},
    'mean_pos_frac': float(d.pos_frac.mean()),
    'mean_dec_idx': float(d.dec_idx.mean()),
}

# ---------- sweep ----------
sweep = []
for e in ETA_GRID:
    tag = f'eta{e}'
    orr = rate(dec[f'overreach_{tag}'].sum(), len(dec))
    orr_std = rate(std[f'overreach_{tag}'].sum(), len(std))
    orr_pr = rate(probe[f'overreach_{tag}'].sum(), len(probe))
    ces = rate((ctrl[f'stopfrac_{tag}'] < 1.0).sum(), len(ctrl))
    sweep.append({'eta': e, 'tau': 2.0 ** (-e),
                  'overreach_all': orr['p'], 'overreach_all_lo': orr['lo'], 'overreach_all_hi': orr['hi'],
                  'overreach_std': orr_std['p'], 'overreach_probe': orr_pr['p'],
                  'control_early': ces['p'],
                  'mean_ctrl_stopfrac': float(ctrl[f'stopfrac_{tag}'].mean())})
S['sweep'] = sweep

# ---------- drivers of over-reach ----------
def boot_pointbiserial(y, x, n=BOOT):
    y = np.asarray(y, float); x = np.asarray(x, float)
    r, p = sps.pointbiserialr(y, x)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(y):
            rs.append(sps.pointbiserialr(y[idx], x[idx])[0])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return {'r': float(r), 'p': float(p), 'lo': float(lo), 'hi': float(hi),
            'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum())}

y = dec.overreach_user_eta4.astype(int).values
S['drivers'] = {
    'predictability': boot_pointbiserial(y, dec.dec_prob.fillna(0).values),
    'dec_bits': boot_pointbiserial(y, (-dec.dec_logprob.fillna(0) / math.log(2)).values),
    'earlier_position': boot_pointbiserial(y, (1 - dec.pos_frac.fillna(0)).values),
    'dec_idx': boot_pointbiserial(y, dec.dec_idx.fillna(0).values),
    'n_tokens': boot_pointbiserial(y, dec.n_tokens.values),
}

# logistic regression: standardized predictors
import numpy.linalg as la
X = np.column_stack([
    dec.dec_prob.fillna(0).values,
    dec.dec_idx.fillna(0).values,
])
Xz = (X - X.mean(0)) / X.std(0)
Xd = np.column_stack([np.ones(len(Xz)), Xz])


def logistic_fit(Xd, y, iters=500):
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xd @ b))
        W = p * (1 - p) + 1e-9
        z = Xd @ b + (y - p) / W
        b = la.solve(Xd.T @ (Xd * W[:, None]) + 1e-8 * np.eye(Xd.shape[1]), Xd.T @ (W * z))
    p = 1 / (1 + np.exp(-Xd @ b))
    W = p * (1 - p) + 1e-9
    cov = la.inv(Xd.T @ (Xd * W[:, None]) + 1e-8 * np.eye(Xd.shape[1]))
    se = np.sqrt(np.diag(cov))
    return b, se


b, se = logistic_fit(Xd, y.astype(float))
names = ['intercept', 'dec_prob_z', 'dec_idx_z']
S['logistic'] = {n: {'beta': float(bb), 'se': float(ss), 'z': float(bb / ss),
                     'p': float(2 * (1 - sps.norm.cdf(abs(bb / ss))))}
                 for n, bb, ss in zip(names, b, se)}

# ---------- is the boundary a pure surprisal detector? ----------
def boundary_cumprod(lp, tau):
    return int(np.sum(np.cumsum(lp) >= math.log(tau)))


al, stops = [], []
for _, r in df.iterrows():
    lp = np.asarray(r['_logprobs'], float)
    bnd = boundary_cumprod(lp, 2 ** -4.0)
    if bnd < r['n_tokens']:
        al.append(abs(bnd - r['argmax_surprisal_idx']) <= 1)
    stops.append(bnd / r['n_tokens'])
S['stop_at_argmax_surprisal'] = {'frac': float(np.mean(al)), 'n': int(len(al))}
df['stop_pos'] = stops
S['label_moves_boundary'] = boot_pointbiserial((~df.is_control).astype(int).values, df.stop_pos.values)
S['mean_stop_pos_dec'] = float(df[~df.is_control].stop_pos.mean())
S['mean_stop_pos_ctrl'] = float(df[df.is_control].stop_pos.mean())

# ---------- budget accounting: how many tokens does eta=4 buy? ----------
# how far does the prefix reach, in tokens, at each operating point
S['prefix_len'] = {}
for tag in ['sim_eta0.32', 'user_eta4']:
    S['prefix_len'][tag] = {
        'dec_mean': float(dec[f'b_{tag}'].mean()),
        'dec_median': float(dec[f'b_{tag}'].median()),
        'ctrl_mean': float(ctrl[f'b_{tag}'].mean()),
        'frac_zero': float((dec[f'b_{tag}'] == 0).mean()),
    }

# among over-reached items, what did the decision token cost?
ov = dec[dec.overreach_user_eta4.astype(bool)]
nv = dec[~dec.overreach_user_eta4.astype(bool)]
S['overreached_items'] = {
    'mean_dec_prob': float(ov.dec_prob.mean()), 'median_dec_prob': float(ov.dec_prob.median()),
    'mean_dec_idx': float(ov.dec_idx.mean()), 'median_dec_idx': float(ov.dec_idx.median()),
}
S['deferred_items'] = {
    'mean_dec_prob': float(nv.dec_prob.mean()), 'median_dec_prob': float(nv.dec_prob.median()),
    'mean_dec_idx': float(nv.dec_idx.mean()), 'median_dec_idx': float(nv.dec_idx.median()),
}
S['mannwhitney_prob'] = {k: float(v) for k, v in
                         zip(['U', 'p'], sps.mannwhitneyu(ov.dec_prob, nv.dec_prob, alternative='greater'))}
S['mannwhitney_idx'] = {k: float(v) for k, v in
                        zip(['U', 'p'], sps.mannwhitneyu(ov.dec_idx, nv.dec_idx, alternative='less'))}

# 2x2: predictable (p>.5) x early (idx <= median) over-reach table
d2 = dec.copy()
d2['predictable'] = d2.dec_prob > 0.5
med_idx = d2.dec_idx.median()
d2['early'] = d2.dec_idx <= med_idx
S['median_dec_idx_split'] = float(med_idx)
cells = {}
for pv in [True, False]:
    for ev in [True, False]:
        g = d2[(d2.predictable == pv) & (d2.early == ev)]
        cells[f'pred={pv},early={ev}'] = rate(g.overreach_user_eta4.sum(), len(g))
S['two_by_two'] = cells

# probe detail
S['probe_detail'] = probe[['id', 'category', 'dec_idx', 'dec_prob', 'n_tokens',
                           'b_user_eta4', 'overreach_user_eta4',
                           'b_sim_eta0.32', 'overreach_sim_eta0.32']].to_dict('records')

json.dump(S, open(P_STATS, "w"), indent=2, default=str)

# ---------- CSV exports for the repo ----------
pd.DataFrame(S['sweep']).to_csv(RESULTS / "sweep.csv", index=False)

rows = []
for tag, eta in [('sim_eta0.32', 0.32), ('user_eta4', 4.0)]:
    o = S['overreach'][tag]
    for name in ['all', 'standard', 'probe']:
        r = o[name]
        rows.append({'eta': eta, 'metric': 'over_reach', 'subset': name,
                     'k': r['k'], 'n': r['n'], 'rate': r['p'],
                     'wilson_lo': r['lo'], 'wilson_hi': r['hi']})
    for c, r in sorted(o['by_category'].items()):
        rows.append({'eta': eta, 'metric': 'over_reach', 'subset': f'category:{c}',
                     'k': r['k'], 'n': r['n'], 'rate': r['p'],
                     'wilson_lo': r['lo'], 'wilson_hi': r['hi']})
    r = S['controls'][tag]['early_stop']
    rows.append({'eta': eta, 'metric': 'control_early_stop', 'subset': 'controls',
                 'k': r['k'], 'n': r['n'], 'rate': r['p'],
                 'wilson_lo': r['lo'], 'wilson_hi': r['hi']})
pd.DataFrame(rows).to_csv(RESULTS / "main_results.csv", index=False)

df.drop(columns=['_logprobs']).to_csv(RESULTS / "per_item.csv", index=False)
print("wrote sweep.csv, main_results.csv, per_item.csv")

# ---------- console report ----------
def pct(r):
    return f"{r['k']}/{r['n']} = {100*r['p']:.1f}% [{100*r['lo']:.1f}, {100*r['hi']:.1f}]"


print("=" * 70)
print(f"composition: {S['n_decision']} decisions ({S['n_standard']} standard + {S['n_probe']} probes), "
      f"{S['n_control']} controls")
print(f"mean continuation length: {S['mean_tokens_dec']:.1f} tokens (decisions), "
      f"{S['mean_tokens_ctrl']:.1f} (controls)")
print("=" * 70)
for tag in ['sim_eta0.32', 'user_eta4']:
    o = S['overreach'][tag]
    print(f"\nOVER-REACH @ eta={o['eta']}")
    print("  all      :", pct(o['all']))
    print("  standard :", pct(o['standard']))
    print("  probe    :", pct(o['probe']))
    for c, r in sorted(o['by_category'].items(), key=lambda kv: -kv[1]['p']):
        print(f"    {c:16s} {pct(r)}")
    print("  control early-stop:", pct(S['controls'][tag]['early_stop']),
          f" mean stopfrac={S['controls'][tag]['mean_stopfrac']:.3f}")
p = S['predictability']
print(f"\nPREDICTABILITY  mean lp={p['mean_logprob']:.3f} (p={p['mean_prob_implied']:.3f}) "
      f"median lp={p['median_logprob']:.3f} (p={p['median_prob_implied']:.3f})")
print(f"  mean p={p['mean_prob']:.3f} median p={p['median_prob']:.3f} mean bits={p['mean_bits']:.2f}")
print(f"  p>0.5: {p['n_above_half']}/{p['n']} = {100*p['frac_above_half']:.1f}%   "
      f"p>0.9: {p['n_above_ninetenths']}/{p['n']}")
for c, g in sorted(p['by_category'].items(), key=lambda kv: -kv[1]['mean_prob']):
    print(f"    {c:16s} n={g['n']:3d} meanP={g['mean_prob']:.3f} medP={g['median_prob']:.3f} "
          f"bits={g['mean_bits']:.2f} p>.5={100*g['frac_above_half']:.0f}%")
print("\nDRIVERS (point-biserial, over-reach @eta=4):")
for k, v in S['drivers'].items():
    print(f"  {k:18s} r={v['r']:+.3f} p={v['p']:.3g} boot95=[{v['lo']:+.3f},{v['hi']:+.3f}] n_pos={v['n_pos']}")
print("LOGISTIC (standardized):")
for k, v in S['logistic'].items():
    print(f"  {k:14s} beta={v['beta']:+.3f} se={v['se']:.3f} z={v['z']:+.2f} p={v['p']:.3g}")
print(f"\nover-reached items: mean p={S['overreached_items']['mean_dec_prob']:.3f}, "
      f"mean idx={S['overreached_items']['mean_dec_idx']:.2f}")
print(f"deferred items    : mean p={S['deferred_items']['mean_dec_prob']:.3f}, "
      f"mean idx={S['deferred_items']['mean_dec_idx']:.2f}")
print(f"MW prob p={S['mannwhitney_prob']['p']:.3g}   MW idx p={S['mannwhitney_idx']['p']:.3g}")
print("\n2x2 (median dec_idx split =", S['median_dec_idx_split'], "):")
for k, v in S['two_by_two'].items():
    print(f"  {k:26s} {pct(v)}")
print(f"\nstop lands within 1 token of most-surprising token: "
      f"{100*S['stop_at_argmax_surprisal']['frac']:.1f}% of {S['stop_at_argmax_surprisal']['n']} stopped items")
lm = S['label_moves_boundary']
print(f"label -> stop_pos: r={lm['r']:+.3f} p={lm['p']:.3g} boot95=[{lm['lo']:+.3f},{lm['hi']:+.3f}]")
print(f"mean stop_pos: decisions={S['mean_stop_pos_dec']:.3f} controls={S['mean_stop_pos_ctrl']:.3f}")
print("\nSWEEP:")
print(f"{'eta':>6} {'tau':>8} {'over_all':>9} {'over_std':>9} {'over_prb':>9} {'ctrl_early':>11}")
for r in S['sweep']:
    print(f"{r['eta']:>6} {r['tau']:>8.4f} {r['overreach_all']:>9.3f} {r['overreach_std']:>9.3f} "
          f"{r['overreach_probe']:>9.3f} {r['control_early']:>11.3f}")
print("\nPREFIX LENGTHS:", json.dumps(S['prefix_len'], indent=2))
print("\nwrote stats.json")
