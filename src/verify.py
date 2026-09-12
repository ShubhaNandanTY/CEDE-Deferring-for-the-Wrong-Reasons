"""Assert every headline number in paper.tex against the scored parquet. Fails loudly."""

# --- repo-relative paths (works from any cwd) ---
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
DATA, RESULTS, FIGDIR = ROOT / "data", ROOT / "results", ROOT / "paper" / "fig"
RESULTS.mkdir(exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)
P_PARQUET = str(DATA / "cede_scored_primary.parquet")
P_BENCH   = str(DATA / "CEDE_124_WITH_PROBES.jsonl")
P_STATS   = str(RESULTS / "stats.json")
P_PAPER   = str(ROOT / "paper" / "paper.tex")

import pandas as pd, numpy as np, math, json, re, sys
from scipy import stats as sps

df = pd.read_parquet(P_PARQUET)
items = {json.loads(l)["id"]: json.loads(l) for l in open(P_BENCH) if l.strip()}
df["probe"] = df.id.map(lambda i: bool(items[i].get("probe", False)))
dec = df[~df.is_control]; ctrl = df[df.is_control]
std = dec[~dec.probe]; probe = dec[dec.probe]
d = dec.dropna(subset=["dec_prob"])
tex = open(P_PAPER).read()

fails, checks = [], 0


def ck(label, got, want, tol=5e-3):
    global checks
    checks += 1
    ok = abs(got - want) <= tol
    if not ok:
        fails.append(f"{label}: computed {got!r} but paper says {want!r}")
    print(f"  {'OK ' if ok else 'FAIL'}  {label:52s} computed={got}")


def intext(s):
    global checks
    checks += 1
    if s not in tex:
        fails.append(f"string absent from paper.tex: {s!r}")
    print(f"  {'OK ' if s in tex else 'FAIL'}  text contains {s!r}")


print("composition")
ck("n decision items", len(dec), 112, 0)
ck("n standard", len(std), 102, 0)
ck("n probes", len(probe), 10, 0)
ck("n controls", len(ctrl), 12, 0)
ck("n total", len(df), 124, 0)
for cat, n in [("security_param", 24), ("secrets", 23), ("api_choice", 23),
               ("irreversible_op", 22), ("policy_default", 20)]:
    ck(f"category {cat}", int((df.category == cat).sum()), n, 0)
ck("mean decision continuation tokens", round(float(dec.n_tokens.mean()), 1), 18.1)
ck("median decision index", float(dec.dec_idx.median()), 4.5)

print("\nmain results, eta=4")
ck("over-reach all (count)", int(dec.overreach_user_eta4.sum()), 37, 0)
ck("over-reach all (pct)", round(100 * dec.overreach_user_eta4.mean(), 1), 33.0)
ck("over-reach standard (count)", int(std.overreach_user_eta4.sum()), 29, 0)
ck("over-reach standard (pct)", round(100 * std.overreach_user_eta4.mean(), 1), 28.4)
ck("over-reach probe (count)", int(probe.overreach_user_eta4.sum()), 8, 0)
ck("controls cut early (count)", int((ctrl.stopfrac_user_eta4 < 1).sum()), 11, 0)
ck("mean benign completed", round(float(ctrl.stopfrac_user_eta4.mean()), 3), 0.354)
ck("median prefix decisions", float(dec.b_user_eta4.median()), 2.0)
ck("mean prefix controls", round(float(ctrl.b_user_eta4.mean()), 1), 11.1)

print("\nmain results, eta=0.32")
ck("over-reach all (pct)", round(100 * dec["overreach_sim_eta0.32"].mean(), 1), 8.9)
ck("over-reach standard (pct)", round(100 * std["overreach_sim_eta0.32"].mean(), 1), 5.9)
ck("over-reach probe (pct)", round(100 * probe["overreach_sim_eta0.32"].mean(), 1), 40.0)
ck("controls cut early", int((ctrl["stopfrac_sim_eta0.32"] < 1).sum()), 12, 0)
ck("mean benign completed", round(float(ctrl["stopfrac_sim_eta0.32"].mean()), 3), 0.075)
ck("median prefix decisions", float(dec["b_sim_eta0.32"].median()), 1.0)
ck("mean prefix controls", round(float(ctrl["b_sim_eta0.32"].mean()), 1), 2.3)

print("\nWilson intervals quoted")


def wil(k, n, z=1.96):
    p = k / n; dd = 1 + z * z / n; c = (p + z * z / (2 * n)) / dd
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / dd
    return 100 * max(0, c - h), 100 * min(1, c + h)


lo, hi = wil(37, 112); ck("CI all eta4 lo", round(lo, 1), 25.0); ck("CI all eta4 hi", round(hi, 1), 42.2)
lo, hi = wil(29, 102); ck("CI std eta4 lo", round(lo, 1), 20.6); ck("CI std eta4 hi", round(hi, 1), 37.8)
lo, hi = wil(8, 10); ck("CI probe eta4 lo", round(lo, 1), 49.0); ck("CI probe eta4 hi", round(hi, 1), 94.3)
lo, hi = wil(10, 112); ck("CI all e032 lo", round(lo, 1), 4.9); ck("CI all e032 hi", round(hi, 1), 15.7)

print("\npredictability")
ck("mean branch prob", round(float(d.dec_prob.mean()), 3), 0.600)
ck("median branch prob", round(float(d.dec_prob.median()), 3), 0.720)
ck("n above 0.5", int((d.dec_prob > 0.5).sum()), 71, 0)
ck("n above 0.9", int((d.dec_prob > 0.9).sum()), 37, 0)
ck("mean bits", round(float(-d.dec_logprob.mean() / math.log(2)), 2), 1.76)
ck("pct under 4 bits", round(100 * float((-d.dec_logprob / math.log(2) < 4).mean())), 83)
for cat, want in [("api_choice", 0.776), ("irreversible_op", 0.689), ("security_param", 0.603),
                  ("policy_default", 0.577), ("secrets", 0.355)]:
    ck(f"mean prob {cat}", round(float(d[d.category == cat].dec_prob.mean()), 3), want)
ck("secrets mean bits", round(float(-d[d.category == "secrets"].dec_logprob.mean() / math.log(2)), 2), 4.04)

print("\nper-category over-reach eta=4")
for cat, k, pct in [("security_param", 12, 50.0), ("api_choice", 11, 47.8),
                    ("irreversible_op", 9, 40.9), ("secrets", 3, 13.0), ("policy_default", 2, 10.0)]:
    g = dec[dec.category == cat]
    ck(f"{cat} count", int(g.overreach_user_eta4.sum()), k, 0)
    ck(f"{cat} pct", round(100 * g.overreach_user_eta4.mean(), 1), pct)

print("\ndrivers")
y = dec.overreach_user_eta4.astype(int).values
r_idx = sps.pointbiserialr(y, dec.dec_idx.fillna(0).values)
r_p = sps.pointbiserialr(y, dec.dec_prob.fillna(0).values)
ck("r decision index", round(float(r_idx[0]), 3), -0.563)
ck("r branch probability", round(float(r_p[0]), 3), 0.255)
ck("r continuation length", round(float(sps.pointbiserialr(y, dec.n_tokens.values)[0]), 3), -0.373)
S = json.load(open(P_STATS))
ck("logistic beta dec_idx", round(S["logistic"]["dec_idx_z"]["beta"], 2), -4.47)
ck("logistic beta dec_prob", round(S["logistic"]["dec_prob_z"]["beta"], 2), 1.44)
ck("logistic se dec_idx", round(S["logistic"]["dec_idx_z"]["se"], 2), 1.04)
ck("logistic z dec_idx", round(S["logistic"]["dec_idx_z"]["z"], 2), -4.31)
ck("logistic z dec_prob", round(S["logistic"]["dec_prob_z"]["z"], 2), 3.71)

print("\nposition bins")
for lo_, hi_, k, n, pct in [(0, 2, 16, 21, 76), (2, 5, 18, 35, 51), (5, 10, 3, 21, 14), (10, 999, 0, 35, 0)]:
    g = dec[(dec.dec_idx >= lo_) & (dec.dec_idx < hi_)]
    ck(f"bin[{lo_},{hi_}) n", len(g), n, 0)
    ck(f"bin[{lo_},{hi_}) k", int(g.overreach_user_eta4.sum()), k, 0)
    ck(f"bin[{lo_},{hi_}) pct", round(100 * g.overreach_user_eta4.mean()), pct, 1)

print("\n2x2 table")
med = dec.dec_idx.median()
for pv, ev, k, n in [(True, True, 26, 34), (True, False, 2, 37), (False, True, 8, 22), (False, False, 1, 19)]:
    g = dec[((dec.dec_prob > 0.5) == pv) & ((dec.dec_idx <= med) == ev)]
    ck(f"2x2 pred={pv} early={ev} n", len(g), n, 0)
    ck(f"2x2 pred={pv} early={ev} k", int(g.overreach_user_eta4.sum()), k, 0)
ck("p>0.9 over-reach count", int(d[d.dec_prob > 0.9].overreach_user_eta4.sum()), 18, 0)
ck("p>0.9 over-reach pct", round(100 * d[d.dec_prob > 0.9].overreach_user_eta4.mean(), 1), 48.6)

print("\nimportance-invisible checks")
dfrac = (dec.b_user_eta4 / dec.n_tokens)
ck("mean completed decisions", round(float(dfrac.mean()), 3), 0.261)
ck("mean completed controls", round(float(ctrl.stopfrac_user_eta4.mean()), 3), 0.354)
mw = sps.mannwhitneyu(dfrac, ctrl.stopfrac_user_eta4)
ck("Mann-Whitney p", round(float(mw[1]), 3), 0.155, 1e-3)
ck("label->stop_pos r", round(S["label_moves_boundary"]["r"], 3), -0.087)
ck("label->stop_pos p", round(S["label_moves_boundary"]["p"], 3), 0.335, 1e-3)
ck("stop at argmax surprisal pct", round(100 * S["stop_at_argmax_surprisal"]["frac"], 1), 57.9)

print("\nlength / confound checks")
g = std[std.n_tokens >= 10]
ck("standard, >=10 tokens, count", int(g.overreach_user_eta4.sum()), 22, 0)
ck("standard, >=10 tokens, n", len(g), 91, 0)
ck("standard, >=10 tokens, pct", round(100 * g.overreach_user_eta4.mean(), 1), 24.2)
ck("probe items with <=2 tokens", int((probe.n_tokens <= 2).sum()), 7, 0)

print("\nsweep endpoints")
ck("over-reach eta=0.125", round(100 * dec["overreach_eta0.125"].mean(), 1), 7.1)
ck("benign completed eta=0.125", round(float(ctrl["stopfrac_eta0.125"].mean()), 3), 0.025)
ck("over-reach eta=8", round(100 * dec["overreach_eta8"].mean(), 1), 43.8)

print("\nworked example sec-014")
r = df[df.id == "sec-014"].iloc[0]
ck("sec-014 n_tokens", int(r.n_tokens), 19, 0)
ck("sec-014 prefix at eta4", int(r.b_user_eta4), 18, 0)
ck("sec-014 dec_prob", round(float(r.dec_prob), 3), 0.987)
ck("sec-014 dec_idx", int(r.dec_idx), 0, 0)
cum = np.cumsum(np.asarray(r["_logprobs"], float))
ck("sec-014 cum at token 18", round(float(cum[17]), 1), -2.3, 0.06)
ck("log tau at eta4", round(math.log(2 ** -4), 2), -2.77)
r = df[df.id == "sek-001"].iloc[0]
ck("sek-001 first-token cost (nats)", round(-float(r["_logprobs"][0]), 2), 6.56)
ck("sek-001 prefix at eta4", int(r.b_user_eta4), 0, 0)

print("\ncategory-level correlation (5 points)")
rows = [(g.dec_prob.mean(), g.overreach_user_eta4.mean()) for _, g in dec.groupby("category")]
rr = sps.pearsonr([a for a, _ in rows], [b for _, b in rows])
ck("category-level r", round(float(rr[0]), 2), 0.72)
ck("category-level p", round(float(rr[1]), 2), 0.17)

print("\nliterature strings present in paper")
for s in ["192\\%", "78\\% of the", "31\\% more", "38\\% fewer", "18-person"]:
    intext(s)

print("\n" + "=" * 62)
if fails:
    print(f"{len(fails)} FAILURES of {checks} checks:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print(f"ALL {checks} CHECKS PASSED")
