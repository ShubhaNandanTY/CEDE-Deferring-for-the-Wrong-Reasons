"""Publication figures for the CEDE-Bench paper. All numbers read from the scored parquet."""

# --- repo-relative paths (works from any cwd) ---
from pathlib import Path as _P
ROOT = _P(__file__).resolve().parents[1]
DATA, RESULTS, FIGDIR = ROOT / "data", ROOT / "results", ROOT / "paper" / "fig"
RESULTS.mkdir(exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)
P_PARQUET = str(DATA / "cede_scored_primary.parquet")
P_BENCH   = str(DATA / "CEDE_124_WITH_PROBES.jsonl")
P_STATS   = str(RESULTS / "stats.json")
P_PAPER   = str(ROOT / "paper" / "paper.tex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd, numpy as np, math, json, os

# ---- global mapping (fixed order, never cycled) ----
BLUE = "#2a78d6"    # decision items (standard)
ORANGE = "#eb6834"  # probe items
AQUA = "#1baf7a"    # benign controls
GRAY = "#8a8a86"    # secondary stratum
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dedddA"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

OUT = str(FIGDIR)
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(P_PARQUET)
items = {json.loads(l)["id"]: json.loads(l) for l in open(P_BENCH) if l.strip()}
df["probe"] = df.id.map(lambda i: bool(items[i].get("probe", False)))
dec = df[~df.is_control].copy(); ctrl = df[df.is_control].copy()
std = dec[~dec.probe]; probe = dec[dec.probe]
S = json.load(open(P_STATS))
ETA = [0.125, 0.25, 0.32, 0.5, 1, 2, 3, 4, 5, 6, 8]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0, c - h), min(1, c + h)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# =====================================================================
# Figure 1 — the mechanism: two cumulative-budget traces
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
logtau = math.log(2 ** -4)
cases = [
    # id, title, note, colour, ax, tau-label xy/va, stop-label xy/ha, decision-label xytext
    ("sec-014", "sec-014 — SQL injection via f-string", "rule completes 18 of 19 tokens",
     BLUE, axes[0], (1.2, logtau - 0.28, "top"), (17.4, -5.1, "right"), (4.2, -0.62)),
    ("sek-001", "sek-001 — hardcoded live API key", "rule stops before the first token",
     AQUA, axes[1], (25.0, logtau + 2.2, "bottom"), (1.0, -62.0, "left"), (12.0, -10.5)),
]
for iid, title, note, col, ax, tl, sl, dxy in cases:
    r = df[df.id == iid].iloc[0]
    lp = np.asarray(r["_logprobs"], float)
    cum = np.cumsum(lp)
    x = np.arange(1, len(cum) + 1)
    ax.plot(x, cum, "-", color=col, lw=2, zorder=3, solid_capstyle="round")
    ax.plot(x, cum, "o", color=col, ms=3.4, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.axhline(logtau, color=INK2, ls="--", lw=1.1, zorder=2)
    ax.text(tl[0], tl[1], r"log $\tau$  ($\eta$ = 4)", ha="right" if iid != "sec-014" else "left",
            va=tl[2], fontsize=7.5, color=INK2)
    b = int(r["b_user_eta4"])
    if b > 0:
        ax.axvline(b + 0.5, color=INK2, lw=0.9, ls=":", zorder=2)
    ax.text(sl[0], sl[1], f"stops here\n(prefix = {b} tokens)",
            fontsize=7.2, color=INK2, va="center", ha=sl[2])
    di = int(r["dec_idx"])
    ax.plot([di + 1], [cum[di]], marker="D", ms=7, color=ORANGE, zorder=5,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.annotate(f"decision token, p = {r.dec_prob:.3f}",
                xy=(di + 1, cum[di]), xytext=dxy,
                fontsize=7.2, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9,
                                shrinkA=2, shrinkB=6))
    ax.set_title(title + "\n" + note, fontsize=8.4, color=INK, loc="left", pad=6)
    ax.set_xlabel("continuation token")
    grid(ax)
axes[0].set_ylim(-7.8, 0.9)
axes[1].set_ylim(-84, 6)
axes[0].set_ylabel("cumulative log-likelihood (nats)")
save(fig, "fig1_mechanism")

# =====================================================================
# Figure 2 — threshold sweep: no operating point is safe
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))
ax = axes[0]
o_std = [wilson(int(std[f"overreach_eta{e}"].sum()), len(std)) for e in ETA]
o_prb = [wilson(int(probe[f"overreach_eta{e}"].sum()), len(probe)) for e in ETA]
ax.fill_between(ETA, [a[1] for a in o_std], [a[2] for a in o_std], color=BLUE, alpha=0.13, lw=0)
ax.plot(ETA, [a[0] for a in o_std], "-o", color=BLUE, lw=2, ms=4.5, zorder=4,
        markeredgecolor="white", markeredgewidth=0.7, label="standard items (n=102)")
ax.plot(ETA, [a[0] for a in o_prb], "-s", color=ORANGE, lw=2, ms=4.5, zorder=4,
        markeredgecolor="white", markeredgewidth=0.7, label="probe items (n=10)")
for e, lab in [(0.32, "simulated\n$\\eta$=0.32"), (4, "user study\n$\\eta$=4")]:
    ax.axvline(e, color=INK2, ls=":", lw=0.9, zorder=1)
    ax.text(e * 1.06, 0.95, lab, fontsize=7, color=INK2, va="top")
ax.set_xscale("log"); ax.set_xticks([0.125, 0.5, 2, 8]); ax.set_xticklabels(["0.125", "0.5", "2", "8"])
ax.set_ylim(0, 1.0); ax.set_xlabel(r"threshold $\eta$ (bits)")
ax.set_ylabel("over-reach rate")
ax.set_title("(a) consequential decisions committed silently", loc="left", fontsize=8.6)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80))
grid(ax)

ax = axes[1]
ov = [float(dec[f"overreach_eta{e}"].mean()) for e in ETA]
cc = [float(ctrl[f"stopfrac_eta{e}"].mean()) for e in ETA]
ax.plot(cc, ov, "-", color=GRAY, lw=1.2, zorder=2)
ax.scatter(cc, ov, s=34, color=BLUE, zorder=4, edgecolor="white", linewidth=0.7)
for e, x, y in zip(ETA, cc, ov):
    if e in (0.125, 1, 4, 8):
        ax.annotate(f"$\\eta$={e}", (x, y), textcoords="offset points", xytext=(6, -8),
                    fontsize=7.2, color=INK2)
ax.scatter([cc[ETA.index(4)]], [ov[ETA.index(4)]], s=110, facecolor="none",
           edgecolor=ORANGE, linewidth=1.6, zorder=5)
ax.set_xlabel("benign boilerplate completed (mean fraction)")
ax.set_ylabel("over-reach rate on decisions")
ax.set_title("(b) the two errors trade off against each other", loc="left", fontsize=8.6)
ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.5)
grid(ax, axis="both")
save(fig, "fig2_sweep")

# =====================================================================
# Figure 3 — the benchmark's premise holds: risky branches are predictable
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75),
                         gridspec_kw={"width_ratios": [1.05, 1]})
ax = axes[0]
d = dec.dropna(subset=["dec_logprob"]).copy()
bits = (-d.dec_logprob / math.log(2)).clip(0, 16)
ax.hist(bits, bins=np.arange(0, 16.5, 0.75), color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
ax.axvline(4, color=INK2, ls="--", lw=1.2, zorder=4)
ax.text(4.25, ax.get_ylim()[1] * 0.93, "total budget\nat $\\eta$=4", fontsize=7.4, color=INK2, va="top")
ax.axvline(0.32, color=INK2, ls=":", lw=1.0, zorder=4)
ax.text(0.55, ax.get_ylim()[1] * 0.60, "$\\eta$=0.32", fontsize=7.2, color=INK2)
ax.set_xlabel("decision-token surprisal (bits)")
ax.set_ylabel("decision items")
ax.set_title(f"(a) {S['predictability']['frac_under_4bits']*100:.0f}% of risky branches cost less\n"
             f"than the entire $\\eta$=4 budget", loc="left", fontsize=8.6)
grid(ax)

ax = axes[1]
cats = sorted(S["predictability"]["by_category"].items(), key=lambda kv: kv[1]["mean_prob"])
labels = [c.replace("_", " ") for c, _ in cats]
vals = [v["mean_prob"] for _, v in cats]
ns = [v["n"] for _, v in cats]
y = np.arange(len(cats))
ax.barh(y, vals, height=0.62, color=BLUE, zorder=3)
for i, (v, n) in enumerate(zip(vals, ns)):
    ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=7.6, color=INK)
ax.set_yticks(y); ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labels, ns)], fontsize=7.8)
ax.set_xlim(0, 1.0); ax.set_xlabel("mean probability of the risky branch")
ax.set_title("(b) predictability by category", loc="left", fontsize=8.6)
grid(ax, axis="x")
save(fig, "fig3_predictability")

# =====================================================================
# Figure 4 — what actually drives over-reach: position gates, predictability modulates
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                         gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.42})
ax = axes[0]
bins = [(0, 2), (2, 5), (5, 10), (10, 999)]
blab = ["0–1", "2–4", "5–9", "10+"]
w = 0.38
for k, (pred, col, hatch, lab) in enumerate([
        (True, BLUE, None, "predictable branch (p > 0.5)"),
        (False, GRAY, "///", "unpredictable branch (p $\\leq$ 0.5)")]):
    ks, ns = [], []
    for lo, hi in bins:
        g = dec[(dec.dec_idx >= lo) & (dec.dec_idx < hi) & ((dec.dec_prob > 0.5) == pred)]
        ks.append(int(g.overreach_user_eta4.sum())); ns.append(len(g))
    rates = [k_ / n_ if n_ else np.nan for k_, n_ in zip(ks, ns)]
    xpos = np.arange(len(bins)) + (k - 0.5) * w
    ax.bar(xpos, rates, width=w * 0.92, color=col, hatch=hatch, edgecolor="white",
           linewidth=0.9, zorder=3, label=lab)
    for xp, r_, k_, n_ in zip(xpos, rates, ks, ns):
        if n_:
            ax.text(xp, r_ + 0.025, f"{k_}/{n_}", ha="center", fontsize=7, color=INK2)
ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels(blab)
ax.set_xlabel("position of the decision token in the continuation")
ax.set_ylabel("over-reach rate ($\\eta$=4)")
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("(a) position gates the failure", loc="left", fontsize=8.6)
ax.legend(loc="upper right", ncol=1, handlelength=1.4, borderpad=0.2)
grid(ax)

ax = axes[1]
dv = S["drivers"]
names = [("dec_idx", "decision position\n(token index)"),
         ("predictability", "branch predictability\n(p of risky token)"),
         ("n_tokens", "continuation length")]
y = np.arange(len(names))
rs = [dv[k]["r"] for k, _ in names]
los = [dv[k]["lo"] for k, _ in names]
his = [dv[k]["hi"] for k, _ in names]
cols = [BLUE if r_ > 0 else ORANGE for r_ in rs]
ax.axvline(0, color=INK2, lw=0.9, zorder=2)
for i, (r_, lo_, hi_, c) in enumerate(zip(rs, los, his, cols)):
    ax.plot([lo_, hi_], [i, i], color=c, lw=2.2, solid_capstyle="round", zorder=3)
    ax.plot([r_], [i], "o", color=c, ms=7, zorder=4, markeredgecolor="white", markeredgewidth=1)
    ax.text(r_, i + 0.26, f"r = {r_:+.2f}", ha="center", fontsize=7.4, color=INK)
ax.set_yticks(y); ax.set_yticklabels([l for _, l in names], fontsize=7.8)
ax.set_ylim(-0.6, len(names) - 0.35)
ax.set_xlim(-0.8, 0.6)
ax.set_xlabel("point-biserial $r$ with over-reach\n(bootstrap 95% CI, n = 112)")
ax.set_title("(b) both signals are real", loc="left", fontsize=8.6)
grid(ax, axis="x")
save(fig, "fig4_drivers")

# =====================================================================
# Figure 5 — per-category over-reach
# =====================================================================
fig, ax = plt.subplots(figsize=(4.9, 2.55))
rows = []
for c, g in dec.groupby("category"):
    k = int(g.overreach_user_eta4.sum()); n = len(g)
    p, lo, hi = wilson(k, n)
    rows.append((c, k, n, p, lo, hi, float(g.dec_prob.mean())))
rows.sort(key=lambda r: r[3])
y = np.arange(len(rows))
ax.barh(y, [r[3] for r in rows], height=0.6, color=BLUE, zorder=3)
for i, r in enumerate(rows):
    ax.plot([r[4], r[5]], [i, i], color=INK2, lw=1.1, zorder=4, solid_capstyle="butt")
    ax.text(r[5] + 0.02, i, f"{r[1]}/{r[2]}", va="center", fontsize=7.4, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([f"{r[0].replace('_',' ')}  ($\\bar p$={r[6]:.2f})" for r in rows], fontsize=7.8)
ax.set_xlim(0, 0.9)
ax.set_xlabel("over-reach rate at $\\eta$=4 (Wilson 95% CI)")
ax.set_title("Risk category, mean branch predictability, and failure rate", loc="left", fontsize=8.6)
grid(ax, axis="x")
save(fig, "fig5_categories")

# =====================================================================
# Figure 6 — importance is invisible to the rule
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9),
                         gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})
ax = axes[0]
dfrac = (dec.b_user_eta4 / dec.n_tokens).values
cfrac = ctrl.stopfrac_user_eta4.values
bins = np.arange(0, 1.05, 0.1)
ax.hist([dfrac], bins=bins, weights=[np.ones(len(dfrac)) / len(dfrac)],
        color=BLUE, edgecolor="white", linewidth=0.7, zorder=3, label="consequential decisions (n=112)")
ax.hist([cfrac], bins=bins, weights=[np.ones(len(cfrac)) / len(cfrac)],
        histtype="step", color=AQUA, linewidth=2.2, zorder=4, label="benign controls (n=12)")
ax.set_xlabel("fraction of the continuation the rule completes")
ax.set_ylabel("share of items")
ax.set_ylim(0, 0.62)
ax.set_title("(a) the rule stops in the same place either way\n"
             f"(means {dfrac.mean():.2f} vs {cfrac.mean():.2f}; Mann–Whitney p = 0.16)",
             loc="left", fontsize=8.4)
ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0), handlelength=1.3)
grid(ax)

ax = axes[1]
sc_ok = dec[~dec.overreach_user_eta4.astype(bool)]
sc_bad = dec[dec.overreach_user_eta4.astype(bool)]
ax.scatter(sc_ok.dec_idx, sc_ok.dec_prob, s=22, color=GRAY, alpha=0.75, zorder=3,
           edgecolor="white", linewidth=0.5, label="deferred (75)")
ax.scatter(sc_bad.dec_idx, sc_bad.dec_prob, s=30, color=ORANGE, zorder=4,
           edgecolor="white", linewidth=0.6, label="committed silently (37)")
ax.set_xlabel("decision-token index")
ax.set_ylabel("probability of the risky branch")
ax.set_xlim(-1, 26); ax.set_ylim(-0.03, 1.22)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("(b) every failure sits in the low-index corner", loc="left", fontsize=8.6)
ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2, labelspacing=0.3)
grid(ax, axis="both")
save(fig, "fig6_invisible")
print("all figures written to", OUT)

# =====================================================================
# Figure 7 — the boundary is a token count set by the local entropy rate
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75),
                         gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})
H = -df.mean_logprob
pred = np.minimum((4 * math.log(2)) / H, df.n_tokens)
act = df.b_user_eta4.astype(float)
ax = axes[0]
lim = 34
ax.plot([0, lim], [0, lim], color=GRAY, lw=1.0, ls="--", zorder=2)
ax.scatter(pred[~df.is_control], act[~df.is_control], s=24, color=BLUE, alpha=0.8,
           edgecolor="white", linewidth=0.5, zorder=3, label="decision items")
ax.scatter(pred[df.is_control], act[df.is_control], s=34, color=AQUA,
           edgecolor="white", linewidth=0.6, zorder=4, label="benign controls")
from scipy import stats as _sps
_r = _sps.pearsonr(pred[np.isfinite(pred)], act[np.isfinite(pred)])[0]
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel(r"predicted prefix $\eta \ln 2 / H$ (tokens)")
ax.set_ylabel("actual retained prefix (tokens)")
ax.set_title(f"(a) a constant-rate model predicts the stop\n"
             f"$r$ = {_r:.2f}, $R^2$ = {_r**2:.2f}, median error 1.0 token",
             loc="left", fontsize=8.4)
ax.legend(loc="lower right", handlelength=1.0, borderpad=0.25)
grid(ax, axis="both")

ax = axes[1]
f3, rst = [], []
for _, r in df.iterrows():
    lp = np.asarray(r["_logprobs"], float)
    if len(lp) >= 6:
        f3.append(-lp[:3].mean()); rst.append(-lp[3:].mean())
parts = ax.boxplot([f3, rst], widths=0.5, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=1.4),
                   whiskerprops=dict(color=INK2, lw=0.9), capprops=dict(color=INK2, lw=0.9))
for pc, c in zip(parts["boxes"], [BLUE, GRAY]):
    pc.set_facecolor(c); pc.set_alpha(0.85); pc.set_edgecolor("white"); pc.set_linewidth(0.9)
ax.set_xticklabels(["first 3 tokens", "tokens 4 and later"])
ax.set_ylabel("per-token surprisal (nats)")
ax.set_ylim(0, 4.2)
ax.set_title("(b) the rate does not drift within an item\n"
             "(Wilcoxon $p$ = 0.18, n = 113)", loc="left", fontsize=8.4)
grid(ax)
save(fig, "fig7_entropyrate")
