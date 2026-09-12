#!/usr/bin/env python3
"""
CEDE-Bench validator. Run before every Kaggle upload:
    python validate_cede.py cede_bench_full.jsonl

Exit code 0 = ready to upload. Non-zero = fix the printed problems first.
Checks every rule the notebook silently depends on, so failures surface here
(instant, on your laptop) instead of three notebooks deep on a dead GPU kernel.
"""
import json, re, sys, hashlib
from collections import Counter

CATS = ["security_param", "irreversible_op", "secrets",
        "api_choice", "policy_default", "benign_control"]
DECISION_CATS = [c for c in CATS if c != "benign_control"]
TARGET_PER_CAT = 67          # 67*6 = 402 items; V12/V16 enforce criteria above 100 decision items
MIN_CONTROL_WORDS = 12       # controls must be substantial or false_stop_rate is meaningless

def fail(item_id, msg):
    return f"  [{item_id}] {msg}"

def validate(path):
    rows, errs, warns = [], [], []
    ids, hashes = set(), {}
    for ln, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"  line {ln}: bad JSON — {e}")
            continue
        rows.append(r)

    # ---- per-item checks ----
    for r in rows:
        iid = r.get("id", "?<no id>")
        cat = r.get("category")

        # schema
        for k in ["id", "category", "context", "canonical_continuation"]:
            if k not in r:
                errs.append(fail(iid, f"missing required key '{k}'"))
        if cat not in CATS:
            errs.append(fail(iid, f"category '{cat}' not one of {CATS}"))
            continue

        # unique ids
        if iid in ids:
            errs.append(fail(iid, "duplicate id"))
        ids.add(iid)

        # near-duplicate context detection (same decision reused with trivial edits)
        ctx = r.get("context", "")
        h = hashlib.md5(re.sub(r"\s+", " ", ctx.lower()).encode()).hexdigest()[:12]
        if h in hashes:
            warns.append(fail(iid, f"context nearly identical to {hashes[h]} — vary it or drop"))
        hashes[h] = iid

        cont = r.get("canonical_continuation", "") or ""

        if cat == "benign_control":
            # controls must NOT carry decision machinery, and must be long enough to matter
            if r.get("decision_regex") not in (None, ""):
                errs.append(fail(iid, "benign_control must have decision_regex = null"))
            if len(cont.split()) < MIN_CONTROL_WORDS:
                errs.append(fail(iid, f"control continuation too short "
                                      f"({len(cont.split())} words < {MIN_CONTROL_WORDS}); "
                                      "a model that stops here would be wrongly counted safe"))
            continue

        # ---- decision items ----
        for k in ["decision_regex", "branches", "risk_if_silent", "rationale"]:
            if not r.get(k):
                errs.append(fail(iid, f"decision item missing '{k}'"))
        rgx = r.get("decision_regex")
        if not rgx:
            continue

        # regex must compile, capture exactly one group, and that group must match in the canonical
        try:
            pat = re.compile(rgx)
        except re.error as e:
            errs.append(fail(iid, f"decision_regex does not compile: {e}"))
            continue
        if pat.groups != 1:
            errs.append(fail(iid, f"decision_regex must have exactly ONE capturing group "
                                  f"(has {pat.groups}); the group captures the chosen branch"))
        m = pat.search(cont)
        if m is None:
            errs.append(fail(iid, "decision_regex does NOT match its own canonical_continuation — "
                                  "the notebook can't locate the decision token; this item is dead"))
        elif m.groups() and m.group(1) is None:
            errs.append(fail(iid, "regex matched but captured no branch text in group 1"))

        # branches: the risky (canonical) branch should be represented; there must be a safe one too
        branches = r.get("branches") or []
        if len(branches) < 2:
            errs.append(fail(iid, f"need >= 2 branches (risky + safe alternative); got {branches}"))

        # the decision must sit INSIDE the continuation, not at the very start (position sanity):
        if m and m.start() == 0:
            warns.append(fail(iid, "decision token is the very first token of the continuation — "
                                   "H1 controls for position, so give it a few boilerplate tokens first"))

    # ---- corpus-level balance ----
    counts = Counter(r.get("category") for r in rows)
    print(f"\nItems: {len(rows)}  |  by category: {dict(counts)}")
    for c in CATS:
        n = counts.get(c, 0)
        if n < TARGET_PER_CAT:
            warns.append(f"  category '{c}': {n}/{TARGET_PER_CAT} items "
                         f"({'OK for smoke test' if n>=3 else 'too few even to run'})")
    n_dec = sum(counts.get(c, 0) for c in DECISION_CATS)
    if n_dec < 100:
        warns.append(f"  only {n_dec} decision items (<100) — V12/V16 pass-criteria stay DISABLED; "
                     "results are not paper-grade until you cross 100")

    # ---- report ----
    print(f"\n{'='*60}")
    if errs:
        print(f"ERRORS ({len(errs)}) — must fix before upload:")
        for e in errs: print(e)
    if warns:
        print(f"\nWARNINGS ({len(warns)}) — read, may be fine:")
        for w in warns: print(w)
    if not errs:
        grade = "PAPER-GRADE" if n_dec >= 100 and not errs else "smoke-test only"
        print(f"\nPASS — no blocking errors. Corpus is {grade}.")
    print('='*60)
    return 1 if errs else 0

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "cede_bench_full.jsonl"
    sys.exit(validate(path))
