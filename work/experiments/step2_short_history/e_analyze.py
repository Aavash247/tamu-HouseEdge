"""Analyze Step 2 CV: per-fold short-history Brier delta vs base, nested selection of prior power."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
res = {v: {int(f): rows for f, rows in d.items()} for v, d in json.load(open(HERE / "d_cv_rows.json")).items()}

# sanity: base must equal the Step-0 residual predictions for the same machine/tick
s0 = json.load(open(HERE.parents[0] / "step0_diagnosis/residual_rows.json"))
s0p = {(r["episode"], r["k"], r["machine"]): r["p"] for r in s0 if r["cond"] == "short_history"}
# rows don't store machine; compare multiset per (episode,k)
by = defaultdict(list)
for r in s0:
    if r["cond"] == "short_history" and r["k"] < 12 and r["y"] is not None:
        by[(r["episode"], r["k"])].append(np.array(r["p"]))
mx = 0.0
for f, rows in res["base"].items():
    grp = defaultdict(list)
    for r in rows: grp[(r["episode"], r["k"])].append(np.array(r["p"]))
    for k, ps in grp.items():
        a = sorted(map(tuple, ps)); b = sorted(map(tuple, by[k])); mx = max(mx, np.abs(np.array(a) - np.array(b)).max())
print(f"sanity: base variant vs Step-0 base predictions, max |diff| = {mx:.2e}")

def ep_mean(rows):
    e = defaultdict(list)
    for r in rows: e[r["episode"]].append(r["brier"])
    return {k: np.mean(v) for k, v in e.items()}

def fold_delta(v, f):
    b = ep_mean(res["base"][f]); x = ep_mean(res[v][f])
    return float(np.mean([b[e] - x[e] for e in b]))

def ep_deltas(v, f):
    b = ep_mean(res["base"][f]); x = ep_mean(res[v][f]); return [b[e] - x[e] for e in b]

table = {}
for v in res:
    table[v] = [fold_delta(v, f) for f in (0, 1, 2)]

# nested selection of prior power (1.0 = base) on the other two folds
def nested(prefix_marg):
    choices = {}
    out = []
    for f in (0, 1, 2):
        cands = {1.0: ("2.0-fix_marg" if prefix_marg else "base")}
        for pw in (0.5, 1.5, 2.0, 3.0):
            cands[pw] = f"2.1b_prior_power_{pw}" + ("+marg" if prefix_marg else "")
        score = {pw: np.mean([fold_delta(v, g) for g in (0, 1, 2) if g != f]) for pw, v in cands.items()}
        best = max(score, key=score.get); choices[f] = best
        out.append(fold_delta(cands[best], f))
    full = {pw: np.mean([fold_delta(v, g) for g in (0, 1, 2)]) for pw, v in
            {1.0: ("2.0-fix_marg" if prefix_marg else "base"), **{pw: f"2.1b_prior_power_{pw}" + ("+marg" if prefix_marg else "") for pw in (0.5, 1.5, 2.0, 3.0)}}.items()}
    return out, choices, max(full, key=full.get)

table["2.1b_nested_power"], ch_b, full_b = nested(False)
table["2.1c_nested_power+marg"], ch_c, full_c = nested(True)
print("\nShort-history Brier delta vs base (positive = better), episode-weighted, per fold; overall effect on the")
print("4-condition mean = delta/4 (other conditions are unchanged by construction: gate n < 100).")
for v, d in table.items():
    sd = np.std(d)
    print(f"  {v:28s} {[f'{x:+.5f}' for x in d]}  mean {np.mean(d):+.5f} ± {sd:.5f}  folds better: {sum(x > 0 for x in d)}/3")
print(f"\nnested power choices per held-out fold: plain={ch_b} (full-data pick {full_b}); with marg={ch_c} (full-data pick {full_c})")
json.dump({"table": table, "nested_choice_plain": ch_b, "nested_choice_marg": ch_c, "full_pick_plain": full_b, "full_pick_marg": full_c},
          open(HERE / "e_analyze.json", "w"), indent=1, default=float)
