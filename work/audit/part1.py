"""Audit part 1: leakage probes, ID independence, lag-pair consistency, train-only size profiles."""
import random, time, json
from common import *
sys.path.insert(0, str(ROOT / "work"))
from slot_forecasting.contest_data import load_contest, score_predictions  # noqa: E402

model = he.Forecaster(FINAL_CFG)
vq, vh, vcond, vlab = load("validation")
tq, th, tcond, tlab = load("train")
rng = random.Random(0)
sample = rng.sample(vq, 60)

def pred(qs, hist, m=model):
    return m.predict(qs, hist)

print("== 1. Leakage / memorization probes on 60 validation queries ==")
base = pred(sample, vh)
# (a) inject fake FUTURE ticks (>= cutoff) for the queried machine and all its episode neighbours
h2 = {k: dict(v) for k, v in vh.items()}
for q in sample:
    for (ep, mid), s in h2.items():
        if ep == q["episode_id"]:
            for t in range(q["cutoff"], q["cutoff"] + 30):
                s[t] = rng.choice(PAY)
fut = pred(sample, h2)
print("max |diff| after adding fake future ticks:", max(np.abs(base[k] - fut[k]).max() for k in base))
# (b) scramble OTHER machines' histories in the same episode
h3 = {k: dict(v) for k, v in vh.items()}
for q in sample:
    for (ep, mid), s in h3.items():
        if ep == q["episode_id"] and mid != q["machine_id"]:
            for t in list(s):
                s[t] = rng.choice(PAY)
nb = pred(sample, h3)
print("max |diff| after scrambling neighbour machines:", max(np.abs(base[k] - nb[k]).max() for k in base))
# (c) rename episode / machine / query IDs
ren_q, ren_h = [], {}
for n, q in enumerate(sample):
    r = dict(q, query_id=f"X{n}", episode_id=f"E{n}", machine_id=f"M{n}")
    ren_q.append(r); ren_h[("E%d" % n, "M%d" % n)] = vh[(q["episode_id"], q["machine_id"])]
rn = pred(ren_q, ren_h)
print("max |diff| after renaming all IDs:", max(np.abs(base[q["query_id"]] - rn[f"X{n}"]).max() for n, q in enumerate(sample)))
# (d) change the cutoff-k observation (an INPUT) to confirm the probe is sensitive
h4 = {k: dict(v) for k, v in vh.items()}
for q in sample:
    s = h4[(q["episode_id"], q["machine_id"])]
    t = q["cutoff"] - 1
    if t in s:
        s[t] = 0 if s[t] > 0 else 2
ch = pred(sample, h4)
print("sanity: max |diff| after flipping the lag-1 INPUT:", round(max(np.abs(base[k] - ch[k]).max() for k in base), 3))
keys = json.dumps(FINAL_CFG)
print("IDs stored in model.json:", any(s in keys for s in ("main-train", "main-validation", "main-test")),
      "| model.json keys:", sorted(FINAL_CFG))
print("prior keys (only machine types):", sorted(FINAL_CFG["prior2"]), sorted(FINAL_CFG["size_profiles"]))

print("\n== 2. Lag-pair consistency: best-evidence pair per machine (histories only) ==")
grid = he.FlipGrid(FINAL_CFG["grid_size"])
t0 = time.time()
for name, qs, hs, cond in (("train", tq, th, tcond), ("validation", vq, vh, vcond)):
    _, _, counts, bym = fit_pairs(qs, hs, FINAL_CFG, grid)
    tot = sum(counts.values())
    top = {f"{p}": round(c / tot, 3) for p, c in sorted(counts.items(), key=lambda x: -x[1])[:5]}
    print(f"{name}: machines={int(tot)} top pairs share={top}")
    for c in CONDS:
        ms = [v for k, v in bym.items() if cond[k] == c]
        in3 = np.mean([v[0] in ((1, 3), (2, 5), (4, 7)) for v in ms])
        print(f"   {c:14s} share in {{(1,3),(2,5),(4,7)}}={in3:.3f}  median evidence margin over 2nd-best pair={np.median([v[1] for v in ms]):.1f} nats")
print(f"({time.time()-t0:.0f}s)")

print("\n== 3. Payout-size profiles: validation targets vs train-only EM ==")
em = size_profiles_em(tq, th)
for typ in sorted(em):
    for k in range(2):
        print(f"  {typ:8s} profile {k}: val-targets={np.round(FINAL_CFG['size_profiles'][typ][k], 4).tolist()}  train-EM={np.round(em[typ][k], 4).tolist()}")
bundle = load_contest(PUB)
cfg_em = dict(FINAL_CFG, size_profiles=em)
for label, m in (("final (val-target profiles)", model), ("same model, train-EM profiles", he.Forecaster(cfg_em))):
    p = m.predict(vq, vh)
    s = score_predictions({k: tuple(float(x) for x in v) for k, v in p.items()}, bundle)
    print(f"  {label:32s} points={s['technical_points']:.3f} R={s['mean_loss_ratio']:.4f} " +
          " ".join(f"{c}={s['by_condition'][c]['loss_ratio']:.3f}" for c in CONDS))
json.dump(em, open(ROOT / "work/audit/size_profiles_train_em.json", "w"), indent=1)
