"""Audit part 2: 4-fold episode-level CV entirely inside the TRAIN split (no validation data at all),
plus the same Brier metric on validation for comparison.

Train has no target probabilities, only sampled outcomes, so models are scored with the Brier score
against outcomes. E[Brier] = sum_k (p_k - q_k)^2 + (1 - sum_k q_k^2): the second term does not depend
on the model, so Brier *differences* between models are unbiased estimates of differences in the
official squared error to the true probabilities.

Held-out examples per machine: the real query (k=0) plus pseudo-queries at cutoff-k for k=1..6, where
the whole episode's history is truncated at cutoff-k and the label is the observed payout at cutoff-k.
"""
import time, json, random
from common import *
from slot_forecasting.schema import Query, Play
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.neighbor import NeighborPooledRecent
from slot_forecasting.blend import BlendNeighborPooled

K_PSEUDO = 6
NFOLD = 4
OLD = {"half_life": 12.0, "prior_strength": 10.0, "neighbor_weight": 0.15, "same_type_only": True,
       "blend_weight": 0.25, "pooled_half_life": 16.0, "pooled_prior_strength": 10.0}


def to_sf(queries, hist):
    qs = tuple(Query(q["query_id"], q["episode_id"], q["machine_id"], q["machine_type"], q["signal"], q["cutoff"]) for q in queries)
    keys = {(q["episode_id"], q["machine_id"]) for q in queries}
    hs = tuple(Play(e, m, t, p) for (e, m), s in hist.items() if (e, m) in keys for t, p in s.items())
    return qs, hs


def fit_baselines(train_q, train_h, labels):
    qs, hs = to_sf(train_q, train_h)
    lab = {q["query_id"]: labels[q["query_id"]] for q in train_q}
    ref = FrequencyForecaster("pooled", prior_strength=20.0).fit(qs, hs, lab)
    nb = NeighborPooledRecent(prior_strength=OLD["prior_strength"], half_life=OLD["half_life"],
                              neighbor_weight=OLD["neighbor_weight"], same_type_only=True).fit(qs, hs, lab)
    pr = FrequencyForecaster("pooled_recent", prior_strength=OLD["pooled_prior_strength"],
                             half_life=OLD["pooled_half_life"]).fit(qs, hs, lab)
    return ref, BlendNeighborPooled(nb, pr, OLD["blend_weight"])


def heldout_sets(queries, hist, labels):
    """Yield (queries, truncated history, {qid: label}) for k = 0..K_PSEUDO."""
    for k in range(K_PSEUDO + 1):
        qs, hs, lab = [], {}, {}
        for q in queries:
            c = q["cutoff"] - k
            s = hist[(q["episode_id"], q["machine_id"])]
            if k == 0:
                y = labels[q["query_id"]]
            elif c in s:
                y = s[c]
            else:
                y = None
            nq = dict(q, cutoff=c, query_id=f'{q["query_id"]}:k{k}')
            qs.append(nq)
            hs[(q["episode_id"], q["machine_id"])] = {t: v for t, v in s.items() if t < c}
            if y is not None:
                lab[nq["query_id"]] = y
        yield k, qs, hs, lab


def evaluate(models, queries, hist, labels, cond, fold):
    rows = []
    for k, qs, hs, lab in heldout_sets(queries, hist, labels):
        preds = {}
        for name, m in models.items():
            if isinstance(m, he.Forecaster):
                preds[name] = {key: np.asarray(v) for key, v in m.predict(qs, hs).items()}
            else:
                sq, sh = to_sf(qs, hs)
                preds[name] = {key: np.asarray(v) for key, v in m.predict(sq, sh).items()}
        for q in qs:
            if q["query_id"] not in lab:
                continue
            base_id = q["query_id"].split(":k")[0]
            r = {"fold": fold, "k": k, "episode": q["episode_id"], "cond": cond[base_id]}
            for name in models:
                r[name] = brier(preds[name][q["query_id"]], lab[q["query_id"]])
            rows.append(r)
    return rows


def main():
    t0 = time.time()
    tq, th, tcond, tlab = load("train")
    # stratified episode folds: within each condition, episodes sorted by id and dealt round-robin
    ep_cond = {}
    for q in tq:
        ep_cond[q["episode_id"]] = tcond[q["query_id"]]
    fold_of = {}
    for c in CONDS:
        for n, e in enumerate(sorted(e for e, cc in ep_cond.items() if cc == c)):
            fold_of[e] = n % NFOLD
    rows = []
    fold_cfgs = {}
    for f in range(NFOLD):
        tr = [q for q in tq if fold_of[q["episode_id"]] != f]
        te = [q for q in tq if fold_of[q["episode_id"]] == f]
        em = size_profiles_em(tr, th)
        cfg = build_cfg(tr, th, em)
        fold_cfgs[f] = {"pairs": cfg["pairs"], "pair_prior": cfg["pair_prior"], "size_profiles": em}
        ref, old = fit_baselines(tr, th, tlab)
        models = {"reference_pooled": ref, "original_blend": old, "new_trainonly": he.Forecaster(cfg)}
        rows += evaluate(models, te, th, tlab, tcond, f)
        print(f"fold {f}: train eps={len({q['episode_id'] for q in tr})} test eps={len({q['episode_id'] for q in te})} "
              f"pairs={cfg['pairs']} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(ROOT / "work/audit/cv_rows.json", "w"))
    json.dump(fold_cfgs, open(ROOT / "work/audit/cv_fold_cfgs.json", "w"), indent=1)

    # same metric on VALIDATION with the submitted model (fit on full train + val-target sizes)
    vq, vh, vcond, vlab = load("validation")
    ref, old = fit_baselines(tq, th, tlab)
    vrows = evaluate({"reference_pooled": ref, "original_blend": old, "new_trainonly": he.Forecaster(FINAL_CFG)},
                     vq, vh, vlab, vcond, -1)
    json.dump(vrows, open(ROOT / "work/audit/val_brier_rows.json", "w"))
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
