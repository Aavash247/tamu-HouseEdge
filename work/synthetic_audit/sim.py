"""Independent synthetic House Edge simulator.

Writes competition-format folders (queries.csv, history.csv, labels.csv, conditions.csv,
targets.csv) so the existing model can be fitted and scored without code changes.

Generative model per machine (x_t = 1 if payout_t > 0, s = +-1 coding of x):
    logit P(win_t) = alpha + b_i*s_{t-i} + b_j*s_{t-j} + g(t)*inter(t) + delta*h_t
    inter(t)       = s_{t-i}*s_{t-j}                      (agree/disagree "XOR" family)
                   = +1 if x_{t-i}=x_{t-j}=1 else -1      (AND family; unseen_pattern only)
    h_t ~ Bernoulli(pi_m) hidden state each tick: raises win logit by delta and selects size profile B
    payout | win ~ size profile A or B of the machine type (per-seed constants)

Scenarios:
    stable          fixed parameters, full history (a few random missing ticks)
    short_history   as stable, but only the 40 most recent ticks + 24 sampled older ticks are observed
    drift           g(t) reverses sign gradually over a machine-specific window; alpha random-walks
    shared_change   one episode-wide tick tau: every machine's g flips and alpha shifts together
    unseen_pattern  TEST ONLY: different lag pairs, AND-type interaction, a third-lag main effect,
                    persistent (Markov) hidden state, and an optional shared flip
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
TYPES = ("steady", "balanced", "jackpot")
TRAIN_PAIRS = ((1, 3), (2, 5), (4, 7))
UNSEEN_PAIRS = ((1, 2), (3, 6), (5, 8))
BASE_ALPHA = {"steady": 0.6, "balanced": 0.0, "jackpot": -1.2}
SIGNAL_EFFECT = {"steady": 0.15, "balanced": 0.25, "jackpot": 0.35}
SIZE_TEMPLATES = {  # (profile A, profile B) over payouts 1, 2, 5, 20; perturbed per seed
    "steady": ([0.72, 0.22, 0.055, 0.005], [0.45, 0.42, 0.12, 0.01]),
    "balanced": ([0.48, 0.36, 0.14, 0.02], [0.22, 0.52, 0.22, 0.04]),
    "jackpot": ([0.36, 0.26, 0.29, 0.09], [0.13, 0.31, 0.42, 0.14]),
}
MACHINES = 8
DELTA = 0.6


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def size_profiles(rng):
    out = {}
    for t, (a, b) in SIZE_TEMPLATES.items():
        out[t] = [rng.dirichlet(np.array(a) * 400).tolist(), rng.dirichlet(np.array(b) * 400).tolist()]
    return out


def machine_params(rng, scenario, mtype, signal):
    unseen = scenario == "unseen_pattern"
    pairs = UNSEEN_PAIRS if unseen else TRAIN_PAIRS
    i, j = pairs[rng.integers(len(pairs))]
    return {
        "type": mtype, "signal": signal, "i": i, "j": j,
        "alpha": BASE_ALPHA[mtype] + SIGNAL_EFFECT[mtype] * signal + rng.normal(0, 0.3),
        "b_i": rng.normal(0, 0.5), "b_j": rng.normal(0, 0.5),
        "b_k": rng.normal(0, 0.8) if unseen else 0.0, "k": 6 if unseen else 0,
        "gamma": rng.uniform(2.5, 4.0) * rng.choice([-1, 1]),
        "pi": rng.beta(2, 2),
        "and_type": unseen,
        "markov_h": unseen,
    }


def simulate_episode(rng, scenario, cutoff, profiles):
    """Returns per-machine dicts with outcomes for ticks 0..cutoff and the true distribution at cutoff."""
    machines = []
    tau = int(rng.integers(100, cutoff - 40)) if scenario in ("shared_change", "unseen_pattern") else None
    if scenario == "unseen_pattern" and rng.random() < 0.5:
        tau = None
    shared_shift = rng.normal(0, 0.5)
    for m in range(MACHINES):
        mtype = TYPES[rng.integers(3)]
        signal = int(rng.integers(-1, 2))
        p = machine_params(rng, scenario, mtype, signal)
        drift_t0 = int(rng.integers(80, max(81, cutoff - 60)))
        drift_len = int(rng.integers(20, 61))
        x = np.zeros(cutoff + 1, dtype=int)
        pay = np.zeros(cutoff + 1, dtype=int)
        h = 0
        alpha_walk = 0.0
        for t in range(cutoff + 1):
            # hidden state
            if p["markov_h"]:
                if t == 0 or rng.random() < 0.15:
                    h = int(rng.random() < p["pi"])
            else:
                h = int(rng.random() < p["pi"])
            # time-varying interaction sign / level
            g = p["gamma"]
            alpha = p["alpha"]
            if scenario == "drift":
                frac = min(max((t - drift_t0) / drift_len, 0.0), 1.0)
                g = p["gamma"] * (1 - 2 * frac)
                alpha_walk += rng.normal(0, 0.03)
                alpha += alpha_walk
            if tau is not None and t >= tau:
                g = -g
                alpha += shared_shift
            if t < 8:
                z = alpha
            else:
                si, sj = 2 * x[t - p["i"]] - 1, 2 * x[t - p["j"]] - 1
                inter = (1 if (x[t - p["i"]] and x[t - p["j"]]) else -1) if p["and_type"] else si * sj
                z = alpha + p["b_i"] * si + p["b_j"] * sj + g * inter
                if p["k"]:
                    z += p["b_k"] * (2 * x[t - p["k"]] - 1)
            z_h1, z_h0 = z + DELTA, z
            prof_a, prof_b = profiles[mtype]
            if t == cutoff:
                pw = sigmoid(z_h1 if h else z_h0)
                prof = prof_b if h else prof_a
                truth = [1 - pw] + [pw * v for v in prof]
                # parameter oracle: knows everything except the hidden state at the query tick
                pi_now = p["pi"]
                pw1, pw0 = sigmoid(z_h1), sigmoid(z_h0)
                marg = [1 - (pi_now * pw1 + (1 - pi_now) * pw0)] + [
                    pi_now * pw1 * vb + (1 - pi_now) * pw0 * va for va, vb in zip(prof_a, prof_b)]
            pw = sigmoid(z_h1 if h else z_h0)
            win = rng.random() < pw
            x[t] = int(win)
            if win:
                prof = prof_b if h else prof_a
                pay[t] = PAYOUTS[1 + rng.choice(4, p=prof)]
            else:
                pay[t] = 0
        machines.append({"params": p, "pay": pay, "truth": truth, "param_oracle": marg})
    return machines


def observed_ticks(rng, scenario, cutoff):
    if scenario == "short_history":
        recent = list(range(cutoff - 40, cutoff))
        older = sorted(rng.choice(np.arange(0, cutoff - 40), 24, replace=False).tolist())
        return older + recent
    keep = [t for t in range(cutoff) if rng.random() >= 0.03]  # ~3% randomly missing ticks
    return keep


def write_split(out, split, prefix, scenarios, n_per, rng, profiles, with_targets=True):
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    q_rows, h_rows, l_rows, c_rows, t_rows, o_rows = [], [], [], [], [], []
    e = 0
    for scen in scenarios:
        for _ in range(n_per):
            ep = f"{prefix}-{split}-e{e:04d}"
            e += 1
            cutoff = int(rng.integers(192, 257))
            for m, mach in enumerate(simulate_episode(rng, scen, cutoff, profiles)):
                mid = f"{ep}-m{m:02d}"
                qid = f"{mid}-next"
                p = mach["params"]
                q_rows.append([qid, ep, mid, p["type"], p["signal"], cutoff])
                for t in observed_ticks(rng, scen, cutoff):
                    h_rows.append([ep, mid, t, int(mach["pay"][t])])
                l_rows.append([qid, int(mach["pay"][cutoff])])
                c_rows.append([qid, scen])
                t_rows.append([qid, ep] + [repr(float(v)) for v in mach["truth"]] + [int(mach["pay"][cutoff])])
                o_rows.append([qid] + [repr(float(v)) for v in mach["param_oracle"]])
    def w(name, header, rows):
        with open(d / name, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n"); wr.writerow(header); wr.writerows(rows)
    w("queries.csv", ["query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"], q_rows)
    w("history.csv", ["episode_id", "machine_id", "tick", "payout"], h_rows)
    w("labels.csv", ["query_id", "outcome"], l_rows)
    w("conditions.csv", ["query_id", "condition"], c_rows)
    w("targets.csv", ["query_id", "episode_id", "p_0", "p_1", "p_2", "p_5", "p_20", "outcome"], t_rows)
    w("param_oracle.csv", ["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"], o_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-scenario", type=int, default=32)
    ap.add_argument("--test-per-scenario", type=int, default=32)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    profiles = size_profiles(rng)
    seen = ["stable", "short_history", "drift", "shared_change"]
    write_split(a.out, "train", f"syn{a.seed}", seen, a.train_per_scenario, rng, profiles)
    write_split(a.out, "test", f"syn{a.seed}", seen + ["unseen_pattern"], a.test_per_scenario, rng, profiles)
    Path(a.out, "sim_config.json").write_text(json.dumps({
        "seed": a.seed, "size_profiles": profiles, "train_pairs": TRAIN_PAIRS, "unseen_pairs": UNSEEN_PAIRS,
        "train_per_scenario": a.train_per_scenario, "test_per_scenario": a.test_per_scenario,
        "machines_per_episode": MACHINES, "delta_hidden": DELTA}, indent=1))


if __name__ == "__main__":
    main()
