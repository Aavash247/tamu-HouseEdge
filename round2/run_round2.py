"""Round two in one command: python round2/run_round2.py --kit /path/to/adaptation-kit --out round2-run

Refits entry/ on the new kit for a few flip rates and families, keeps the main-round model
if it still scores best, then builds, checks and zips the entry and confirms the ZIP reproduces.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "entry"
RHOS = (0.0005, 0.001, 0.002, 0.005)
FAMILIES = {"blend": {"2": 0.5, "4": 0.5}, "2cell": {"2": 1.0}, "4cell": {"4": 1.0}}


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def kit_score(kit, csv_path):
    """Validation points and condition ratios from the kit's own checker."""
    out = sh([sys.executable, "check_submission.py", str(csv_path), "--split", "validation"], cwd=kit)
    m = re.search(r"Validation points: ([0-9.]+)", out)
    return (float(m.group(1)) if m else float("nan")), dict(re.findall(r"(\w+): loss ratio ([0-9.]+)", out))


def fit_and_predict(job):
    """Worker: fit with one flip rate, predict validation for each family."""
    public, rho, workdir = job
    sys.path.insert(0, str(SRC))
    from data import read_history, read_queries, write_submission
    from forecaster import DEFAULT_CONFIG, Forecaster
    model = Forecaster.fit(public, dict(DEFAULT_CONFIG, rho=rho))
    workdir.mkdir(parents=True, exist_ok=True)
    model.save(workdir / "model.json")
    q = read_queries(public / "validation" / "queries.csv")
    h = read_history(public / "validation" / "history.csv")
    preds = {}
    for name, weights in FAMILIES.items():
        path = workdir / f"validation_{name}.csv"
        write_submission(path, Forecaster(dict(model.cfg, family_weights=weights)).predict(q, h))
        preds[name] = path
    return rho, workdir / "model.json", preds, model.cfg["pairs"]


def pick_best(kit, public, out):
    fallback = out / "fallback_validation.csv"
    sh([sys.executable, str(SRC / "predict.py"), "--data", str(public), "--split", "validation", "--out", str(fallback)])
    pts, conds = kit_score(kit, fallback)
    print(f"main-round model, unchanged: {pts:.3f} {conds}")
    best = (pts, None, "main-round", SRC / "model.json", conds)

    jobs = [(public, r, out / f"rho_{r}") for r in RHOS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for rho, model_path, preds, pairs in ex.map(fit_and_predict, jobs):
            for fam, path in preds.items():
                pts, conds = kit_score(kit, path)
                print(f"rho={rho:<7} family={fam:6s} pairs={pairs} points={pts:.3f} {conds}")
                if pts > best[0] or (pts == best[0] and best[1] is None):  # ties go to the refit
                    best = (pts, rho, fam, model_path, conds)
    print(f"\nBEST: points={best[0]:.3f} rho={best[1]} family={best[2]} {best[4]}")
    return best


def build_entry(best, meta, out):
    pts, rho, fam, model_path, conds = best
    entry = out / "entry"
    shutil.copytree(SRC, entry, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = json.loads(Path(model_path).read_text())
    if fam in FAMILIES:
        cfg["family_weights"] = FAMILIES[fam]
    (entry / "model.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    readme = (entry / "README.md").read_text()
    readme = re.sub(r"- Round: .*", "- Round: adaptation", readme)
    readme = re.sub(r"- Season ID: .*", f"- Season ID: {meta.get('season_id')}", readme)
    readme += (f"\n## Round two\n\nRefitted on the adaptation kit: rho={rho}, family={fam}. "
               f"Validation points: {pts:.3f} / {meta.get('max_points')}. Conditions: {conds}.\n")
    (entry / "README.md").write_text(readme)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    kit, out = a.kit.resolve(), a.out.resolve()
    public = kit / "public"
    if out.exists():
        raise SystemExit(f"Choose a fresh --out folder: {out}")
    meta = json.loads((public / "metadata.json").read_text())
    print(f"kit round={meta.get('round')} season={meta.get('season_id')} max_points={meta.get('max_points')}")
    for s in ("train", "validation", "test"):
        if not (public / s / "queries.csv").exists():
            raise SystemExit(f"kit is missing {s}/queries.csv")
    out.mkdir(parents=True)

    entry = build_entry(pick_best(kit, public, out), meta, out)
    sub = out / "submission.csv"
    sh([sys.executable, "predict.py", "--data", str(public), "--out", str(sub)], cwd=entry)
    print(sh([sys.executable, "check_submission.py", str(sub)], cwd=kit).splitlines()[0])
    zip_path = out / "entry.zip"
    sh([sys.executable, "build_entry.py", "--source", str(entry), "--out", str(zip_path)], cwd=kit)

    fresh = out / "_repro"
    shutil.unpack_archive(zip_path, fresh)
    sh([sys.executable, "predict.py", "--data", str(public), "--out", str(fresh / "submission.csv")], cwd=fresh)
    same = (fresh / "submission.csv").read_bytes() == sub.read_bytes()
    shutil.rmtree(fresh)
    print(f"reproduction from ZIP byte-identical: {same}")
    print(f"\nUPLOAD: {sub}\n        {zip_path}")


if __name__ == "__main__":
    main()
