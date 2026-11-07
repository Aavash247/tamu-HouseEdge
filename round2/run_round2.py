"""One-command round-two (adaptation) pipeline. Never touches entry-final/ or submissions/final/.

    python round2/run_round2.py --kit /path/to/adaptation-kit --out round2-run

Steps:
  1. Check the kit's metadata (round, season) and that it has train/validation/test.
  2. Fit the entry-final architecture on the NEW kit (lag pairs, rate priors, size profiles),
     for a small grid of flip rates (rho) in parallel.
  3. Score each candidate, plus the unchanged main-round model as a fallback, on the NEW
     validation set using the kit's own check_submission.py.
  4. For the best rho, also compare family weights (2-cell / 4-cell / blend).
  5. Build the entry folder, predict test, check it, zip it, and verify a fresh-folder
     reproduction from the ZIP.
The kit folder is the extracted participant kit containing check_submission.py,
build_entry.py, slot_forecasting/ and public/.
"""
import argparse, json, re, shutil, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "entry-final"
RHOS = (0.0005, 0.001, 0.002, 0.005)
FAMILIES = {"blend": {"2": 0.5, "4": 0.5}, "2cell": {"2": 1.0}, "4cell": {"4": 1.0}}


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def kit_score(kit, csv_path):
    out = sh([sys.executable, "check_submission.py", str(csv_path), "--split", "validation"], cwd=kit)
    m = re.search(r"Validation points: ([0-9.]+)", out)
    conds = dict(re.findall(r"(\w+): loss ratio ([0-9.]+)", out))
    return (float(m.group(1)) if m else float("nan")), conds, out


def fit_and_predict(args):
    """Worker: fit on the new kit with the given rho and predict validation."""
    public, rho, workdir = args
    sys.path.insert(0, str(SRC))
    import houseedge as he
    cfg = dict(he.DEFAULT_CONFIG, rho=rho)
    model = he.Forecaster.fit(public, cfg)
    workdir.mkdir(parents=True, exist_ok=True)
    model.save(workdir / "model.json")
    q = he.read_queries(public / "validation" / "queries.csv")
    h = he.read_history(public / "validation" / "history.csv")
    preds = {}
    for name, fw in FAMILIES.items():
        m = he.Forecaster(dict(model.cfg, family_weights=fw))
        p = workdir / f"validation_{name}.csv"
        he.write_submission(p, m.predict(q, h))
        preds[name] = str(p)
    return rho, str(workdir / "model.json"), preds, model.cfg["pairs"]


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
        assert (public / s / "queries.csv").exists(), f"missing {s}"
    out.mkdir(parents=True)

    # main-round model unchanged, as a fallback benchmark on the new validation set
    fb = out / "fallback_main_model_validation.csv"
    sh([sys.executable, str(SRC / "predict.py"), "--data", str(public), "--split", "validation", "--out", str(fb)])
    fb_pts, fb_c, _ = kit_score(kit, fb)
    print(f"fallback (main-round model.json, unchanged): {fb_pts:.3f}  {fb_c}")

    jobs = [(public, r, out / f"rho_{r}") for r in RHOS]
    results = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for rho, model_path, preds, pairs in ex.map(fit_and_predict, jobs):
            for fam, p in preds.items():
                pts, c, _ = kit_score(kit, p)
                results.append((pts, rho, fam, model_path, c))
                print(f"rho={rho:<7} family={fam:6s} pairs={pairs} points={pts:.3f} {c}")
    results.sort(key=lambda r: -r[0])
    best_pts, rho, fam, model_path, conds = results[0]
    if fb_pts > best_pts:
        print("NOTE: the unchanged main-round model scored best; using it.")
        best_pts, rho, fam, model_path, conds = fb_pts, None, "main-round", str(SRC / "model.json"), fb_c
    print(f"\nBEST: points={best_pts:.3f} rho={rho} family={fam} {conds}")

    # build the round-two entry folder
    entry = out / "entry"
    shutil.copytree(SRC, entry, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = json.loads(Path(model_path).read_text())
    if fam in FAMILIES:
        cfg["family_weights"] = FAMILIES[fam]
    (entry / "model.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    readme = (entry / "README.md").read_text()
    readme = re.sub(r"- Round: .*", "- Round: adaptation", readme)
    readme = re.sub(r"- Season ID: .*", f"- Season ID: {meta.get('season_id')}", readme)
    readme += (f"\n## Round two\n\nRefitted on the adaptation kit with `round2/run_round2.py`: rho={rho}, "
               f"family={fam}. Adaptation validation points: {best_pts:.3f} / {meta.get('max_points')}. "
               f"Conditions: {conds}.\n")
    (entry / "README.md").write_text(readme)

    sub = out / "submission.csv"
    sh([sys.executable, "predict.py", "--data", str(public), "--out", str(sub)], cwd=entry)
    print(sh([sys.executable, "check_submission.py", str(sub)], cwd=kit).splitlines()[0])
    val = out / "validation_predictions.csv"
    sh([sys.executable, "predict.py", "--data", str(public), "--split", "validation", "--out", str(val)], cwd=entry)
    print("final entry validation:", kit_score(kit, val)[0])
    zip_path = out / "entry.zip"
    sh([sys.executable, "build_entry.py", "--source", str(entry), "--out", str(zip_path)], cwd=kit)

    # fresh-folder reproduction from the ZIP
    fresh = out / "_repro"
    fresh.mkdir()
    shutil.unpack_archive(zip_path, fresh)
    sh([sys.executable, "predict.py", "--data", str(public), "--out", str(fresh / "submission.csv")], cwd=fresh)
    same = (fresh / "submission.csv").read_bytes() == sub.read_bytes()
    shutil.rmtree(fresh)
    print(f"reproduction from ZIP byte-identical: {same}")
    print(f"\nUPLOAD: {sub}\n        {zip_path}")


if __name__ == "__main__":
    main()
