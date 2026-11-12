"""Week-long split-catalogue evaluation, grid vs refined screening.

Usage: PYTHONPATH=src python research/run_refined_eval.py --hours 168 --screening refined --pc-model circular
Writes research/results/eval_<screening>_<model>_<hours>h.json
"""
import argparse, json, time, pathlib, collections
from satcollision.real_evaluation import run_split_catalog_evaluation
from satcollision.pc import FosterPcModel

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=168.0)
ap.add_argument("--step", type=float, default=30.0)
ap.add_argument("--screening", choices=("grid", "refined"), default="refined")
ap.add_argument("--pc-model", choices=("circular", "foster"), default="circular")
ap.add_argument("--along-track-ratio", type=float, default=8.0)
ap.add_argument("--tle", default="data/starlink.tle")
a = ap.parse_args()
model = FosterPcModel(along_track_ratio=a.along_track_ratio) if a.pc_model == "foster" else None
t0 = time.time()
res = run_split_catalog_evaluation(a.tle, "Starlink", hours=a.hours, step_s=a.step,
                                   epoch="catalogue", pc_model=model, screening=a.screening)
res["runtime_s"] = round(time.time() - t0, 1)
res["geometry_counts"] = dict(collections.Counter(e["geometry_class"] for e in res["encounters"]))
out = pathlib.Path("research/results"); out.mkdir(parents=True, exist_ok=True)
name = f"eval_{a.screening}_{a.pc_model}{'' if model is None else f'k{a.along_track_ratio:g}'}_{a.hours:g}h.json"
(out / name).write_text(json.dumps(res, indent=1, default=str))
print(name, res["runtime_s"], "s")
print({k: res[k] for k in ("n_cross_operator", "n_above_threshold", "n_scored", "n_excluded_burn_in")})
print(res["geometry_counts"]); print(res["summary"])
