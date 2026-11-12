"""Compare the grid screen (30 s, 5 km, min over samples) with a refined
screen (30 s grid, large candidate radius, linear closest-approach between
samples, then exact SGP4 refinement)."""
import sys, time, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "src")
from satcollision.fleets import load_tle_file, catalogue_epoch, _epoch_from_calendar
from satcollision.twin import propagate_window

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
DT = 30.0
objs = load_tle_file("data/starlink.tle", operator_name="Starlink")
start = catalogue_epoch(objs)
_, jd0, fr0 = _epoch_from_calendar(start.year, start.month, start.day, start.hour + start.minute/60)
t0 = time.time()
P = propagate_window(objs, jd0, fr0, HOURS*3600, DT)
r, v, ok = P.positions, P.velocities, P.valid
n, T = ok.shape
print(f"{n} objects, {T} steps, propagate {time.time()-t0:.1f}s")

# grid method (as in twin.screen_conjunctions): min sampled distance within 5 km
grid = {}
# refined: candidates within R at step k, linear relative motion over [-DT/2, DT/2]
R = 15.0 * DT / 2 + 5.0
ref = {}
t0 = time.time()
for k in range(T):
    idx = np.nonzero(ok[:, k])[0]
    tree = cKDTree(r[idx, k])
    pr = tree.query_pairs(R, output_type="ndarray")
    if pr.size == 0: continue
    a, b = idx[pr[:, 0]], idx[pr[:, 1]]
    dr = r[a, k] - r[b, k]; dv = v[a, k] - v[b, k]
    d0 = np.linalg.norm(dr, axis=1)
    m = d0 < 5.0
    for i, j, d in zip(a[m], b[m], d0[m]):
        key = (i, j)
        if d < grid.get(key, 1e9): grid[key] = d
    tau = -np.einsum("ij,ij->i", dr, dv) / np.maximum(np.einsum("ij,ij->i", dv, dv), 1e-12)
    tau = np.clip(tau, -DT/2, DT/2)
    dmin = np.linalg.norm(dr + dv * tau[:, None], axis=1)
    m = dmin < 5.0
    for i, j, d, tt in zip(a[m], b[m], dmin[m], tau[m]):
        key = (i, j)
        if d < ref.get(key, (1e9,))[0]: ref[key] = (d, k*DT + tt, np.linalg.norm(dv[a==i][0]) if False else 0)
print(f"screen {time.time()-t0:.1f}s")
sub1 = lambda D: sum(1 for x in D.values() if (x if isinstance(x, float) or isinstance(x, np.floating) else x[0]) < 1.0)
print("pairs <5km  grid:", len(grid), " refined:", len(ref))
print("pairs <1km  grid:", sub1(grid), " refined:", sub1(ref))
both = [k for k in ref if k in grid]
err = [grid[k] - ref[k][0] for k in both]
print("pairs found by both:", len(both), " median grid overestimate of miss (km):", np.median(err) if err else None)

# verify refined sub-km approaches with direct SGP4 + bounded minimisation
from scipy.optimize import minimize_scalar
def dist(i, j, ts):
    fr = fr0 + ts/86400.0
    e1, r1, _ = objs[i].satrec.sgp4(jd0 + np.floor(fr), fr - np.floor(fr))
    e2, r2, _ = objs[j].satrec.sgp4(jd0 + np.floor(fr), fr - np.floor(fr))
    return float(np.linalg.norm(np.subtract(r1, r2)))
chk = sorted(((d, t, k) for k, (d, t, _) in ref.items() if d < 1.0))[:15]
for d, t, (i, j) in chk:
    res = minimize_scalar(lambda s: dist(i, j, s), bounds=(t-20, t+20), method="bounded")
    vrel = None
    print(f"linear {d*1000:7.1f} m  sgp4 {res.fun*1000:7.1f} m  dt {res.x-t:+.2f}s")
