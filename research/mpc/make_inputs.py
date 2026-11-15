"""Inputs for secure_pc.mpc from real hindcast encounters, plus the plaintext answer.

    PYTHONPATH=src python research/mpc/make_inputs.py <n> <out_dir> [--seed 0]

Takes n assessments from the cached four-week hindcast (research/cache/
hindcast_w4_*.npz): the in-plane predicted miss and each object's in-plane
covariance (own satellite precise 0.354 km; the other object's public
sigma from the reference scale at its age). The miss is split between the
two operators as offsets from a random public reference point, as each
would input them. Writes Player-Data/Input-P0-0, Input-P1-0 under out_dir
and expected_bits.txt (plaintext decision, Pc >= 1e-4).
"""
import glob
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from satcollision import error_learning as el  # noqa: E402
from satcollision import hindcast as hc  # noqa: E402


def main(n, out_dir, seed=0):
    f = sorted(glob.glob(str(ROOT / "research" / "cache" / "hindcast_w4_*.npz")))[0]
    z = np.load(f)
    rng = np.random.default_rng(seed)
    N = len(z["asm_enc"])
    # half near-misses (so both decisions occur), half random
    near = np.nonzero(z["miss_km"][z["asm_enc"]] < 2.0)[0]
    pick = np.concatenate([rng.choice(near, n // 2, replace=False), rng.choice(N, n - n // 2, replace=False)])
    e = z["asm_enc"][pick]
    side = z["asm_side"][pick]
    m_true = z["m_true"][e]
    err = z["err_err_rtn"][pick]
    P_oth = np.where((side == 0)[:, None, None], z["Pb"][e], z["Pa"][e])
    shift = np.einsum("nk,nkj->nj", err, P_oth)
    m = np.where((side == 0)[:, None], m_true - shift, m_true + shift)
    sig_o = el.reference_scale(z["err_age_h"][pick]) * np.array([0.5, 0.5, 0.5])
    C_b = hc.plane_cov(P_oth, sig_o)
    C_a = np.broadcast_to(0.354 ** 2 * np.eye(2), C_b.shape).copy()
    ref = rng.normal(0, 5.0, (n, 2))
    m_a = ref + m            # operator A's own position offset from the public reference
    m_b = ref                # operator B's (the miss is m_a - m_b)
    pc = hc.pc_small(m, C_a + C_b)
    out = Path(out_dir)
    (out / "Player-Data").mkdir(parents=True, exist_ok=True)

    def write(path, cols):
        with open(path, "w") as fh:
            for c in cols:                         # column-major: the program reads whole vectors
                fh.write(" ".join(f"{v:.9f}" for v in c) + "\n")

    write(out / "Player-Data" / "Input-P0-0", [m_a[:, 0], m_a[:, 1], C_a[:, 0, 0], C_a[:, 0, 1], C_a[:, 1, 1]])
    write(out / "Player-Data" / "Input-P1-0", [m_b[:, 0], m_b[:, 1], C_b[:, 0, 0], C_b[:, 0, 1], C_b[:, 1, 1]])
    (out / "expected_bits.txt").write_text(" ".join(str(int(v)) for v in (pc >= 1e-4)) + "\n")
    print(f"{n} encounters, {int((pc >= 1e-4).sum())} act in plaintext; Pc range {pc.min():.1e} to {pc.max():.1e}")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(int(args[0]), args[1], int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0)
