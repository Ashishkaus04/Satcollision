"""Hindcast on a TLE history (hindcast.py) and the learned error model
(error_learning.py), on the small synthetic history of history_fixture."""

import math

import numpy as np
import pytest

from satcollision import error_learning as el
from satcollision import hindcast as hc
from satcollision.history_world import jd_of
from satcollision.pc import foster_pc, rtn_covariance
from satcollision.tle_history import TLEHistory
from tests.history_fixture import BASE, FAST_TCA_AFTER_S, TCA_AFTER_S, write_history

OPS = {"Alpha": ["ALPHA"], "Beta": ["BETA"]}


@pytest.fixture(scope="module")
def hist(tmp_path_factory):
    return TLEHistory.load(write_history(tmp_path_factory.mktemp("hc") / "history.tle", along_track_noise_km=0.3))


@pytest.fixture(scope="module")
def screened(hist):
    owner, ops = hc.owners_of(hist.names, OPS)
    j0 = jd_of(BASE)
    sc = hc.screen_truth(hist, owner, j0, j0 + 3 / 24, screen_km=10.0)
    return owner, sc, hc.approach_geometry(hist, sc)


# ---------------------------------------------------------------------------
# Truth and prediction
# ---------------------------------------------------------------------------

def test_truth_never_uses_the_set_a_prediction_would_use(hist):
    rng = np.random.default_rng(0)
    rows = rng.integers(0, hist.n_objects, 500)
    jd = hist.first_epoch + rng.uniform(0.3, 3.5, 500)
    a, b, w = hc.truth_weights(hist, rows, jd, blend_s=600.0)
    asof = hc.asof_sets(hist, rows, jd)
    far = (jd - hist.epochs[asof]) * 86400.0 > 300.0       # outside the blend after an epoch
    assert np.all(a[far] != asof[far]) and np.all(b[far] != asof[far])
    # and the truth set is always the first one published after t
    assert np.all(hist.epochs[a[far]] > jd[far])


def test_truth_is_continuous_across_a_switch(hist):
    row = 0
    ep = hist.epochs[hist.start[row] + 3]
    t = ep + np.linspace(-900, 900, 361) / 86400.0
    r, _, ok = hc.truth_pairs(hist, np.full(len(t), row), t)
    assert ok.all()
    step = np.linalg.norm(np.diff(r, axis=0), axis=1)
    # 5 s apart at ~7.6 km/s: 38 km per step, no jump on top of it
    assert step.max() - step.min() < 0.5


def test_propagate_pairs_matches_sgp4(hist):
    sets = np.array([0, 0, 5, hist.n_sets - 1])
    jd = hist.epochs[sets] + np.array([0.1, 0.5, 0.2, 0.0])
    r, v, ok = hc.propagate_pairs(hist, sets, jd)
    assert ok.all()
    for k, s in enumerate(sets):
        sat = hist.satrec(int(s))
        whole = math.floor(jd[k] - 0.5) + 0.5
        e, rr, vv = sat.sgp4(whole, jd[k] - whole)
        assert e == 0
        assert np.allclose(r[k], rr, atol=1e-6) and np.allclose(v[k], vv, atol=1e-9)


# ---------------------------------------------------------------------------
# Screening and geometry
# ---------------------------------------------------------------------------

def test_screen_finds_both_engineered_crossings_and_respects_owners(hist, screened):
    owner, sc, g = screened
    j0 = jd_of(BASE)
    names = {(hist.names[a], hist.names[b]) for a, b in zip(g["row_a"], g["row_b"])}
    assert ("ALPHA-1", "BETA-1") in names
    assert ("ALPHA-2", "BETA-3") in names
    k = [i for i, (a, b) in enumerate(zip(g["row_a"], g["row_b"]))
         if hist.names[a] == "ALPHA-1" and hist.names[b] == "BETA-1"][0]
    assert abs((g["tca_jd"][k] - j0) * 86400.0 - TCA_AFTER_S) < 60.0
    assert g["miss_km"][k] < 2.0
    f = [i for i, (a, b) in enumerate(zip(g["row_a"], g["row_b"]))
         if hist.names[a] == "ALPHA-2" and hist.names[b] == "BETA-3"]
    # (the pair meets more than once in three hours; the engineered one is fast and close)
    assert min(abs((g["tca_jd"][i] - j0) * 86400.0 - FAST_TCA_AFTER_S) for i in f) < 60.0
    # the member comes first, and an operator is never paired with itself
    assert np.all(owner[g["row_a"]] >= 0)
    assert np.all(owner[g["row_a"]] != owner[g["row_b"]])


def test_thinning_is_deterministic_and_weighted(hist):
    owner, _ = hc.owners_of(hist.names, OPS)
    j0 = jd_of(BASE)
    strata = ((2.0, 1.0), (2000.0, 0.5))
    s1 = hc.screen_truth(hist, owner, j0, j0 + 3 / 24, strata=strata)
    s2 = hc.screen_truth(hist, owner, j0, j0 + 3 / 24, strata=strata)
    assert np.array_equal(s1["row_a"], s2["row_a"]) and np.array_equal(s1["tca_jd"], s2["tca_jd"])
    assert set(np.unique(s1["weight"])) <= {1.0, 2.0}
    assert np.all(s1["weight"][s1["miss_lin_km"] < 2.0] == 1.0)


def test_plane_projections_are_orthonormal(screened):
    _, _, g = screened
    for P in (g["Pa"], g["Pb"]):
        S = np.einsum("nka,nkb->nab", P, P)          # sum over RTN axes of p p^T
        assert np.allclose(S, np.eye(2)[None], atol=1e-9)
    assert np.allclose(np.linalg.norm(g["m_true"], axis=1), g["miss_km"])


def test_small_body_pc_matches_foster(screened):
    _, _, g = screened
    k = int(np.argmin(g["miss_km"]))
    sa, sb = (0.354, 0.354, 0.354), (0.3, 4.0, 0.2)
    ref = foster_pc(g["ra"][k] - g["rb"][k], g["va"][k] - g["vb"][k],
                    rtn_covariance(g["ra"][k], g["va"][k], *sa), rtn_covariance(g["rb"][k], g["vb"][k], *sb),
                    hbr_km=0.02)
    C = hc.plane_cov(g["Pa"][k:k + 1], np.array([sa])) + hc.plane_cov(g["Pb"][k:k + 1], np.array([sb]))
    mine = hc.pc_small(g["m_true"][k:k + 1], C)[0]
    assert ref > 0 and abs(mine / ref - 1) < 0.02


def test_predicted_miss_ignores_error_along_relative_velocity(screened):
    owner, _, g = screened
    upd = np.array([g["tca_jd"].min() - 0.2])
    asm = hc.assessments(g, owner, upd)
    n = len(asm["enc"])
    zero = {"err_rtn": np.zeros((n, 3))}
    assert np.allclose(hc.predicted_miss(g, asm, zero), g["m_true"][asm["enc"]])
    # an error exactly along v_rel moves the TCA, not the miss
    e = asm["enc"]
    vrel = g["va"][e] - g["vb"][e]
    from satcollision.tle_history import to_rtn
    r_o = np.where((asm["side"] == 0)[:, None], g["rb"][e], g["ra"][e])
    v_o = np.where((asm["side"] == 0)[:, None], g["vb"][e], g["va"][e])
    along_vrel = {"err_rtn": to_rtn(3.0 * vrel / np.linalg.norm(vrel, axis=1, keepdims=True), r_o, v_o)}
    assert np.allclose(hc.predicted_miss(g, asm, along_vrel), g["m_true"][e], atol=1e-9)


def test_assessments_cover_each_member_side_and_the_horizon(screened):
    owner, _, g = screened
    upd = hc.update_times(g["tca_jd"].min() - 4, g["tca_jd"].max(), 8.0)
    asm = hc.assessments(g, owner, upd, horizon_h=72.0)
    assert np.all((asm["lead_h"] > 0) & (asm["lead_h"] <= 72.0))
    # both ALPHA-1 and BETA-1 are members: two sides, nine 8-hourly updates each
    k = [i for i, (a, b) in enumerate(zip(g["row_a"], g["row_b"]))
         if {owner[a], owner[b]} == {0, 1}][0]
    sel = asm["enc"] == k
    assert set(asm["side"][sel]) == {0, 1}
    assert sel.sum() == 18


def test_score_counts_catches_false_alarms_and_lead():
    asm = {"enc": np.array([0, 0, 1, 1, 2]), "side": np.zeros(5, int), "lead_h": np.array([30.0, 6.0, 30.0, 6.0, 10.0])}
    pc = np.array([1e-6, 1e-3, 1e-3, 1e-3, 1e-6])
    dangerous = np.array([True, False, True])
    s = hc.score(asm, pc, dangerous, n_days=1.0)
    assert s["dangerous"] == 2 and s["caught"] == 1 and s["false_alarms"] == 1
    assert s["median_warning_h"] == 6.0 and s["caught_24h_frac"] == 0.0
    s = hc.score(asm, pc, dangerous, n_days=1.0, enc_weight=np.array([1.0, 10.0, 1.0]))
    assert s["false_alarms"] == 10.0


def test_public_errors_are_the_real_tle_error(hist, screened):
    owner, _, g = screened
    upd = hc.update_times(hist.first_epoch, g["tca_jd"].max(), 8.0)
    asm = hc.assessments(g, owner, upd)
    err = hc.public_errors(hist, g, asm)
    assert err["ok"].all() and len(err["ok"]) > 20
    assert np.all(err["age_h"] > 0)
    # fixture sets carry ~0.3 km along-track noise: errors are mostly along-track
    t = np.abs(err["err_rtn"][:, 1])
    assert 0.02 < np.median(t) < 2.0
    assert np.median(np.abs(err["err_rtn"][:, 2])) < 0.2 * np.median(t)


def test_subset_history_drops_later_sets(hist):
    cut = hist.first_epoch + 1.0
    sub = hc.subset_history(hist, cut)
    assert sub.epochs.max() < cut and sub.n_objects == hist.n_objects
    assert sub.n_sets == int((hist.epochs < cut).sum())


# ---------------------------------------------------------------------------
# The error model
# ---------------------------------------------------------------------------

def test_features_and_samples(hist):
    rows = np.arange(hist.n_objects)
    hi = hist.first_epoch + 2.0
    s = el.build_samples(hist, rows, hist.first_epoch, hi, ages_per_set=4, seed=1)
    assert len(s) > 50 and s.X.shape == (len(s), el.FEATURE_DIM)
    assert np.all(np.isfinite(s.X)) and np.all(np.isfinite(s.y))
    assert np.median(np.abs(s.y[:, 1])) > 0.02            # real, non-zero labels
    assert np.all(s.age_h > 0)
    # no look-ahead: a label needs the set published after its time, so a
    # window ending before the second set of every object has no labels
    second = min(hist.epochs[hist.start[r] + 1] for r in rows)
    assert len(el.build_samples(hist, rows, hist.first_epoch, second, seed=1)) == 0


def _synthetic(n, scale_t, seed):
    """Errors whose along-track spread grows with age at a rate set by altitude."""
    rng = np.random.default_rng(seed)
    age = np.exp(rng.uniform(math.log(0.5), math.log(72), n))
    X = np.zeros((n, el.FEATURE_DIM), np.float32)
    alt = rng.uniform(-1, 1, n)
    X[:, 0] = np.log(age + 0.1) / math.log(72)
    X[:, 1] = age / 72
    X[:, 2] = alt
    ref = el.reference_scale(age)
    sd = ref * np.array([0.5, 1.0, 0.5]) * (scale_t * (1.5 + alt))[:, None]
    y = rng.standard_normal((n, 3)) * sd + np.array([0, 0.3, 0]) * ref
    return el.ErrorSamples(X, y, age, np.zeros(n, np.int64))


def test_training_beats_an_untrained_model():
    d = _synthetic(4000, 1.0, 0)
    test = _synthetic(2000, 1.0, 1)
    before = el.evaluate(el.new_model(0), test)["nll"]
    after = el.evaluate(el.train_local(d, epochs=15), test)
    assert after["nll"] < before - 0.2
    assert 0.5 < after["cover1"][1] < 0.85


def test_secure_fedavg_equals_plain_fedavg():
    data = {"A": _synthetic(600, 1.0, 2), "B": _synthetic(300, 0.5, 3), "C": _synthetic(100, 2.0, 4)}
    plain, log1 = el.fedavg(data, rounds=2, seed=0)
    secure, log2 = el.fedavg(data, rounds=2, seed=0, secure=True)
    assert log2.secure and log1.bytes_per_update == el.flat(plain).size * 4
    assert np.abs(el.flat(plain) - el.flat(secure)).max() < 1e-5


def test_federation_helps_a_small_operator():
    # a small operator with few samples, a big one with many from a related population
    small = _synthetic(60, 1.0, 5)
    big = _synthetic(5000, 1.0, 6)
    test = _synthetic(3000, 1.0, 7)
    local = el.evaluate(el.train_local(small, epochs=10, seed=0), test)["nll"]
    fed, _ = el.fedavg({"small": small, "big": big}, rounds=10, seed=0)
    assert el.evaluate(fed, test)["nll"] < local


def test_clipped_updates_move_the_model_less():
    data = {"A": _synthetic(800, 1.0, 8), "B": _synthetic(800, 1.0, 9)}
    start = el.flat(el.new_model(0))
    free, _ = el.fedavg(data, rounds=1, seed=0)
    clipped, log = el.fedavg(data, rounds=1, seed=0, clip=0.01)
    assert log.dp == {"clip": 0.01, "noise_multiplier": 0.0}
    assert np.linalg.norm(el.flat(clipped) - start) <= 0.01 + 1e-6
    assert np.linalg.norm(el.flat(free) - start) > 0.01


def test_table_metrics_agree_with_model_metrics_for_the_same_distribution():
    d = _synthetic(3000, 1.0, 10)
    ref = el.reference_scale(d.age_h)
    sd = ref * np.array([0.5, 1.0, 0.5]) * (1.5 + d.X[:, 2:3].astype(float))
    good = el.table_nll(sd * 0.6745 * 1.4826, np.array([0, 0.3, 0]) * ref, d)
    bad = el.table_nll(sd * 10, np.zeros_like(sd), d)
    assert good["nll"] < bad["nll"]
    assert 0.6 < good["cover1"][1] < 0.8


def test_early_stopping_keeps_the_best_round_and_split_is_by_object():
    d = _synthetic(3000, 1.0, 11)
    d.row = np.arange(len(d)) % 300
    tr, va = el.split_by_object(d, 0.2, seed=0)
    assert not set(tr.row) & set(va.row) and 0.1 < len(va) / len(d) < 0.3
    m, log = el.fedavg({"A": d, "B": _synthetic(500, 1.0, 12)}, rounds=8, seed=0, patience=2)
    vals = [h["val_nll"] for h in log.history]
    assert log.best_round == log.history[int(np.argmin(vals))]["round"]


def test_dp_epsilon_shrinks_with_noise_and_grows_with_rounds():
    e1 = el.gaussian_rdp_epsilon(1.0, 10)
    assert el.gaussian_rdp_epsilon(2.0, 10) < e1 < el.gaussian_rdp_epsilon(1.0, 40)
    assert math.isinf(el.gaussian_rdp_epsilon(0.0, 10))
