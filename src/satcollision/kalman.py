"""
Kalman-filter state estimation: replacing ``twin.py``'s fixed
``DEFAULT_COMBINED_SIGMA_KM`` assumption with genuinely evolving per-object
position uncertainty.

``twin.compute_pc`` uses one constant combined sigma (0.5km) for every
conjunction, regardless of how long it's actually been since either
operator last refined its tracking of the object. Real orbit determination
does not work that way: uncertainty is smallest right after a tracking
update (a radar pass, an optical observation, a fresh TLE) and grows
between updates as unmodeled forces (drag, solar radiation pressure,
third-body effects -- none of which a pure SGP4/Keplerian propagation
captures) accumulate error. This module implements a standard linear
Kalman filter -- constant-velocity process model, position-only
measurement model -- that produces a covariance which actually reflects
that growth-then-shrink pattern, and :func:`refine_pc_for_conjunction`
plugs a time-varying combined sigma straight into the existing
``twin.compute_pc`` formula in place of the constant.

This is deliberately scoped as a refinement layer callable per-encounter,
not a rewrite of ``twin.screen_conjunctions``'s broad-phase KD-tree pass:
that pass exists specifically to stay fast against a multi-thousand-object
real catalog (see ``twin.py``'s module docstring), and running a full
per-object Kalman track through every screening timestep for every object
would give up exactly the performance property that pass was built for.
Running it only for conjunctions that already crossed the screening
distance -- a small, sparse set by construction -- keeps the benefit
without the cost. Promoting this from a per-encounter refinement to the
default screening path is listed as future work in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fleets import TrackedObject
from .twin import Conjunction, DEFAULT_COMBINED_HBR_KM, compute_pc, propagate_window


@dataclass
class KalmanState:
    x: np.ndarray  # shape (6,): [rx, ry, rz, vx, vy, vz], km and km/s
    P: np.ndarray  # shape (6, 6): state covariance


def initial_state(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    position_sigma_km: float = 0.05,
    velocity_sigma_km_s: float = 0.0005,
) -> KalmanState:
    """Starting belief right after a fresh tracking update: small, fixed
    uncertainty, since this is the moment the filter is initialized from an
    (assumed accurate) observation."""
    x = np.concatenate([np.asarray(position_km, dtype=float), np.asarray(velocity_km_s, dtype=float)])
    P = np.diag([position_sigma_km ** 2] * 3 + [velocity_sigma_km_s ** 2] * 3)
    return KalmanState(x=x, P=P)


def _transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity process model over a short step ``dt``: position
    advances by velocity*dt, velocity is unchanged by the model itself
    (any real change in velocity between steps is treated as process
    noise -- see :func:`_process_noise` -- rather than modeled directly,
    since modeling actual orbital dynamics here would just be re-deriving
    SGP4, which is what this filter is deliberately estimating *without*
    relying on."""
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F


def _process_noise(dt: float, accel_noise_km_s2: float) -> np.ndarray:
    """Standard discrete white-noise-acceleration process noise: models
    everything SGP4/Keplerian propagation does *not* capture (drag, solar
    radiation pressure, third-body perturbations, maneuvers) as a small
    random acceleration of magnitude ``accel_noise_km_s2`` acting
    continuously between updates. This -- not the constant in
    ``twin.compute_pc`` -- is where uncertainty growth actually comes
    from in this model.
    """
    q = accel_noise_km_s2 ** 2
    dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
    Q = np.zeros((6, 6))
    for i in range(3):
        pi, vi = i, i + 3
        Q[pi, pi] = q * dt4 / 4.0
        Q[pi, vi] = Q[vi, pi] = q * dt3 / 2.0
        Q[vi, vi] = q * dt2
    return Q


def predict(state: KalmanState, dt: float, accel_noise_km_s2: float = 1e-7) -> KalmanState:
    """Advance the state estimate by ``dt`` seconds with no new
    measurement -- uncertainty only grows here, never shrinks; that
    asymmetry is exactly what a fixed sigma cannot represent."""
    F = _transition_matrix(dt)
    Q = _process_noise(dt, accel_noise_km_s2)
    x = F @ state.x
    P = F @ state.P @ F.T + Q
    return KalmanState(x=x, P=P)


def update(state: KalmanState, measured_position_km: np.ndarray, position_noise_km: float) -> KalmanState:
    """Fold in one new noisy position measurement (a tracking update) via
    the standard Kalman gain -- this is the step that shrinks uncertainty
    back down, the counterpart to the growth in :func:`predict`."""
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.eye(3) * position_noise_km ** 2
    innovation = np.asarray(measured_position_km, dtype=float) - H @ state.x
    innovation_cov = H @ state.P @ H.T + R
    K = state.P @ H.T @ np.linalg.inv(innovation_cov)
    x = state.x + K @ innovation
    P = (np.eye(6) - K @ H) @ state.P
    return KalmanState(x=x, P=P)


def track_object(
    true_positions_km: np.ndarray,
    true_velocities_km_s: np.ndarray,
    t_offsets_s: np.ndarray,
    measurement_interval_s: float,
    position_noise_km: float = 0.05,
    accel_noise_km_s2: float = 1e-7,
    rng: np.random.Generator | None = None,
) -> list[KalmanState]:
    """Run predict/update across a real SGP4-propagated ground-truth
    trajectory (``true_positions_km``/``true_velocities_km_s``, e.g. from
    :func:`~satcollision.twin.propagate_window`), feeding one simulated
    noisy position measurement every ``measurement_interval_s`` and
    predicting-only (growing uncertainty) at every timestep in between.
    Returns one :class:`KalmanState` per input timestep.
    """
    rng = rng if rng is not None else np.random.default_rng()
    state = initial_state(true_positions_km[0], true_velocities_km_s[0])
    states = [state]
    last_measurement_t = float(t_offsets_s[0])
    for i in range(1, len(t_offsets_s)):
        dt = float(t_offsets_s[i] - t_offsets_s[i - 1])
        state = predict(state, dt, accel_noise_km_s2)
        if float(t_offsets_s[i]) - last_measurement_t >= measurement_interval_s:
            noisy_measurement = true_positions_km[i] + rng.normal(scale=position_noise_km, size=3)
            state = update(state, noisy_measurement, position_noise_km)
            last_measurement_t = float(t_offsets_s[i])
        states.append(state)
    return states


def isotropic_sigma_km(state: KalmanState) -> float:
    """Collapse a 3x3 position covariance block into the single isotropic
    scalar sigma the simplified circular Pc formula in
    ``twin.compute_pc`` expects -- the average of the diagonal (trace/3),
    i.e. the RMS position uncertainty averaged over the three axes.
    """
    return float(np.sqrt(np.trace(state.P[:3, :3]) / 3.0))


def combined_sigma_km(state_a: KalmanState, state_b: KalmanState) -> float:
    """Independent-tracking combined sigma: two operators' tracking errors
    are assumed independent (they run their own filters on their own
    observations), so variances add before taking the square root -- the
    direct drop-in replacement for ``twin.DEFAULT_COMBINED_SIGMA_KM``,
    except now it is a genuine function of each object's actual tracking
    history instead of a constant.
    """
    var_a = np.trace(state_a.P[:3, :3]) / 3.0
    var_b = np.trace(state_b.P[:3, :3]) / 3.0
    return float(np.sqrt(var_a + var_b))


def refine_pc_for_conjunction(
    conjunction: Conjunction,
    jd0: float,
    fr0: float,
    measurement_interval_s: float = 120.0,
    position_noise_km: float = 0.05,
    accel_noise_km_s2: float = 1e-7,
    fine_step_s: float = 5.0,
    seed: int | None = None,
) -> dict:
    """Recompute one already-screened conjunction's Pc using a Kalman-
    filter-derived combined sigma at TCA instead of the fixed
    ``DEFAULT_COMBINED_SIGMA_KM``.

    Re-propagates just the conjunction's own two objects (not the whole
    catalog -- see module docstring) at a fine step from t=0 out to the
    conjunction's TCA, runs an independent Kalman track for each against
    that same SGP4 ground truth standing in for "what the operator's own
    sensors would have observed," and evaluates ``twin.compute_pc`` with
    the resulting time-varying combined sigma. Returns a dict with both
    Pc values side by side so the difference is visible, plus the
    Kalman-derived sigma that produced the new one.
    """
    rng = np.random.default_rng(seed)
    objects = [conjunction.object_a, conjunction.object_b]
    tca_s = conjunction.tca_offset_s
    duration_s = max(tca_s, fine_step_s)
    prop = propagate_window(objects, jd0, fr0, duration_s=duration_s, step_s=fine_step_s)

    if not np.all(prop.valid):
        raise ValueError("SGP4 reported invalid propagation for one of the two objects in this window")

    states_by_object = []
    for row in range(2):
        states = track_object(
            true_positions_km=prop.positions[row],
            true_velocities_km_s=prop.velocities[row],
            t_offsets_s=prop.t_offsets_s,
            measurement_interval_s=measurement_interval_s,
            position_noise_km=position_noise_km,
            accel_noise_km_s2=accel_noise_km_s2,
            rng=rng,
        )
        states_by_object.append(states)

    # nearest sampled index to the true TCA offset
    tca_idx = int(np.argmin(np.abs(prop.t_offsets_s - tca_s)))
    state_a_at_tca = states_by_object[0][tca_idx]
    state_b_at_tca = states_by_object[1][tca_idx]
    kalman_sigma_km = combined_sigma_km(state_a_at_tca, state_b_at_tca)

    kalman_pc = compute_pc(conjunction.miss_distance_km, combined_sigma_km=kalman_sigma_km,
                            combined_hbr_km=DEFAULT_COMBINED_HBR_KM)

    return {
        "pair_label": conjunction.pair_label,
        "fixed_sigma_pc": conjunction.pc,
        "kalman_sigma_km": kalman_sigma_km,
        "kalman_sigma_pc": kalman_pc,
        "seconds_since_last_measurement": _time_since_last_measurement(
            prop.t_offsets_s, tca_idx, measurement_interval_s
        ),
    }


def _time_since_last_measurement(t_offsets_s: np.ndarray, tca_idx: int, measurement_interval_s: float) -> float:
    """How long (in simulated seconds) it had been since the most recent
    tracking update at TCA -- purely descriptive context for why the
    Kalman-derived sigma came out the way it did."""
    tca_t = float(t_offsets_s[tca_idx])
    return tca_t % measurement_interval_s
