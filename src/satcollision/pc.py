"""
Non-circular probability of collision: the Foster/Estes 2D encounter-plane
method, with a full 3x3 position covariance per object.

The twin layer's :func:`satcollision.twin.compute_pc` models position
uncertainty as a single scalar sigma — a circular error in the encounter
plane. That is the standard teaching formula and it is fine for ranking
encounters, but it cannot represent the one fact that matters most for this
project: the position error of a TLE-propagated object is overwhelmingly
**along-track**, typically 5-10x its radial and cross-track components. A
circular model averages that anisotropy away.

It is not a detail. It is the physical reason the federated scheme wins. An
operator's own ephemeris collapses the along-track error on its own
satellite, and along-track is exactly the direction the circular model
cannot see. With a real covariance the cooperative advantage stops being
"the number got better" and becomes a statement about *which axis of the
error ellipsoid each regime shrinks*.

The method (Foster & Estes, NASA JSC-25898, 1992) is the standard
short-encounter formulation:

1. At TCA, form the relative position ``r_rel`` and relative velocity
   ``v_rel`` of the two objects.
2. The **encounter plane** (B-plane) is the plane normal to ``v_rel``. Under
   the short-encounter assumption — relative motion linear and covariance
   static across the encounter — the 3D collision integral collapses to a 2D
   integral in this plane.
3. Project the combined covariance ``C = C_a + C_b`` into that plane, giving
   a 2x2 matrix; diagonalise it for the principal axes sigma_x, sigma_y.
4. Project ``r_rel`` into the plane and rotate into those principal axes,
   giving the miss vector ``(x_m, y_m)``.
5. Integrate the 2D Gaussian over the hard-body circle of radius ``R``:

       Pc = 1/(2 pi sx sy)
            * int_0^R int_0^2pi
              exp( -0.5 * [ ((p cos t + x_m)/sx)^2 + ((p sin t + y_m)/sy)^2 ] )
              * p dt dp

   in polar coordinates centred on the hard-body circle, so the domain is a
   rectangle in ``(rho, theta)`` and a fixed-order tensor Gauss-Legendre rule
   applies directly.

**Why fixed-order quadrature and not ``scipy.integrate.dblquad``.** Adaptive
quadrature is slower here and, more to the point, is not bit-reproducible
across runs and versions — its refinement path depends on floating-point
comparisons against a tolerance. Every figure and PDF in this project is
deterministic on purpose so that a re-render does not churn git; the Pc model
underneath them has to be deterministic too. A fixed tensor rule evaluates
the same nodes in the same order every time.

``twin.compute_pc`` is deliberately left untouched. This module is additive:
the circular model remains the default everywhere, and the two are meant to
be compared rather than silently substituted. The selectable-model layer at
the end of this file (:class:`CircularPcModel`, :class:`FosterPcModel`,
:func:`resolve_pc_model`) is how that comparison is run: the pipeline takes a
``pc_model`` argument that defaults to ``None``, meaning circular, on exactly
the code path it used before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_HBR_KM = 0.02           # ~20 m combined hard-body radius; matches twin.py
DEFAULT_QUAD_ORDER = 64         # Gauss-Legendre nodes per panel, per dimension
MAX_PANELS = 32                 # cap on the composite-rule subdivision
N_SIGMA_SUPPORT = 10.0          # beyond this many sigma the Gaussian contributes nothing
MAX_HBR_SIGMA_RATIO = 250.0     # resolvable envelope: R / smallest in-plane sigma
MIN_RELATIVE_SPEED_KM_S = 1.0e-6  # below this the encounter plane is ill-conditioned

# Typical TLE/catalogue-grade anisotropy: along-track error dominates radial
# and cross-track by roughly an order of magnitude. Used as the default when
# constructing a covariance from a single scalar sigma.
DEFAULT_ALONG_TRACK_RATIO = 8.0


def rtn_covariance(position_km,
                   velocity_km_s,
                   sigma_radial_km: float,
                   sigma_along_km: float,
                   sigma_cross_km: float) -> np.ndarray:
    """Build a 3x3 ECI position covariance from RTN-frame 1-sigma components.

    The RTN (radial / along-track / cross-track) frame is the natural one for
    orbital position error, because the error is anisotropic *in that frame*
    and roughly stationary there over an encounter. The covariance is diagonal
    in RTN by construction — cross-correlations between the three axes are
    second-order next to the along-track dominance this is here to capture —
    and is then rotated into ECI, where the encounter geometry lives:

        C_eci = M diag(sr^2, st^2, sc^2) M^T

    with ``M``'s columns the RTN basis vectors expressed in ECI:

        R_hat = r / |r|                     (radial, outward)
        C_hat = (r x v) / |r x v|           (cross-track, orbit normal)
        T_hat = C_hat x R_hat               (along-track, completes the triad)

    ``T_hat`` is built as ``C_hat x R_hat`` rather than from ``v`` directly so
    that the triad is exactly orthonormal even when the orbit is eccentric and
    the velocity is therefore not perpendicular to the radius.

    Raises ``ValueError`` on a degenerate state (zero radius, or position and
    velocity parallel, which leaves the orbit normal undefined) and on a
    non-positive sigma.
    """
    r = np.asarray(position_km, dtype=float).reshape(3)
    v = np.asarray(velocity_km_s, dtype=float).reshape(3)

    for name, value in (("sigma_radial_km", sigma_radial_km),
                        ("sigma_along_km", sigma_along_km),
                        ("sigma_cross_km", sigma_cross_km)):
        if not (value > 0.0) or not math.isfinite(value):
            raise ValueError(f"{name} must be a positive finite value, got {value!r}")

    r_norm = float(np.linalg.norm(r))
    if r_norm == 0.0:
        raise ValueError("position_km has zero magnitude; RTN frame is undefined")
    r_hat = r / r_norm

    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if h_norm <= 1.0e-12 * r_norm * max(float(np.linalg.norm(v)), 1.0e-12):
        raise ValueError(
            "position and velocity are parallel (zero angular momentum); "
            "the cross-track direction is undefined"
        )
    c_hat = h / h_norm
    t_hat = np.cross(c_hat, r_hat)

    basis = np.column_stack((r_hat, t_hat, c_hat))  # ECI columns, RTN order
    variances = np.diag(np.array([sigma_radial_km, sigma_along_km, sigma_cross_km]) ** 2)
    cov = basis @ variances @ basis.T
    return 0.5 * (cov + cov.T)  # symmetrise away round-off


def rtn_covariance_from_isotropic(position_km,
                                  velocity_km_s,
                                  sigma_km: float,
                                  along_track_ratio: float = DEFAULT_ALONG_TRACK_RATIO,
                                  cross_to_radial_ratio: float = 1.0) -> np.ndarray:
    """RTN covariance calibrated to match a scalar sigma in total variance.

    This is the continuity hinge with the rest of the project. Every existing
    result is stated against a single combined sigma (``twin.DEFAULT_COMBINED_
    SIGMA_KM`` = 0.5 km), so an anisotropic covariance that is simply "bigger"
    or "smaller" would not be comparable to any of them. Here the anisotropy
    is introduced at **constant total variance**: the shape of the error
    ellipsoid changes, its overall size does not.

    Concretely, with ``k = along_track_ratio`` and ``m = cross_to_radial_ratio``,
    the components satisfy ``sr^2 + st^2 + sc^2 = 3 sigma^2`` — the trace of an
    isotropic covariance with the same sigma — which gives

        sr = sigma * sqrt(3 / (1 + k^2 + m^2)),   st = k sr,   sc = m sr

    So ``along_track_ratio=1.0`` reproduces the isotropic case exactly, and
    raising it redistributes existing uncertainty into the along-track axis
    rather than adding any. That is the honest comparison to run: it isolates
    the effect of the *shape* of the uncertainty, which is the claim being
    made, from the effect of its magnitude, which is not.
    """
    if not (sigma_km > 0.0) or not math.isfinite(sigma_km):
        raise ValueError(f"sigma_km must be a positive finite value, got {sigma_km!r}")
    if not (along_track_ratio > 0.0) or not math.isfinite(along_track_ratio):
        raise ValueError(
            f"along_track_ratio must be a positive finite value, got {along_track_ratio!r}"
        )
    if not (cross_to_radial_ratio > 0.0) or not math.isfinite(cross_to_radial_ratio):
        raise ValueError(
            "cross_to_radial_ratio must be a positive finite value, "
            f"got {cross_to_radial_ratio!r}"
        )

    scale = math.sqrt(3.0 / (1.0 + along_track_ratio ** 2 + cross_to_radial_ratio ** 2))
    sigma_radial = sigma_km * scale
    return rtn_covariance(
        position_km,
        velocity_km_s,
        sigma_radial_km=sigma_radial,
        sigma_along_km=sigma_radial * along_track_ratio,
        sigma_cross_km=sigma_radial * cross_to_radial_ratio,
    )


def encounter_plane_basis(v_rel_km_s) -> np.ndarray:
    """Two orthonormal ECI vectors spanning the plane normal to ``v_rel``.

    Returned as a 3x2 matrix whose columns are the basis vectors. Which pair
    of vectors is chosen inside the plane does not matter — the Pc integral is
    invariant to rotation within the encounter plane, since the covariance is
    diagonalised afterwards — so the only requirement is numerical
    conditioning. The seed vector is therefore taken as whichever axis is
    *least* aligned with ``v_rel``, which bounds the cross-product magnitude
    away from zero.
    """
    v = np.asarray(v_rel_km_s, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))
    if not math.isfinite(speed) or speed < MIN_RELATIVE_SPEED_KM_S:
        raise ValueError(
            f"relative speed {speed!r} km/s is below {MIN_RELATIVE_SPEED_KM_S} km/s: "
            "the short-encounter assumption (linear relative motion, static "
            "covariance) does not hold and the encounter plane is undefined"
        )
    v_hat = v / speed

    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(v_hat)))] = 1.0

    u = np.cross(v_hat, seed)
    u /= np.linalg.norm(u)
    w = np.cross(v_hat, u)
    w /= np.linalg.norm(w)
    return np.column_stack((u, w))


def encounter_plane_miss_km(r_rel_km, v_rel_km_s) -> float:
    """The miss distance the short-encounter model actually uses.

    Only the component of the separation *perpendicular* to the relative
    velocity matters for a fast encounter: the along-velocity component says
    where the two objects are on their approach, not how close they will pass.
    Under linear relative motion the true closest approach happens at
    ``t* = -(r . v) / |v|^2``, and the separation there is exactly this
    perpendicular component. So projecting into the encounter plane and
    refining TCA are the same operation.

    This matters for reading a model comparison. ``twin.compute_pc`` is handed
    the raw 3D separation at the closest *sampled* timestep, and the screening
    grid is 30 s wide — at a relative speed of several km/s a sample can sit a
    long way along the track from the true minimum, and all of that error lands
    in the along-velocity direction. The Foster path projects it out; the
    circular path cannot, because a scalar distance has no direction left in
    it.

    Which means a Foster-vs-circular run moves for **two** independent reasons:
    the error ellipsoid is no longer circular, and the miss distance is no
    longer contaminated by grid resolution. They are separable — compare
    ``compute_pc`` at this in-plane miss against ``compute_pc`` at the raw miss
    to isolate the second — and a report that attributes the whole difference
    to anisotropy would be wrong.
    """
    r = np.asarray(r_rel_km, dtype=float).reshape(3)
    v = np.asarray(v_rel_km_s, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))
    if not math.isfinite(speed) or speed < MIN_RELATIVE_SPEED_KM_S:
        raise ValueError(
            f"relative speed {speed!r} km/s is below {MIN_RELATIVE_SPEED_KM_S} km/s: "
            "the encounter plane is undefined"
        )
    v_hat = v / speed
    return float(np.linalg.norm(r - np.dot(r, v_hat) * v_hat))


def _project_to_encounter_plane(r_rel_km, v_rel_km_s, cov_combined):
    """Reduce the 3D geometry to principal-axis 2D: ``(x_m, y_m, sx, sy)``."""
    basis = encounter_plane_basis(v_rel_km_s)
    r = np.asarray(r_rel_km, dtype=float).reshape(3)

    cov2 = basis.T @ cov_combined @ basis
    cov2 = 0.5 * (cov2 + cov2.T)
    miss2 = basis.T @ r

    # Symmetric 2x2: eigh gives an orthonormal eigenbasis and real eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(cov2)
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            "combined covariance projects to a non-positive-definite 2x2 matrix "
            f"(eigenvalues {eigenvalues.tolist()}); Pc is undefined"
        )

    sigma_x, sigma_y = np.sqrt(eigenvalues)
    x_m, y_m = eigenvectors.T @ miss2  # miss vector in principal axes
    return float(x_m), float(y_m), float(sigma_x), float(sigma_y)


def _panel_count(hbr_km, sigma_min_km) -> int:
    """How many quadrature panels each dimension needs, deterministically.

    A single Gauss-Legendre panel is a polynomial fit across the whole domain,
    so it is excellent while the integrand varies smoothly over the disc and
    degrades once the disc spans many standard deviations — there the Gaussian
    falls by orders of magnitude from one edge to the other and no moderate
    polynomial follows it. In this project's regime that never happens (a 20 m
    hard-body radius against uncertainties of hundreds of metres is a disc over
    which the Gaussian is very nearly constant), but the module should not be
    quietly wrong outside it, and the closed-form validation deliberately
    probes cases with R >> sigma.

    So the domain is split into panels sized at a quarter of the smaller
    sigma, capped at ``MAX_PANELS``. The count is a pure function of the
    geometry, not of an error estimate, which keeps the rule fixed-order and
    bit-reproducible: the same inputs evaluate the same nodes in the same
    order on every run.
    """
    if sigma_min_km <= 0.0:
        return 1
    needed = math.ceil(hbr_km / (0.25 * sigma_min_km))
    return int(min(MAX_PANELS, max(1, needed)))


def _composite_nodes(lower, upper, n_panels, nodes, weights):
    """Map a Gauss-Legendre rule onto ``n_panels`` equal sub-intervals."""
    edges = np.linspace(lower, upper, n_panels + 1)
    half = 0.5 * (edges[1] - edges[0])
    centres = 0.5 * (edges[:-1] + edges[1:])
    x = (centres[:, None] + half * nodes[None, :]).ravel()
    w = np.tile(half * weights, n_panels)
    return x, w


@lru_cache(maxsize=16)
def _leggauss(order: int):
    """Cached Gauss-Legendre nodes and weights.

    ``leggauss`` solves an eigenvalue problem, which at order 64 costs more
    than the integrand evaluations it is used for — and screening a real
    catalogue calls this thousands of times with the same order. The arrays
    are made read-only so a caller cannot mutate the shared copy.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return nodes, weights


def _integration_radius(x_m, y_m, sigma_x, sigma_y, hbr_km) -> float:
    """Outer radius the rho integration actually needs to cover.

    The integrand is the Gaussian restricted to the hard-body disc, and the
    Gaussian's own support is finite in any practical sense: beyond ten
    standard deviations from its centre the density is below 1e-22 of its peak,
    far under the precision any probability here is quoted to. So when the
    disc is much larger than the error ellipse — a big hard body, or an
    extremely well-determined state — most of the disc contributes nothing, and
    integrating over all of it wastes every node on empty space. That is not
    merely slow: with a fixed node budget it is *inaccurate*, because the
    handful of nodes that land near the Gaussian cannot resolve it.

    Clipping the outer radius to where the Gaussian has died concentrates the
    same nodes on the region that carries the mass. The discarded annulus holds
    less than the ten-sigma tail, so nothing measurable is lost, and in the
    project's own regime (a 20 m hard body inside a covariance hundreds of
    metres across) the clip is inactive and this returns ``hbr_km`` unchanged.
    """
    centre_offset = math.hypot(x_m, y_m)
    support = centre_offset + N_SIGMA_SUPPORT * max(sigma_x, sigma_y)
    return min(hbr_km, support)


def _gauss_legendre_2d(x_m, y_m, sigma_x, sigma_y, hbr_km, order):
    """Composite tensor Gauss-Legendre over the hard-body disc, in polar form.

    Two cases are settled before any quadrature runs.

    *Containment.* If the whole error ellipse sits inside the hard-body circle
    — the objects are known far more precisely than their combined size — then
    collision is certain and the answer is 1. Stating that directly is both
    exact (to within the ten-sigma tail outside the circle) and better
    conditioned than asking a quadrature rule to integrate a narrow spike
    inside a vast domain and hope it lands on it.

    *Resolvability.* Outside that case, a fixed-order rule has an honest limit:
    once the disc spans more than a few hundred standard deviations, the panel
    cap can no longer resolve the Gaussian and the rule would return a
    confident wrong number. It raises instead. The limit is three orders of
    magnitude beyond anything this project encounters — a 20 m hard body
    against a 500 m uncertainty is a ratio of 0.04, against the limit of 250 —
    so it exists to make the failure mode explicit, not to be hit.
    """
    sigma_max = max(sigma_x, sigma_y)
    sigma_min = min(sigma_x, sigma_y)
    centre_offset = math.hypot(x_m, y_m)

    if centre_offset + N_SIGMA_SUPPORT * sigma_max <= hbr_km:
        return 1.0
    if hbr_km > MAX_HBR_SIGMA_RATIO * sigma_min:
        raise ValueError(
            f"hard-body radius {hbr_km} km is more than {MAX_HBR_SIGMA_RATIO}x the "
            f"smallest encounter-plane sigma ({sigma_min} km) without containing the "
            "error ellipse; the fixed-order quadrature cannot resolve this geometry"
        )

    nodes, weights = _leggauss(order)
    rho_outer = _integration_radius(x_m, y_m, sigma_x, sigma_y, hbr_km)
    n_panels = _panel_count(rho_outer, min(sigma_x, sigma_y))

    # rho in [0, rho_outer]; theta in [0, 2 pi]
    rho, w_rho = _composite_nodes(0.0, rho_outer, n_panels, nodes, weights)
    theta, w_theta = _composite_nodes(0.0, 2.0 * math.pi, n_panels, nodes, weights)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Outer products over the (rho, theta) grid.
    dx = (rho[:, None] * cos_t[None, :] + x_m) / sigma_x
    dy = (rho[:, None] * sin_t[None, :] + y_m) / sigma_y
    integrand = np.exp(-0.5 * (dx * dx + dy * dy)) * rho[:, None]

    integral = float(w_rho @ integrand @ w_theta)
    return integral / (2.0 * math.pi * sigma_x * sigma_y)


def foster_pc(r_rel_km,
              v_rel_km_s,
              cov_a,
              cov_b,
              hbr_km: float = DEFAULT_HBR_KM,
              order: int = DEFAULT_QUAD_ORDER) -> float:
    """Probability of collision by the Foster/Estes 2D encounter-plane method.

    ``r_rel_km`` and ``v_rel_km_s`` are the relative position and velocity at
    TCA (object A minus object B, in ECI, km and km/s). ``cov_a`` and ``cov_b``
    are the two objects' 3x3 ECI *position* covariances in km^2; they are
    assumed independent and are summed. ``hbr_km`` is the combined hard-body
    radius. ``order`` is the Gauss-Legendre node count per dimension.

    The result is clipped to 1.0. Raises ``ValueError`` on a relative speed too
    low for the short-encounter assumption, on a non-positive-definite combined
    covariance, and on a non-positive hard-body radius or quadrature order.
    """
    if not (hbr_km > 0.0) or not math.isfinite(hbr_km):
        raise ValueError(f"hbr_km must be a positive finite value, got {hbr_km!r}")
    if int(order) < 2:
        raise ValueError(f"order must be at least 2, got {order!r}")

    cov_a = np.asarray(cov_a, dtype=float).reshape(3, 3)
    cov_b = np.asarray(cov_b, dtype=float).reshape(3, 3)
    cov = cov_a + cov_b
    cov = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(cov)):
        raise ValueError("combined covariance contains non-finite entries")

    x_m, y_m, sigma_x, sigma_y = _project_to_encounter_plane(r_rel_km, v_rel_km_s, cov)
    pc = _gauss_legendre_2d(x_m, y_m, sigma_x, sigma_y, hbr_km, int(order))
    return float(min(max(pc, 0.0), 1.0))


def foster_pc_2d(miss_x_km: float,
                 miss_y_km: float,
                 sigma_x_km: float,
                 sigma_y_km: float,
                 hbr_km: float = DEFAULT_HBR_KM,
                 order: int = DEFAULT_QUAD_ORDER) -> float:
    """Foster Pc directly from encounter-plane quantities.

    The 3D entry point above exists to turn orbital geometry into these four
    numbers; this one takes them as given. Useful for validating the quadrature
    against closed forms and published test cases, which are stated in the
    plane rather than in ECI, and for the encounter-plane figure.
    """
    if not (hbr_km > 0.0) or not math.isfinite(hbr_km):
        raise ValueError(f"hbr_km must be a positive finite value, got {hbr_km!r}")
    if not (sigma_x_km > 0.0) or not (sigma_y_km > 0.0):
        raise ValueError(
            f"sigmas must be positive, got ({sigma_x_km!r}, {sigma_y_km!r})"
        )
    if int(order) < 2:
        raise ValueError(f"order must be at least 2, got {order!r}")
    pc = _gauss_legendre_2d(
        float(miss_x_km), float(miss_y_km),
        float(sigma_x_km), float(sigma_y_km),
        float(hbr_km), int(order),
    )
    return float(min(max(pc, 0.0), 1.0))


# --------------------------------------------------------------------------
# Selectable Pc models
# --------------------------------------------------------------------------
#
# The pipeline evaluates Pc in two places: once per candidate pair in
# ``twin.screen_conjunctions`` (the number stamped on each Conjunction and used
# for threshold filtering), and thousands of times per encounter inside
# ``evaluate.detection_lead_time``, which is what actually produces the
# warning-time result. Both took the circular formula as a hard-coded call.
#
# These classes make that choice a parameter without changing the default.
# ``pc_model=None`` everywhere means "circular", on the same code path as
# before, so every existing figure, table and test is untouched. Passing a
# model swaps the estimator and nothing else — same screening, same regimes,
# same thresholds — which is what makes running the real evaluation both ways
# an honest comparison rather than two different experiments.
#
# **The sigma knob keeps its meaning.** The regimes (``SIGMA_GOOD_KM``,
# ``sigma_public_tracking``) express uncertainty as one scalar combined sigma,
# and that stays the single dial. A model receives that scalar and decides how
# to turn it into covariance. ``FosterPcModel`` splits it symmetrically between
# the two objects and redistributes each into an RTN ellipsoid at constant
# trace, so ``along_track_ratio=1.0`` reproduces the circular answer and any
# difference at a higher ratio is attributable to the *shape* of the error
# ellipsoid alone. That is the claim the project is making, isolated.
#
# **The asymmetric case is the point, and the hook is here for it.** What the
# federated scheme actually buys an operator is a collapsed along-track error
# on *its own* satellite while the other object stays catalogue-grade — "I have
# a filtered covariance for mine and a public one for yours". That needs a
# per-object sigma, which the current regime functions do not supply, so it is
# not wired yet. ``sigma_a_km``/``sigma_b_km`` on the model interface are where
# it goes: supply them and the symmetric split is bypassed, with no change to
# any call site that does not.


@dataclass(frozen=True)
class EncounterGeometry:
    """What a Pc model may need about one encounter at TCA.

    The miss distance is always present; the state vectors are present only
    when the encounter came from ``screen_conjunctions`` (which records them)
    rather than being constructed by hand from a scalar. Models that need the
    geometry check :attr:`has_state_vectors` and say so when it is missing,
    rather than quietly falling back to a different model — a silent fallback
    inside a comparison run would mean comparing the circular model against
    itself for some fraction of the encounters and never knowing which.

    Positions and velocities are ECI, km and km/s. Both objects' own states are
    carried, not just the relative ones, because each object's error ellipsoid
    is built in *its own* RTN frame.
    """

    miss_distance_km: float
    position_a_km: np.ndarray | None = None
    position_b_km: np.ndarray | None = None
    velocity_a_km_s: np.ndarray | None = None
    velocity_b_km_s: np.ndarray | None = None

    @property
    def has_state_vectors(self) -> bool:
        return all(v is not None for v in (self.position_a_km, self.position_b_km,
                                           self.velocity_a_km_s, self.velocity_b_km_s))

    @property
    def r_rel_km(self) -> np.ndarray:
        return np.asarray(self.position_a_km, dtype=float) - np.asarray(self.position_b_km, dtype=float)

    @property
    def v_rel_km_s(self) -> np.ndarray:
        return np.asarray(self.velocity_a_km_s, dtype=float) - np.asarray(self.velocity_b_km_s, dtype=float)

    @property
    def in_plane_miss_km(self) -> float:
        """Miss distance projected into the encounter plane.

        What a short-encounter Pc model is really scoring, and generally
        smaller than :attr:`miss_distance_km`, which still carries whatever
        along-track offset the screening grid left in. See
        :func:`encounter_plane_miss_km`.
        """
        if not self.has_state_vectors:
            raise ValueError("no state vectors: the encounter plane is undefined")
        return encounter_plane_miss_km(self.r_rel_km, self.v_rel_km_s)

    @classmethod
    def from_conjunction(cls, conjunction) -> "EncounterGeometry":
        """Build from a :class:`twin.Conjunction`, state vectors included if it has them."""
        return cls(
            miss_distance_km=conjunction.miss_distance_km,
            position_a_km=getattr(conjunction, "position_a_km", None),
            position_b_km=getattr(conjunction, "position_b_km", None),
            velocity_a_km_s=getattr(conjunction, "velocity_a_km_s", None),
            velocity_b_km_s=getattr(conjunction, "velocity_b_km_s", None),
        )


@dataclass(frozen=True)
class CircularPcModel:
    """The project's existing estimator, behind the model interface.

    Delegates to ``twin.compute_pc`` rather than reimplementing it, so there is
    exactly one circular formula in the codebase and this cannot drift from the
    default path. Geometry beyond the miss distance is ignored — that is the
    model's defining limitation, not an oversight.
    """

    hbr_km: float = DEFAULT_HBR_KM
    name: str = "circular"

    def pc(self, geometry: EncounterGeometry, sigma_km: float,
           sigma_a_km: float | None = None, sigma_b_km: float | None = None) -> float:
        from .twin import compute_pc  # local: twin imports this module's foster_pc path

        combined = _combined_sigma(sigma_km, sigma_a_km, sigma_b_km)
        return compute_pc(geometry.miss_distance_km, combined, self.hbr_km)


@dataclass(frozen=True)
class FosterPcModel:
    """Foster/Estes with an anisotropic RTN covariance built per object.

    ``along_track_ratio`` is the one parameter that matters. At 1.0 this is the
    circular model to within the small-hard-body approximation (about 0.1% at
    the project's operating point), which is the continuity check worth running
    before trusting any comparison. The default of 8.0 is the typical
    TLE-propagated anisotropy.

    The scalar ``sigma_km`` is read as the *combined* uncertainty, so each
    object gets ``sigma_km / sqrt(2)`` — that is what makes ``C_a + C_b`` come
    back to ``sigma_km`` and keeps the regimes comparable across models. Pass
    ``sigma_a_km``/``sigma_b_km`` to model the two objects asymmetrically
    instead.

    **This model does not see the same miss distance the circular one does.**
    It projects the separation into the encounter plane, which drops the
    along-velocity component — and that component is where the screening grid's
    resolution error sits. So a Foster-vs-circular comparison moves for two
    reasons, not one; :func:`encounter_plane_miss_km` documents how to separate
    them, and :attr:`EncounterGeometry.in_plane_miss_km` reports the projected
    value per encounter so a run can quantify it rather than assume it is
    small.
    """

    along_track_ratio: float = DEFAULT_ALONG_TRACK_RATIO
    cross_to_radial_ratio: float = 1.0
    hbr_km: float = DEFAULT_HBR_KM
    order: int = DEFAULT_QUAD_ORDER

    @property
    def name(self) -> str:
        return f"foster(k={self.along_track_ratio:g})"

    def pc(self, geometry: EncounterGeometry, sigma_km: float,
           sigma_a_km: float | None = None, sigma_b_km: float | None = None) -> float:
        if not geometry.has_state_vectors:
            raise ValueError(
                "FosterPcModel needs the encounter's state vectors at TCA and this "
                "geometry has none (miss distance "
                f"{geometry.miss_distance_km} km only). Screen the encounter with "
                "twin.screen_conjunctions, which records them, or use CircularPcModel."
            )

        if sigma_a_km is None and sigma_b_km is None:
            sigma_a_km = sigma_b_km = sigma_km / math.sqrt(2.0)
        elif sigma_a_km is None or sigma_b_km is None:
            raise ValueError(
                "supply both sigma_a_km and sigma_b_km for an asymmetric encounter, "
                "or neither for the symmetric split of sigma_km"
            )

        cov_a = rtn_covariance_from_isotropic(
            geometry.position_a_km, geometry.velocity_a_km_s, sigma_a_km,
            along_track_ratio=self.along_track_ratio,
            cross_to_radial_ratio=self.cross_to_radial_ratio,
        )
        cov_b = rtn_covariance_from_isotropic(
            geometry.position_b_km, geometry.velocity_b_km_s, sigma_b_km,
            along_track_ratio=self.along_track_ratio,
            cross_to_radial_ratio=self.cross_to_radial_ratio,
        )
        return foster_pc(geometry.r_rel_km, geometry.v_rel_km_s, cov_a, cov_b,
                         hbr_km=self.hbr_km, order=self.order)


def _combined_sigma(sigma_km, sigma_a_km, sigma_b_km) -> float:
    """Collapse per-object sigmas to one combined sigma, or pass the scalar through.

    Independent errors add in quadrature, so the combined 1-sigma is
    ``sqrt(sa^2 + sb^2)``. When neither is given the scalar is returned
    untouched — not recomputed — so the default path is bit-identical to the
    code before the model interface existed.
    """
    if sigma_a_km is None and sigma_b_km is None:
        return sigma_km
    if sigma_a_km is None or sigma_b_km is None:
        raise ValueError("supply both per-object sigmas or neither")
    return math.hypot(sigma_a_km, sigma_b_km)


def resolve_pc_model(pc_model):
    """``None`` means circular. Anything else is returned unchanged.

    One place decides what the default is, so the meaning of ``pc_model=None``
    cannot diverge between ``screen_conjunctions``, ``detection_lead_time`` and
    the evaluation entry points.
    """
    return CIRCULAR_PC_MODEL if pc_model is None else pc_model


def pc_model_name(pc_model) -> str:
    """Label for the model, for run metadata and result rows."""
    return resolve_pc_model(pc_model).name


CIRCULAR_PC_MODEL = CircularPcModel()
