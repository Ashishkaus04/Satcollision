"""
Validation for the Foster/Estes non-circular Pc model.

The double integral in ``pc.py`` is short enough to look obviously right and
subtle enough to be wrong — a dropped Jacobian, an axis swapped between the
covariance and the miss vector, a factor of two in the exponent — in ways that
still produce plausible small numbers. Plausible small numbers are exactly what
this project is full of, so "the answer looked reasonable" is not evidence.

These tests pin it down four independent ways:

* **A closed form.** For a circular covariance the integral has an exact
  analytic value — the non-central chi-squared CDF with two degrees of freedom
  — so a whole regime is checkable to machine precision with no sampling at
  all. This is the single strongest test here and it covers 400+ combinations
  of sigma, miss distance and hard-body radius.
* **Monte Carlo**, for the anisotropic case, where no closed form exists.
  Seeded, so a failure is reproducible rather than a coin flip.
* **Internal consistency**: convergence under refinement, invariance under
  rotation of the whole geometry, agreement with the circular model in the
  limit where the circular model is valid.
* **Guards**: the physical assumptions the method rests on are asserted to
  fail loudly rather than return a plausible-looking wrong number.

Note on the published Alfano benchmark cases, which the design brief listed as
a possible external check: those are specified as full state vectors and 6x6
covariances at TCA, and no source with the complete tabulated inputs and
reference probabilities could be verified. Rather than transcribe numbers of
uncertain provenance, the exact non-central chi-squared check below is used as
the reference — it is a closed form rather than a published value, so it is
strictly stronger where it applies, and Monte Carlo covers the rest.
"""

import math

import numpy as np
import pytest
from scipy.stats import ncx2

from satcollision.pc import (
    DEFAULT_ALONG_TRACK_RATIO,
    MAX_HBR_SIGMA_RATIO,
    MIN_RELATIVE_SPEED_KM_S,
    N_SIGMA_SUPPORT,
    encounter_plane_basis,
    foster_pc,
    foster_pc_2d,
    rtn_covariance,
    rtn_covariance_from_isotropic,
)
from satcollision.twin import (
    DEFAULT_COMBINED_HBR_KM,
    DEFAULT_COMBINED_SIGMA_KM,
    compute_pc,
)


def _exact_circular_pc(miss_km, sigma_km, hbr_km):
    """Exact Pc for a circular covariance, in closed form.

    For ``X ~ N(0, sigma^2 I)`` in 2D and a hard-body circle of radius ``R``
    whose centre is a distance ``d`` away, the collision probability is
    ``P(|X - m| < R)`` with ``|m| = d``. Scaling by sigma makes
    ``|X - m|^2 / sigma^2`` a non-central chi-squared variate with 2 degrees of
    freedom and non-centrality ``(d/sigma)^2``, so

        Pc = ncx2.cdf( (R/sigma)^2, df=2, nc=(d/sigma)^2 )

    exactly. No approximation and no sampling — this is the value the
    quadrature has to reproduce.
    """
    return float(ncx2.cdf((hbr_km / sigma_km) ** 2, df=2, nc=(miss_km / sigma_km) ** 2))


# --------------------------------------------------------------------------
# 1. Exact closed-form agreement in the isotropic case
# --------------------------------------------------------------------------

def test_matches_closed_form_across_the_isotropic_regime():
    """Sweep sigma, miss distance and HBR; demand near machine precision.

    The sweep spans six orders of magnitude in sigma, R/sigma from 2e-6 to
    2e5, and miss distances from dead-centre to a thousand sigma out — far
    beyond this project's operating point (a 20 m hard body against hundreds
    of metres of uncertainty), because the interesting question is not whether
    the rule works where it is comfortable but whether it ever lies.

    Every case must land in exactly one of three buckets:

    * the error ellipse is swallowed by the hard-body circle, so Pc is 1;
    * the geometry is past the fixed-order rule's resolving power, so it
      raises rather than guesses;
    * otherwise the quadrature matches the closed form to 1e-11.

    Cases where the reference itself underflows to zero in double precision
    are skipped — there it is the reference that has lost precision.
    """
    sigmas = [1.0e-4, 0.002, 0.02, 0.05, 0.2, 0.5, 2.0, 5.0, 50.0]
    misses = [0.0, 1.0e-4, 0.01, 0.02, 0.1, 0.5, 0.8, 1.02, 3.0, 10.0, 100.0]
    radii = [1.0e-4, 0.001, 0.005, 0.02, 0.1, 0.5, 2.0, 20.0]

    checked = 0
    contained = 0
    refused = 0
    for sigma in sigmas:
        for miss in misses:
            for radius in radii:
                exact = _exact_circular_pc(miss, sigma, radius)
                if exact < 1.0e-280:  # reference underflowed, not a fair comparison
                    continue
                where = f"sigma={sigma} miss={miss} R={radius}"

                if miss + N_SIGMA_SUPPORT * sigma <= radius:
                    assert foster_pc_2d(miss, 0.0, sigma, sigma, radius) == 1.0, where
                    assert exact == pytest.approx(1.0, abs=1.0e-12), where
                    contained += 1
                elif radius > MAX_HBR_SIGMA_RATIO * sigma:
                    with pytest.raises(ValueError, match="cannot resolve"):
                        foster_pc_2d(miss, 0.0, sigma, sigma, radius)
                    refused += 1
                else:
                    got = foster_pc_2d(miss, 0.0, sigma, sigma, radius)
                    assert got == pytest.approx(exact, rel=1.0e-11), (
                        f"{where}: {got} vs exact {exact}"
                    )
                    checked += 1

    assert checked > 400, f"sweep degenerated to {checked} quadrature comparisons"
    assert contained > 0 and refused > 0, "sweep missed the edge-case branches"


def test_closed_form_agreement_is_orientation_independent():
    """The miss vector's direction in the plane must not matter when sx == sy."""
    sigma, radius, distance = 0.3, 0.02, 0.4
    exact = _exact_circular_pc(distance, sigma, radius)
    for angle in np.linspace(0.0, 2.0 * math.pi, 9):
        got = foster_pc_2d(
            distance * math.cos(angle), distance * math.sin(angle),
            sigma, sigma, radius,
        )
        assert got == pytest.approx(exact, rel=1.0e-11)


# --------------------------------------------------------------------------
# 2. Reduction to the circular model the rest of the project uses
# --------------------------------------------------------------------------

def test_reduces_to_twin_compute_pc_when_hbr_is_small():
    """Agreement with ``twin.compute_pc`` in the regime that formula assumes.

    ``compute_pc`` is the small-R limit: it evaluates the Gaussian density at
    the miss distance and multiplies by the area of the hard-body circle,
    which is exact only as R -> 0. At the project's operating point (R = 20 m,
    sigma = 500 m) the two agree to better than 0.1%, which is why every
    existing result in the project stands unchanged under the refined model.
    """
    sigma = DEFAULT_COMBINED_SIGMA_KM
    hbr = DEFAULT_COMBINED_HBR_KM
    for miss in (0.0, 0.5, 0.828, 1.02):
        refined = foster_pc_2d(miss, 0.0, sigma, sigma, hbr)
        circular = compute_pc(miss, sigma, hbr)
        assert refined == pytest.approx(circular, rel=1.0e-3)


def test_small_hbr_approximation_degrades_predictably_with_miss_distance():
    """Document where the circular formula starts to drift, rather than assume.

    The point-density approximation is worst far out in the tail, because
    there the density varies most sharply across the hard-body circle. The
    drift is still small in absolute terms, but it grows monotonically — worth
    asserting so that the report can state the limit of the old model instead
    of hand-waving it.
    """
    sigma, hbr = DEFAULT_COMBINED_SIGMA_KM, DEFAULT_COMBINED_HBR_KM
    errors = []
    for miss in (0.828, 1.5, 2.0, 3.0):
        refined = foster_pc_2d(miss, 0.0, sigma, sigma, hbr)
        circular = compute_pc(miss, sigma, hbr)
        errors.append(abs(refined - circular) / refined)
    assert errors == sorted(errors), f"expected monotone drift, got {errors}"
    assert errors[0] < 1.0e-3
    assert errors[-1] < 1.0e-2


# --------------------------------------------------------------------------
# 3. Monte Carlo cross-check for the anisotropic case
# --------------------------------------------------------------------------

@pytest.mark.parametrize("miss_x, miss_y, sigma_x, sigma_y, hbr", [
    (0.05, 0.05, 0.10, 0.50, 0.05),
    (0.12, 0.00, 0.08, 0.60, 0.04),
    (0.00, 0.30, 0.20, 0.90, 0.10),
])
def test_anisotropic_agrees_with_monte_carlo(miss_x, miss_y, sigma_x, sigma_y, hbr):
    """Sample the encounter-plane Gaussian and count hits inside the circle.

    There is no closed form once sx != sy, so this is the only independent
    reference available for the case the whole module exists to handle. The
    geometries are chosen to put Pc around 1e-2, high enough that a few million
    samples resolve it to well under a percent; a realistic 1e-4 encounter
    would need billions of samples to say anything at all, which is precisely
    why Monte Carlo is the cross-check and not the method.

    The tolerance is four standard errors of the sample proportion. The seed is
    fixed, so this either passes every time or fails every time.
    """
    rng = np.random.default_rng(20260920)
    n_samples = 2_000_000
    x = rng.normal(0.0, sigma_x, n_samples) - miss_x
    y = rng.normal(0.0, sigma_y, n_samples) - miss_y
    hits = int(np.count_nonzero(x * x + y * y < hbr * hbr))

    proportion = hits / n_samples
    std_error = math.sqrt(proportion * (1.0 - proportion) / n_samples)
    assert hits > 1000, "geometry too improbable for Monte Carlo to resolve"

    refined = foster_pc_2d(miss_x, miss_y, sigma_x, sigma_y, hbr)
    assert abs(refined - proportion) < 4.0 * std_error, (
        f"foster={refined:.6e} monte-carlo={proportion:.6e} +-{std_error:.2e}"
    )


# --------------------------------------------------------------------------
# 4. Quadrature convergence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("miss_x, miss_y, sigma_x, sigma_y, hbr", [
    (0.828, 0.0, 0.06, 0.50, 0.02),   # the real Starlink x Kuiper miss distance
    (0.000, 0.0, 0.05, 0.50, 0.50),   # hard body comparable to the error ellipse
    (2.000, 1.0, 0.10, 1.20, 0.02),   # deep tail
    (0.300, 0.2, 0.40, 0.40, 1.00),   # near-certain collision
])
def test_quadrature_has_converged_at_the_default_order(miss_x, miss_y, sigma_x, sigma_y, hbr):
    """Doubling the node count must not move the answer.

    If it does, the default order is not converged and every number downstream
    inherits a quadrature error of unknown size.
    """
    coarse = foster_pc_2d(miss_x, miss_y, sigma_x, sigma_y, hbr, order=32)
    default = foster_pc_2d(miss_x, miss_y, sigma_x, sigma_y, hbr, order=64)
    fine = foster_pc_2d(miss_x, miss_y, sigma_x, sigma_y, hbr, order=128)

    assert default == pytest.approx(coarse, rel=1.0e-9)
    assert default == pytest.approx(fine, rel=1.0e-9)


def test_result_is_bit_reproducible():
    """Same inputs, same bits — the determinism the figures and PDFs rely on."""
    args = (0.828, 0.1, 0.06, 0.5, 0.02)
    first = foster_pc_2d(*args)
    assert all(foster_pc_2d(*args) == first for _ in range(5))


# --------------------------------------------------------------------------
# 5. Monotonicity, and the dilution maximum
# --------------------------------------------------------------------------

def test_pc_falls_with_miss_distance_and_rises_with_hard_body_radius():
    sigma_x, sigma_y = 0.1, 0.8
    misses = [0.0, 0.2, 0.5, 1.0, 2.0]
    values = [foster_pc_2d(m, 0.0, sigma_x, sigma_y, 0.02) for m in misses]
    assert values == sorted(values, reverse=True), values

    radii = [0.005, 0.01, 0.02, 0.05, 0.1]
    values = [foster_pc_2d(0.5, 0.0, sigma_x, sigma_y, r) for r in radii]
    assert values == sorted(values), values


def test_pc_is_not_monotone_in_sigma_and_peaks_at_the_dilution_maximum():
    """The probability dilution trap, asserted rather than described.

    Pc does *not* simply rise as uncertainty shrinks. For a fixed non-zero miss
    distance the curve has a maximum: with a very large covariance the
    probability mass is spread thin and little of it lands on the hard body,
    and with a very small one the hard body sits far out in a now-sharp tail.
    In between there is a worst case.

    This matters operationally and is the standard argument against treating a
    low Pc as automatically safe — a low number can mean "well tracked and
    genuinely missing" or "barely tracked at all", and those call for opposite
    responses. For a small hard body the maximum sits at sigma = d / sqrt(2),
    which is what is checked here.
    """
    miss, hbr = 1.0, 0.02
    sigmas = np.geomspace(0.05, 20.0, 400)
    values = np.array([foster_pc_2d(miss, 0.0, s, s, hbr) for s in sigmas])

    peak_index = int(np.argmax(values))
    assert 0 < peak_index < len(sigmas) - 1, "maximum fell on the edge of the sweep"
    assert sigmas[peak_index] == pytest.approx(miss / math.sqrt(2.0), rel=0.02)

    # Strictly increasing up to the peak, strictly decreasing after it.
    assert np.all(np.diff(values[: peak_index + 1]) > 0.0)
    assert np.all(np.diff(values[peak_index:]) < 0.0)


# --------------------------------------------------------------------------
# 6. The 3D entry point: frame construction and invariance
# --------------------------------------------------------------------------

def _circular_orbit_state(radius_km=7000.0, speed_km_s=7.5):
    return np.array([radius_km, 0.0, 0.0]), np.array([0.0, speed_km_s, 0.0])


def test_rtn_covariance_is_symmetric_positive_definite_with_the_right_axes():
    position, velocity = _circular_orbit_state()
    cov = rtn_covariance(position, velocity, 0.1, 0.8, 0.12)

    assert np.allclose(cov, cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > 0.0
    # The principal 1-sigmas are the RTN components, recovered in any order.
    assert sorted(np.sqrt(eigenvalues)) == pytest.approx([0.1, 0.12, 0.8], rel=1.0e-12)

    # For this circular orbit the RTN axes align with ECI, so the covariance is
    # diagonal and each axis is identifiable by name.
    assert cov[0, 0] == pytest.approx(0.1 ** 2, rel=1.0e-12)   # radial  -> x
    assert cov[1, 1] == pytest.approx(0.8 ** 2, rel=1.0e-12)   # along   -> y
    assert cov[2, 2] == pytest.approx(0.12 ** 2, rel=1.0e-12)  # cross   -> z


def test_isotropic_calibration_preserves_total_variance():
    """Anisotropy is introduced at constant trace, so sizes stay comparable."""
    position, velocity = _circular_orbit_state()
    sigma = DEFAULT_COMBINED_SIGMA_KM

    isotropic = rtn_covariance_from_isotropic(position, velocity, sigma, along_track_ratio=1.0)
    assert np.allclose(isotropic, sigma ** 2 * np.eye(3), atol=1.0e-12)

    for ratio in (2.0, DEFAULT_ALONG_TRACK_RATIO, 20.0):
        cov = rtn_covariance_from_isotropic(position, velocity, sigma, along_track_ratio=ratio)
        assert np.trace(cov) == pytest.approx(3.0 * sigma ** 2, rel=1.0e-12)
        sigmas = np.sqrt(np.linalg.eigvalsh(cov))
        assert sigmas.max() / sigmas.min() == pytest.approx(ratio, rel=1.0e-12)


def test_encounter_plane_basis_is_orthonormal_and_normal_to_relative_velocity():
    for v_rel in (np.array([0.0, 0.0, 10.0]),
                  np.array([1.0, 1.0, 1.0]),
                  np.array([1.0e-3, -5.0, 0.2])):
        basis = encounter_plane_basis(v_rel)
        assert basis.shape == (3, 2)
        assert np.allclose(basis.T @ basis, np.eye(2), atol=1.0e-14)
        assert np.allclose(basis.T @ v_rel, np.zeros(2), atol=1.0e-12)


def test_foster_pc_is_invariant_under_rotation_of_the_whole_geometry():
    """Pc is a physical quantity; it cannot depend on the choice of ECI axes.

    This is the test that catches a basis handled inconsistently between the
    covariance and the miss vector — the most likely way to get a wrong answer
    that still looks like a probability.
    """
    position, velocity = _circular_orbit_state()
    cov = rtn_covariance_from_isotropic(position, velocity, 0.35)
    r_rel = np.array([0.1, 0.828, -0.05])
    v_rel = np.array([0.4, -1.2, 10.0])

    baseline = foster_pc(r_rel, v_rel, cov, cov)

    axis = np.array([0.3, -0.7, 0.5])
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    for angle in (0.7, 1.234, 2.9):
        rotation = (np.eye(3) + math.sin(angle) * skew
                    + (1.0 - math.cos(angle)) * skew @ skew)
        rotated = foster_pc(
            rotation @ r_rel,
            rotation @ v_rel,
            rotation @ cov @ rotation.T,
            rotation @ cov @ rotation.T,
        )
        assert rotated == pytest.approx(baseline, rel=1.0e-12)


def test_along_track_anisotropy_changes_pc_for_an_along_track_miss():
    """The whole point of the module, as an assertion.

    Same total uncertainty, same 828 m miss — the real Starlink x Kuiper
    separation — but redistributed into the along-track axis, which is the
    axis the miss lies along. The circular model returns one number for both
    and so cannot see the difference at all; the refined model must.
    """
    position, velocity = _circular_orbit_state()
    sigma = DEFAULT_COMBINED_SIGMA_KM
    # Separation purely along-track, relative velocity across the orbit plane,
    # so the encounter plane contains the radial and along-track axes.
    r_rel = np.array([0.0, 0.828, 0.0])
    v_rel = np.array([0.0, 0.0, 10.0])

    isotropic = rtn_covariance_from_isotropic(position, velocity, sigma, along_track_ratio=1.0)
    anisotropic = rtn_covariance_from_isotropic(position, velocity, sigma,
                                                along_track_ratio=DEFAULT_ALONG_TRACK_RATIO)

    pc_isotropic = foster_pc(r_rel, v_rel, isotropic / 2.0, isotropic / 2.0)
    pc_anisotropic = foster_pc(r_rel, v_rel, anisotropic / 2.0, anisotropic / 2.0)

    # The isotropic case must still reproduce the project's existing number.
    assert pc_isotropic == pytest.approx(compute_pc(0.828, sigma), rel=1.0e-3)
    # Stretching the ellipse along the miss direction thins the transverse
    # axis less than it fattens the tail along the miss, so Pc rises here.
    assert pc_anisotropic > 5.0 * pc_isotropic


# --------------------------------------------------------------------------
# 7. Guards on the method's assumptions
# --------------------------------------------------------------------------

def test_rejects_relative_velocity_too_low_for_the_short_encounter_assumption():
    """A near-zero relative velocity is not a hard case; it is a different problem.

    The encounter plane is defined as the plane normal to the relative
    velocity, and the 3D-to-2D reduction assumes the objects sweep past each
    other fast enough that the covariance is static during the encounter. At
    very low relative speed neither holds. Returning a number there would be
    worse than failing, because the number would look ordinary.
    """
    cov = np.eye(3) * 0.25
    r_rel = np.array([0.0, 0.828, 0.0])
    with pytest.raises(ValueError, match="short-encounter"):
        foster_pc(r_rel, np.array([0.0, 0.0, MIN_RELATIVE_SPEED_KM_S / 10.0]), cov, cov)
    with pytest.raises(ValueError, match="short-encounter"):
        foster_pc(r_rel, np.zeros(3), cov, cov)


def test_rejects_non_positive_definite_covariance():
    r_rel = np.array([0.0, 0.828, 0.0])
    v_rel = np.array([0.0, 0.0, 10.0])
    with pytest.raises(ValueError, match="positive-definite"):
        foster_pc(r_rel, v_rel, np.zeros((3, 3)), np.zeros((3, 3)))

    # Rank-deficient: no uncertainty at all along one ECI axis.
    degenerate = np.diag([0.25, 0.25, 0.0])
    with pytest.raises(ValueError, match="positive-definite"):
        foster_pc(r_rel, np.array([0.0, 10.0, 0.0]), degenerate / 2.0, degenerate / 2.0)


def test_rejects_invalid_hbr_and_quadrature_order():
    cov = np.eye(3) * 0.25
    r_rel = np.array([0.0, 0.828, 0.0])
    v_rel = np.array([0.0, 0.0, 10.0])
    for bad_hbr in (0.0, -0.02, float("nan")):
        with pytest.raises(ValueError, match="hbr_km"):
            foster_pc(r_rel, v_rel, cov, cov, hbr_km=bad_hbr)
    with pytest.raises(ValueError, match="order"):
        foster_pc(r_rel, v_rel, cov, cov, order=1)


def test_rtn_covariance_rejects_degenerate_states_and_sigmas():
    position, velocity = _circular_orbit_state()
    with pytest.raises(ValueError, match="zero magnitude"):
        rtn_covariance(np.zeros(3), velocity, 0.1, 0.8, 0.1)
    with pytest.raises(ValueError, match="parallel"):
        rtn_covariance(position, position * 1.0e-3, 0.1, 0.8, 0.1)
    with pytest.raises(ValueError, match="sigma_along_km"):
        rtn_covariance(position, velocity, 0.1, 0.0, 0.1)
    with pytest.raises(ValueError, match="along_track_ratio"):
        rtn_covariance_from_isotropic(position, velocity, 0.5, along_track_ratio=-1.0)


def test_certain_collision_returns_exactly_one():
    """Enormous hard body, tiny covariance — the answer is 1, not 1.0000001.

    This is the containment branch: the entire error ellipse lies inside the
    hard-body circle, so the objects are known well enough that the collision
    is not in doubt. Handling it in closed form rather than by quadrature also
    removes the one case a fixed-order rule genuinely cannot integrate — a
    needle-thin Gaussian somewhere inside a vast disc.
    """
    cov = np.eye(3) * 1.0e-8
    pc = foster_pc(np.zeros(3), np.array([0.0, 0.0, 10.0]), cov, cov, hbr_km=5.0)
    assert pc == 1.0

    # Still exact when the ellipse is off-centre but comfortably inside.
    assert foster_pc_2d(1.0, 0.5, 0.002, 0.003, 2.0) == 1.0


def test_refuses_geometries_the_fixed_order_rule_cannot_resolve():
    """Past the envelope the module raises instead of returning a wrong number.

    A hard body hundreds of sigma across, not containing the error ellipse, is
    not a satellite conjunction — but it is the geometry where a fixed node
    budget silently stops resolving the integrand, and a silent wrong Pc is
    the worst possible failure here. The boundary is asserted from both sides
    so the envelope stays documented rather than incidental.
    """
    sigma = 0.001
    inside = MAX_HBR_SIGMA_RATIO * sigma * 0.9
    outside = MAX_HBR_SIGMA_RATIO * sigma * 1.1
    miss = 2.0 * MAX_HBR_SIGMA_RATIO * sigma  # far enough not to be contained

    ok = foster_pc_2d(miss, 0.0, sigma, sigma, inside)
    assert ok == pytest.approx(_exact_circular_pc(miss, sigma, inside), rel=1.0e-11)

    with pytest.raises(ValueError, match="cannot resolve"):
        foster_pc_2d(miss, 0.0, sigma, sigma, outside)


def test_probability_is_bounded_everywhere_it_returns():
    """No geometry in the sweep may produce a value outside [0, 1]."""
    for sigma in (0.002, 0.05, 0.5, 5.0):
        for miss in (0.0, 0.1, 1.0, 10.0):
            for hbr in (0.001, 0.02, 0.2):
                if hbr > MAX_HBR_SIGMA_RATIO * sigma:
                    continue
                assert 0.0 <= foster_pc_2d(miss, 0.0, sigma, sigma, hbr) <= 1.0
