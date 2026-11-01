"""Gemeinsamer Kern `kegel_kern.py` (wortgleich in allen Repos der Reihe): jede Einheit gegen Handrechnung, finite Differenzen oder ein Mini-Programm mit bekanntem Optimum."""

import numpy as np
import pytest

import kegel_kern as K


def _num_grad(f, s, h=1e-6):
    g = np.zeros(len(s))
    for i in range(len(s)):
        e = np.zeros(len(s))
        e[i] = h
        g[i] = (f(s + e) - f(s - e)) / (2 * h)
    return g


def _interior_point(kind, m, rng):
    if kind == "l":
        return rng.uniform(0.5, 2.0, m)
    if kind == "q":
        rest = rng.normal(size=m - 1)
        return np.concatenate([[np.linalg.norm(rest) + rng.uniform(0.5, 1.5)], rest])
    S = rng.normal(size=(m, m))
    return K.svec(S @ S.T + 0.5 * np.eye(m))


@pytest.mark.parametrize("kind,m", [("l", 4), ("q", 4), ("q", 3), ("s", 2), ("s", 3)])
def test_barrier_gradient_and_hessian_match_finite_differences(kind, m):
    rng = np.random.default_rng(3)
    d = K.cone_dim(kind, m)
    s = _interior_point(kind, m, rng)
    f = lambda v: K.barrier_value(((kind, m),), v)
    g, H = K.block_grad_hess(kind, s, m)
    assert g == pytest.approx(_num_grad(f, s), rel=1e-5, abs=1e-7)
    num_h = np.stack([_num_grad(lambda v: K.block_grad_hess(kind, v, m)[0][j], s) for j in range(d)])
    assert H == pytest.approx(num_h, rel=1e-4, abs=1e-6)
    assert np.allclose(H, H.T) and np.linalg.eigvalsh(H).min() > 0


def test_grouped_barrier_equals_the_sum_of_single_blocks():
    rng = np.random.default_rng(4)
    cones = (("q", 3),) * 3 + (("l", 2),) + (("s", 2),) * 2
    s = np.concatenate([_interior_point(k, m, rng) for k, m in cones])
    singles, off = 0.0, 0
    for k, m in cones:
        d = K.cone_dim(k, m)
        singles += K.barrier_value(((k, m),), s[off:off + d])
        off += d
    assert K.barrier_value(cones, s) == pytest.approx(singles, rel=1e-12)


def test_groups_merge_only_consecutive_identical_cones():
    assert K.groups((("l", 5), ("q", 3), ("q", 3), ("s", 2), ("q", 3))) == [("l", 5, 1, 0), ("q", 3, 2, 5), ("s", 2, 1, 11), ("q", 3, 1, 14)]


def test_newton_system_matches_finite_differences_of_f_t(tiny_cones):
    prob, _ = tiny_cones["sdp"]
    x, t = np.array([1.7]), 3.0
    f = lambda v: t * prob.c @ v + K.barrier_value(prob.cones, prob.b - prob.A @ v)
    grad, hess = K.newton_system(prob, x, t)
    assert grad[0] == pytest.approx(_num_grad(f, x)[0], rel=1e-5)
    assert hess[0, 0] == pytest.approx((f(x + 1e-4) - 2 * f(x) + f(x - 1e-4)) / 1e-8, rel=1e-3)


def test_svec_is_an_isometry_and_smat_inverts_it():
    rng = np.random.default_rng(5)
    for k in (1, 2, 3, 5):
        S = rng.normal(size=(k, k)); S = S + S.T
        T = rng.normal(size=(k, k)); T = T + T.T
        assert K.svec(S) @ K.svec(T) == pytest.approx(float(np.sum(S * T)), rel=1e-12)
        assert K.smat(K.svec(S), k) == pytest.approx(S, abs=1e-12)
        assert len(K.svec(S)) == K.cone_dim("s", k)


def test_dimensions_and_barrier_parameter_by_hand():
    assert [K.cone_dim("l", 5), K.cone_dim("q", 4), K.cone_dim("s", 3)] == [5, 4, 6]
    assert K.barrier_parameter((("l", 5), ("q", 4), ("q", 3), ("s", 3))) == 5 + 2 + 2 + 3
    with pytest.raises(ValueError, match="unbekannter Kegel"):
        K.cone_dim("x", 2)


def test_interior_and_margin_at_the_cone_boundary():
    assert K.in_interior("l", np.array([1.0, 0.1]), 2) and not K.in_interior("l", np.array([1.0, 0.0]), 2)
    assert K.in_interior("q", np.array([2.0, 1.0, 1.0]), 3)
    assert not K.in_interior("q", np.array([5.0, 3.0, 4.0]), 3)                # genau auf dem Rand (25 - 9 - 16 = 0): kein Inneres
    assert not K.in_interior("q", np.array([1.0, 1.0, 1.0]), 3)                 # s0 < Norm des Rests
    assert not K.in_interior("q", np.array([-3.0, 1.0, 0.0]), 3)                # negatives s0
    assert K.in_interior("s", K.svec(np.eye(2)), 2) and not K.in_interior("s", K.svec(np.array([[1.0, 1.0], [1.0, 1.0]])), 2)
    assert K.cone_margin("l", np.array([3.0, 1.0]), 2) == pytest.approx(1.0)
    assert K.cone_margin("q", np.array([5.0, 3.0, 4.0]), 3) == pytest.approx(0.0)
    assert K.cone_margin("q", np.array([4.0, 3.0, 4.0]), 3) == pytest.approx(-1.0)
    assert K.cone_margin("s", K.svec(np.array([[2.0, 0.0], [0.0, 1.0]])), 2) == pytest.approx(1.0)


def test_mini_programs_have_their_known_optimum_and_the_gap_bound_holds(tiny_cones):
    for name, (prob, opt) in tiny_cones.items():
        res = K.solve_barrier(prob, tol=1e-9)
        assert res.value >= opt - 1e-12, name                                 # nie unter dem Optimum (zulässig)
        assert res.value - opt <= res.gap_bound + 1e-12, name                  # die Lückenschranke stimmt
        assert res.value == pytest.approx(opt, abs=1e-8), name
        assert K.check_solution(prob, res.x) == [], name


def test_gap_bound_falls_by_the_factor_mu_per_round_and_steps_accumulate(tiny_cones):
    prob, _ = tiny_cones["soc"]
    res = K.solve_barrier(prob, tol=1e-6, mu=8.0)
    ts = [h[0] for h in res.history]
    assert all(b / a == pytest.approx(8.0) for a, b in zip(ts, ts[1:]))
    steps = [h[1] for h in res.history]
    assert steps == sorted(steps) and res.newton_steps == steps[-1] and res.rounds == len(res.history)
    assert res.gap_bound == pytest.approx(K.barrier_parameter(prob.cones) / ts[-1])


def test_a_start_outside_the_cone_is_rejected(tiny_cones):
    prob, _ = tiny_cones["lp"]
    bad = K.ConeProblem(prob.c, prob.A, prob.b, prob.cones, np.array([0.5]))      # s = 0.5 - 1 < 0
    with pytest.raises(ValueError, match="Start nicht streng zulässig"):
        K.solve_barrier(bad)


def test_initial_t_makes_the_start_nearly_central(tiny_cones):
    prob, _ = tiny_cones["lp"]
    t = K.initial_t(prob)
    grad, _ = K.newton_system(prob, prob.x0, t)
    assert t > 0 and abs(grad[0]) < 1e-12                                        # eine Variable: der Gradient lässt sich auf null stellen


def test_check_solution_names_the_broken_cone(tiny_cones):
    prob, _ = tiny_cones["soc"]
    assert K.check_solution(prob, np.array([5.0])) == []                          # Rand: s = (5; 3, 4)
    problems = K.check_solution(prob, np.array([4.0]))
    assert len(problems) == 1 and problems[0].startswith("Lorentz-Kegel 1 verletzt um 1.00e+00")
    assert K.check_solution(prob, np.array([4.0]), tol=2.0) == []                 # Toleranz greift


def test_stacked_psd_hessian_equals_the_reference_loop():
    """Gebündelte Hesse-Matrix gegen die einfache Schleife über die Basismatrizen (S^-1 E_j S^-1 in svec)."""
    rng = np.random.default_rng(6)
    k = 3
    S = rng.normal(size=(k, k)); S = S @ S.T + np.eye(k)
    Si = np.linalg.inv(S)
    d = K.cone_dim("s", k)
    ref = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = 1.0
        ref[:, j] = K.svec(Si @ K.smat(e, k) @ Si)
    _, H = K.block_grad_hess("s", K.svec(S), k)
    assert H == pytest.approx(ref, abs=1e-12)


def test_tolerance_below_rounding_is_clamped(tiny_cones):
    prob, opt = tiny_cones["lp"]
    res = K.solve_barrier(prob, tol=1e-30)
    assert K.MIN_TOL / 8.0 <= res.gap_bound <= K.MIN_TOL          # die letzte Runde fällt um den Faktor mu = 8 unter die Grenze
    assert res.converged and abs(res.value - opt) <= 1e-12


def test_singular_hessian_ends_the_run_with_the_last_proven_gap(tiny_cones, monkeypatch):
    """Simuliert eine numerisch singuläre Hesse-Matrix bei großem t (echter Fall: Raute in d = 3, siehe test_algorithm): kein Absturz, converged = False,
    die zuletzt zentrierte Runde ist das Ergebnis, und ihre Lücke nu / t gilt."""
    prob, opt = tiny_cones["soc"]
    real = K.newton_step

    def failing(grad, hess):
        if hess[0, 0] > 1e3:                  # große Krümmung heißt großes t
            raise np.linalg.LinAlgError("Singular matrix")
        return real(grad, hess)

    monkeypatch.setattr(K, "newton_step", failing)
    res = K.solve_barrier(prob, tol=1e-9)
    assert not res.converged and res.gap_bound > 1e-9
    assert res.value - opt <= res.gap_bound and res.value >= opt - 1e-12
    assert res.history[-1][2] == res.gap_bound and res.rounds == len(res.history)


def test_failure_in_the_very_first_round_raises(tiny_cones, monkeypatch):
    prob, _ = tiny_cones["lp"]
    monkeypatch.setattr(K, "newton_step", lambda g, h: (_ for _ in ()).throw(np.linalg.LinAlgError("Singular matrix")))
    with pytest.raises(np.linalg.LinAlgError, match="erste Runde"):
        K.solve_barrier(prob)


def test_non_finite_systems_are_rejected():
    with pytest.raises(np.linalg.LinAlgError, match="nicht endlich"):
        K.newton_step(np.array([np.nan]), np.eye(1))
