"""Die sechs Formen gegen Handrechnung auf der Handinstanz; Aufbau, Rückumformung und LP-Nachbildung je Einheit (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kgf_algorithm as A

HAND_VALUES = {"cube": 1.0, "diamond": 2.0, "ball_soc": math.sqrt(2), "ball_psd": math.sqrt(2), "ellipse_axis": 4.0, "ellipse_psd": 2.5}


@pytest.mark.parametrize("form", A.FORMS)
def test_hand_instance_values(hand, form):
    r = A.solve_form(form, hand)
    assert r.result.value == pytest.approx(HAND_VALUES[form], abs=1e-6)
    assert r.cone_problems == [] and r.original_problems == []


def test_start_point_is_strictly_feasible_for_every_form(hand):
    for form in A.FORMS:
        p = A.build_form(form, hand)
        assert math.isfinite(K.barrier_value(p.cones, p.b - p.A @ p.x0)), form


def test_sizes_by_hand_for_four_points_in_the_plane(hand):
    """n = 4, d = 2: Würfel 2 n d = 16 Zeilen; Raute 16 + 4 = 20 Zeilen, 3 + 8 Variablen; Kugel 4 Kegel (3); als SDP 4 Kegel (3) = 24 Zeilen;
    Ellipse achsenparallel 4 + 2 * 3 = 10 Zeilen und 4 Variablen; allgemein 4 + 10 = 14 Zeilen und 6 Variablen."""
    sizes = {f: A.build_form(f, hand).sizes() for f in A.FORMS}
    assert (sizes["cube"]["variables"], sizes["cube"]["rows"]) == (3, 16)
    assert (sizes["diamond"]["variables"], sizes["diamond"]["rows"]) == (3 + 8, 20)
    assert sizes["ball_soc"]["soc"] == [3] * 4 and sizes["ball_soc"]["rows"] == 12
    assert sizes["ball_psd"]["psd"] == [3] * 4 and sizes["ball_psd"]["rows"] == 24
    assert (sizes["ellipse_axis"]["variables"], sizes["ellipse_axis"]["rows"], sizes["ellipse_axis"]["orthant"], sizes["ellipse_axis"]["soc"]) == (4, 10, 4, [3, 3])
    assert (sizes["ellipse_psd"]["variables"], sizes["ellipse_psd"]["rows"], sizes["ellipse_psd"]["psd"]) == (6, 14, [4])


def test_cube_rows_by_hand():
    """Ein Punkt (3, 5): Zeilen s = R - (x - c_j) und R + (x - c_j) je Koordinate; bei c = (3, 5), R = 1 sind alle s gleich 1."""
    p = A.build_form("cube", np.array([[3.0, 5.0]]))
    s = p.b - p.A @ np.array([3.0, 5.0, 1.0])
    assert s.tolist() == [1.0, 1.0, 1.0, 1.0]
    s2 = p.b - p.A @ np.array([2.0, 5.0, 1.0])                                    # c_1 = 2: R - (3 - 2) = 0 und R + (3 - 2) = 2
    assert sorted(s2[[0, 2]]) == [0.0, 2.0]


def test_ball_psd_matrix_has_the_schur_structure():
    """Punkt (3, 4), c = (0, 0), R = 5: [[5, 0, 3],[0, 5, 4],[3, 4, 5]] hat Determinante 0 (Rand), bei R = 6 positiv definit."""
    p = A.build_form("ball_psd", np.array([[3.0, 4.0]]))
    for R, definite in ((5.0, False), (6.0, True)):
        S = K.smat(p.b - p.A @ np.array([0.0, 0.0, R]), 3)
        assert S == pytest.approx(np.array([[R, 0, 3], [0, R, 4], [3, 4, R]]))
        eig = np.linalg.eigvalsh(S).min()
        assert (eig > 1e-9) == definite and (definite or abs(eig) < 1e-9)


def test_ellipse_axis_rotated_cone_means_p_times_t_at_least_one():
    p = A.build_form("ellipse_axis", np.array([[1.0, 0.0], [0.0, 1.0]]))
    ok = np.array([0.5, 0.5, 2.5, 2.5])                                            # p t = 1,25 >= 1
    bad = np.array([0.5, 0.5, 1.5, 1.5])                                           # p t = 0,75 < 1
    assert K.check_solution(p, ok) == [] and len(K.check_solution(p, bad)) == 2


def test_ellipse_psd_block_matrix_by_hand():
    """P = diag(1, 4), T = diag(1, 1/4) = P^-1: der Block [[P, I],[I, T]] ist am Rand (Determinante 0), mit T = 2 P^-1 definit."""
    p = A.build_form("ellipse_psd", np.array([[1.0, 0.0], [0.0, 0.5]]))
    def x_of(P, T):
        return np.concatenate([[P[0, 0], P[0, 1], P[1, 1]], [T[0, 0], T[0, 1], T[1, 1]]])
    P = np.diag([1.0, 4.0])
    s_edge = (p.b - p.A @ x_of(P, np.linalg.inv(P)))[2:]
    s_wide = (p.b - p.A @ x_of(P, 2 * np.linalg.inv(P)))[2:]
    assert np.linalg.eigvalsh(K.smat(s_edge, 4)).min() == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.eigvalsh(K.smat(s_wide, 4)).min() > 0.1
    assert p.c @ x_of(P, np.linalg.inv(P)) == pytest.approx(1.25)                  # trace(T) = 1 + 1/4


def test_decode_returns_the_shapes(hand):
    ball = A.decode("ball_soc", hand, np.array([0.1, -0.2, 3.0]))
    assert (ball.kind, ball.radius) == ("ball", 3.0) and ball.center.tolist() == [0.1, -0.2]
    assert A.decode("cube", hand, np.array([0, 0, 1.0])).kind == "box" and A.decode("diamond", hand, np.array([0, 0, 1.0] + [0.0] * 8)).kind == "diamond"
    assert A.decode("ellipse_axis", hand, np.array([0.5, 0.25, 0, 0])).matrix.tolist() == [[0.5, 0], [0, 0.25]]
    E = A.decode("ellipse_psd", hand, np.array([2.0, 0.5 * K.SQ2, 3.0, 0, 0, 0]))        # svec: Außerdiagonale mal sqrt 2
    assert E.matrix == pytest.approx(np.array([[2.0, 0.5], [0.5, 3.0]]))


def test_original_problem_check_names_each_violation(hand):
    good = np.array([0.0, 0.0, math.sqrt(2) + 1e-9])
    assert A.original_violations("ball_soc", hand, good, math.sqrt(2) + 1e-9) == []
    out = A.original_violations("ball_soc", hand, np.array([0.0, 0.0, 1.0]), 1.0)
    assert len(out) == 1 and "außerhalb" in out[0]
    assert "Radius und Zielwert" in A.original_violations("ball_soc", hand, good, 5.0)[0]
    # Würfel: Radius 1 reicht in l-unendlich, in l1 nicht
    assert A.original_violations("cube", hand, np.array([0.0, 0.0, 1.0]), 1.0) == []
    assert A.original_violations("diamond", hand, np.array([0.0, 0.0, 1.0] + [0.0] * 8), 1.0) != []
    # Ellipse: P = diag(1/2, 1/2) enthält alle vier Punkte (x^T P x = 1), trace(P^-1) = 4
    assert A.original_violations("ellipse_axis", hand, np.array([0.5, 0.5, 2.0, 2.0]), 4.0) == []
    assert "außerhalb der Ellipse" in A.original_violations("ellipse_axis", hand, np.array([0.6, 0.6, 2.5, 2.5]), 5.0)[0]
    assert "nicht positiv definit" in A.original_violations("ellipse_axis", hand, np.array([0.5, -0.5, 2.0, 2.0]), 4.0)[0]
    assert "Summe der quadrierten Halbachsen" in A.original_violations("ellipse_axis", hand, np.array([0.5, 0.5, 2.0, 2.0]), 3.0)[0]


def test_unknown_form_and_bad_points_are_rejected(hand):
    with pytest.raises(ValueError, match="unbekannte Form"):
        A.build_form("dreieck", hand)
    with pytest.raises(ValueError, match="nicht leere Matrix"):
        A.build_form("cube", np.zeros((0, 2)))
    with pytest.raises(ValueError, match="nicht leere Matrix"):
        A.build_form("cube", np.array([1.0, 2.0]))


def test_directions_in_the_plane_are_a_regular_polygon():
    u = A.directions(2, 8)
    assert np.linalg.norm(u, axis=1) == pytest.approx(1.0)
    ang = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    assert np.mod(np.diff(ang), 360.0) == pytest.approx(45.0)
    r = A.directions(5, 100, seed=1)
    assert r.shape == (100, 5) and np.linalg.norm(r, axis=1) == pytest.approx(1.0)
    assert np.array_equal(A.directions(5, 100, seed=1), r) and not np.array_equal(A.directions(5, 100, seed=2), r)
    with pytest.raises(ValueError, match="mindestens eine Richtung"):
        A.directions(2, 0)


def test_polyhedral_ball_by_hand():
    """Zwei Punkte (+-1, 0), vier Richtungen (Achsen): LP-Radius 1 (gleich dem Würfel); acht Richtungen auf dem Achsenkreuz plus Diagonalen: Radius 1."""
    pts = np.array([[1.0, 0.0], [-1.0, 0.0]])
    axes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert A.polyhedral_radius(pts, axes) == pytest.approx(1.0, abs=1e-6)
    assert A.polyhedral_radius(pts, A.directions(2, 64)) <= 1.0 + 1e-6


def test_polyhedral_radius_never_exceeds_the_ball_and_is_not_negative():
    rng = np.random.default_rng(1)
    for d in (2, 3):
        pts = rng.normal(size=(15, d))
        ball = A.solve_form("ball_soc", pts).result.value
        for k in (4, 16, 64):
            r = A.polyhedral_radius(pts, A.directions(d, k))
            assert -1e-9 <= r <= ball + 1e-7, (d, k)


def test_polyhedral_problem_has_the_radius_nonnegativity_row():
    pts = np.array([[1.0, 0.0], [-1.0, 0.0]])
    p = A.build_ball_polyhedral(pts, A.directions(2, 4))
    assert p.A.shape == (2 * 4 + 1, 3) and p.A[-1].tolist() == [0.0, 0.0, -1.0] and p.b[-1] == 0.0


def test_diamond_in_three_dimensions_ends_gracefully_when_rounding_stops_it():
    """Regression: die Raute (viele Hilfsvariablen, entartetes Optimum) machte die Hesse-Matrix bei t ~ 2e11 singulär und löste LinAlgError aus.
    Jetzt: kein Absturz, converged = False, die zuletzt bewiesene Lücke bleibt unter 1e-8, beide Prüfer bestehen."""
    pts = np.random.default_rng(1).normal(0, 1, (9, 3))
    pts -= pts.mean(axis=0)
    r = A.solve_form("diamond", pts)
    assert not r.result.converged and 1e-9 < r.result.gap_bound < 1e-6
    assert r.cone_problems == [] and r.original_problems == []
    assert abs(r.result.value - 3.4000665) < 1e-5
