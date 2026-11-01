"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): die sechs Formen und die LP-Nachbildung gegen einen anderen Rechenweg.

- cvxpy mit Clarabel (anderer Löser, anderes Verfahren: Innere Punkte mit Skalierung, andere Modellierung über Normen und Matrixvariable)
- geschlossene Formeln: Würfel = halbe größte Spannweite; Kugel in der Ebene = Kreis durch zwei oder drei Punkte
- scipy linprog (HiGHS, Simplex/Innere Punkte im LP) für Raute und LP-Nachbildung
- Eigenschaften, die auch ein Fehler in beiden Rechenwegen nicht überlebt (Kette der Normen, Skalierung, Drehinvarianz, Sonderfall)

Jeder Test läuft in unter 10 s. Fehlen cvxpy oder scipy im Test-venv, überspringen sich die jeweiligen Tests (pytest.importorskip), sie raten nie."""

import itertools
import math

import numpy as np
import pytest

import kgf_algorithm as A
from kgf_scenario import make_cloud, rotation


def _cloud(seed, n, d=2):
    pts = np.random.default_rng(seed).normal(0.0, 1.0, (n, d))
    return pts - pts.mean(axis=0)


def _value(form, pts):
    return A.solve_form(form, pts).result.value


def _circle_radius(pts):
    """Unabhängige exakte kleinste Kugel in der Ebene: Kreise durch zwei (Durchmesser) und drei Punkte (Umkreis), kleinster, der alles enthält."""
    best = math.inf
    for i, j in itertools.combinations(range(len(pts)), 2):
        c, r = (pts[i] + pts[j]) / 2, np.linalg.norm(pts[i] - pts[j]) / 2
        if r < best and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = r
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        a, b, c3 = pts[i], pts[j], pts[k]
        d = 2 * (a[0] * (b[1] - c3[1]) + b[0] * (c3[1] - a[1]) + c3[0] * (a[1] - b[1]))
        if abs(d) < 1e-12:
            continue
        ux = ((a @ a) * (b[1] - c3[1]) + (b @ b) * (c3[1] - a[1]) + (c3 @ c3) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c3[0] - b[0]) + (b @ b) * (a[0] - c3[0]) + (c3 @ c3) * (b[0] - a[0])) / d
        c = np.array([ux, uy])
        r = np.linalg.norm(a - c)
        if r < best and np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9):
            best = r
    return best


def test_ball_matches_the_circle_through_two_or_three_points():
    for seed in range(30):
        pts = _cloud(seed, 6 + seed % 8)
        for form in ("ball_soc", "ball_psd"):
            assert _value(form, pts) == pytest.approx(_circle_radius(pts), rel=1e-6), (form, seed)


def test_cube_matches_half_the_largest_extent():
    for seed in range(30):
        d = 2 + seed % 3
        pts = _cloud(seed, 8, d)
        assert _value("cube", pts) == pytest.approx(((pts.max(axis=0) - pts.min(axis=0)) / 2).max(), rel=1e-7), seed


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_cvxpy_agrees_with_all_six_forms():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    kw = dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    for seed in range(12):
        d = 2 + seed % 2
        pts = _cloud(seed, 8 + seed % 4, d)
        c, R = cp.Variable(d), cp.Variable()
        for form, norm in (("cube", "inf"), ("diamond", 1), ("ball_soc", 2), ("ball_psd", 2)):
            cp.Problem(cp.Minimize(R), [cp.norm(p - c, norm) <= R for p in pts]).solve(**kw)
            assert _value(form, pts) == pytest.approx(float(R.value), rel=1e-6), (form, seed)
        pv = cp.Variable(d, nonneg=True)
        cp.Problem(cp.Minimize(cp.sum(cp.inv_pos(pv))), [(pts ** 2) @ pv <= 1]).solve(**kw)
        assert _value("ellipse_axis", pts) == pytest.approx(float(np.sum(1 / pv.value)), rel=1e-5), seed
        M = cp.Variable((d, d), PSD=True)
        cp.Problem(cp.Minimize(cp.tr_inv(M)), [cp.sum(cp.multiply(np.outer(p, p), M)) <= 1 for p in pts]).solve(**kw)
        assert _value("ellipse_psd", pts) == pytest.approx(float(np.trace(np.linalg.inv(M.value))), rel=1e-5), seed


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_gap_bound_covers_the_distance_to_cvxpy():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    for seed in range(6):
        pts = _cloud(seed, 10)
        c, R = cp.Variable(2), cp.Variable()
        cp.Problem(cp.Minimize(R), [cp.norm(p - c, 2) <= R for p in pts]).solve(solver=cp.CLARABEL, tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
        for tol in (1e-3, 1e-6):
            res = A.solve_form("ball_soc", pts, tol=tol).result
            assert res.value - float(R.value) <= res.gap_bound + 1e-9 and res.value >= float(R.value) - 1e-9, (seed, tol)


def test_diamond_and_polyhedral_ball_match_scipy_linprog():
    linprog = pytest.importorskip("scipy.optimize").linprog
    for seed in range(10):
        d = 2 + seed % 2
        pts = _cloud(seed, 8, d)
        n = len(pts)
        # Raute: Variablen (c, R, t_ij), min R mit t_ij >= |x_ij - c_j|, R >= Summe_j t_ij
        nv = d + 1 + n * d
        Aub, bub = [], []
        for i in range(n):
            for j in range(d):
                for sign in (1.0, -1.0):
                    row = np.zeros(nv); row[d + 1 + i * d + j] = -1.0; row[j] = -sign
                    Aub.append(row); bub.append(-sign * pts[i, j])
            row = np.zeros(nv); row[d] = -1.0; row[d + 1 + i * d:d + 1 + (i + 1) * d] = 1.0
            Aub.append(row); bub.append(0.0)
        cvec = np.zeros(nv); cvec[d] = 1.0
        res = linprog(cvec, A_ub=np.array(Aub), b_ub=np.array(bub), bounds=[(None, None)] * (d + 1) + [(0, None)] * (n * d), method="highs")
        assert _value("diamond", pts) == pytest.approx(res.fun, rel=1e-6), seed
        # LP-Nachbildung der Kugel mit k Richtungen
        dirs = A.directions(d, 24, seed)
        Au = np.hstack([-np.tile(dirs, (n, 1)), -np.ones((n * len(dirs), 1))])
        bu = -(dirs @ pts.T).T.reshape(-1)
        res2 = linprog(np.eye(d + 1)[d], A_ub=Au, b_ub=bu, bounds=[(None, None)] * d + [(0, None)], method="highs")
        assert A.polyhedral_radius(pts, dirs) == pytest.approx(res2.fun, rel=1e-6, abs=1e-8), seed


def test_chain_of_norms_gives_cube_below_ball_below_diamond():
    for seed in range(20):
        pts = _cloud(seed, 10, 2 + seed % 2)
        cube, ball, diamond = _value("cube", pts), _value("ball_soc", pts), _value("diamond", pts)
        assert cube <= ball + 1e-8 and ball <= diamond + 1e-8, seed


def test_general_ellipse_is_never_worse_than_the_axis_parallel_one_and_ball_forms_agree():
    for seed in range(20):
        pts = _cloud(seed, 12, 2 + seed % 2)
        assert _value("ellipse_psd", pts) <= _value("ellipse_axis", pts) * (1 + 1e-7), seed
        assert _value("ball_soc", pts) == pytest.approx(_value("ball_psd", pts), rel=1e-7), seed


def test_scaling_laws():
    pts = _cloud(2, 10)
    for form in A.FORMS:
        exponent = 2 if form in A.ELLIPSES else 1
        assert _value(form, 3.0 * pts) == pytest.approx(3.0 ** exponent * _value(form, pts), rel=1e-6), form


def test_order_of_the_points_does_not_matter():
    pts = _cloud(4, 10)
    perm = np.random.default_rng(1).permutation(len(pts))
    for form in A.FORMS:
        assert _value(form, pts[perm]) == pytest.approx(_value(form, pts), rel=1e-6), form


def test_rotation_leaves_ball_and_general_ellipse_unchanged_but_not_the_axis_parallel_one():
    pts = make_cloud(14, 4.0, 0, 7).points
    turned = pts @ rotation(45).T
    for form in ("ball_soc", "ellipse_psd"):
        assert _value(form, turned) == pytest.approx(_value(form, pts), rel=1e-5), form
    assert _value("ellipse_axis", turned) > 1.2 * _value("ellipse_axis", pts)


def test_one_dimensional_clouds_agree_across_forms():
    """d = 1: Würfel, Raute und Kugel sind dasselbe (halbe Spannweite); die Ellipse hat eine Halbachse, Wert = größter Abstand zum Ursprung hoch 2."""
    pts = _cloud(3, 9, 1)
    half = float((pts.max() - pts.min()) / 2)
    for form in ("cube", "diamond", "ball_soc", "ball_psd"):
        assert _value(form, pts) == pytest.approx(half, rel=1e-6), form
    assert _value("ellipse_axis", pts) == pytest.approx(float(np.abs(pts).max() ** 2), rel=1e-6)
    assert _value("ellipse_psd", pts) == pytest.approx(float(np.abs(pts).max() ** 2), rel=1e-6)
