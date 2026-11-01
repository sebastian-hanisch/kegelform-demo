"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kegel_kern as K
import kgf_algorithm as A
from kgf_scenario import make_cloud, make_gaussian


def test_two_points_give_the_half_distance_in_every_ball_form():
    pts = np.array([[0.0, 0.0], [6.0, 8.0]])
    for form, expected in (("ball_soc", 5.0), ("ball_psd", 5.0), ("cube", 4.0), ("diamond", 7.0)):
        assert A.solve_form(form, pts).result.value == pytest.approx(expected, abs=1e-5), form       # die Raute endet je nach CPU bei Lücke 1e-9 oder 1,4e-6


def test_single_point_gives_radius_zero_up_to_the_gap():
    """n = 1: das Optimum R = 0 liegt auf dem Rand, die Barrierenmethode nähert sich ihm bis auf die bewiesene Lücke."""
    for form in ("cube", "diamond", "ball_soc", "ball_psd"):
        r = A.solve_form(form, np.array([[3.0, -2.0]]))
        assert 0.0 <= r.result.value <= r.result.gap_bound + 1e-9, form
        assert r.result.value <= 1e-7 and r.original_problems == []


def test_identical_points_behave_like_a_single_point():
    pts = np.array([[1.0, 1.0]] * 5)
    for form in ("cube", "ball_soc", "ball_psd"):
        r = A.solve_form(form, pts)
        assert r.result.value <= 1e-7 and r.cone_problems == [], form


def test_collinear_points_use_the_two_outer_ones():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert A.solve_form("ball_soc", pts).result.value == pytest.approx(3 * 2 ** 0.5 / 2, abs=1e-6)


def test_one_dimensional_clouds_work_in_every_form():
    pts = np.array([[-2.0], [1.0], [3.0]])
    pts = pts - pts.mean()
    for form in A.FORMS:
        r = A.solve_form(form, pts)
        assert r.cone_problems == [] and r.original_problems == [], form
    assert A.solve_form("cube", pts).result.value == pytest.approx(2.5, abs=1e-6)


def test_three_dimensions_work_in_every_form():
    pts = make_gaussian(10, 3, 5)
    pts = pts - pts.mean(axis=0)
    for form in A.FORMS:
        r = A.solve_form(form, pts)
        assert r.cone_problems == [] and r.original_problems == [] and r.result.gap_bound <= 1e-6, form


def test_a_point_at_the_origin_does_not_break_the_ellipse_rows():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    for form in A.ELLIPSES:
        r = A.solve_form(form, pts)
        assert r.cone_problems == [] and r.original_problems == [], form


def test_cloud_inputs_are_validated():
    with pytest.raises(ValueError, match="mindestens zwei Punkte"):
        make_cloud(1, 2.0, 0, 1)
    with pytest.raises(ValueError, match="Streckung muss mindestens 1 sein"):
        make_cloud(5, 0.5, 0, 1)
    c = make_cloud(12, 3.0, 30, 7)
    assert c.n == 12 and c.points.mean(axis=0) == pytest.approx(0.0, abs=1e-12)
    assert np.array_equal(c.points, make_cloud(12, 3.0, 30, 7).points) and not np.array_equal(c.points, make_cloud(12, 3.0, 30, 8).points)


def test_stretch_one_would_be_round_but_the_minimum_slider_value_still_responds_to_the_angle():
    from kgf_constants import STRETCH_MIN

    assert STRETCH_MIN > 1.0
    a, b = make_cloud(10, STRETCH_MIN, 0, 3).points, make_cloud(10, STRETCH_MIN, 45, 3).points
    assert not np.allclose(a, b)


def test_the_solver_rejects_a_start_outside_the_cone():
    p = A.build_form("ball_soc", np.array([[0.0, 0.0], [2.0, 0.0]]))
    bad = K.ConeProblem(p.c, p.A, p.b, p.cones, np.array([1.0, 0.0, 0.5]))        # R = 0,5 < Abstand 1
    with pytest.raises(ValueError, match="Start nicht streng zulässig"):
        K.solve_barrier(bad)
