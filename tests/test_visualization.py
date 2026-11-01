"""Abbildungen: Achsen gesperrt, Umrisse stimmen mit den Formen überein, Diagramme bauen an den Rändern."""

import numpy as np
import pytest

import kgf_algorithm as A
import kgf_evaluation as ev
import kgf_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


@pytest.fixture(scope="module")
def a():
    return ev.analyse(ev.Settings(n=12))


def test_outlines_by_hand():
    s = A.Shape("box", np.array([1.0, 2.0]), 3.0)
    x, y = V.outline(s)
    assert (x.min(), x.max(), y.min(), y.max()) == (-2.0, 4.0, -1.0, 5.0)
    x, y = V.outline(A.Shape("diamond", np.array([0.0, 0.0]), 2.0))
    assert sorted(set(zip(x.round(6), y.round(6)))) == [(-2.0, 0.0), (0.0, -2.0), (0.0, 2.0), (2.0, 0.0)]
    x, y = V.outline(A.Shape("ball", np.array([1.0, 1.0]), 2.0))
    assert np.hypot(x - 1.0, y - 1.0) == pytest.approx(2.0)
    # Ellipse x^T P x = 1 mit P = diag(1, 4): Halbachsen 1 und 1/2
    x, y = V.outline(A.Shape("ellipse", np.zeros(2), matrix=np.diag([1.0, 4.0])))
    assert (abs(x).max(), abs(y).max()) == pytest.approx((1.0, 0.5), abs=1e-3) and np.allclose(x ** 2 + 4 * y ** 2, 1.0)


def test_rotated_ellipse_outline_lies_on_x_t_p_x_equals_one():
    P = np.array([[2.0, 0.6], [0.6, 1.0]])
    x, y = V.outline(A.Shape("ellipse", np.zeros(2), matrix=P))
    pts = np.stack([x, y], axis=1)
    assert np.einsum("ij,jk,ik->i", pts, P, pts) == pytest.approx(1.0)


def test_shape_chart_has_all_points_and_is_locked(a):
    for form in A.FORMS:
        fig = V.build_shape(a.cloud.points, form, a.results[form].shape)
        assert _locked(fig) and fig.layout.yaxis.scaleanchor == "x"
        assert [t.name for t in fig.data][:2] == [A.FORM_NAMES[form], "Punkte"]
        assert len(fig.data[1].x) == 12


def test_forms_gap_lp_psd_and_rotation_charts_build(a):
    rows = ev.form_rows(a)
    bars = V.build_forms(rows)
    assert _locked(bars) and list(bars.data[0].y) == [r["rows"] for r in rows]
    gap = V.build_gap(a.selected.result)
    assert _locked(gap) and gap.layout.yaxis.type == "log" and len(gap.data[0].x) == a.selected.result.rounds
    lp = V.build_lp(ev.lp_experiment(dims=(2,), ks=(4, 8, 16), n=8, seeds=(1,)))
    assert _locked(lp) and lp.layout.xaxis.type == "log" and lp.data[-1].name == "1 %"
    psd = V.build_psd(ev.psd_experiment(sizes=(6, 8), seeds=(1,)))
    assert _locked(psd) and [t.name for t in psd.data] == ["Lorentz-Kegel", "PSD-Kegel"]
    rot = V.build_rotation(ev.rotation_experiment(angles=(0, 45), n=8, seeds=(1, 2)))
    assert _locked(rot) and rot.data[-1].name == "gleich groß"
