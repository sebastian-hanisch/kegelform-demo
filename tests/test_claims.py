"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Wolken (SWEEP_SEEDS), nie über einen einzelnen Seed als Befund; die Standardwolke der Primäransicht (Seed 35) ist ausdrücklich eine einzelne.
Bänder statt Punktwerte, wo ein Zufallsstrom oder eine Bibliotheksversion die letzte Stelle bewegt (CI installiert ungepinnt, Linux rundet anders als Windows)."""

import math
from statistics import median

import numpy as np
import pytest

import kgf_algorithm as A
import kgf_constants as C
import kgf_evaluation as ev
from kgf_scenario import HAND_POINTS, make_cloud


@pytest.fixture(scope="module")
def std():
    return ev.analyse(ev.Settings())


@pytest.fixture(scope="module")
def planar_runs():
    """Alle sechs Formen auf 15 festen Wolken (5 Seeds, 3 Zuschnitte)."""
    runs = {f: [] for f in A.FORMS}
    for seed in C.SWEEP_SEEDS:
        for n, stretch, angle in ((20, 4.0, 45), (12, 2.0, 10), (30, 5.0, 80)):
            pts = make_cloud(n, stretch, angle, seed).points
            for f in A.FORMS:
                runs[f].append(A.solve_form(f, pts))
    return runs


def test_hand_instance_values_and_the_ratio():
    v = {f: A.solve_form(f, HAND_POINTS).result.value for f in A.FORMS}
    assert (v["cube"], v["diamond"], v["ball_soc"], v["ball_psd"], v["ellipse_axis"], v["ellipse_psd"]) == pytest.approx((1.0, 2.0, math.sqrt(2), math.sqrt(2), 4.0, 2.5), abs=1e-6)
    assert v["ellipse_psd"] / v["ellipse_axis"] == pytest.approx(0.625, abs=1e-6)
    assert ev.verdict_hand(v)[0] == "success"


def test_standard_cloud_values_and_sizes(std):
    rows = {r["form"]: r for r in ev.form_rows(std)}
    assert {f: round(r["value"], 3) for f, r in rows.items()} == pytest.approx({"cube": 5.784, "diamond": 10.090, "ball_soc": 7.168, "ball_psd": 7.168, "ellipse_axis": 134.265, "ellipse_psd": 78.517}, abs=0.002)
    assert {f: (r["variables"], r["rows"]) for f, r in rows.items()} == {"cube": (3, 80), "diamond": (43, 100), "ball_soc": (3, 60), "ball_psd": (3, 120), "ellipse_axis": (4, 26), "ellipse_psd": (6, 30)}
    assert std.ellipse_ratio == pytest.approx(0.585, abs=0.002) and all(r["ok"] for r in rows.values())
    assert rows["ball_soc"]["cones"] == "20 × Lorentz-Kegel (3)" and rows["ellipse_psd"]["cones"] == "Orthant (20) + PSD-Kegel (4)"


def test_every_planar_solve_converges_to_the_requested_gap_and_passes_both_checks(planar_runs):
    """README: alle 90 Läufe (6 Formen, 15 Wolken) erreichen die Lücke 1e-9 und bestehen die Rückumformung."""
    runs = [r for rs in planar_runs.values() for r in rs]
    assert len(runs) == 90 and all(r.result.converged for r in runs)
    assert not any(r.result.far_from_path for r in runs)                 # README: der Kern meldet in diesen 90 Läufen nie „weit vom Pfad“
    assert max(r.result.gap_bound for r in runs) <= 1.0e-9 * 1.01
    assert all(not r.cone_problems and not r.original_problems for r in runs)


def test_newton_steps_stay_in_the_documented_band(planar_runs):
    """README: 29 bis 84 Newton-Schritte je Lauf, Median je Form 31 bis 57 (diese Implementierung, nicht eine Eigenschaft der Kegel)."""
    steps = {f: [r.result.newton_steps for r in rs] for f, rs in planar_runs.items()}
    assert min(min(v) for v in steps.values()) >= 25 and max(max(v) for v in steps.values()) <= 100
    meds = {f: median(v) for f, v in steps.items()}
    assert 28 <= min(meds.values()) and max(meds.values()) <= 65
    assert meds["ellipse_psd"] > meds["ball_soc"] and abs(meds["ball_soc"] - meds["ball_psd"]) <= 3


def test_ball_forms_agree_and_the_norm_chain_holds_on_every_planar_cloud(planar_runs):
    for i in range(15):
        cube, diamond = planar_runs["cube"][i].result.value, planar_runs["diamond"][i].result.value
        soc, psd = planar_runs["ball_soc"][i].result.value, planar_runs["ball_psd"][i].result.value
        assert abs(soc - psd) <= 1e-8 * soc and cube <= soc <= diamond
        assert planar_runs["ellipse_psd"][i].result.value <= planar_runs["ellipse_axis"][i].result.value * (1 + 1e-8)


def test_psd_form_doubles_the_rows_and_needs_about_the_same_steps():
    """README: Kugel als PSD-Kegel hat bei jeder Punktzahl genau doppelt so viele Zeilen wie im Lorentz-Kegel (je Punkt 6 statt 3), gleicher Radius, Newton-Schritte höchstens 3 auseinander."""
    rows = ev.psd_experiment()
    assert [r["n"] for r in rows] == list(C.PSD_SIZES)
    for r in rows:
        assert r["rows_psd"] == 2 * r["rows_soc"] == 6 * r["n"] and r["max_dev"] < 1e-9
        assert abs(r["newton_soc"] - r["newton_psd"]) <= 3 and 28 <= r["newton_soc"] <= 36


def test_lp_replica_needs_16_256_and_2048_directions_for_one_percent():
    """README: für 1 % Fehler beim Radius braucht die LP-Nachbildung 16 (d = 2, regelmäßiges Vieleck), 256 (d = 3) und 2048 Richtungen (d = 4); die Fehler fallen mit k."""
    rows = {r["d"]: r for r in ev.lp_experiment()}
    assert [rows[d]["first"] for d in (2, 3, 4)] == [16, 256, 2048]
    for d in (2, 3, 4):
        e = rows[d]["errors"]
        assert e[-1] < e[2] and all(0.0 <= x <= 1.0 + 1e-9 for x in e)
    assert rows[2]["errors"][-1] < 1e-6 and rows[3]["errors"][-1] == pytest.approx(6.7e-4, abs=3e-4) and rows[4]["errors"][-1] == pytest.approx(6.7e-3, abs=3e-3)
    assert rows[3]["errors"][0] == pytest.approx(1.0, abs=1e-6)          # vier Richtungen im Raum: R fällt auf null (Fehler 100 %)


def test_rotation_ratio_by_angle_and_never_worse():
    """README: Verhältnis allgemein / achsenparallel (Wolke 4 : 1 gestreckt, 20 Punkte): 0,88 / 0,83 / 0,62 / 0,53 / 0,56 / 0,66 / 0,88 bei 0 bis 90 Grad in 15-Grad-Schritten; nie über 1."""
    rows = ev.rotation_experiment()
    assert [r["median"] for r in rows] == pytest.approx([0.875, 0.831, 0.622, 0.532, 0.562, 0.664, 0.875], abs=0.01)
    assert all(r["max"] < 1.0 for r in rows) and rows[0]["median"] == pytest.approx(rows[-1]["median"], abs=1e-6)       # 0 und 90 Grad sind dieselbe Wolke vertauscht
    assert min(r["min"] for r in rows) == pytest.approx(0.523, abs=0.01)


def test_only_the_diamond_ever_stops_at_the_rounding_limit():
    """README: in 240 Läufen (40 Wolken mit n = 8 bis 13 und d = 2 bis 4, sechs Formen) endet die Raute höchstens 14-mal (auf dem Entwicklungsrechner neunmal, auf GitHub-Runnern auch dreimal) vor der Lücke 1e-9 an der Rechengenauigkeit (bewiesene Lücke dann höchstens 1,5e-7), alle anderen Läufe erreichen sie."""
    stopped, worst = [], 0.0
    for seed in range(40):
        d, n = 2 + seed % 3, 8 + seed % 6
        pts = np.random.default_rng(seed).normal(0, 1, (n, d))
        pts -= pts.mean(axis=0)
        for f in A.FORMS:
            r = A.solve_form(f, pts)
            assert not r.cone_problems and not r.original_problems, (seed, f)
            if not r.result.converged:
                stopped.append(f)
                worst = max(worst, r.result.gap_bound)
    assert set(stopped) <= {"diamond"} and len(stopped) <= 14             # wie viele es sind, hängt von der CPU ab (Windows-Rechner: 9, GitHub-Runner: 3 und mehr)
    assert worst < 2e-7
