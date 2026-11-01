"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import math

import pytest

import kgf_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.pct(0.0051) == "0,51 %" and ev.pct(0.532) == "53,2 %" and ev.pct(1.0) == "100,0 %"
    assert ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(0.0) == "0,0" and ev.sci(9.96e-3) == "1,0·10⁻²"
    assert ev.sci(float("inf")) == "unendlich"
    sizes = {"orthant": 20, "soc": [3, 3, 3], "psd": [4]}
    assert ev.cones_text(sizes) == "Orthant (20) + 3 × Lorentz-Kegel (3) + PSD-Kegel (4)"
    assert ev.cones_text({"orthant": 0, "soc": [3], "psd": []}) == "Lorentz-Kegel (3)"
    assert ev.cones_text({"orthant": 80, "soc": [], "psd": []}) == "Orthant (80)"


def _row(form="ball_soc", ok=True, value=7.1675, rows=60, variables=3, cones="20 × Lorentz-Kegel (3)", gap=1.7e-10, name="Kugel (Lorentz-Kegel)", objective="Radius"):
    return {"form": form, "name": name, "objective": objective, "value": value, "variables": variables, "rows": rows, "cones": cones, "newton": 34, "gap": gap, "ok": ok}


def test_form_verdict_names_size_value_gap_and_the_back_check():
    kind, text = ev.verdict_form(_row(), 0.585)
    assert kind == "success"
    for part in ("einem Kegelprogramm mit 3 Variablen und 60 Zeilen", "(20 × Lorentz-Kegel (3))", "Radius = **7,1675**", "höchstens 1,7·10⁻¹⁰", "Die Rückumformung besteht"):
        assert part in text, part
    assert "allgemeine Ellipse" not in text


def test_form_verdict_for_ellipses_names_the_ratio_and_a_failed_check_is_an_error():
    kind, text = ev.verdict_form(_row("ellipse_psd", name="Ellipse, allgemein", objective="Summe der quadrierten Halbachsen"), 0.585)
    assert kind == "success" and "braucht die allgemeine Ellipse 58,5 % der Summe der achsenparallelen" in text
    kind, text = ev.verdict_form(_row(ok=False), 1.0)
    assert kind == "error" and "besteht nicht" in text and "Die Rückumformung besteht (" not in text


def test_forms_verdict_compares_ball_forms_and_orders_the_norms():
    rows = [_row("cube", rows=80, value=5.784, name="Würfel (l-unendlich)"), _row("diamond", rows=100, value=10.0895, name="Raute (l1)"),
            _row("ball_soc", rows=60, value=7.1675), _row("ball_psd", rows=120, value=7.1675, name="Kugel (PSD-Kegel)"),
            _row("ellipse_axis", rows=26, value=134.0, name="Ellipse, achsenparallel"), _row("ellipse_psd", rows=30, value=78.5, name="Ellipse, allgemein")]
    kind, text = ev.verdict_forms(rows)
    assert kind == "info"
    for part in ("von 26 bis 120 Zeilen", "**denselben Radius** (7,1675)", "braucht 120 statt 60 Zeilen", "Der Würfel (5,784)", "die Raute (10,089)"):
        assert part in text, part


def test_lp_verdict_names_each_dimension_and_flags_the_missing_ones():
    rows = [{"d": 2, "ks": [4, 16, 64], "errors": [0.1, 0.005, 0.001], "first": 16},
            {"d": 3, "ks": [4, 16, 64], "errors": [1.0, 0.2, 0.05], "first": None}]
    kind, text = ev.verdict_lp(rows)
    assert kind == "warning"
    for part in ("höchstens 1,0 %", "d = 2: 16 Richtungen", "d = 3: mehr als 64 Richtungen (Fehler 5,0 % bei 64)", "Ein einziger Lorentz-Kegel je Punkt ersetzt das exakt"):
        assert part in text, part


def _psd_rows():
    return [{"n": 6, "rows_soc": 18, "rows_psd": 36, "newton_soc": 28.0, "newton_psd": 29.0, "max_dev": 1e-11, "seconds_soc": 0.01, "seconds_psd": 0.03},
            {"n": 30, "rows_soc": 90, "rows_psd": 180, "newton_soc": 31.0, "newton_psd": 33.0, "max_dev": 1.8e-11, "seconds_soc": 0.02, "seconds_psd": 0.1}]


def test_psd_verdict_names_deviation_rows_and_steps():
    kind, text = ev.verdict_psd(_psd_rows())
    assert kind == "info"
    for part in ("größte Abweichung 1.8e-11", "bei n = 30 Punkten 180 statt 90 Zeilen", "(31 gegen 33)"):
        assert part in text, part


def test_rotation_verdict_names_best_and_worst_angle():
    rows = [{"angle": 0, "median": 0.875, "min": 0.79, "max": 0.95}, {"angle": 45, "median": 0.532, "min": 0.52, "max": 0.64}, {"angle": 90, "median": 0.875, "min": 0.79, "max": 0.95}]
    kind, text = ev.verdict_rotation(rows, 4.0)
    assert kind == "info"
    for part in ("um den Faktor 4,0 gestreckten", "**53,2 %**", "um 45° gedreht", "87,5 % bei 0°"):
        assert part in text, part


def test_hand_verdict_accepts_the_hand_values_and_names_every_deviation():
    ok = {"cube": 1.0, "diamond": 2.0, "ball_soc": math.sqrt(2), "ball_psd": math.sqrt(2), "ellipse_axis": 4.0, "ellipse_psd": 2.5}
    assert ev.verdict_hand(ok)[0] == "success"
    bad = dict(ok, ellipse_psd=2.6, diamond=2.0000001)
    kind, text = ev.verdict_hand(bad)
    assert kind == "error" and "Ellipse, allgemein (1.0e-01)" in text and "Raute" not in text      # 1e-7 liegt unter der Grenze 1e-6
    assert ev.verdict_hand(dict(ok, cube=1.000002))[0] == "error"
