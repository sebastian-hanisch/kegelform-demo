"""Pfade der App, die der Rauchtest nicht erreicht: Experimente 2 und 4, PDF, jede Form, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kgf_algorithm as A
import kgf_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        (at.sidebar.selectbox if key == "form_select" else at.sidebar.number_input if key == "seed_input" else at.sidebar.slider)(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_standard_numbers():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Radius"] == "7,1675" and values["Variablen"] == "3" and values["Zeilen von A"] == "60" and values["Bewiesene Lücke"].endswith("⁻¹⁰")
    text = _texts(at)
    assert "einem Kegelprogramm mit 3 Variablen und 60 Zeilen" in text and "Alle sechs Werte stimmen mit der Handrechnung überein" in text


def test_experiment_2_runs_and_names_the_directions():
    at = _app()
    assert any("Noch nicht gerechnet" in i.value for i in at.info)
    at.button(key="lp_run").click().run()
    assert not at.exception
    assert "d = 2: 16 Richtungen; d = 3: 256 Richtungen; d = 4: 2048 Richtungen" in _texts(at)


def test_experiment_4_runs_and_names_the_best_angle():
    at = _app()
    at.button(key="rot_run").click().run()
    assert not at.exception
    assert "um 45° gedreht" in _texts(at) and "**53,2 %**" in _texts(at)


def test_experiment_4_result_is_dropped_when_the_point_number_changes():
    at = _app()
    at.button(key="rot_run").click().run()
    at.sidebar.slider(key="n_slider").set_value(12).run()
    assert not at.exception and any("Noch nicht gerechnet für diese Punktzahl" in i.value for i in at.info)


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("form", A.FORMS)
def test_every_form_renders_without_error_and_passes_the_back_check(form):
    at = _app(form_select=form)
    assert not at.error
    assert any("Die Rückumformung besteht" in s.value for s in at.success)


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sliders_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"n_slider": pick(C.N_MIN, C.N_MAX), "stretch_slider": pick(C.STRETCH_MIN, C.STRETCH_MAX), "angle_slider": pick(C.ANGLE_MIN, C.ANGLE_MAX)}
    at = _app(**state)
    assert not at.exception and not at.error


def test_the_smallest_cloud_works_in_every_form():
    for form in A.FORMS:
        at = _app(form_select=form, n_slider=C.N_MIN)
        assert not at.error, form
